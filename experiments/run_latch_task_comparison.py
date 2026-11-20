"""Compare observable latch behavior using windows in native firing periods."""
import argparse
from hashlib import sha256
import json
from math import floor
from pathlib import Path
from statistics import median

from experiments import run_spaeth_latch as spaeth
from experiments.run_locked_erasure_scale import plans_at
from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec
from integer_snn.topology_certificate import MODEL
from tasks.delayed_query import advance, fresh_network


ROOT = Path(__file__).resolve().parents[1]
SPEC = NetworkSpec(MODEL, ((0,),))
ARMS = ("full", "skip_reset", "skip_rewrite")


def native_period(events, start):
    times = [t for t, cell in events if t >= start and cell == 0]
    if len(times) < 5:
        raise ValueError("insufficient unforced E1 spikes for period calibration")
    intervals = [b - a for a, b in zip(times, times[1:])]
    return dict(period=median(intervals), intervals=intervals)


def integer_trace(times, arm, plan):
    reset, rewrite, finish = times
    state = fresh_network(SPEC)
    amplitude, pressure, recovery = plan
    events = []
    zero_at_certificate = None
    for t in range(finish):
        inputs = ((14, 0),) if t < 2 or (arm != "skip_rewrite" and rewrite <= t < rewrite+2) else None
        erase = amplitude if arm != "skip_reset" and reset <= t < reset+pressure else 0
        state = advance(SPEC, state, 1, inputs=inputs, erase=erase, reference=True)
        # Timestamp is the update slot [t,t+1). Commands are set at its left edge.
        events.extend((t, cell) for cell, fired in enumerate(state[0].spike) if fired)
        if t+1 == reset+pressure+recovery:
            zero_at_certificate = state == (zero_state(),)
    return events, zero_at_certificate


def spaeth_trace(times, arm, amplitudes, tol=1e-8):
    reset, rewrite, finish = times
    commands = [(0., 0), (reset, 2 if arm != "skip_reset" else None),
                (rewrite, 0 if arm != "skip_rewrite" else None)]
    state, events = spaeth.resting(), []
    for (start, cell), stop in zip(commands, (reset, rewrite, finish)):
        pulse_end = start + spaeth.PULSE_MS
        state = spaeth.evolve(state, start, pulse_end, 5., events, cell,
                              pulse_pa=amplitudes.get(cell, 0.), tol=tol)
        state = spaeth.evolve(state, pulse_end, stop, 5., events, tol=tol)
    return events, None


def collect(period, integer, trace):
    rows = []
    # Each integer boundary is placed at the preceding update edge. The
    # 16 nominal offsets therefore collapse to two distinct integer schedules.
    edge = floor if integer else float
    for phase in range(16):
        reset = edge((12 + phase/16) * period)
        times = (reset, reset+edge(12*period), reset+2*edge(12*period))
        windows = [(t-edge(5*period), t) for t in times]
        for arm in ARMS:
            events, exact_zero = trace(times, arm)
            counts = [spaeth.count(events, lo, hi, (0, 1)) for lo, hi in windows]
            expected = {"full": [True, False, True], "skip_reset": [True, True, True],
                        "skip_rewrite": [True, False, False]}[arm]
            active = [c > 0 for c in counts]
            rows.append(dict(phase=phase, arm=arm, edges=times, windows=windows,
                             excitatory_counts=counts, observed=active,
                             on_off_on=active==[True, False, True], matches_arm=active==expected,
                             exact_zero_at_guaranteed_time=exact_zero,
                             events=events, all_cell_spikes=len(events)))
    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"]==arm]
        totals = [r["all_cell_spikes"] for r in selected]
        summary[arm] = dict(n=len(selected), distinct_schedules=len(set(r["edges"] for r in selected)),
                            on_off_on=sum(r["on_off_on"] for r in selected),
                            matches_arm=sum(r["matches_arm"] for r in selected),
                            phase_pass_counts=[sum(r["observed"][j] == target for r in selected)
                                               for j, target in enumerate([True, False, True])],
                            all_cell_spikes_min_median_max=[min(totals), median(totals), max(totals)])
    return dict(summary=summary, rows=rows)


def main(out):
    design = ROOT / "configs/latch_task_comparison_20261119.json"
    settings = json.loads(design.read_text(encoding="utf8"))
    if settings["hold_cycles"] != 12 or settings["window_cycles"] != 5 or settings["phase_count"] != 16:
        raise ValueError("this protocol implements 12/5/16 only")
    if out.exists():
        raise FileExistsError(out)
    plans = plans_at(SPEC, 60)
    plan = plans["locked"]
    amplitudes, _ = spaeth.calibrate()
    events = []
    state = spaeth.evolve(spaeth.resting(), 0., 2., 5., events, 0, pulse_pa=amplitudes[0])
    spaeth.evolve(state, 2., 1000., 5., events)
    bio = native_period(events, 500.)
    state = advance(SPEC, fresh_network(SPEC), 2, inputs=((14, 0),), reference=True)
    events = []
    for tick in range(2, 300):
        state = advance(SPEC, state, 1, reference=True)
        if state[0].spike[0]:
            events.append((tick, 0))
    discrete = native_period(events, 150)
    result = dict(design=settings, design_sha256=sha256(design.read_bytes()).hexdigest(),
                  author_source_sha256=sha256(spaeth.NOTEBOOK.read_bytes()).hexdigest(),
                  source_hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__).resolve(), Path(spaeth.__file__).resolve()]},
                  spaeth_current_pA=amplitudes, integer_plan=plan, integer_row=plans["row_tight"],
                  period_calibration=dict(spaeth_ms=bio, integer_ticks=discrete),
                  integer=collect(discrete["period"], True, lambda t,a: integer_trace(t,a,plan)),
                  spaeth=collect(bio["period"], False, lambda t,a: spaeth_trace(t,a,amplitudes)))
    checks = []
    for row in result["spaeth"]["rows"]:
        if row["phase"] in (0, 5, 10, 15):
            strict, _ = spaeth_trace(row["edges"], row["arm"], amplitudes, tol=1e-10)
            counts = [spaeth.count(strict, lo, hi, (0, 1)) for lo, hi in row["windows"]]
            checks.append(dict(phase=row["phase"], arm=row["arm"], strict_counts=counts,
                               counts_agree=counts == row["excitatory_counts"]))
    result["tolerance_checks"] = checks
    out.parent.mkdir(exist_ok=True, parents=True)
    with out.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    print({m:result[m]["summary"] for m in ("integer","spaeth")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
