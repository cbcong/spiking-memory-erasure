"""Free-duration PicoSAT amplitude search for a common erasure deadline."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pycosat

from experiments.run_core_scale_comparison import examine
from experiments.run_external_sat_erasure import query
from integer_snn.model import State, zero_state
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL, certify_row_sum_erasure


# Full factorial of three preselected weight levels up to exchange of the two modules.
WEIGHTS = (0, 1, 3)
DEADLINES = (18, 24)
AMPLITUDE_MAX = 8
PROP_LIMIT = 3_000_000


def protocol_library(spec):
    choices = examine(spec)
    fast = certify_row_sum_erasure(spec, shorter_pulse=True)
    return dict(row=choices["base"], topology=choices["proposed"],
                short_row=[fast["amplitude"], fast["duration"], fast["recovery"]])


def bad_durations(spec, start, amplitude, horizon, durations):
    """Replay each requested pulse length with the independent reference dynamics."""
    zero = (zero_state(),) * spec.module_count
    pressure = start
    needed = set(durations)
    disproved = []
    for d in range(max(needed) + 1):
        if d in needed:
            released = pressure
            for _ in range(horizon - d):
                released = network_reference_step(spec, released)
                if released == zero:
                    break
            if released != zero:
                disproved.append(d)
        if d < max(needed):
            pressure = network_reference_step(spec, pressure, erase=amplitude)
    return disproved


def solve(spec, horizon):
    witnesses, decisions, remaining_gaps = [], [], []
    found = None
    for amplitude in range(AMPLITUDE_MAX + 1):
        pending = set(range(horizon + 1))
        # A single E=0 sequence represents every duration; assign all its statuses below.
        for wi, witness in enumerate(witnesses):
            if not pending:
                break
            rejected = bad_durations(spec, witness, amplitude, horizon, pending)
            decisions.extend(dict(e=amplitude, d=d, status="replay", witness=wi) for d in rejected)
            pending.difference_update(rejected)
        while pending:
            d = min(pending)
            r = query(spec, amplitude, d, horizon - d, PROP_LIMIT)
            status = r["status"]
            entry = dict(e=amplitude, d=d, status=status,
                         encoding_s=r["encoding_s"], solving_s=r["solving_s"])
            decisions.append(entry)
            pending.remove(d)
            if status == "UNSAT":
                found = dict(e=amplitude, d=d, h=horizon-d)
                break
            if status == "UNKNOWN":
                remaining_gaps.append([amplitude, d])
                continue
            if status != "SAT":
                raise ValueError(status)
            witness = tuple(State(tuple(x["v"]), tuple(x["current"]), tuple(x["spike"]))
                            for x in r["initial"])
            wi = len(witnesses)
            witnesses.append(witness)
            entry["witness"] = wi
            if pending:
                rejected = bad_durations(spec, witness, amplitude, horizon, pending)
                decisions.extend(dict(e=amplitude, d=j, status="replay", witness=wi) for j in rejected)
                pending.difference_update(rejected)
        if found:
            break
    if found:
        optimal = not any(e < found["e"] for e, _ in remaining_gaps)
        if optimal:
            for e in range(found["e"]):
                proved = {x["d"] for x in decisions if x["e"] == e and x["status"] in ("SAT","replay")}
                assert proved == set(range(horizon+1))
    else:
        optimal = False
    return dict(first_feasible=found, minimum_proved=optimal, unknown=remaining_gaps,
                counts=Counter(x["status"] for x in decisions), checks=decisions,
                witnesses=[[asdict(s) for s in w] for w in witnesses])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        parser.error("choose a new result path")
    rows = []
    for a in WEIGHTS:
        for b in WEIGHTS:
            if a > b:
                continue
            spec = NetworkSpec(MODEL, ((0, a), (b, 0)))
            plans = protocol_library(spec)
            for cutoff in DEADLINES:
                feasible = [(e,d,h,name) for name,(e,d,h) in plans.items() if d+h<=cutoff]
                start = perf_counter()
                answer = solve(spec, cutoff)
                rows.append(dict(weights=spec.weights, horizon=cutoff, plans=plans,
                                 best_known_library=min(feasible) if feasible else None,
                                 solver_seconds=perf_counter()-start, result=answer))
                print(a,b,cutoff,answer["first_feasible"],answer["minimum_proved"],flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_free_duration_sat.py","experiments/run_external_sat_erasure.py",
               "experiments/run_core_scale_comparison.py","integer_snn/network.py",
               "integer_snn/reference.py","integer_snn/topology_certificate.py",
               "integer_snn/core_erasure.py","experiments/run_core_endpoint_witnesses.py"]
    report = dict(deadlines=DEADLINES, weights=WEIGHTS, amplitude_max=AMPLITUDE_MAX,
                  prop_limit_per_query=PROP_LIMIT, solver=f"PicoSAT via pycosat {pycosat.__version__}",
                  domain="all legal post-update states; fixed parameters; no new input or noise",
                  search="D=0..T, H=T-D; zero is absorbing after release; ascending E",
                  cases=rows, sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in sources})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report,stream,indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
