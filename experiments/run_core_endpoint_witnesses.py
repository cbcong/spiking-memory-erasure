"""Certify endpoint feasibility using bounded consecutive spike runs."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.run_topology_certificate import initials
from integer_snn.core_erasure import core_erasure_bounds
from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "results/topology_certificate_20261113_v1/case_results.json"
SEED = 2026111917
COUNTEREXAMPLE = ((0, 0, 1), (0, 0, 1), (2, 2, 0))


def endpoint_certificate(spec, amplitude):
    """All legal states; press and release have no external writes or noise."""
    if spec.params != MODEL or type(amplitude) is not int or amplitude < 3:
        raise ValueError("requires the fixed model and an integer amplitude >=3")
    weights = spec.weights
    remaining = set(range(spec.module_count))
    start = 0
    stages = []
    while remaining:
        nodes = sorted(remaining)
        sums = {i: sum(weights[i][j] for j in nodes) for i in nodes}
        runs = {i: 1 for i in nodes if sums[i] <= amplitude - 2}
        via = {i: None for i in runs}
        # Breadth-first levels keep every dependency on a preceding finite bound.
        while True:
            additions = {}
            for i in nodes:
                if i in runs:
                    continue
                candidates = [(b + 1, j) for j, b in runs.items()
                              if sums[i] - weights[i][j] < amplitude - 1]
                if candidates:
                    additions[i] = min(candidates)
            if not additions:
                break
            for i, (bound, source) in additions.items():
                runs[i], via[i] = bound, source
        eliminated = {}
        for i in nodes:
            if sums[i] <= amplitude - 3:
                eliminated[i] = dict(offset=4, required_run_source=None)
            elif sums[i] == amplitude - 2:
                candidates = [(bound, j) for j, bound in runs.items() if weights[i][j] > 0]
                if candidates:
                    bound, source = min(candidates)
                    eliminated[i] = dict(offset=6 + 2 * (bound // 2), required_run_source=source)
        stages.append(dict(start=start, residual_row_sums=sums, run_bounds=runs,
                           run_bound_parents=via, eliminated=eliminated))
        if not eliminated:
            return dict(certified=False, stages=stages, unresolved=nodes)
        last = start + max(record["offset"] for record in eliminated.values())
        start = last + 1
        remaining.difference_update(eliminated)
    return dict(certified=True, stages=stages, last_spike_upper=last, duration=last+4,
                recovery=10)


def cycle(spec, initial, erase):
    orbit = []
    seen = {}
    state = initial
    while state not in seen:
        if len(orbit) >= 256:
            raise RuntimeError("counterexample did not close within diagnostic cap")
        seen[state] = len(orbit)
        orbit.append(state)
        state = network_reference_step(spec, state, erase=erase)
    return orbit, seen[state]


def infinite_witness(spec):
    if spec.weights != COUNTEREXAMPLE:
        raise ValueError("witness applies to the named 3-node motif")
    zero = (zero_state(),) * 3
    state = zero
    for t in range(1, 10):
        active = (t in (1, 2), t in (2, 3), t <= 5)
        state = network_reference_step(spec, state, inputs=tuple((14*on, 0) for on in active))
    orbit, entry = cycle(spec, state, 3)
    released = []
    for duration, phase in enumerate(orbit):
        release, first = cycle(spec, phase, 0)
        if zero in release or not any(any(any(s.spike) for s in x) for x in release[first:]):
            raise AssertionError(("release reached zero or lost firing", duration))
        released.append(dict(duration=duration, transient=first, period=len(release)-first))
    return dict(written_from_zero=True, pressure_transient=entry,
                pressure_period=len(orbit)-entry, release_all_phases=released)


def sample(spec, case_id, e, duration):
    state_count = 0
    initial_states = initials(spec.module_count, SEED + case_id)
    zero = (zero_state(),) * spec.module_count
    last = 0
    for state in initial_states:
        for t in range(1, duration + 11):
            state = network_reference_step(spec, state, erase=e if t <= duration else 0)
            if any(any(x.spike) for x in state):
                last = max(last, t)
        if state != zero:
            raise AssertionError(("protocol failure", case_id, state))
        state_count += 1
    return dict(initial_states=state_count, fraction_network_updates=state_count*(duration+10),
                latest_sampled_spike=last)


def run():
    rows = []
    for case in json.loads(ATLAS.read_bytes()):
        spec = NetworkSpec(MODEL, tuple(map(tuple, case["weights"])))
        bounds = core_erasure_bounds(spec)
        cert = endpoint_certificate(spec, bounds["lower"])
        old = case["topology_certificate"].get("minimum_amplitude")
        if old is not None and cert["certified"] and old != bounds["lower"]:
            raise AssertionError(("conflicting full-domain thresholds", case["id"]))
        row = dict(id=case["id"], label=case["label"], lower=bounds["lower"],
                   previous_exact=old, endpoint=cert)
        if cert["certified"]:
            row["exact"] = bounds["lower"]
            if old is None:
                row["audit"] = sample(spec, case["id"], bounds["lower"], cert["duration"])
                if row["audit"]["latest_sampled_spike"] > cert["last_spike_upper"]:
                    raise AssertionError(("last spike violated bound",case["id"]))
        elif spec.weights == COUNTEREXAMPLE:
            if (bounds["lower"], bounds["upper"]) != (3, 4):
                raise AssertionError("unexpected motif bracket")
            row["obstruction"] = infinite_witness(spec)
            row["exact"] = bounds["upper"]
            row["audit"] = sample(spec, case["id"], bounds["upper"], bounds["duration_at_upper"])
        else:
            row["exact"] = old
        rows.append(row)
    sources = [ATLAS, Path(__file__), ROOT / "experiments/run_topology_certificate.py",
               ROOT / "integer_snn/core_erasure.py", ROOT / "integer_snn/topology_certificate.py",
               ROOT / "integer_snn/network.py", ROOT / "integer_snn/model.py"]
    return dict(seed=SEED, source_hashes={p.relative_to(ROOT).as_posix():
                hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    dest = parser.parse_args().output
    report = run()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, separators=(",", ":"))
        stream.write("\n")
