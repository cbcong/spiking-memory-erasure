"""Solve the disjoint two-module benchmark and replay its counterexamples."""

import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
from time import perf_counter

import pycosat

from experiments.run_free_duration_sat import AMPLITUDE_MAX, DEADLINES, PROP_LIMIT, solve
from integer_snn.model import State, zero_state
from integer_snn.network import NetworkSpec, network_reference_step, validate_network_state
from integer_snn.topology_certificate import MODEL


WEIGHT_LEVELS = (2, 4, 5)


def replay_checks(spec, deadline, answer):
    witnesses = [
        tuple(State(tuple(s["v"]), tuple(s["current"]), tuple(s["spike"])) for s in witness)
        for witness in answer["witnesses"]
    ]
    for witness in witnesses:
        validate_network_state(spec, witness)

    zero = (zero_state(),) * spec.module_count
    disproved = set()
    for check in answer["checks"]:
        if check["status"] not in ("SAT", "replay"):
            continue
        amplitude, duration = check["e"], check["d"]
        state = witnesses[check["witness"]]
        for tick in range(deadline):
            state = network_reference_step(spec, state, erase=amplitude if tick < duration else 0)
        if state == zero:
            raise AssertionError((spec.weights, deadline, check))
        disproved.add((amplitude, duration))

    if answer["minimum_proved"]:
        plan = answer["first_feasible"]
        expected = {
            (amplitude, duration)
            for amplitude in range(plan["e"])
            for duration in range(deadline + 1)
        }
        if not expected <= disproved or plan["d"] + plan["h"] != deadline:
            raise AssertionError((spec.weights, deadline, plan))
        if not any(
            c["status"] == "UNSAT" and c["e"] == plan["e"] and c["d"] == plan["d"]
            for c in answer["checks"]
        ):
            raise AssertionError((spec.weights, deadline, plan))
    return len(disproved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        parser.error("choose a new result path")

    rows = []
    for a, b in combinations_with_replacement(WEIGHT_LEVELS, 2):
        spec = NetworkSpec(MODEL, ((0, a), (b, 0)))
        for deadline in DEADLINES:
            start = perf_counter()
            answer = solve(spec, deadline)
            solver_seconds = perf_counter() - start
            checked = replay_checks(spec, deadline, answer)
            rows.append(
                dict(weights=spec.weights, horizon=deadline, result=answer,
                     solver_seconds=solver_seconds, independently_replayed=checked)
            )
            print(a, b, deadline, answer["first_feasible"], answer["minimum_proved"],
                  "replayed", checked, flush=True)

    root = Path(__file__).resolve().parents[1]
    sources = (
        "experiments/run_free_duration_disjoint.py",
        "experiments/run_free_duration_sat.py",
        "experiments/run_external_sat_erasure.py",
        "integer_snn/model.py",
        "integer_snn/network.py",
        "integer_snn/reference.py",
        "integer_snn/topology_certificate.py",
    )
    report = dict(
        weight_levels=WEIGHT_LEVELS,
        deadlines=DEADLINES,
        amplitude_max=AMPLITUDE_MAX,
        prop_limit_per_query=PROP_LIMIT,
        solver=f"PicoSAT via pycosat {pycosat.__version__}",
        model=asdict(MODEL),
        domain="all legal post-update states; fixed model; no new input or noise",
        search="D=0..T, H=T-D; zero is absorbing after release; ascending E",
        cases=rows,
        sha256={source: hashlib.sha256((root / source).read_bytes()).hexdigest()
                for source in sources},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
