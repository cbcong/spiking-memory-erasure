"""Find exact minimum finite-deadline peaks on two three-module non-ring graphs."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pycosat

from experiments.run_box_erasure_scale import MarginalIntervals, SATURATION_MARGIN
from experiments.run_core_endpoint_witnesses import COUNTEREXAMPLE
from experiments.run_external_sat_erasure import query
from experiments.run_free_duration_disjoint import replay_checks
from experiments.run_free_duration_sat import AMPLITUDE_MAX, PROP_LIMIT, solve
from experiments.run_locked_erasure_scale import plans_at, rank
from integer_snn.network import NetworkSpec
from integer_snn.topology_certificate import MODEL


ROOT = Path(__file__).resolve().parents[1]
DEADLINES = (18, 20, 24)
CASES = (
    ("feedback_fork", COUNTEREXAMPLE),
    ("feedforward_merge", ((0, 0, 0), (3, 0, 0), (2, 3, 0))),
)


def run():
    records = []
    for label, weights in CASES:
        spec = NetworkSpec(MODEL, weights)
        marginal = MarginalIntervals(weights)
        for deadline in DEADLINES:
            start = perf_counter()
            exact = solve(spec, deadline)
            count = replay_checks(spec, deadline, exact)
            if not exact["minimum_proved"] or exact["unknown"]:
                raise RuntimeError(f"Unresolved optimum: {label}, T={deadline}")
            minimum = exact["first_feasible"]["e"]

            certified = plans_at(spec, deadline)
            marginal_plan = marginal.search(deadline, max(spec.row_sums) + SATURATION_MARGIN)
            comparator = min((certified["row_tight"], marginal_plan), key=rank)
            plans = dict(row=certified["row_tight"], marginal=marginal_plan,
                         comparator=comparator, net_inhibition=certified["timed"],
                         proposed=certified["locked"])
            gaps = {name: plan[0] - minimum for name, plan in plans.items()}
            selected_checks = []
            for plan in dict.fromkeys(plans.values()):
                if plan[0] < minimum or sum(plan[1:]) > deadline:
                    raise AssertionError((label, deadline, plan))
                check = query(spec, *plan, PROP_LIMIT)
                if check["status"] != "UNSAT":
                    raise RuntimeError((label, deadline, plan, check))
                selected_checks.append(dict(plan=plan, **check))
            print(label, deadline, exact["first_feasible"], gaps, flush=True)
            records.append(dict(graph=label, weights=weights, deadline=deadline, exact=exact,
                                total_replayed=count, plans=plans, gaps=gaps,
                                independent_schedule_checks=selected_checks,
                                wall_seconds=perf_counter() - start))

    sources = (
        Path(__file__),
        ROOT / "experiments/run_free_duration_sat.py",
        ROOT / "experiments/run_free_duration_disjoint.py",
        ROOT / "experiments/run_external_sat_erasure.py",
        ROOT / "experiments/run_locked_erasure_scale.py",
        ROOT / "experiments/run_box_erasure_scale.py",
        ROOT / "integer_snn/network.py",
        ROOT / "integer_snn/reference.py",
        ROOT / "integer_snn/topology_certificate.py",
    )
    return dict(
        domain="all legal post-update states of the specified fixed three-module model",
        objective="minimize E under D+H <= deadline; search D=0..T and H=T-D",
        solver=f"PicoSAT via pycosat {pycosat.__version__}",
        amplitude_cap=AMPLITUDE_MAX, propagation_limit=PROP_LIMIT, records=records,
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        parser.error("choose a new result path")
    result = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
