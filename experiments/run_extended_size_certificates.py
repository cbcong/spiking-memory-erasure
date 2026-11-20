"""Larger paired graph-certificate benchmark with fixed network-step budgets."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.run_core_scale_comparison import FAMILIES, ROOT, audit, examine, summary, weights
from integer_snn.network import NetworkSpec
from integer_snn.topology_certificate import MODEL


class ExtendedSpec(NetworkSpec):
    """Experiment-only limits; the fixed model and integer-weight domain are unchanged."""

    def __post_init__(self):
        if self.params != MODEL:
            raise ValueError("benchmark requires the fixed topology model")
        if not isinstance(self.weights, tuple) or not 1 <= len(self.weights) <= 512:
            raise ValueError("expected a tuple matrix with 1..512 modules")
        n = len(self.weights)
        for row in self.weights:
            if not isinstance(row, tuple) or len(row) != n:
                raise ValueError("expected square tuple matrix")
            if any(type(value) is not int or value < 0 for value in row):
                raise ValueError("expected nonnegative integer weights")


SEED = 2026111919
BUDGETS = (20, 40, 60, 100)


def aggregate(rows):
    result = summary(rows)
    result["distinct_matrices"] = len({r["matrix_sha256"] for r in rows})
    result["fixed_budgets"] = {}
    for cutoff in BUDGETS:
        baseline_sum = 0
        selected_sum = 0
        wins = 0
        for row in rows:
            base, proposed = row["base"], row["proposed"]
            if sum(base[1:]) > cutoff:
                raise ValueError("baseline infeasible; comparison denominator must be revised")
            winner = base[0]
            if sum(proposed[1:]) <= cutoff:
                winner = min(winner, proposed[0])
            wins += winner < base[0]
            selected_sum += winner
            baseline_sum += base[0]
        result["fixed_budgets"][cutoff] = dict(wins=wins, amplitude_sum=selected_sum,
                                              base_sum=baseline_sum, reduction=1-selected_sum/baseline_sum)
    return result


def run(sizes, replicates, audit_sizes):
    rows = []
    for n in sizes:
        for index, family in enumerate(FAMILIES):
            for trial in range(replicates):
                seed = SEED + n*10000 + index*100 + trial
                matrix = weights(n, family, seed)
                spec = ExtendedSpec(MODEL, matrix)
                row = dict(n=n, family=family, trial=trial, seed=seed, **examine(spec))
                row["matrix_sha256"] = hashlib.sha256(json.dumps(matrix).encode()).hexdigest()
                verify = trial == 0 and n in audit_sizes
                row["fraction_updates"] = audit(spec, row, seed) if verify else 0
                rows.append(row)
            print(f"completed {n} modules, {family}, {replicates} matrices", flush=True)
    names = ["experiments/run_extended_size_certificates.py", "experiments/run_core_scale_comparison.py",
             "experiments/run_core_endpoint_witnesses.py", "experiments/run_topology_certificate.py",
             "integer_snn/core_erasure.py", "integer_snn/topology_certificate.py",
             "integer_snn/model.py", "integer_snn/network.py", "integer_snn/reference.py"]
    return dict(seed=SEED, sizes=sizes, replicates=replicates, families=FAMILIES, audit_sizes=audit_sizes,
                model=repr(MODEL), certificate_domain="all legal initial states, no writes/noise",
                fraction_trajectories=4*sum(r["fraction_updates"]>0 for r in rows),
                fraction_updates=sum(r["fraction_updates"] for r in rows),
                sizes_by_family={n:{f:aggregate([r for r in rows if r["n"]==n and r["family"]==f])
                                   for f in FAMILIES} for n in sizes},
                by_size={n:aggregate([r for r in rows if r["n"]==n]) for n in sizes},
                overall=aggregate(rows), sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                                                  for f in names}, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--audit-sizes", nargs="*", type=int, default=[128, 256, 512])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100 or not all(1 <= n <= 512 for n in args.sizes):
        parser.error("sizes must be 1..512 and replicates must be 1..100")
    if len(set(args.sizes)) != len(args.sizes) or not set(args.audit_sizes) <= set(args.sizes):
        parser.error("sizes must be distinct and audit sizes must be included in sizes")
    if args.output.exists():
        parser.error("output exists")
    result = run(args.sizes, args.replicates, args.audit_sizes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
