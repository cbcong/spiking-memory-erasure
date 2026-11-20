"""Paired fixed-model erasure comparison on seeded 4--64-module networks."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from experiments.run_core_endpoint_witnesses import endpoint_certificate
from experiments.run_topology_certificate import initials
from integer_snn.core_erasure import core_erasure_bounds
from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL, certify_row_sum_erasure


ROOT = Path(__file__).resolve().parents[1]
SIZES = (4, 8, 16, 32, 64)
FAMILIES = ("sparse", "dense", "regular")
SEED = 2026111908
REPLICATES = 20


def weights(n, family, seed):
    rng = random.Random(seed)
    if family == "regular":
        # Equal weighted row sums by construction; cancellation gives an exact baseline.
        degrees = rng.randint(1, min(6, n-1))
        strength = rng.randint(1, 4)
        w = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in rng.sample([j for j in range(n) if j != i], degrees):
                w[i][j] = strength
        return tuple(map(tuple, w))
    probability = min(1, 2/(n-1)) if family == "sparse" else 0.4
    return tuple(tuple(rng.randint(1,4) if i != j and rng.random() < probability else 0
                       for j in range(n)) for i in range(n))


def examine(spec):
    base = certify_row_sum_erasure(spec)
    bound = core_erasure_bounds(spec)
    cert = endpoint_certificate(spec, bound["lower"])
    plans = [(bound["upper"], bound["duration_at_upper"], bound["recovery_at_upper"])]
    if cert["certified"]:
        plans.append((bound["lower"], cert["duration"], cert["recovery"]))
    amplitude, duration, recovery = min(plans)
    return dict(base=[base["amplitude"], base["duration"], base["recovery"]],
                proposed=[amplitude, duration, recovery], lower=bound["lower"],
                exact=amplitude == bound["lower"])


def audit(spec, plans, seed):
    starts = initials(spec.module_count, seed)
    total = 0
    for name in ("base", "proposed"):
        e, d, h = plans[name]
        for i in (3, 8):
            state = starts[i]
            for tick in range(d+h):
                state = network_reference_step(spec, state, erase=e if tick < d else 0)
            if state != (zero_state(),) * spec.module_count:
                raise AssertionError((name, seed, i))
            total += d+h
    return total


def summary(rows):
    base = [r["base"][0] for r in rows]
    prop = [r["proposed"][0] for r in rows]
    return dict(n=len(rows), lower_amplitude=sum(p < b for p,b in zip(prop,base)),
                exact=sum(r["exact"] for r in rows), base_amplitude=statistics.mean(base),
                new_amplitude=statistics.mean(prop), reduction=1-sum(prop)/sum(base),
                base_total=statistics.mean(sum(r["base"][1:]) for r in rows),
                new_total=statistics.mean(sum(r["proposed"][1:]) for r in rows))


def run():
    rows = []
    for n in SIZES:
        for f, family in enumerate(FAMILIES):
            for trial in range(REPLICATES):
                seed = SEED + n*10000 + f*100 + trial
                matrix = weights(n, family, seed)
                spec = NetworkSpec(MODEL, matrix)
                row = dict(n=n, family=family, trial=trial, seed=seed, **examine(spec))
                row["matrix_sha256"] = hashlib.sha256(json.dumps(matrix).encode()).hexdigest()
                row["fraction_updates"] = audit(spec, row, seed) if trial == 0 else 0
                rows.append(row)
    names = ["experiments/run_core_endpoint_witnesses.py", "experiments/run_core_scale_comparison.py",
             "experiments/run_topology_certificate.py", "integer_snn/core_erasure.py",
             "integer_snn/topology_certificate.py", "integer_snn/model.py",
             "integer_snn/network.py", "integer_snn/reference.py"]
    return dict(seed=SEED, replicates=REPLICATES, sizes=SIZES, families=FAMILIES,
                scope="no loops, same model, no writes/noise, two full-state starts per arm of each first graph",
                sources={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
                overall=summary(rows), by_size={n: summary([r for r in rows if r["n"]==n]) for n in SIZES},
                by_family={f: summary([r for r in rows if r["family"]==f]) for f in FAMILIES},
                fraction_trajectories=4*sum(r["fraction_updates"]>0 for r in rows),
                fraction_updates=sum(r["fraction_updates"] for r in rows), rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    result = run()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
