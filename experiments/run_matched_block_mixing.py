"""Run a fixed directed-block experiment paired on incoming weight profiles."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

from experiments.compare_erasure_baselines import rank, summarize
from experiments.run_box_erasure_scale import MarginalIntervals, SATURATION_MARGIN
from experiments.run_locked_erasure_scale import audit, plans_at
from integer_snn.network import NetworkSpec
from integer_snn.topology_certificate import MODEL


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "results/matched_block_mixing_v1/design.json"


def wiring(seed, arm, profile, blocks):
    rng = random.Random(f"{seed}:{arm}")
    n = len(profile)
    block_size = n // blocks
    matrix = []
    within = 0
    for target, weights in enumerate(profile):
        row = [0] * n
        available = [j for j in range(n) if j != target]
        for weight in weights:
            if arm == "block":
                same = rng.random() < .8
                sources = [j for j in available
                           if (j // block_size == target // block_size) == same]
            else:
                sources = available
            j = rng.choice(sources)
            available.remove(j)
            row[j] = weight
            within += (j // block_size == target // block_size)
        if row[target] or sorted(v for v in row if v) != sorted(weights):
            raise AssertionError((target, weights, row))
        matrix.append(tuple(row))
    return tuple(matrix), within


def main(design_path, output):
    design = json.loads(design_path.read_bytes())
    graph_budget = design["deadline"]
    rows = []
    contrasts = []
    hashes = set()
    for seed in range(design["first_seed"], design["first_seed"] + design["pairs"]):
        rng = random.Random(f"{seed}:profile")
        profile = [[rng.randint(1, 4) for _ in range(rng.randint(1, 4))]
                   for _ in range(design["modules"])]
        pair = []
        for arm in design["arms"]:
            matrix, within = wiring(seed, arm, profile, design["blocks"])
            digest = hashlib.sha256(json.dumps(matrix).encode()).hexdigest()
            if digest in hashes:
                raise AssertionError("Repeated graph")
            hashes.add(digest)
            spec = NetworkSpec(MODEL, matrix)
            plans = plans_at(spec, graph_budget)
            box = MarginalIntervals(matrix).search(graph_budget, spec.max_row_sum + SATURATION_MARGIN)
            locked, row = plans["locked"], plans["row_tight"]
            if any(p is None or sum(p[1:]) > graph_budget for p in (locked, row, box)):
                raise AssertionError((seed, arm, locked, row, box))
            strong = min((box, row), key=rank)
            pair.append(dict(seed=seed, arm=arm, matrix_sha256=digest, row=row, box=box,
                             topology=locked, strong=strong, peak_saving=strong[0]-locked[0],
                             within_edges=within, edges=sum(map(len, profile)),
                             replay=audit(spec, locked, seed)))
        global_row, block_row = pair
        if global_row["row"] != block_row["row"]:
            raise AssertionError("Matched profiles should have identical row plans")
        contrasts.append(block_row["peak_saving"] - global_row["peak_saving"])
        rows.extend(pair)
        print("completed pair", seed, flush=True)
    by_arm = {}
    for arm in design["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm]
        group = summarize(arm_rows)
        group["within_edges"] = sum(r["within_edges"] for r in arm_rows)
        group["edges"] = sum(r["edges"] for r in arm_rows)
        by_arm[arm] = group
    result = dict(design_sha256=hashlib.sha256(design_path.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summary=by_arm, paired_contrast=dict(values=contrasts, mean=mean(contrasts)),
                  rows=rows)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(by_arm), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DESIGN)
    parser.add_argument("--output", type=Path, default=DESIGN.with_name("report.json"))
    args = parser.parse_args()
    main(args.design, args.output)
