"""Prospective, local frozen-panel replication of the T=60 strong comparator."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from experiments.audit_locked_acceptance import interval
from experiments.compare_erasure_baselines import rank
from experiments.run_box_erasure_scale import MarginalIntervals, SATURATION_MARGIN
from experiments.run_core_scale_comparison import FAMILIES, ROOT, weights
from experiments.run_extended_size_certificates import ExtendedSpec
from experiments.run_locked_erasure_scale import MODEL, OLD_PANELS, SOURCE_FILES, plans_at, write


PRIOR = (*OLD_PANELS, "results/locked_erasure_scale_20261119_v1/report.json")
FILES = (*SOURCE_FILES, "experiments/run_confirm_strong.py", "experiments/run_box_erasure_scale.py",
         "experiments/compare_erasure_baselines.py", "experiments/audit_locked_acceptance.py")
DEADLINE = 60


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_hash(matrix):
    return hashlib.sha256(json.dumps(matrix).encode()).hexdigest()


def freeze(folder):
    old = set()
    for name in PRIOR:
        data = json.loads((ROOT / name).read_bytes())
        rows = data.get("rows", data.get("graphs")) if isinstance(data, dict) else data
        old.update(row["matrix_sha256"] for row in rows)
    panel = []
    seen = set(old)
    for n in (128, 256, 512):
        for family_index, family in enumerate(FAMILIES):
            for rep in range(8):
                seed = 731090000 + 10000*n + 100*family_index + rep
                hashed = matrix_hash(weights(n, family, seed))
                if hashed in seen:
                    raise ValueError("graph identity collision")
                seen.add(hashed)
                panel.append(dict(n=n, family=family, seed=seed, matrix_sha256=hashed))
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "design.json", dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        registration="local prospective freeze; no external registry",
        primary="T60 pooled 1-sum(locked peak)/sum(per-graph better(row_tight,box) peak)",
        ranking="peak, guaranteed total steps, pressure steps",
        population="9 equally sampled strata of 3 existing generators and 3 sizes",
        interval="percentile 95%, paired graph resampling within size x family, 10000 draws, seed 20261119",
        decision="positive primary if primary interval lower bound > 0 with full coverage; otherwise inconclusive",
        secondary="same statistic by family; descriptive unadjusted intervals; paired wins/ties/losses, D+H and E*D",
        missing="report all failed cases and no positive primary verdict on incomplete data",
        deadline=DEADLINE, model=repr(MODEL), prior_unique_graphs=len(old),
        prior_reports_sha256={name:digest(ROOT/name) for name in PRIOR},
        source_sha256={name:digest(ROOT/name) for name in FILES}, rows=panel))


def summary(rows):
    output = {}
    for field in ("row","box","baseline","topology"):
        plans = [r[field] for r in rows]
        output[field] = dict(peak_sum=sum(p[0] for p in plans),
                             mean_pressure=mean(p[1] for p in plans),
                             mean_time=mean(sum(p[1:]) for p in plans),
                             amplitude_pressure_sum=sum(p[0]*p[1] for p in plans))
    differences = [r["baseline"][0]-r["topology"][0] for r in rows]
    return dict(n=len(rows), protocols=output,
                reduction=1-output["topology"]["peak_sum"]/output["baseline"]["peak_sum"],
                paired_ci95=interval(rows),
                wins=sum(d>0 for d in differences), ties=differences.count(0),
                losses=sum(d<0 for d in differences))


def run(folder):
    path = folder / "design.json"
    design = json.loads(path.read_bytes())
    for name, hashed in {**design["source_sha256"], **design["prior_reports_sha256"]}.items():
        if digest(ROOT/name) != hashed:
            raise ValueError("source changed: " + name)
    assert design["deadline"] == DEADLINE and design["model"] == repr(MODEL)
    rows = []
    for entry in design["rows"]:
        matrix = weights(entry["n"],entry["family"],entry["seed"])
        assert matrix_hash(matrix) == entry["matrix_sha256"]
        plans = plans_at(ExtendedSpec(MODEL, matrix), DEADLINE)
        box = MarginalIntervals(matrix).search(DEADLINE,max(map(sum,matrix))+SATURATION_MARGIN)
        topo, row = plans["locked"], plans["row_tight"]
        outcome = dict(**entry, topology=topo, row=row, box=box)
        outcome["complete"] = all(p is not None and sum(p[1:])<=DEADLINE for p in (topo,row,box))
        if outcome["complete"]:
            outcome["baseline"] = min((row,box),key=rank)
        rows.append(outcome)
    complete = [r for r in rows if r["complete"]]
    totals = summary(complete) if complete else None
    positive = len(rows)==len(complete) and totals["paired_ci95"][0]>0
    result = dict(design_sha256=digest(path), completed_utc=datetime.now(timezone.utc).isoformat(),
                  incomplete=len(rows)-len(complete), primary_positive=positive, summary=totals,
                  by_family={f:summary([r for r in complete if r["family"]==f])
                             for f in FAMILIES if any(r["family"]==f for r in complete)}, rows=rows)
    write(folder / "report.json",result)
    print(json.dumps({k:v["reduction"] for k,v in result["by_family"].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze","run"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    (freeze if args.action=="freeze" else run)(args.output)
