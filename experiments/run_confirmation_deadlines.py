"""Replay the frozen confirmation panel at four deadlines on the NN source tree."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys


DEADLINES = (20, 40, 60, 100)
ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "confirm_strong_20261119_v1" / "report.json"


def evaluate(entry, source_root):
    sys.path.insert(0, str(source_root))
    from experiments.compare_erasure_baselines import rank
    from experiments.run_box_erasure_scale import MarginalIntervals, SATURATION_MARGIN
    from experiments.run_core_scale_comparison import weights
    from experiments.run_extended_size_certificates import ExtendedSpec
    from experiments.run_locked_erasure_scale import MODEL, plans_at

    n, family, seed = entry["n"], entry["family"], entry["seed"]
    matrix = weights(n, family, seed)
    digest = sha256(json.dumps(matrix).encode()).hexdigest()
    if digest != entry["matrix_sha256"]:
        raise ValueError(f"Unexpected coupling matrix at seed {seed}")
    checker = MarginalIntervals(matrix)
    cap = max(map(sum, matrix)) + SATURATION_MARGIN
    spec = ExtendedSpec(MODEL, matrix)
    records = {}
    for deadline in DEADLINES:
        candidates = plans_at(spec, deadline)
        topology = candidates["locked"]
        row = candidates["row_tight"]
        box = checker.search(deadline, cap)
        baseline = min((p for p in (row, box) if p is not None), key=rank, default=None)
        records[deadline] = dict(topology=topology, row=row, box=box, baseline=baseline)
        if any(sum(p[1:]) > deadline for p in (topology, row, box) if p is not None):
            raise ValueError(f"Certificate missed the deadline at seed {seed}, T={deadline}")
        if deadline == 60:
            for arm, plan in records[deadline].items():
                if json.dumps(plan) != json.dumps(entry[arm]):
                    raise ValueError(f"Confirmation mismatch at seed {seed}, arm {arm}")
    return dict(n=n, family=family, seed=seed, matrix_sha256=digest, plans=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    archive = ARCHIVE.read_bytes()
    archived_report = json.loads(archive)
    root = args.source_root.resolve()
    raw_design = (root / "results" / ARCHIVE.parent.name / "design.json").read_bytes()
    if sha256(raw_design).hexdigest() != archived_report["design_sha256"]:
        raise ValueError("Unexpected frozen design")
    design = json.loads(raw_design)
    for path, digest in design["source_sha256"].items():
        if sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {path}")
    entries = archived_report["rows"]
    if archived_report["incomplete"] or len(entries) != 72 or any(not r["complete"] for r in entries):
        raise ValueError("Expected a complete 72-graph confirmation archive")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, entry, root) for entry in entries]
        for i, future in enumerate(futures, 1):
            rows.append(future.result())
            if i % 8 == 0:
                print(f"Validated T=60 archive and recomputed four deadlines: {i}/{len(entries)} graphs.",
                      flush=True)
    report = dict(
        status="exploratory deadline sensitivity on the frozen 72-graph confirmation panel",
        archive_sha256=sha256(archive).hexdigest(),
        design_sha256=sha256(raw_design).hexdigest(),
        deadlines=DEADLINES,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
