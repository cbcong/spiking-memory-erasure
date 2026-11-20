"""Tabulate the frozen 72-graph comparison of certified schedules at T=60."""

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


REPORT = Path(__file__).resolve().parents[1] / "results/confirm_strong_20261119_v1/report.json"
REPORT_SHA256 = "30300060e2a3a2989b8cb51d8e5edbe14d4bfc26d52b6b97b25a55c5c231bda1"
FAMILIES = ("sparse", "dense", "regular")
METHODS = {
    "row": "Tightened row-sum",
    "box": "Marginal interval",
    "baseline": "Per-graph comparator",
    "topology": "Topology-aware (ours)",
}


def table(path):
    archive = REPORT.read_bytes()
    if hashlib.sha256(archive).hexdigest() != REPORT_SHA256:
        raise ValueError("The archived 72-graph report failed its checksum.")
    report = json.loads(archive)
    if path.resolve() != REPORT.resolve():
        rerun = json.loads(path.read_bytes())
        # Wall-clock timestamps change when a reader repeats the experiment.
        for record in (report, rerun):
            del record["completed_utc"]
        if rerun != report:
            raise ValueError("The rerun differs from the archived design or results.")
        report = rerun
    rows = report["rows"]
    counts = Counter((r["family"], r["n"]) for r in rows)
    if counts != {(f, n): 8 for f in FAMILIES for n in (128, 256, 512)}:
        raise ValueError("Incomplete family-size strata.")
    if len({r["matrix_sha256"] for r in rows}) != len(rows):
        raise ValueError("Repeated graph.")
    for row in rows:
        for key in METHODS:
            e, d, h = row[key]
            if not row["complete"] or min(e, d, h) < 0 or d+h > 60:
                raise ValueError("Missing or ineligible schedule.")
        best = min((row["row"], row["box"]), key=lambda p: (p[0], p[1]+p[2], p[1]))
        if row["baseline"] != best:
            raise ValueError("Per-graph comparator differs from the prespecified rule.")

    for method, label in METHODS.items():
        family_peaks = [sum(r[method][0] for r in rows if r["family"] == f) for f in FAMILIES]
        total_peak = sum(family_peaks)
        mean_time = sum(r[method][1]+r[method][2] for r in rows) / len(rows)
        total_drive = sum(r[method][0]*r[method][1] for r in rows)
        frozen = report["summary"]["protocols"][method]
        if (total_peak, mean_time, total_drive) != (
            frozen["peak_sum"], frozen["mean_time"], frozen["amplitude_pressure_sum"]
        ):
            raise ValueError("Recomputed endpoints differ from the archived summary.")
        yield [label, *family_peaks, total_peak, f"{mean_time:.2f}", total_drive]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, help="Optional CSV output path (default: stdout)")
    args = parser.parse_args()
    header = ["method", "sparse_peak_sum", "dense_peak_sum", "regular_peak_sum",
              "total_peak_sum", "mean_completion_updates", "total_cumulative_drive"]
    records = [header, *table(args.report)]
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as out:
            csv.writer(out).writerows(records)
    else:
        csv.writer(sys.stdout, lineterminator="\n").writerows(records)
