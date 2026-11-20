"""Compare frozen T=60 certificates with the weighted-core lower bound."""
import csv
import hashlib
import json
import sys
from pathlib import Path

from experiments.run_confirm_strong import matrix_hash
from experiments.run_core_scale_comparison import weights
from experiments.run_extended_size_certificates import ExtendedSpec
from integer_snn.core_erasure import core_erasure_bounds
from integer_snn.topology_certificate import MODEL


def main():
    panel = Path(__file__).resolve().parents[1] / "results/confirm_strong_20261119_v1"
    design_bytes = (panel / "design.json").read_bytes()
    design = json.loads(design_bytes)
    report = json.loads((panel / "report.json").read_bytes())
    assert hashlib.sha256(design_bytes).hexdigest() == report["design_sha256"]
    assert design["deadline"] == 60 and report["incomplete"] == 0
    assert len(design["rows"]) == len(report["rows"]) == 72
    writer = csv.writer(sys.stdout)
    writer.writerow(("family", "modules", "seed", "matrix_sha256", "peak", "D", "H",
                     "core_lower_bound", "gap_upper_bound"))
    for entry, row in zip(design["rows"], report["rows"], strict=True):
        assert all(row[key] == value for key, value in entry.items())
        matrix = weights(entry["n"], entry["family"], entry["seed"])
        assert matrix_hash(matrix) == entry["matrix_sha256"]
        peak, duration, recovery = row["topology"]
        assert row["complete"] and duration + recovery <= design["deadline"]
        lower = core_erasure_bounds(ExtendedSpec(MODEL, matrix))["lower"]
        assert peak >= lower
        writer.writerow((entry["family"], entry["n"], entry["seed"], entry["matrix_sha256"],
                         peak, duration, recovery, lower, peak - lower))


if __name__ == "__main__":
    main()
