"""Compare archived protocols against the better feasible baseline on each graph."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def rank(plan):
    amplitude, duration, recovery = plan
    return amplitude, duration + recovery, duration


def summarize(rows):
    proposed = [r["topology"] for r in rows]
    baseline = [min((r["row"], r["box"]), key=rank) for r in rows]
    gains = [b[0] - a[0] for a, b in zip(proposed, baseline)]
    def totals(plans):
        return dict(peak_sum=sum(e for e, _, _ in plans),
                    mean_duration=mean(d for _, d, _ in plans),
                    mean_total=mean(d+h for _, d, h in plans),
                    amplitude_steps=sum(e*d for e, d, _ in plans))
    a, b = totals(proposed), totals(baseline)
    return dict(proposed=a, best_baseline=b, n=len(rows), lower=sum(g > 0 for g in gains),
                equal=gains.count(0), higher=sum(g < 0 for g in gains),
                peak_reduction=1-a["peak_sum"]/b["peak_sum"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    result = {}
    graph_ids = None
    for deadline in (20, 60):
        source = ROOT / "results" / f"box_erasure_scale_20261119_t{deadline}" / "report.json"
        content = source.read_bytes()
        panel = json.loads(content)
        rows = panel["rows"]
        ids = [r["matrix_sha256"] for r in rows]
        if len(set(ids)) != len(ids) or (graph_ids is not None and ids != graph_ids):
            raise ValueError("graph panel mismatch")
        graph_ids = ids
        for row in rows:
            for name in ("topology", "row", "box"):
                if row[name] is None or sum(row[name][1:]) > deadline:
                    raise ValueError((row["matrix_sha256"], name, deadline))
        result[deadline] = dict(input_sha256=hashlib.sha256(content).hexdigest(),
                                summary=summarize(rows), by_family={
                                    family: summarize([r for r in rows if r["family"] == family])
                                    for family in sorted({r["family"] for r in rows})})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
