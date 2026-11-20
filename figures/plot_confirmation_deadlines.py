"""Plot deadline sensitivity of the paired 72-graph confirmation results."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "results" / "confirm_strong_20261119_v1" / "report.json"
REPORT = ROOT / "results" / "confirmation_deadline_sensitivity" / "report.json"
DEADLINES = (20, 40, 60, 100)
SIZES = (128, 256, 512)
PANELS = (*SIZES, "pooled")
STYLES = {
    "sparse": ("#30465A", "o", "-"),
    "dense": ("#7D3D49", "s", "--"),
    "regular": ("#686A6C", "D", ":"),
}
LABELS = (
    "Peak saving (%)",
    "Completion-time\nchange (updates)",
    "Cumulative-drive\nchange (%)",
    "Graphs improving\nboth costs (%)",
)


def rank(plan):
    e, d, h = plan
    return e, d + h, d


def summaries(report, archive):
    data = json.loads(report.read_bytes())
    original = archive.read_bytes()
    if tuple(data["deadlines"]) != DEADLINES or data["archive_sha256"] != sha256(original).hexdigest():
        raise ValueError("Deadline report does not match the confirmation archive")
    frozen_rows = {r["matrix_sha256"]: r for r in json.loads(original)["rows"]}
    hashes = [r["matrix_sha256"] for r in data["rows"]]
    if len(hashes) != 72 or len(set(hashes)) != 72 or set(hashes) != set(frozen_rows):
        raise ValueError("Expected all 72 confirmation graphs, exactly once")

    groups = defaultdict(list)
    for row in data["rows"]:
        frozen = frozen_rows[row["matrix_sha256"]]
        if any(row[k] != frozen[k] for k in ("n", "family", "seed")):
            raise ValueError("Graph identity differs from confirmation archive")
        for t in DEADLINES:
            plans = row["plans"][str(t)]
            p, b = plans["topology"], plans["baseline"]
            baseline_arms = [plans[k] for k in ("row", "box") if plans[k] is not None]
            if p is None or b is None or b != min(baseline_arms, key=rank):
                raise ValueError(f"Incomplete pair at seed {row['seed']}, T={t}")
            if rank(p)[1] > t or rank(b)[1] > t:
                raise ValueError(f"Pair misses the deadline at seed {row['seed']}, T={t}")
            if t == 60 and any(plan != frozen[arm] for arm, plan in plans.items()):
                raise ValueError("T=60 pair differs from confirmation archive")
            groups[row["n"], row["family"], t].append((p, b))
            groups["pooled", row["family"], t].append((p, b))

    results = {}
    for n in PANELS:
        for family in STYLES:
            for t in DEADLINES:
                pairs = groups[n, family, t]
                expected = 24 if n == "pooled" else 8
                if len(pairs) != expected:
                    raise ValueError(f"Expected {expected} paired graphs for {n}, {family}, T={t}")
                proposed, baseline = zip(*pairs)
                peak_saving = 100 * (
                    1 - sum(p[0] for p in proposed) / sum(b[0] for b in baseline)
                )
                time_change = mean(rank(p)[1] - rank(b)[1] for p, b in pairs)
                drive_change = 100 * (
                    sum(e * d for e, d, _ in proposed) / sum(e * d for e, d, _ in baseline) - 1
                )
                joint_improvement = 100 * mean(p[0] < b[0] and p[0]*p[1] < b[0]*b[1]
                                               for p, b in pairs)
                results[n, family, t] = peak_saving, time_change, drive_change, joint_improvement
    return results


def render(output, png=None):
    data = summaries(REPORT, ARCHIVE)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#686A6C",
            "axes.labelcolor": "#242426",
            "xtick.color": "#242426",
            "ytick.color": "#242426",
            "text.color": "#242426",
        }
    ):
        fig, axes = plt.subplots(4, 4, figsize=(7.05, 6.45), sharex=True, sharey="row")
        fig.subplots_adjust(left=.125, right=.991, top=.93, bottom=.102, wspace=.17, hspace=.47)
        for metric, label in enumerate(LABELS):
            for i, n in enumerate(PANELS):
                ax = axes[metric, i]
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#E4E4E4", lw=.55)
                ax.axhline(0, color="#828284", lw=.8, linestyle=(0, (3, 2)), zorder=1)
                ax.set(xlim=(14, 106), xticks=DEADLINES)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                ax.tick_params(length=2.4, width=.6, labelbottom=metric == 3)
                group_label = "Sizes pooled" if n == "pooled" else f"$n={n}$"
                ax.text(.5, -.23 if metric == 3 else -.105, f"({chr(97 + metric * 4 + i)})  {group_label}",
                        transform=ax.transAxes, ha="center", va="top")
                for family, (color, marker, line) in STYLES.items():
                    values = [data[n, family, t][metric] for t in DEADLINES]
                    ax.plot(DEADLINES, values, color=color, marker=marker, linestyle=line, lw=1,
                            markersize=3.5, markerfacecolor="white", markeredgewidth=.9,
                            label=family.title(), zorder=2)
            axes[metric, 0].set_ylabel(label, labelpad=4)
            pooled = [values[metric] for values in data.values()]
            low, high = min(0, *pooled), max(0, *pooled)
            margin = .09 * (high - low) or 1
            axes[metric, 0].set_ylim(low - margin, high + margin)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=3, frameon=False,
                   loc="upper center", bbox_to_anchor=(.56, .996), columnspacing=1.75)
        fig.text(.55, .015, "Prescribed deadline (updates)", ha="center")
        fig.savefig(output, metadata={"CreationDate": None, "Title": "Deadline sensitivity"})
        if png is not None:
            fig.savefig(png, dpi=260)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    render(args.output, args.png)
