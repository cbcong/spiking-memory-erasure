"""Render the graph-level completion-time differences for the supplement."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["incomplete"] or len(report["rows"]) != 72:
        raise ValueError("Expected all 72 archived graph pairs.")

    differences = defaultdict(list)
    graph_ids = set()
    for row in report["rows"]:
        graph_ids.add(row["matrix_sha256"])
        baseline = min(
            (row["row"], row["box"]),
            key=lambda plan: (plan[0], sum(plan[1:]), plan[1]),
        )
        if not row["complete"] or row["baseline"] != baseline:
            raise ValueError("Graph pair does not match the archived design.")
        change = sum(row["topology"][1:]) - sum(baseline[1:])
        differences[row["family"], row["n"]].append(change)

    styles = {
        "sparse": ("#30465A", "o"),
        "dense": ("#7D3D49", "s"),
        "regular": ("#686A6C", "D"),
    }
    groups = [(family, size) for family in styles for size in (128, 256, 512)]
    if len(graph_ids) != 72 or set(differences) != set(groups):
        raise ValueError("Expected 72 distinct graphs distributed across nine strata.")
    if any(len(differences[group]) != 8 for group in groups):
        raise ValueError("Each stratum must contain eight graph pairs.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#686A6C",
            "axes.labelcolor": "#242426",
            "xtick.color": "#242426",
            "ytick.color": "#242426",
            "text.color": "#242426",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.975, top=.82, bottom=.29, wspace=.15)
    for index, (ax, size) in enumerate(zip(axes, (128, 256, 512))):
        ax.axhline(0, color="#828284", lw=.8, linestyle=(0, (3, 2)))
        ax.grid(axis="y", color="#E4E4E4", lw=.55)
        ax.set_axisbelow(True)
        for family, (color, marker) in styles.items():
            values = sorted(differences[family, size])
            ax.plot(range(1, 9), values, color=color, label=family.title(),
                    marker=marker, ms=4.0, markerfacecolor="white", markeredgewidth=.95,
                    linestyle={"sparse":"-", "dense":"--", "regular":":"}[family], lw=1)
        ax.tick_params(length=3, width=.6)
        ax.set(xlim=(.6,8.4), ylim=(-16,37), xticks=[1,2,3,4,5,6,7,8],
               yticks=[-10,0,10,20,30], xlabel=f"({chr(97+index)})  $n = {size}$")
    axes[0].set_ylabel("Completion-time\nchange (updates)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, frameon=False, fontsize=8,
               loc="upper center", bbox_to_anchor=(.53,1.005), columnspacing=1.8)
    fig.text(.54,.035,"Graph rank within each size and family", ha="center", fontsize=9)
    fig.savefig(
        args.output,
        metadata={"Title": "Paired completion times on the 72 confirmation graphs", "CreationDate": None},
    )


if __name__ == "__main__":
    main()
