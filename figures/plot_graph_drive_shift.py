"""Matched changes in cumulative-drive ratio across deadlines on 72 graphs."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
from statistics import median

from .plot_confirmation_deadlines import ARCHIVE, DEADLINES, REPORT, STYLES, summaries
from matplotlib import pyplot as plt


def graph_outcomes(report=REPORT, archive=ARCHIVE):
    summaries(report, archive)  # Verify identities, arm selection and all deadlines.
    counts = defaultdict(Counter)
    shifts = defaultdict(list)
    for row in json.loads(report.read_bytes())["rows"]:
        family = row["family"]
        ratios = {}
        for deadline in DEADLINES:
            plans = row["plans"][str(deadline)]
            proposed, baseline = plans["topology"], plans["baseline"]
            e, d, _ = proposed
            b, duration, _ = baseline
            ratios[deadline] = Fraction(e * d, b * duration)
            peak_sign = (e > b) - (e < b)
            drive_sign = (ratios[deadline] > 1) - (ratios[deadline] < 1)
            counts[family, deadline][peak_sign, drive_sign] += 1
        change = 100 * (ratios[100] - ratios[20])
        shifts[family].append((float(change), row["matrix_sha256"]))
    return counts, shifts


def render(output, png=None):
    _, shifts = graph_outcomes()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#686A6C",
                         "axes.labelcolor": "#242426", "text.color": "#242426"}):
        fig, ax = plt.subplots(figsize=(5.25, 2.95))
        fig.subplots_adjust(left=.19, bottom=.14, top=.98, right=.98)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dedede", linewidth=.55)
        ax.axhline(0, color="#777777", linewidth=.7)
        for position, (family, (color, marker, _)) in enumerate(STYLES.items(), 1):
            records = shifts[family]
            # Hash order controls only horizontal displacement. Each column spans
            # the same width and every original vertical coordinate is retained.
            values = [v for v, _ in sorted(records, key=lambda record: record[1])]
            if len(set(values)) > 1:
                shape = ax.violinplot(values, positions=[position], widths=.70,
                                      showextrema=False, bw_method="scott")
                body = shape["bodies"][0]
                body.set(facecolor=color, edgecolor=color, linewidth=.6, alpha=.20)
                offsets = [(i % 7 - 3) * .046 for i in range(len(values))]
            else:
                offsets = [(i - (len(values)-1)/2) * .03 for i in range(len(values))]
            ax.plot([position-.23, position+.23], [median(values)] * 2,
                    color=color, lw=2.2, solid_capstyle="butt", zorder=2)
            ax.scatter([position + x for x in offsets], values, s=19 if family != "regular" else 9,
                       color=color, marker=marker, edgecolor="white", linewidth=.3,
                       zorder=3)
        ax.set(xticks=[1, 2, 3], xticklabels=[f.title() for f in STYLES],
               xlim=(.48, 3.52), ylim=(-18, 340), yticks=range(0, 301, 50),
               ylabel="Change in drive ratio\n(percentage points)")
        ax.tick_params(length=3, width=.6)
        fig.savefig(output, metadata={"CreationDate": None, "Title": "Matched cumulative-drive changes"})
        if png is not None:
            fig.savefig(png, dpi=240)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    render(args.output, args.png)
