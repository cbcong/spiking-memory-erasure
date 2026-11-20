"""Rebuild the box-and-points candidate from the frozen 72-graph panel."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "results" / "confirm_strong_20261119_v1" / "report.json"
FAMILIES = ("sparse", "dense", "regular")
SIZES = (128, 256, 512)


def main(output_pdf=None):
    rows = json.loads(REPORT.read_text(encoding="utf-8"))["rows"]
    if len(rows) != 72 or len({row["matrix_sha256"] for row in rows}) != 72:
        raise ValueError("The prospective panel must contain 72 distinct graphs.")
    groups = defaultdict(list)
    for row in rows:
        comparator = min((row["box"], row["row"]), key=lambda p: (p[0], sum(p[1:]), p[1]))
        if not row["complete"] or row["baseline"] != comparator:
            raise ValueError("The archived comparator differs from the design.")
        e0, d0, _ = comparator
        e, d, _ = row["topology"]
        groups[row["family"], row["n"]].append(100 * (e * d / (e0 * d0) - 1))

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#242426", "axes.linewidth": .8,
                         "axes.labelcolor": "#242426", "text.color": "#242426",
                         "xtick.color": "#242426", "ytick.color": "#242426"}):
        fig, ax = plt.subplots(figsize=(7.10, 3.25))
        fig.subplots_adjust(left=.135, right=.982, top=.96, bottom=.235)
        ax.axhline(0, color="#686A6C", linestyle=(0, (3, 3)), lw=.65, zorder=0)
        ticks, labels = [], []
        for i, family in enumerate(FAMILIES):
            for j, n in enumerate(SIZES):
                values = sorted(groups[family, n])
                if len(values) != 8:
                    raise ValueError("Expected eight paired graphs per stratum.")
                x = i * 3.85 + j + 1
                ax.boxplot(values, positions=[x], widths=.61, patch_artist=True,
                           showfliers=False, showcaps=False,
                           boxprops=dict(facecolor="#E8E7E4", edgecolor="#686A6C", linewidth=.9),
                           medianprops=dict(color="#242426", linewidth=1.65),
                           whiskerprops=dict(color="#686A6C", linewidth=.85))
                # Offsets carry no measurement. Spread tied values to reveal all observations.
                offsets = np.zeros(len(values))
                for y in set(values):
                    idx = np.flatnonzero(np.asarray(values) == y)
                    if len(idx) > 1:
                        offsets[idx] = np.linspace(-.195, .195, len(idx))
                unique = np.array([values.count(y) == 1 for y in values])
                offsets[unique] = np.random.default_rng(82+i*10+j).uniform(-.19,.19,unique.sum())
                ax.scatter(x+offsets, values, color="#30465A", alpha=.79, s=12,
                           linewidths=0, zorder=4)
                ticks.append(x)
                labels.append(str(n))
            ax.text(i*3.85+2, -.139, family.title(), transform=ax.get_xaxis_transform(),
                    fontsize=10, ha="center", va="top")
        ax.set_xticks(ticks, labels)
        ax.tick_params(axis="x", labelsize=8.5, length=3)
        ax.set_xlim(.35, ticks[-1]+.65)
        ax.set_ylim(-59,183)
        ax.set_yticks([-50,0,50,100,150])
        ax.set_ylabel("Change in cumulative drive $ED$ (%)")
        fig.savefig(output_pdf or HERE / "3D.pdf", metadata={"CreationDate": None})
        if output_pdf is None:
            fig.savefig(HERE / "3D.png", dpi=190)
        plt.close(fig)


if __name__ == "__main__":
    main()
