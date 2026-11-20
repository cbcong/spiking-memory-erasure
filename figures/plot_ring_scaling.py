"""Plot last-spike times from the archived weak-edge ring experiments."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, StrMethodFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data"
SLATE = "#30465A"
GARNET = "#7D3D49"
INK = "#242426"
GRID = "#E4E3E1"
LINK = "#777775"
GAP = "#D9D7D2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.4,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#686A6C",
    "axes.linewidth": .65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": .65,
    "ytick.major.width": .65,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
})


def read_records(filename):
    return json.loads((RESULTS / filename).read_text(encoding="utf-8"))["records"]


def last_spikes():
    rows = read_records("weak_ring_scaling_v120.json")
    sizes = sorted({row["n"] for row in rows})
    lookup = {(row["n"], row["amplitude"]): row for row in rows}
    if len(rows) != 18 or len(lookup) != 18 or sizes != [2 ** i for i in range(1, 10)]:
        raise ValueError("Expected two amplitudes at each of nine tested ring sizes")
    for (size, _), row in lookup.items():
        if len(row["last_spike"]) != size or max(row["last_spike"]) != row["network_last_spike"]:
            raise ValueError("The network last spike does not agree with the module records")
    nodes = [np.array(lookup[32, e]["last_spike"]) for e in (3, 4)]
    network = [np.array([lookup[n, e]["network_last_spike"] for n in sizes]) for e in (3, 4)]
    return np.array(sizes), np.array(nodes), np.array(network)


def panels():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.88))
    fig.subplots_adjust(left=.102, right=.975, bottom=.22, top=.932, wspace=.37)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=.6)
    return fig, axes


def save(fig, name, output_pdf=None):
    fig.savefig(output_pdf or HERE / f"{name}.pdf", metadata={"CreationDate": None})
    if output_pdf is None:
        fig.savefig(HERE / f"{name}.png", dpi=200)
    plt.close(fig)


def ring_scale(ax, sizes, times):
    ax.set(xscale="log", yscale="log", xlim=(1.75, 920), ylim=(2.7, 1650))
    ax.xaxis.set_major_locator(FixedLocator(sizes[::2]))
    ax.yaxis.set_major_locator(FixedLocator([4, 16, 64, 256, 1024]))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set(xlabel="(b) Ring size (modules, log scale)", ylabel="Network last spike (tick, log scale)")
    ax.fill_between(sizes, times[0], times[1], color=GAP, alpha=.34, linewidth=0)
    ax.vlines(sizes[:-1], times[1, :-1], times[0, :-1], color=LINK, alpha=.58, lw=.75)
    ax.vlines(sizes[-1], times[1, -1], times[0, -1], color=LINK, lw=1.0)
    ax.annotate(f"{times[0, -1]-times[1, -1]:,}\nticks", (sizes[-1], 72),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=7.2, color=INK)


def word_range(ax):
    rows = sorted(read_records("weak_ring_words_v150.json"), key=lambda row: row["modules"])
    sizes = np.array([row["modules"] for row in rows])
    if list(sizes) != [2, 4, 6, 8, 10, 12]:
        raise ValueError("Unexpected sizes in the complete-word census")
    for peak, color, marker, style in ((3, SLATE, "o", "-"), (4, GARNET, "s", "--")):
        values = []
        for row in rows:
            dist = row["amplitudes"][str(peak)]["last_spike"]
            counts = dist["counts"]
            if sum(counts.values()) != row["words"] or row["words"] != 2**row["modules"]:
                raise ValueError("Incomplete word census")
            values.append([dist["minimum"], dist["median"], dist["maximum"]])
        low, med, high = np.array(values).T
        ax.errorbar(sizes, med, yerr=(med-low, high-med), color=color, fmt=marker,
                    ls=style, lw=1.25, markersize=3.8, markerfacecolor="white",
                    markeredgewidth=.95, elinewidth=.85, capsize=3.1)
    ax.set(xlim=(1.3, 12.7), ylim=(0, 25), xticks=sizes, yticks=[0, 5, 10, 15, 20, 25],
           xlabel="(c) Ring size (modules)", ylabel="Network last spike (tick)")


def fold_lines(sizes, nodes, network, output_pdf=None, output_png=None):
    fig, (left, right, words) = plt.subplots(1, 3, figsize=(7.85, 2.45))
    fig.subplots_adjust(left=.073, right=.985, bottom=.245, top=.955, wspace=.48)
    for ax in (left, right, words):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=.6)
    position = np.arange(nodes.shape[1])
    left.set(xlim=(-1.4, 33), ylim=(-1, 70), xticks=[0, 8, 16, 24, 31],
             yticks=[0, 16, 32, 48, 64], xlabel="(a) Distance from node 0 (edges)",
             ylabel="Module last spike (tick)")
    left.fill_between(position, nodes[0], nodes[1], color=GAP, alpha=.34, linewidth=0)
    selected = np.array([0, 8, 16, 24, 31])
    left.vlines(selected, nodes[1, selected], nodes[0, selected], color=LINK, alpha=.62, lw=.8)
    ring_scale(right, sizes, network)
    right.set_ylabel("Network last spike\n(tick, log scale)")
    word_range(words)
    for ax, x, traces, marked in (
        (left, position, nodes, selected.tolist()),
        (right, sizes, network, None),
    ):
        for values, color, marker, line_style in (
            (traces[0], SLATE, "o", "-"),
            (traces[1], GARNET, "s", "--"),
        ):
            ax.plot(x, values, color=color, marker=marker, linestyle=line_style,
                    markevery=marked, markerfacecolor="white", markeredgewidth=.95,
                    markersize=3.8, lw=1.45, zorder=3)
    left.annotate(r"$E=3$", (5, nodes[0, 5]), xytext=(0, 5), textcoords="offset points",
                  ha="center", color=SLATE)
    left.annotate(r"$E=4$", (5, nodes[1, 5]), xytext=(0, 5), textcoords="offset points",
                  ha="center", color=GARNET)
    right.annotate(f"{network[0, -1]:,}", (sizes[-1], network[0, -1]), xytext=(-5, 3),
                   textcoords="offset points", ha="right", va="bottom", color=SLATE)
    right.annotate(f"{network[1, -1]}", (sizes[-1], network[1, -1]), xytext=(6, 0),
                   textcoords="offset points", va="center", color=GARNET)
    if output_png is not None:
        fig.savefig(output_png, dpi=240)
    save(fig, "4A", output_pdf)


def points_and_stems(sizes, nodes, network):
    fig, (left, right) = panels()
    position = np.arange(nodes.shape[1])
    left.set(xlim=(-1.2, 33), ylim=(-1, 70), xticks=[0, 8, 16, 24, 31],
             yticks=[0, 16, 32, 48, 64], xlabel="(a) Distance from node 0 (edges)",
             ylabel="Module last spike (tick)")
    for values, color, marker, label in (
        (nodes[0], SLATE, "o", r"$E=3$"),
        (nodes[1], GARNET, "s", r"$E=4$"),
    ):
        left.scatter(position, values, color=color, marker=marker, s=57, alpha=.12, linewidths=0)
        left.scatter(position, values, color=color, marker=marker, s=13, alpha=.85,
                     linewidths=0, label=label, zorder=3)
    left.legend(frameon=False, loc="upper left", handletextpad=.2, borderaxespad=.2)
    delta = network[0] - network[1]
    ys = np.arange(len(sizes))
    right.set(xscale="log", xlim=(1, 2000), ylim=(-.45, 8.45), yticks=ys,
              yticklabels=sizes, xlabel="(b) Advance in last-spike time (ticks)",
              ylabel="Ring size (modules)")
    right.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    right.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    right.tick_params(which="minor", length=0)
    right.hlines(ys, 1, delta, lw=.95, color=LINK)
    right.scatter(delta, ys, color=SLATE, s=18, linewidth=0, zorder=3)
    for x, y in zip(delta, ys):
        right.annotate(f"{x:,}", (x, y), xytext=(4, 0), textcoords="offset points",
                       fontsize=7.4, va="center")
    save(fig, "4B")


def discrete_words():
    fig, axes = panels()
    rows = sorted(read_records("weak_ring_words_v120.json"), key=lambda row: row["modules"])
    for idx, (ax, row) in enumerate(zip(axes, rows, strict=True)):
        if row["words"] != 2 ** row["modules"]:
            raise ValueError("The archived word census is incomplete")
        distribution = row["paired_last_spike_reduction"]
        counts = sorted((int(tick), count) for tick, count in distribution["counts"].items())
        if sum(count for _, count in counts) != row["words"]:
            raise ValueError("The observed word counts do not sum to the census size")
        x, y = np.array(counts).T
        percent = 100 * y / row["words"]
        ax.set(xlim=(.3, 20.2), ylim=(0, 43), xticks=[1, 5, 10, 15, 19],
               yticks=[0, 10, 20, 30, 40],
               xlabel=f"({chr(97+idx)}) Advance in last-spike time (ticks)")
        ax.text(.96, .93, f"{row['modules']} modules\n{row['words']:,} words",
                transform=ax.transAxes, ha="right", va="top", linespacing=1.22)
        ax.vlines(x, 0, percent, color="#6BA5CD", lw=1)
        ax.scatter(x, percent, color="#1E76B2", s=17, facecolor="white", lw=.95,
                   zorder=3, clip_on=False)
        ax.annotate(f"{y[-1]} uniform words", (x[-1], percent[-1]), xytext=(-3, 29),
                    textcoords="offset points", ha="right", fontsize=7.4, color=SLATE,
                    arrowprops={"arrowstyle": "-", "color": LINK, "lw": .7})
    axes[0].set_ylabel("Share of written words (%)")
    axes[1].tick_params(labelleft=False)
    save(fig, "4C")


if __name__ == "__main__":
    sizes, nodes, network = last_spikes()
    fold_lines(sizes, nodes, network)
    points_and_stems(sizes, nodes, network)
    discrete_words()
