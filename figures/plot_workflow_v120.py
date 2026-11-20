"""Illustrate one certificate with the same routines used in the experiments."""

import argparse
from pathlib import Path

from matplotlib.patches import Circle, FancyArrowPatch

from .plot_manuscript_v100 import GARNET, GREY, INK, plt
from experiments.run_core_endpoint_witnesses import endpoint_certificate
from experiments.run_locked_erasure_scale import plans_at
from integer_snn.core_erasure import core_erasure_bounds
from integer_snn.network import NetworkSpec
from integer_snn.release_tail import release_tail
from integer_snn.topology_certificate import MODEL


def arrow(ax, start, end, color=INK, radius=0, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, color=color,
                                 linewidth=1.15, mutation_scale=9, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={radius}"))


def cell(ax):
    ax.set(xlim=(0, 6.9), ylim=(-.2, 3.5), aspect="equal")
    ax.axis("off")
    for c, y, side in [(0, 2.5, 1), (1, .85, -1)]:
        ax.text(.05, y, rf"$\sum_j W_{{ij}}s_{{j{c}}}$", va="center")
        arrow(ax, (1.55, y), (3.05, y))
        ax.add_patch(Circle((3.36, y), .30, fc="white", ec=INK, lw=1.5))
        ax.text(3.36, y, rf"$c_{c}$", ha="center", va="center")
        arrow(ax, (3.68, y), (5.50, y))
        ax.text(5.62, y, rf"$s_{{i{c}}}'$", va="center")
        ax.plot([4.92, 4.92, 3.60], [y, y+.48*side, y+.48*side], lw=1.0, color=INK)
        arrow(ax, (3.60, y+.48*side), (3.49, y+.29*side))
        ax.text(5.03, y+.48*side, "+6", va="center", fontsize=8)
    for x, y1, y2 in [(3.26, 2.18, 1.19), (3.46, 1.17, 2.15)]:
        arrow(ax, (x, y1), (x, y2), GARNET, style="-[")
    ax.text(3.75, 1.66, r"$-4$", va="center", color=GARNET)
    ax.text(.08, 1.67, r"$-E(t)$", va="center", color=GARNET)
    ax.plot([1.07, 2.14], [1.67, 1.67], "--", color=GARNET, lw=1)
    for target in (2.29, 1.06):
        arrow(ax, (2.14, 1.67), (3.08, target), GARNET)
    ax.text(.48, -.065, "(a) Two channels in module $i$", transform=ax.transAxes, ha="center")


def ring(ax, spec):
    ax.set(xlim=(-1.95, 1.95), ylim=(-.94, 1.83), aspect="equal")
    ax.axis("off")
    pos = [(0, 1.32), (1.15, -.38), (-1.15, -.38)]
    labels = ((-.82, .59), (.82, .59), (0, -.62))
    for j, row in enumerate(spec.weights):
        i = next(i for i, weight in enumerate(row) if weight)
        x, y = pos[i]
        u, v = pos[j]
        dx, dy = u-x, v-y
        gap = .255 / (dx*dx+dy*dy)**.5
        arrow(ax, (x+gap*dx, y+gap*dy), (u-gap*dx, v-gap*dy))
        ax.text(*labels[j], str(row[i]), ha="center", va="center")
    for i, xy in enumerate(pos):
        ax.add_patch(Circle(xy, .215, facecolor="white", edgecolor=INK, lw=1.5))
        ax.text(*xy, str(i), ha="center", va="center")
    ax.text(.5, -.065, "(b) Directed coupling weights", transform=ax.transAxes, ha="center")


def module(folder):
    spec = NetworkSpec(MODEL, ((0, 0, 1), (2, 0, 0), (0, 2, 0)))
    lower = core_erasure_bounds(spec)["lower"]
    e, d, h = plans_at(spec, 60)["locked"]
    certificate = endpoint_certificate(spec, e)
    limits = {node: stage["start"] + info["offset"] for stage in certificate["stages"]
              for node, info in stage["eliminated"].items()}
    extra, recover = release_tail(e)
    assert e == lower and (max(limits.values())+extra, recover) == (d, h)

    fig = plt.figure(figsize=(6.45, 3.60))
    upper = fig.add_gridspec(1, 2, left=.025, right=.986, bottom=.50, top=.98,
                             width_ratios=(1.46, 1), wspace=.05)
    cell(fig.add_subplot(upper[0]))
    ring(fig.add_subplot(upper[1]), spec)
    ax = fig.add_axes([.143, .145, .797, .221])
    ax.set_xlim(-.4, d+h+.4)
    ax.set_ylim(0, 3.44)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_bounds(0, d+h)
    ax.set_yticks([])
    ax.set_xticks([0, *limits.values(), d, d+h])
    ax.tick_params(axis="x", labelsize=8, pad=2, length=3)
    ax.plot([0, d], [2.84, 2.84], color=GARNET, lw=2.3, solid_capstyle="butt")
    ax.plot([d, d+h], [2.84, 2.84], color=GREY, lw=1.2, ls="--")
    ax.text(d/2, 3.07, rf"Applied pulse $-E=-{e}$", ha="center", va="bottom", fontsize=8)
    ax.text(d+h/2, 3.07, "Release", ha="center", va="bottom", fontsize=8)
    for node, tick in limits.items():
        height = .49 + (2-node)*.76
        ax.text(-.72, height, f"Module {node}", ha="right", va="center", fontsize=8)
        ax.hlines(height, tick, d+h, colors=INK, lw=1.05)
        ax.plot(tick, height, marker="o", ms=4.1, mfc="white", mec=INK)
    ax.plot(d+h, 2.84, marker="D", color=INK, ms=4, clip_on=False)
    ax.annotate("All states zero", (d+h, 2.84), xytext=(0, 16),
                textcoords="offset points", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("(c) Last-spike bounds and release schedule (ticks)", labelpad=5)
    fig.savefig(Path(folder)/"module.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Existing manuscript figures folder")
    module(parser.parse_args().out)
