"""Matched ring layouts under E=3 and a fixed-longest-segment control."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

INPUT = Path(__file__).resolve().parent / "data"
BLUE, RED = "#30465A", "#7D3D49"
STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#686A6C", "axes.labelcolor": "#242426",
         "text.color": "#242426", "axes.linewidth": .65}


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=.55, color="#dedede")
    ax.tick_params(length=3.1, width=.7)


def render_main(output, preview=None):
    with (INPUT / "multiweak_word_pilot.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sizes = [16, 32, 64]
    series = {}
    for n in sizes:
        for layout, length in (("even", n // 4 - 1), ("clustered", n - 4)):
            match, = [r for r in rows if int(r["n"]) == n and int(r["weak_count"]) == 4
                      and int(r["longest_strong_run"]) == length and int(r["word_count"]) == 1]
            series[layout, n] = int(match["E3_word_opt_total"])
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.85, 2.78))
        fig.subplots_adjust(left=.098, right=.985, bottom=.28, top=.96, wspace=.43)
        for sub in (ax, bx):
            style_axis(sub)
        ax.vlines(sizes, [series["even", n] for n in sizes],
                  [series["clustered", n] for n in sizes], lw=.7, color="#a7a9aa")
        for layout, name, color, marker, ls in [
                ("even", "Even spacing", BLUE, "o", "-"),
                ("clustered", "Clustered", RED, "s", "--")]:
            values = [series[layout, n] for n in sizes]
            ax.plot(sizes, values, label=name, color=color, marker=marker, ls=ls, lw=1.35,
                    ms=4.6, mfc="white", mew=1.05)
        ax.legend(frameon=False, loc="upper left", fontsize=8, borderaxespad=.18)
        ax.set(xlim=(12,68), xticks=sizes, ylim=(14,84), yticks=[20,40,60,80],
               ylabel="Minimum total time (updates)", xlabel="Ring size (modules)\n(a)  Four weak edges")
        even_sizes = list(range(16, 81, 2))
        bx.plot(even_sizes, [27] * len(even_sizes), color=BLUE, lw=1.45,
                label="Alternating: upper bound")
        bx.plot(even_sizes, [n // 2 + 1 for n in even_sizes], color=RED, ls="--", lw=1.45,
                label="Clustered: lower bound")
        bx.vlines(64, 27, 33, color="#777777", lw=.8)
        for y, color, marker, dy in ((27, BLUE, "o", -10), (33, RED, "s", 3)):
            bx.plot(64, y, color=color, marker=marker, ms=4.6, mfc="white", mew=1.05)
            dx = -5 if y == 33 else 2
            bx.annotate(str(y), (64,y), xytext=(dx,dy), textcoords="offset points",
                        ha="right" if y == 33 else "left", color=color, fontsize=8)
        bx.set(xlim=(13,82), xticks=[16,32,48,64,80], ylim=(6,48), yticks=[10,20,30,40],
               ylabel="All-state time bound (updates)", xlabel="Ring size (modules)\n(b)  Half the edges weak")
        bx.legend(loc="upper left", frameon=False, fontsize=7.5, borderaxespad=.15, handlelength=1.8)
        fig.savefig(output, metadata={"CreationDate": None})
        if preview:
            fig.savefig(preview, dpi=160)
        plt.close(fig)


def render_control(output, preview=None):
    records = json.loads((INPUT / "multiweak_size_control.json").read_text(encoding="utf-8"))["records"]
    sizes = [32,64,128,256,512]
    series = {(r["layout"],r["n"],r["amplitude"]):r["minimum_witness_D"] for r in records}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.72,2.36))
        fig.subplots_adjust(left=.17, right=.965, top=.94, bottom=.22)
        style_axis(ax)
        ax.set_xscale("log",base=2)
        for layout, name, color, marker, ls in [
                ("repeated","Periodic",BLUE,"o","-"),("shuffled","Shuffled",RED,"s","--")]:
            # Hollow surrounds a smaller filled marker wherever the settings coincide.
            ax.plot(sizes,[series[layout,n,3] for n in sizes], color=color, marker=marker,
                    label=name, linestyle=ls, linewidth=1.35, markersize=6 if layout=="repeated" else 2.7,
                    markerfacecolor="white" if layout=="repeated" else color, markeredgewidth=1.2)
        ax.set(xlim=(26,620),xticks=sizes,xticklabels=[str(n) for n in sizes],
               ylim=(15.35,19),yticks=[16,17,18,19],xlabel="Ring size (modules)",
               ylabel="Minimum pulse width\n(updates)")
        ax.legend(loc="upper left",ncol=2,frameon=False,fontsize=8)
        fig.savefig(output, metadata={"CreationDate":None})
        if preview:
            fig.savefig(preview,dpi=160)
        plt.close(fig)
