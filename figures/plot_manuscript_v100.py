"""Generate the single-module trajectory and prospective graph comparisons."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from experiments.run_locked_erasure_scale import plans_at
from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec
from integer_snn.topology_certificate import MODEL
from tasks.delayed_query import advance, fresh_network


ROOT = Path(__file__).resolve().parents[1]
SLATE, GARNET, INK, GREY = "#30465A", "#7D3D49", "#242426", "#686A6C"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "axes.spines.right": False,
                     "axes.spines.top": False, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "text.color": INK, "axes.linewidth": .7})


def module(folder):
    fig, ax = plt.subplots(figsize=(6.35, 2.85))
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    ax.set(xlim=(0, 10), ylim=(0, 4.45), aspect="equal")
    ax.axis("off")
    def arrow(p, q, color=INK, style="-|>", curve=0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, color=color, lw=1.35,
                                     mutation_scale=10, shrinkA=0, shrinkB=0,
                                     connectionstyle=f"arc3,rad={curve}"))

    ax.add_patch(Rectangle((3.1,.57), 4.05,3.12, lw=.85, ec="#909092", fc="#F7F7F6"))
    ax.text(5.12,3.42,"Memory module $i$",ha="center",weight="bold")
    for y, channel in [(2.72,0),(1.40,1)]:
        ax.add_patch(Circle((5.1,y), .32, facecolor="white", ec=INK, lw=1.6))
        ax.text(5.1,y,rf"$c_{channel}$",ha="center",va="center",fontsize=11)
        ax.text(.15,y+.18,rf"channel {channel} input",fontsize=8.5)
        ax.text(.15,y-.15,rf"$\sum_j W_{{ij}}s_{{j{channel}}}(t)$",fontsize=10)
        arrow((2.38,y),(4.75,y))
        arrow((5.43,y+.03),(7.62,y+.03),INK)
        ax.text(7.78,y+.03,rf"$s_{{i{channel}}}(t+1)$",va="center",fontsize=11)
        # Delayed positive feedback connects the spike output to its own cell.
        offset = .39 if channel == 0 else -.39
        ax.plot([6.35,6.35,5.55],[y+.03,y+offset,y+offset], color=INK,lw=1.15)
        arrow((5.55,y+offset),(5.32,y+offset*.55),INK)
        ax.text(6.48,y+offset,"+6",ha="left",va="center",color=INK,fontsize=8)
    arrow((5.00,2.41),(5.00,1.74),GARNET,style="-[")
    arrow((5.22,1.72),(5.22,2.38),GARNET,style="-[")
    ax.text(5.50,2.06,"−4",color=GARNET,va="center")
    ax.text(.16,.42,r"Common $-E(t)$",va="center",color=GARNET)
    ax.plot([2.55,3.86,3.86],[.42,.42,2.42],color=GARNET,ls="--",lw=1.2)
    for y in (2.42,1.18):
        arrow((3.86,y),(4.80,y+.12),GARNET)
    fig.savefig(folder/"module.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def trajectory(folder):
    spec = NetworkSpec(MODEL, ((0,),))
    amplitude, duration, recovery = plans_at(spec,60)["locked"]
    assert (amplitude, duration, recovery)==(3,9,8)
    reset, rewrite, finish = 16, 39, 54
    state = fresh_network(spec)
    history = [state[0]]
    signed_input = []
    for t in range(finish):
        write = t < 2 or rewrite <= t < rewrite+2
        erase = amplitude if reset <= t < reset+duration else 0
        state = advance(spec, state, 1, inputs=((14,0),) if write else None,
                        erase=erase, reference=True)
        history.append(state[0])
        signed_input.append(14 if write else -erase)
    zero_tick = next(t for t in range(reset+1,rewrite) if history[t]==zero_state())
    guaranteed = reset+duration+recovery
    last_spike = max(t for t in range(reset, rewrite) if any(history[t].spike))
    assert zero_tick <= guaranteed and all(s==zero_state() for s in history[zero_tick:rewrite+1])
    fig, (command,raster,states) = plt.subplots(3,1,figsize=(6.35,4.05),sharex=True,
        gridspec_kw={"height_ratios":[.9,.70,1.55]})
    fig.subplots_adjust(left=.145,right=.975,top=.92,bottom=.15,hspace=.58)
    for ax in (command,raster,states):
        ax.axvspan(reset,reset+duration, color="#DAD7D8",zorder=0)
        ax.axvspan(reset+duration,guaranteed, color="#F1F0EF",zorder=0)
        ax.spines["left"].set_color(GREY)
        ax.spines["bottom"].set_color(GREY)
        ax.tick_params(length=3, width=.6)
    time = np.arange(finish+1)
    command.step(time, signed_input+[0],where="post",color=INK,lw=1.25)
    command.set(yticks=[-3,0,14],ylim=(-7,18),ylabel="Input (code)")
    command.set_xlabel("(a) Applied input", labelpad=4)
    for x,s in [(reset+duration/2,"Pressure"),((reset+duration+guaranteed)/2,"Release")]:
        command.text(x,1.07,s,transform=command.get_xaxis_transform(),ha="center",fontsize=8)
    for c in (0,1):
        times=[t for t,s in enumerate(history) if s.spike[c]]
        raster.vlines(times,c-.23,c+.23,lw=1.25,color=INK)
    raster.set(yticks=[0,1],yticklabels=["0","1"],ylim=(-.5,1.5),ylabel="Channel")
    raster.set_xlabel("(b) Spike raster", labelpad=4)
    labels = [("current",r"$\max_c I_c$",SLATE,"-"), ("v",r"$\max_c |v_c|$",GARNET,"--")]
    for field,label,color,ls in labels:
        y=[max(abs(v) for v in getattr(s,field)) for s in history]
        states.step(time,y,where="post",ls=ls,color=color,lw=1.3,label=label)
    states.plot(zero_tick,0,marker="o",ls="none",mfc="white",mec=INK,ms=4.2,zorder=3)
    states.annotate("Full state zero",(zero_tick,0),xytext=(zero_tick+1.7,6),
                    fontsize=8,ha="center",arrowprops={"arrowstyle":"-","color":GREY,"lw":.7})
    states.set(yticks=[0,5,10,15],ylim=(-1.2,16),ylabel="State magnitude",xlabel="(c) State by update index")
    states.legend(loc="upper right",ncol=2,handlelength=1.5,fontsize=7.8,frameon=False)
    states.set_xlim(0,finish)
    fig.savefig(folder/"protocol.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"pressure [{reset},{reset+duration}); guarantee {guaranteed}; last spike state "
          f"{last_spike}; observed zero {zero_tick}; rewrite {rewrite}")


def replication(folder, *, preview=False):
    report = json.loads((ROOT / "results/confirm_strong_20261119_v1/report.json").read_bytes())
    rows = report["rows"]
    assert len(rows) == 72 and all(row["complete"] for row in rows)
    fig, (sparse, dense) = plt.subplots(1, 2, figsize=(6.35, 2.85), sharey=True)
    fig.subplots_adjust(left=.135, right=.975, top=.95, bottom=.245, wspace=.23)
    sparse.set(xlim=(11, 36), xticks=[15, 20, 25, 30, 35],
               xlabel="(a) Sparse: peak reduction (%)", ylabel="Change in applied drive $ED$ (%)")
    dense.set(xlim=(-.18, 2.25), xticks=[0, .5, 1, 1.5, 2],
              xlabel="(b) Dense and regular: peak reduction (%)")
    sparse.set(ylim=(-60, 183), yticks=[-50, 0, 50, 100, 150])
    for ax in (sparse, dense):
        ax.grid(axis="y", color="#DDE2E3", linewidth=.55)
        ax.axhline(0, color=GREY, lw=.85, ls="--")
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.6)

    from fractions import Fraction
    from collections import Counter
    for family, ax, color, marker in [("sparse", sparse, SLATE, "o"),
                                      ("dense", dense, GARNET, "^"),
                                      ("regular", dense, INK, "s")]:
        points = Counter()
        for row in rows:
            if row["family"] != family:
                continue
            e0, d0, _ = row["baseline"]
            e, d, _ = row["topology"]
            points[Fraction(e0-e, e0), Fraction(e*d-e0*d0, e0*d0)] += 1
        assert sum(points.values()) == 24
        for (gain, drive), count in points.items():
            x, y = 100*float(gain), 100*float(drive)
            ax.plot(x, y, marker=marker, ms=4.5, linestyle="none", markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.05, zorder=3)
            if count > 1:
                label = f"Regular ×{count}" if family == "regular" else f"×{count}"
                ax.annotate(label, (x, y), xytext=(5, -10 if family=="regular" else 4),
                            textcoords="offset points", color=color, fontsize=7.5)
    fig.savefig(folder/"replication.pdf", metadata={"CreationDate": None})
    if preview:
        fig.savefig(folder / "replication.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    folder=parser.parse_args().output
    folder.mkdir(parents=True,exist_ok=True)
    trajectory(folder)
    replication(folder)
