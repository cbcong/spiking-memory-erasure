"""Reconstruct every emitted spike in the 32-module weak-ring example."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
from experiments.run_weak_ring_witness import transition


def events(amplitude, expected):
    n = 32
    weights = np.full((n, 1), 2, dtype=np.int64)
    weights[0] = 1
    state = tuple(np.zeros((n, 2), dtype=np.int64) for _ in range(3))
    for _ in range(5):
        state = transition(state, weights, (14, 0), 0)
    history = [state[2].any(axis=1)]
    for _ in range(78):
        state = transition(state, weights, (0, 0), amplitude)
        history.append(state[2].any(axis=1))
    if np.any(state[1]) or np.any(state[2]):
        raise ValueError("Recording ended before currents and flags decayed.")
    tick, node = np.nonzero(np.asarray(history))
    last = np.full(n,-1)
    np.maximum.at(last,node,tick)
    np.testing.assert_array_equal(last, expected)
    return tick,node


def main(output_pdf=None):
    archive = json.loads((HERE/"data"/"weak_ring_scaling_v120.json").read_text(encoding="utf-8"))
    expected = {r["amplitude"]:r["last_spike"] for r in archive["records"] if r["n"]==32}
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.linewidth":.7,"axes.edgecolor":"#686A6C",
                         "axes.labelcolor":"#242426","text.color":"#242426",
                         "xtick.color":"#242426","ytick.color":"#242426"}):
        fig,axes=plt.subplots(1,2,figsize=(7.05,3.02),sharex=True,sharey=True)
        fig.subplots_adjust(left=.095,right=.984,top=.90,bottom=.23,wspace=.17)
        for ax,peak,color,tag in zip(axes,(3,4),("#30465A","#7D3D49"),("a","b")):
            t,j=events(peak,expected[peak])
            ax.scatter(t,j,marker="|",s=8,linewidths=.60,c=color,alpha=.90)
            last=max(expected[peak])
            ax.axvline(last,ls=(0,(2,3)),color="#686A6C",lw=.7,zorder=0)
            ax.text(last,.982,f"tick {last}",transform=ax.get_xaxis_transform(),
                    ha="right" if peak==3 else "left",va="bottom",color=color,fontsize=8)
            ax.set(xlim=(-2,67),ylim=(-1.5,34),xticks=[0,16,32,48,64],
                   yticks=[0,8,16,24,31],xlabel=f"({tag})   $E={peak}$   ·   Tick after write")
            ax.tick_params(length=3)
        axes[0].set_ylabel("Module along the ring")
        fig.savefig(output_pdf or HERE / "4D.pdf", metadata={"CreationDate":None})
        if output_pdf is None:
            fig.savefig(HERE / "4D.png", dpi=190)
        plt.close(fig)


if __name__=="__main__":
    main()
