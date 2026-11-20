"""Plot actual spike times from the writable three-module obstruction."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experiments.check_three_module_phases import COUNTEREXAMPLE, MODEL, NetworkSpec, checked_step, written_state
from integer_snn.model import zero_state


def render(output):
    spec = NetworkSpec(MODEL, COUNTEREXAMPLE)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.9, 2.45))
        fig.subplots_adjust(left=.13, right=.98, bottom=.19, top=.93, hspace=.30)
        for ax, peak in zip(axes, (3, 4)):
            state = written_state(spec)
            for tick in range(24):
                if tick:
                    state = checked_step(spec, state, erase=peak if tick <= 13 else 0)
                firing = [index for index, cell in enumerate(state) if any(cell.spike)]
                ax.scatter([tick] * len(firing), firing, marker="|", s=37,
                           linewidths=1.2, color="#30465A")
                if peak == 4 and tick == 21:
                    assert state == (zero_state(),) * 3
            ax.axvspan(.5, 13.5, color="#e5e5e4", zorder=-2)
            ax.axvline(13.5, color="#686a6c", ls="--", lw=.7)
            ax.set(yticks=(0, 1, 2), ylim=(-.5, 2.5), xlim=(-.6, 23.6),
                   ylabel=f"$E={peak}$\nModule")
            ax.tick_params(length=3)
        axes[-1].set(xticks=(0, 4, 8, 12, 16, 20, 23), xlabel="Update since end of write")
        fig.savefig(output, metadata={"CreationDate": None})
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    render(parser.parse_args().output)
