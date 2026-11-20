"""Hold four weak edges fixed in number while changing their spacing."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.run_weak_ring_witness import run as run_single_weak_edge
from experiments.run_weak_ring_witness import transition
from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL


def designs(n):
    """Use a fixed nine-level grid including even spacing and full clustering."""
    for longest in sorted({n // 4 + i * (n - 3 - n // 4) // 8 for i in range(9)}):
        total = n - longest
        width, remainder = divmod(total, 3)
        dispersed = tuple(width + (i < remainder) for i in range(3))
        concentrated = []
        for slots in (2, 1, 0):
            gap = min(longest, total - slots)
            concentrated.append(gap)
            total -= gap
        yield (longest, *dispersed), (longest, *concentrated)


def measure(gaps, amplitude, check_reference=False):
    n = sum(gaps)
    nodes = np.cumsum((0, *gaps[:-1]))
    weights = np.full((n, 1), 2, dtype=np.int64)
    weights[nodes] = 1
    states = tuple(np.zeros((n, 2), dtype=np.int64) for _ in range(3))
    ref_state = (zero_state(),) * n
    checked = 0
    if check_reference:
        spec = NetworkSpec(MODEL, tuple(
            tuple(int(weights[i, 0]) if j == (i - 1) % n else 0 for j in range(n))
            for i in range(n)
        ))

    def advance(write=(0, 0), peak=0):
        nonlocal states, ref_state, checked
        states = transition(states, weights, write, peak)
        if check_reference:
            ref_state = network_reference_step(spec, ref_state, peak, inputs=(write,) * n)
            for got, field in zip(states, ("v", "current", "spike")):
                np.testing.assert_array_equal(got, [getattr(cell, field) for cell in ref_state])
            checked += 1

    for _ in range(5):
        advance(write=(14, 0))
    assert np.all(states[1][:, 0] == 15) and np.all(states[2][:, 0] == 1)
    assert not np.any(states[1][:, 1]) and not np.any(states[2][:, 1])
    written_state, written_ref = states, ref_state
    last_spikes = np.zeros(n, dtype=int)
    for tick in range(1, 10 * n + 64):
        advance(peak=amplitude)
        last_spikes[np.any(states[2], axis=1)] = tick
        if not (np.any(states[1]) or np.any(states[2])):
            break
    else:
        raise RuntimeError(f"Persistent rate at {gaps=}, {amplitude=}")

    states, ref_state = written_state, written_ref
    for tick in range(20):
        advance(peak=amplitude if tick < 10 else 0)
    return dict(n=n, weak_targets=nodes.tolist(), gaps=list(gaps), amplitude=amplitude,
                longest_strong_run=max(gaps) - 1, last_spikes=last_spikes.tolist(),
                last_network_spike=int(last_spikes.max()),
                fixed_10_10_zero=not any(np.any(register) for register in states),
                reference_updates=checked)


def build_report():
    records = []
    for n in (32, 64):
        for dispersed, concentrated in designs(n):
            for amplitude in (3, 4):
                for gaps in dict.fromkeys((dispersed, concentrated)):
                    endpoint = gaps in ((n // 4,) * 4, (n - 3, 1, 1, 1))
                    row = measure(gaps, amplitude, check_reference=endpoint)
                    row["layout"] = ("both" if dispersed == concentrated else
                                     "dispersed" if gaps == dispersed else "concentrated")
                    if amplitude == 4:
                        assert row["fixed_10_10_zero"]
                    records.append(row)
        for amplitude in (3, 4):
            old = run_single_weak_edge(n, amplitude)
            check = measure((n,), amplitude)
            assert check["last_spikes"] == old["last_spike"]
    return {
        "design": (
            "32- and 64-module directed rings with four weak (weight 1) edges and all "
            "remaining edges of weight 2. Gaps enumerate nine fixed integer values from n/4 "
            "to n-3 for the largest weak-to-weak distance, allocating remaining gaps either "
            "as evenly as possible (dispersed) or greedily as long as possible (concentrated). "
            "Common five-update (14,0) write from zero. Continuous pressure measures the last "
            "network spike. A separate 10-driven, 10-released schedule is checked for full zero."
        ),
        "records": records,
    }


def plot(records, output):
    from matplotlib import use
    use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    ink, blue, coral = "#202A37", "#20639B", "#BB5B39"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink, "text.color": ink}):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.85), sharey=True)
        fig.subplots_adjust(left=.10, right=.975, bottom=.25, top=.83, wspace=.16)
        for panel, (ax, n) in enumerate(zip(axes, (32, 64))):
            series = {
                (amplitude, layout): [r for r in records if r["n"] == n
                                      and r["amplitude"] == amplitude
                                      and r["layout"] in (layout, "both")]
                for amplitude in (3, 4) for layout in ("dispersed", "concentrated")
            }
            ax.set_axisbelow(True)
            ax.grid(axis="y", lw=.6, color="#E0E6EB")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            low = series[4, "dispersed"]
            high = series[3, "dispersed"]
            x = [r["longest_strong_run"] for r in high]
            upper = [r["last_network_spike"] for r in high]
            lower = [r["last_network_spike"] for r in low]
            ax.vlines(x, lower, upper, color="#B2BCC7", lw=.75, zorder=1)
            for amplitude, color, marker, linestyle in ((3, blue, "o", "-"),
                                                         (4, coral, "s", "--")):
                rows = series[amplitude, "dispersed"]
                ax.plot(x, [r["last_network_spike"] for r in rows], color=color,
                        marker=marker, linestyle=linestyle, lw=1.5, ms=3.8, zorder=3)
                alternatives = [r["last_network_spike"] for r in series[amplitude, "concentrated"]]
                changed = [(a, b) for a, b, r in zip(x, alternatives, rows)
                           if b != r["last_network_spike"]]
                if changed:
                    ax.scatter(*zip(*changed), edgecolor=color, facecolor="white",
                               marker=marker, s=18, lw=1, zorder=4)
            if n == 64:
                example_index = x.index(31)
                same_run_time = series[3, "concentrated"][example_index]["last_network_spike"]
                ax.vlines(x[example_index], upper[example_index], same_run_time,
                          color=blue, lw=.9, ls=":", zorder=2)
                ax.annotate(f"+{same_run_time - upper[example_index]} ticks",
                            (x[example_index], same_run_time), xytext=(-2, 8),
                            textcoords="offset points", ha="center", color=blue)
            for index, align in ((0, "left"), (-1, "right")):
                point = (x[index], upper[index])
                ax.annotate(str(point[1]), point, xytext=(3 if index == 0 else -3, 5),
                            textcoords="offset points", ha=align, color=blue, fontsize=8)
            ax.set(ylim=(0, 76), xlim=(x[0] - 3, x[-1] + 4))
            ax.text(.5, -.29, f"({chr(97 + panel)}) {n} modules", ha="center",
                    transform=ax.transAxes)
            ticks = [x[0], *[a for a in (16, 24, 30, 40, 50) if x[0] + 2 < a < x[-1] - 2], x[-1]]
            ax.set_xticks(ticks)
        axes[0].set_ylabel("Last network spike (ticks)")
        fig.supxlabel("Longest run of strong edges", x=.54, y=.093, fontsize=9)
        legend = [Line2D([0], [0], color=blue, marker="o", label="$E=3$"),
                  Line2D([0], [0], color=coral, marker="s", ls="--", label="$E=4$"),
                  Line2D([0], [0], color=blue, marker="o", mfc="white", lw=0,
                         label="Other gap layout, if different")]
        fig.legend(handles=legend, frameon=False, ncol=3, loc="upper center",
                   bbox_to_anchor=(.53, .985), columnspacing=1.6)
        fig.savefig(output.with_suffix(".png"), dpi=185)
        fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None})
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="Output stem for PNG and vector PDF.")
    args = parser.parse_args()
    report = build_report()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plot(report["records"], args.figure)
