"""Simulate paired trajectories on the weak-edge rings of the time-scale theorem."""

import argparse
import json
from pathlib import Path

import numpy as np

from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL


def transition(state, weights, write, erase):
    voltage, current, spike = state
    p = MODEL
    incoming = weights * np.roll(spike, 1, axis=0)
    current = np.clip(p.beta_num * current // (1 << p.beta_shift) + p.weight * spike, 0, p.qi)
    decay = np.sign(voltage) * (p.alpha_num * np.abs(voltage) // (1 << p.alpha_shift))
    voltage = np.clip(decay + current - p.inhibition * spike[:, ::-1] + incoming
                      + write - erase, -p.qv, p.qv)
    spike = (voltage >= p.threshold).astype(np.int64)
    return np.where(spike, 0, voltage), current, spike


def run(n, erase):
    weights = np.full((n, 1), 2, dtype=np.int64)
    weights[0] = 1
    state = tuple(np.zeros((n, 2), dtype=np.int64) for _ in range(3))
    reference = (zero_state(),) * n
    checked = 0
    if n <= 16:
        matrix = tuple(tuple(int(weights[i, 0]) if j == (i - 1) % n else 0
                             for j in range(n)) for i in range(n))
        spec = NetworkSpec(MODEL, matrix)

    def advance(write=(0, 0), peak=0):
        nonlocal state, reference, checked
        state = transition(state, weights, write, peak)
        if n <= 16:
            reference = network_reference_step(spec, reference, peak, inputs=(write,) * n)
            for actual, field in zip(state, ('v', 'current', 'spike')):
                expected = [getattr(item, field) for item in reference]
                np.testing.assert_array_equal(actual, expected)
            checked += 1

    for _ in range(5):
        advance(write=(14, 0))
    if not (np.all(state[1][:, 0] == 15) and np.all(state[2][:, 0] == 1)
            and not np.any(state[1][:, 1]) and not np.any(state[2][:, 1])):
        raise AssertionError("Write witness did not reach its specified state")
    written_state, written_ref = state, reference
    # Tick zero contains the last write. A module can fire for the last time there.
    last = np.zeros(n, dtype=np.int64)
    for d in range(1, 10 * n + 64):
        advance(peak=erase)
        last[np.any(state[2], axis=1)] = d
        if not np.any(state[1]) and not np.any(state[2]):
            break
    else:
        raise RuntimeError("Drive guard reached")
    if np.any(state[0] > 0):
        raise AssertionError("Release rule expected nonpositive voltages")
    h = 0
    while any(np.any(field) for field in state):
        advance()
        h += 1
        if h > 30 or np.any(state[2]):
            raise AssertionError("Unexpected release trajectory")
    fixed = None
    if erase == 4:
        state, reference = written_state, written_ref
        for t in range(20):
            advance(peak=4 if t < 10 else 0)
        fixed = not any(np.any(field) for field in state)
        if not fixed:
            raise AssertionError("Fixed certificate failed on the write witness")
    return dict(n=n, amplitude=erase, last_spike=last.tolist(),
                network_last_spike=int(last.max()), drive_ticks=d, release_ticks=h,
                total_ticks=d + h, reference_updates=checked, fixed_10_10_zero=fixed)


def plot(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))
        for e, color, marker, line in [(3, '#255379', 'o', '-'), (4, '#a44526', 's', '--')]:
            rows = [row for row in records if row['amplitude'] == e]
            nodes = next(row['last_spike'] for row in rows if row['n'] == 32)
            fmt = dict(color=color, marker=marker, lw=1.15, ms=3.8,
                       mfc='white', linestyle=line, label=f'$E={e}$')
            axes[0].plot(range(len(nodes)), nodes, **fmt)
            axes[1].plot([r['n'] for r in rows], [r['network_last_spike'] for r in rows],
                         **fmt)
        axes[0].set(xlabel='(a) Distance from node 0 (edges)', ylabel='Last spike (tick)',
                    xticks=[0, 8, 16, 24, 31])
        axes[1].set(xscale='log', yscale='log', xlabel='(b) Ring size (modules)',
                    ylabel='Network last spike (tick)', xticks=[2, 8, 32, 128, 512])
        axes[1].xaxis.set_major_formatter(ScalarFormatter())
        axes[1].minorticks_off()
        for ax in axes:
            ax.grid(axis='y', color='0.86', linewidth=0.5)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False, loc='upper left')
        fig.subplots_adjust(left=0.080, right=0.98, bottom=0.22, top=0.95, wspace=0.37)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    records = [run(2 ** k, e) for k in range(1, 10) for e in (3, 4)]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        'definition': 'Tick zero follows five (14,0) writes from zero. Drive to first zero '
                      'current and spike flags in all modules; release to full zero. '
                      'Last spike includes tick zero. Each record is one deterministic trajectory.',
        'records': records,
    }, indent=2) + '\n', encoding='utf-8')
    plot(records, args.figure)
    for r in records:
        print(*(r[k] for k in ('n', 'amplitude', 'network_last_spike', 'drive_ticks',
                               'release_ticks', 'total_ticks', 'reference_updates', 'fixed_10_10_zero')))
