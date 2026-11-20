"""Scalable spike-partitioned interval propagation on the frozen graph panel."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

from experiments.run_core_scale_comparison import weights
from integer_snn.topology_certificate import MODEL


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PANEL = ROOT / "results/locked_erasure_scale_20261119_v1/report.json"
SATURATION_MARGIN = MODEL.qv + MODEL.qi + (
    MODEL.alpha_num * (MODEL.threshold - 1) // (1 << MODEL.alpha_shift)
)
SOURCE_FILES = (
    "experiments/run_box_erasure_scale.py",
    "experiments/run_core_scale_comparison.py",
    "integer_snn/topology_certificate.py",
    "integer_snn/model.py",
    "integer_snn/network.py",
)


class MarginalIntervals:
    """Per-neuron intervals conditioned on its preceding spike bit.

    The two sides have identical bounds by symmetry of both the model and the
    complete initial set. Inter-neuron correlations are relaxed. The silent
    interval is never empty: starting at zero gives a quiet trajectory for
    every protocol in the search. A negative spike-conditioned current bound
    represents an empty firing interval.
    """

    def __init__(self, matrix):
        if any(row[i] for i, row in enumerate(matrix)):
            raise ValueError("this benchmark expects zero diagonal")
        dense = np.array(matrix, dtype=np.int64)
        if np.any(dense < 0) or np.max(dense.sum(axis=1)) >= 2**60:
            raise ValueError("expected nonnegative weights with bounded row sums")
        self.weights = csr_matrix(dense)
        self.shape = (len(matrix),)
        self.ticks = 0
        self.initial = (
            np.full(self.shape, MODEL.threshold - 1, dtype=np.int64),
            np.full(self.shape, MODEL.qi, dtype=np.int64),
            np.full(self.shape, MODEL.qi, dtype=np.int64),
        )

    def advance(self, upper, amplitude):
        self.ticks += 1
        voltage, quiet_i, spike_i = upper
        may_spike = spike_i >= 0
        incoming = self.weights @ may_spike.astype(np.int64)
        quiet_i = MODEL.beta_num * quiet_i // (1 << MODEL.beta_shift)
        fired_i = np.minimum(MODEL.qi, MODEL.beta_num * spike_i
                             // (1 << MODEL.beta_shift) + MODEL.weight)
        decay = (np.sign(voltage) * (MODEL.alpha_num * np.abs(voltage)
                                    // (1 << MODEL.alpha_shift)))
        quiet_v = decay + quiet_i + incoming - amplitude
        fired_v = fired_i + incoming - amplitude
        next_voltage = np.clip(np.maximum(quiet_v, np.where(may_spike, fired_v, -MODEL.qv)),
                               -MODEL.qv, MODEL.threshold - 1)
        next_quiet_i = np.maximum(quiet_i, np.where(may_spike, fired_i, -1))
        next_spike_i = np.maximum(
            np.where(quiet_v >= MODEL.threshold, quiet_i, -1),
            np.where(may_spike & (fired_v >= MODEL.threshold), fired_i, -1),
        )
        return next_voltage, next_quiet_i, next_spike_i

    def recovery(self, upper, budget):
        lower = np.full(self.shape, -MODEL.qv, dtype=np.int64)
        for tick in range(1, budget + 1):
            # Current and network excitation are nonnegative. Local mutual
            # inhibition is as large as its presynaptic partner may fire.
            lower = np.minimum(0, -(MODEL.alpha_num * -lower // (1 << MODEL.alpha_shift))
                               - MODEL.inhibition * (upper[2] >= 0))
            lower = np.maximum(-MODEL.qv, lower)
            upper = self.advance(upper, 0)
            if not (np.any(lower) or np.any(upper[0]) or np.any(upper[1])
                    or np.any(upper[2] >= 0)):
                return tick
        return None

    def plan_at_amplitude(self, amplitude, deadline):
        pulse = self.initial
        candidates = []
        recovery_floor = 0
        magnitude = MODEL.qv
        while magnitude:
            magnitude = MODEL.alpha_num * magnitude // (1 << MODEL.alpha_shift)
            recovery_floor += 1
        # The recovery lower interval starts at -qv regardless of pulse
        # length or magnitude. The rebound check can never close it sooner.
        for d in range(deadline - recovery_floor + 1):
            budget = min([deadline - d] + [total - d for total, _ in candidates])
            if budget < recovery_floor:
                break
            recovery = self.recovery(pulse, budget)
            if recovery is not None:
                candidates.append((d + recovery, d))
            if d + recovery_floor < deadline:
                pulse = self.advance(pulse, amplitude)
        if candidates:
            total, duration = min(candidates)
            return amplitude, duration, total - duration
        return None

    def search(self, deadline, max_amplitude):
        # All upper bounds are componentwise nonincreasing in E, ordered
        # with the empty spike partition below a nonempty one. The recovery
        # lower bound moves towards zero when a possible spike disappears.
        # Thus existence over all D is monotone within this abstraction.
        high_plan = self.plan_at_amplitude(max_amplitude, deadline)
        if high_plan is None:
            return None
        lower, upper = 0, max_amplitude - 1
        best = high_plan
        while lower <= upper:
            amplitude = (lower + upper) // 2
            plan = self.plan_at_amplitude(amplitude, deadline)
            if plan is None:
                lower = amplitude + 1
            else:
                best = plan
                upper = amplitude - 1
        return best


def summarize(records):
    paired = [r for r in records if r["box"] is not None]
    box_sum = sum(r["box"][0] for r in paired)
    topology_sum = sum(r["topology"][0] for r in paired)
    return dict(
        n=len(records), paired=len(paired),
        box_amplitude_sum=box_sum, topology_amplitude_sum=topology_sum,
        topology_relative_reduction=1 - topology_sum / box_sum if box_sum else None,
        topology_lower=sum(r["topology"][0] < r["box"][0] for r in paired),
        equal=sum(r["topology"][0] == r["box"][0] for r in paired),
        box_lower=sum(r["box"][0] < r["topology"][0] for r in paired),
    )


def run(output, deadline):
    source = SOURCE_PANEL.read_bytes()
    panel = json.loads(source)
    output.mkdir(exist_ok=False, parents=True)
    design = dict(
        deadline=deadline,
        source_panel_sha256=hashlib.sha256(source).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in SOURCE_FILES},
        model=repr(MODEL),
        search="all D+H<=T within the marginal interval abstraction; 0 through saturation",
        saturation_margin=SATURATION_MARGIN,
        abstraction="separate spike-conditioned per-neuron V and I upper bounds; symmetry; "
                    "recovery lower V starts at register floor on every release",
    )
    (output / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    records = []
    for row in panel["rows"]:
        n, family, seed = row["n"], row["family"], row["seed"]
        matrix = weights(n, family, seed)
        digest = hashlib.sha256(json.dumps(matrix).encode()).hexdigest()
        if digest != row["matrix_sha256"]:
            raise AssertionError((n, family, seed, digest))
        start = perf_counter()
        checker = MarginalIntervals(matrix)
        # At this cap raw voltages saturate even under maximum excitation,
        # so all larger amplitudes have the same pulse abstraction.
        cap = max(map(sum, matrix)) + SATURATION_MARGIN
        plan = checker.search(deadline, cap)
        runtime = perf_counter() - start
        plans = row["plans"][str(deadline)]
        records.append(dict(n=n, family=family, seed=seed, matrix_sha256=digest,
                            box=plan, topology=plans["locked"], row=plans["row_tight"],
                            solve_seconds=runtime, abstract_steps=checker.ticks))
        if row["trial"] == 7:
            print(n, family, summarize(records[-8:]), flush=True)
    report = dict(
        design=design,
        summary=summarize(records),
        by_family={family: summarize([r for r in records if r["family"] == family])
                   for family in panel["design"]["families"]},
        rows=records,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=int, choices=(20, 40, 60, 100), default=20)
    args = parser.parse_args()
    run(args.output, args.deadline)
