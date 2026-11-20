"""Audit frozen protocols, paired intervals and finite-deadline lower witnesses."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from experiments.compare_erasure_baselines import rank
from experiments.run_core_scale_comparison import weights
from integer_snn.topology_certificate import MODEL


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261119
RESAMPLES = 10000
T = 60


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["n"], row["family"]].append([row["baseline"][0], row["topology"][0]])
    rng = np.random.default_rng(SEED)
    draws = np.zeros((RESAMPLES, 2), dtype=np.int64)
    for key in sorted(groups):
        group = np.asarray(groups[key])
        indices = rng.integers(len(group), size=(RESAMPLES, len(group)))
        draws += group[indices].sum(axis=1)
    return np.quantile(1 - draws[:, 1] / draws[:, 0], [.025, .975]).tolist()


def static_witness(matrix):
    n = len(matrix)
    residual = matrix.sum(axis=1)
    present = np.ones(n, dtype=bool)
    k = -1
    for _ in range(n):
        ids = np.flatnonzero(present)
        node = ids[np.argmin(residual[ids])]
        if residual[node] > k:
            k = int(residual[node])
            core = ids
        present[node] = False
        residual -= matrix[:, node]
    assert matrix[np.ix_(core, core)].sum(axis=1).min() == k
    return max(3, k + 2), k, core.tolist()


def advance(edges, state, amplitude):
    # One active channel suffices for a counterexample. On the other channel
    # all spikes and currents stay zero and all voltages stay nonpositive.
    voltage, current, spikes = state
    current = np.minimum(MODEL.qi, (MODEL.beta_num * current >> MODEL.beta_shift)
                         + MODEL.weight * spikes)
    decayed = np.sign(voltage) * (MODEL.alpha_num * abs(voltage) >> MODEL.alpha_shift)
    voltage = np.clip(decayed + current + edges @ spikes - amplitude, -MODEL.qv, MODEL.qv)
    spikes = (voltage >= MODEL.threshold).astype(np.int64)
    return np.where(spikes, 0, voltage), current, spikes


def check_lower_amplitude(edges, amplitude):
    # Zero is absorbing. A nonzero state at T rules out every shorter recovery
    # at this D. Enumerate D including 0 and T; assume no monotonicity in E.
    n = edges.shape[0]
    held = (np.zeros(n, dtype=np.int64), np.full(n, MODEL.qi),
            np.ones(n, dtype=np.int64))
    terminal_counts = []
    for duration in range(T + 1):
        released = held
        for _ in range(T - duration):
            released = advance(edges, released, 0)
        terminal_counts.append(int(np.count_nonzero(released[0] | released[1] | released[2])))
        if duration < T:
            held = advance(edges, held, amplitude)
    return terminal_counts


def summarize(rows):
    baseline = sum(row["baseline"][0] for row in rows)
    upper = sum(row["topology"][0] for row in rows)
    lower = sum(row["lower"] for row in rows)
    return dict(graphs=len(rows), baseline_sum=baseline, upper_sum=upper, lower_sum=lower,
                static_lower_sum=sum(row["static_lower"] for row in rows),
                reduction=1-upper/baseline, reduction_ci95=interval(rows),
                max_possible_reduction=1-lower/baseline,
                remaining_fraction_of_baseline=(upper-lower)/baseline,
                gap_upper_histogram=dict(sorted(Counter(row["topology"][0]-row["lower"]
                                                         for row in rows).items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    panel = ROOT / "results/box_erasure_scale_20261119_t60/report.json"
    source = json.loads(panel.read_bytes())
    assert source["design"]["deadline"] == T and source["design"]["model"] == repr(MODEL)
    rows = []
    for row in source["rows"]:
        network = weights(row["n"], row["family"], row["seed"])
        assert hashlib.sha256(json.dumps(network).encode()).hexdigest() == row["matrix_sha256"]
        matrix = np.asarray(network, dtype=np.int64)
        static_lower, k, core = static_witness(matrix)
        edges = csr_matrix(matrix)
        baseline = min(row["box"], row["row"], key=rank)
        protocol = row["topology"]
        assert sum(protocol[1:]) <= T and sum(baseline[1:]) <= T
        assert static_lower <= protocol[0] <= baseline[0]
        lower = static_lower
        replay = {}
        for amplitude in range(static_lower, protocol[0]):
            counts = check_lower_amplitude(edges, amplitude)
            replay[amplitude] = counts
            if min(counts) == 0:
                break
            lower += 1
        rows.append(dict(n=row["n"], family=row["family"], seed=row["seed"],
                         graph_sha256=row["matrix_sha256"], baseline=baseline, topology=protocol,
                         weighted_core=k, witness_vertices=core, static_lower=static_lower,
                         lower=lower, active_nonzero_counts_at_T_by_E_and_D=replay))
    assert len({row["graph_sha256"] for row in rows}) == 72
    assert set(Counter((row["n"],row["family"]) for row in rows).values()) == {8}
    result = dict(source_sha256=file_digest(panel), deadline=T,
                  interval_method="paired stratified percentile, strata=(module count, family)",
                  resamples=RESAMPLES, seed=SEED, post_hoc=True,
                  saturated_initial="all active channels V=0, I=15, s=1; partners zero",
                  replay_definition="terminal nonzero active channels at T; index is D=0..60",
                  script_sha256=file_digest(Path(__file__)), summary=summarize(rows),
                  by_family={f:summarize([r for r in rows if r["family"] == f])
                             for f in ("sparse","dense","regular")}, rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
    print(json.dumps([result["summary"], result["by_family"]]))


if __name__ == "__main__":
    main()
