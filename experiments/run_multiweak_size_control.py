"""Exact writable-state release tests across ring sizes at fixed largest gap."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from experiments.run_weak_ring_witness import transition


def release_orbit(state, weights):
    """Classify release by exact rest or the first repeated full register state."""
    seen = {}
    for h in range(1025):
        if not any(np.any(field) for field in state):
            return "zero", h, 0
        key = tuple(field.tobytes() for field in state)
        if key in seen:
            return "cycle", h, h - seen[key]
        seen[key] = h
        state = transition(state, weights, (0, 0), 0)
    raise RuntimeError("Unclassified release")


def evaluate(layout, gaps, amplitude):
    n = sum(gaps)
    weights = np.full((n, 1), 2, dtype=np.int64)
    weights[np.cumsum([0] + gaps[:-1])] = 1
    state = tuple(np.zeros((n, 2), dtype=np.int64) for _ in range(3))
    for _ in range(5):
        state = transition(state, weights, (14, 0), 0)
    assert np.all(state[1][:, 0] == 15) and np.all(state[2][:, 0] == 1)
    rejected = []
    first_safe = None
    last_spike = 0
    fixed_endpoint = None
    for d in range(8 * max(gaps) + 48):
        if np.any(state[2]):
            last_spike = d
        if first_safe is None:
            status, duration, period = release_orbit(state, weights)
            if status == "zero":
                first_safe = [d, duration]
            else:
                rejected.append(dict(d=d, repeat_tick=duration, period=period))
        if d == 10:
            released = state
            for _ in range(10):
                released = transition(released, weights, (0, 0), 0)
            fixed_endpoint = not any(np.any(field) for field in released)
        if d >= 10 and first_safe and not np.any(state[1]) and not np.any(state[2]):
            assert not np.any(state[0] > 0)
            break
        state = transition(state, weights, (0, 0), amplitude)
    else:
        raise RuntimeError(f"Guard reached for {n=} {layout=} {amplitude=}")
    if amplitude == 4 and not fixed_endpoint:
        raise AssertionError("Published 10+10 certificate contradicted")
    return dict(layout=layout, n=n, gaps=gaps, longest_strong_run=max(gaps) - 1,
                amplitude=amplitude, last_spike=last_spike, minimum_witness_D=first_safe[0],
                H_at_first_safe_D=first_safe[1], fixed_10_10_zero=fixed_endpoint,
                rejected_durations_from_zero=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for n in (32, 64, 128, 256, 512):
        repeated = [8, 4, 3, 1] * (n // 16)
        shuffled = repeated.copy()
        random.Random(56170 + n).shuffle(shuffled)
        cases.extend([("repeated", repeated), ("shuffled", shuffled)])
    cases.extend([("four_even", [16] * 4), ("four_clustered", [61, 1, 1, 1])])
    records = [evaluate(label, gaps, amplitude)
               for label, gaps in cases for amplitude in (3, 4)]
    args.output.write_text(json.dumps({
        "state": "From entire ring at zero, five synchronous (14,0) writes",
        "time": "d=0 at last write. Minimum D scans all durations from zero until exact "
                "zero is reachable after release. H is the number of unforced ticks until "
                "all registers first equal zero for that D. Neither optimizes over states.",
        "controls": "Periodic repeats are a symmetry check. Independent of that check, "
                    "each shuffled ring permutes the same gap multiset with random.Random(56170+n).",
        "records": records
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
