"""Test rectangular erasure on cyclic weight-1/2 rings and reachable words."""

import csv
from pathlib import Path

import numpy as np

from experiments.run_weak_ring_witness import transition
from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL


def rotations(bits):
    return {bits[i:] + bits[:i] for i in range(len(bits))}


def trial(mask, all_words, check_reference):
    n = len(mask)
    weak = [i for i, entry in enumerate(mask) if entry == "1"]
    distances = [b - a for a, b in zip(weak, weak[1:] + [weak[0] + n])]
    longest = max(distances) - 1
    weights = np.array([1 if x == "1" else 2 for x in mask])[:, None, None]
    if all_words:
        words = np.arange(2**n)
        digits = (words[None, :] >> np.arange(n - 1, -1, -1)[:, None]) & 1
    else:
        digits = np.zeros((n, 1), dtype=int)
    write = 14 * (digits[:, None, :] == np.arange(2)[None, :, None])
    state = tuple(np.zeros_like(write) for _ in range(3))
    for _ in range(5):
        state = transition(state, weights, write, 0)
    assert np.array_equal(state[1], 15 * (write > 0))
    assert np.array_equal(state[2], write > 0)
    upper_last = 6 + 2 * ((longest + 1) // 2) + 5 * longest
    upper_pulse = upper_last + 4
    best = upper_pulse + 11
    best_d, best_h = None, None

    # Every D below the incumbent is considered. A feasible protocol with
    # equally good total duration need not replace the first minimizer.
    driven = state
    for duration in range(upper_pulse + 11):
        if duration >= best:
            break
        if duration <= longest:
            assert np.any((driven[1][:, 0, 0] == 15) & (driven[2][:, 0, 0] == 1))
        else:
            released = driven
            for waiting in range(best - duration):
                if not any(np.any(field) for field in released):
                    best = duration + waiting
                    best_d, best_h = duration, waiting
                    break
                released = transition(released, weights, 0, 0)
        driven = transition(driven, weights, 0, 3)
    if best_d is None or not (longest + 1 <= best <= upper_pulse + 10):
        raise AssertionError(f"E=3 witness/upper completion bound failed for {mask}")

    # Directly run the proposed universal schedules as well as the minimizer.
    ref_updates = 0
    if check_reference:
        test_word = digits.shape[1] // 3
        matrix = tuple(tuple(int(weights[i, 0, 0]) if j == (i - 1) % n else 0
                             for j in range(n)) for i in range(n))
        spec = NetworkSpec(MODEL, matrix)
        reference = (zero_state(),) * n
        inputs = tuple(tuple(map(int, v)) for v in write[:, :, test_word])
        for _ in range(5):
            reference = network_reference_step(spec, reference, 0, inputs=inputs)
            ref_updates += 1
    for amplitude, duration, waiting in [(3, upper_pulse, 10), (4, 10, 10), (3, best_d, best_h)]:
        joint_state = state
        if check_reference:
            ref_state = reference
        for tick in range(duration + waiting):
            drive = amplitude if tick < duration else 0
            joint_state = transition(joint_state, weights, 0, drive)
            if check_reference:
                ref_state = network_reference_step(spec, ref_state, drive)
                for field, key in zip(joint_state, ("v", "current", "spike")):
                    np.testing.assert_array_equal(field[:, :, test_word],
                                                  [getattr(cell, key) for cell in ref_state])
                ref_updates += 1
        if any(np.any(field) for field in joint_state):
            raise AssertionError(f"Full-register endpoint failed for {mask} at E={amplitude}")
    return dict(n=n, weak_mask=mask, weak_count=len(weak), cyclic_orbit=len(rotations(mask)),
                word_count=digits.shape[1], longest_strong_run=longest, witness_D_lower_bound=longest + 1,
                E3_constructive_D=upper_pulse, E3_constructive_H=10,
                E3_word_opt_D=best_d, E3_word_opt_H=best_h, E3_word_opt_total=best,
                E4_fixed_D10_H10_cleared=digits.shape[1], reference_updates=ref_updates)


def main():
    records = []
    for n in range(2, 9):
        for value in range(1, 2**n):
            mask = format(value, f"0{n}b")
            if mask == min(rotations(mask)):
                records.append(trial(mask, all_words=True, check_reference=n <= 5))
    for n in (16, 32, 64):
        for targets in ((0, n // 4, n // 2, 3 * n // 4), (0, 1, 2, 3)):
            mask = "".join("1" if i in targets else "0" for i in range(n))
            records.append(trial(mask, all_words=False, check_reference=False))
    # For half weak edges, the all-state upper bound of the alternating ring
    # falls below the clustered ring's lower bound at this size.
    for mask in ("10" * 32, "1" * 32 + "0" * 32):
        records.append(trial(mask, all_words=False, check_reference=False))
    output = Path(__file__).resolve().parents[1] / "results" / "multiweak_word_pilot.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} rows in {output}")


if __name__ == "__main__":
    main()
