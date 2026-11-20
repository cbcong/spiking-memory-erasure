"""Enumerate the 8- and 12-bit words of the weak-edge ring on CPU."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.run_weak_ring_witness import MODEL, run, transition
from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec, network_reference_step


def bits(word, size):
    """Format a word with module 0 as the most significant bit."""
    return format(int(word), f"0{size}b")


def distribution(values):
    return dict(minimum=int(np.min(values)), median=float(np.median(values)),
                maximum=int(np.max(values)),
                counts=dict(sorted(Counter(values.tolist()).items())))


def run_census(size):
    words = np.arange(2**size, dtype=np.int64)
    digits = (words[None, :] >> np.arange(size - 1, -1, -1)[:, None]) & 1
    write = 14 * (digits[:, None, :] == np.arange(2)[None, :, None])
    weights = np.full((size, 1, 1), 2, dtype=np.int64)
    weights[0] = 1
    matrix = tuple(tuple(int(weights[i, 0, 0]) if j == (i - 1) % size else 0
                         for j in range(size)) for i in range(size))
    spec = NetworkSpec(MODEL, matrix)
    # Check representative mixed words with different arrangements of the same bits.
    reference_words = [int("01" * (size // 2), 2), (1 << (size // 2)) - 1,
                       (1 << (size - 1)) + 1]
    reference_inputs = {
        word: tuple(tuple(map(int, row)) for row in write[:, :, word])
        for word in reference_words
    }
    zeros = ((0, 0),) * size
    initial_references = {word: (zero_state(),) * size for word in reference_words}
    checks = 0

    def advance(state, references, drive=0, writing=False):
        nonlocal checks
        # Use the witness kernel with its module and channel axes unchanged.
        state = transition(state, weights, write if writing else 0, drive)
        next_references = {}
        for word, ref in references.items():
            ref = network_reference_step(
                spec, ref, drive, inputs=reference_inputs[word] if writing else zeros
            )
            for field, value in zip(("v", "current", "spike"), state):
                np.testing.assert_array_equal(value[:, :, word],
                                              [getattr(module, field) for module in ref])
            next_references[word] = ref
            checks += 1
        return state, next_references

    written = tuple(np.zeros_like(write) for _ in range(3))
    written_reference = initial_references
    for _ in range(5):
        written, written_reference = advance(written, written_reference, writing=True)
    np.testing.assert_array_equal(written[1], 15 * (write > 0))
    np.testing.assert_array_equal(written[2], write > 0)

    records = {}
    by_peak = {}
    for peak in (3, 4):
        state, references = written, written_reference
        last_spike = np.zeros_like(words)
        for tick in range(1, 10 * size + 64):
            state, references = advance(state, references, drive=peak)
            last_spike[np.any(state[2], axis=(0, 1))] = tick
            # Once both registers are zero, under sustained pressure voltage alone
            # stays below threshold for every later tick.
            if not (np.any(state[1]) or np.any(state[2])):
                break
        else:
            raise RuntimeError("A continuously driven word did not become quiescent")

        uniform_result = run(size, peak)["network_last_spike"]
        assert last_spike[0] == last_spike[-1] == uniform_result
        np.testing.assert_array_equal(last_spike, last_spike[::-1])
        by_peak[peak] = last_spike

        state, references = written, written_reference
        for tick in range(20):
            state, references = advance(state, references, drive=peak if tick < 10 else 0)
        cleared = ~np.any(np.stack(state), axis=(0, 1, 2))
        np.testing.assert_array_equal(cleared, cleared[::-1])
        if peak == 4 and not np.all(cleared):
            raise AssertionError("The fixed certified protocol left a nonzero register")
        records[str(peak)] = {
            "last_spike": distribution(last_spike),
            "uniform_word_last_spike": uniform_result,
            "maximum_word_examples": [bits(word, size)
                                      for word in words[last_spike == last_spike.max()][:4]],
            "20_tick_protocol_cleared": int(cleared.sum()),
        }

    return dict(modules=size, words=len(words), amplitudes=records,
                paired_last_spike_reduction=distribution(by_peak[3] - by_peak[4]),
                reference_words=[bits(word, size) for word in reference_words],
                reference_word_updates=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12],
                        choices=range(2, 13), help="Ring sizes to enumerate (2 through 12).")
    args = parser.parse_args()
    report = {
        "definition": "Enumerate each n-module binary word from the zero state. Write "
                      "amplitude 14 to the chosen channel of each module for five ticks. "
                      "Tick zero is the end of these writes. Measure the last network "
                      "spike under constant pressure. In a separate paired run, check "
                      "all registers after 10 driven ticks and 10 released ticks.",
        "records": [run_census(size) for size in args.sizes],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["records"], indent=2))


if __name__ == "__main__":
    main()
