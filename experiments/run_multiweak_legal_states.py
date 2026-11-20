"""Replay matched legal states on half-weak rings, including a translated active domain."""
import json
from pathlib import Path

import numpy as np

from experiments.run_weak_ring_witness import transition
from integer_snn.model import State
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL


def initial_states(n, profile, seed):
    """One independent background per translation, shared across the layouts."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, profile]))
    shape = (n, 2, n)
    code = rng.integers(480, size=shape)
    current = code % 16
    level = code // 16
    spike = (level == 29).astype(np.int64)
    voltage = np.where(spike == 1, 0, level - 15)
    if profile == 1:
        spike = rng.integers(2, size=shape)
        current = rng.integers(12, 16, size=shape)
        voltage = np.where(spike == 1, 0, rng.integers(-15, 14, size=shape))
    elif profile == 2:
        # Every offset occurs once. A dormant opposite channel inside the block
        # makes this a coherent saturated domain amid a random legal background.
        block = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n < n // 2
        voltage[:, 0][block] = 0
        current[:, 0][block] = 15
        spike[:, 0][block] = 1
        voltage[:, 1][block] = 0
        current[:, 1][block] = 0
        spike[:, 1][block] = 0
    return voltage, current, spike


def replay(initial, weights, amplitude, drive, release, check=False):
    state = initial
    n = len(weights)
    reference_updates = 0
    if check:
        spec = NetworkSpec(MODEL, tuple(tuple(int(weights[i, 0, 0])
                            if j == (i-1) % n else 0 for j in range(n)) for i in range(n)))
        reference = tuple(State(*(tuple(map(int, field[i, :, 0])) for field in initial))
                          for i in range(n))
    for tick in range(drive + release):
        force = amplitude if tick < drive else 0
        state = transition(state, weights, 0, force)
        if check:
            reference = network_reference_step(spec, reference, force)
            for field, key in zip(state, ("v", "current", "spike")):
                np.testing.assert_array_equal(field[:, :, 0], [getattr(r, key) for r in reference])
            reference_updates += 1
    cleared = ~np.any(np.stack(state), axis=(0, 1, 2))
    return cleared, reference_updates


def main():
    seed = 291613
    rows = []
    names = ["uniform_legal", "high_current", "translated_domain"]
    for n in (32, 64, 128):
        for profile, name in enumerate(names):
            initial = initial_states(n, profile, seed)
            for layout, weak in (("alternating", np.arange(0, n, 2)),
                                 ("clustered", np.arange(n // 2))):
                weights = np.full((n, 1, 1), 2)
                weights[weak] = 1
                longest = 1 if layout == "alternating" else n // 2
                derived_d = 10 + 2 * ((longest + 1) // 2) + 5 * longest
                results = {}
                reference_updates = 0
                for amplitude, drive, release in dict.fromkeys(((3, 17, 10),
                                                               (3, derived_d, 10), (4, 10, 10))):
                    ok, count = replay(initial, weights, amplitude, drive, release,
                                       check=(n == 32 and drive == 17))
                    if (drive == derived_d or amplitude == 4) and not np.all(ok):
                        raise AssertionError("Certified protocol failed")
                    reference_updates += count
                    results[f"{amplitude},{drive},{release}"] = dict(
                        zero=int(ok.sum()), nonzero_indices=np.flatnonzero(~ok).tolist())
                rows.append(dict(n=n, profile=name, layout=layout, cases=n, L=longest,
                                 certified_D=derived_d, results=results,
                                 reference_updates=reference_updates))
    report = dict(seed=seed, generator="numpy.default_rng(SeedSequence([seed,n,profile_index]))",
                  numpy_version=np.__version__, profile_order=names,
                  protocol_key="E,D,H", records=rows)
    target = Path(__file__).resolve().parents[1] / "results" / "multiweak_legal_states_v160.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
