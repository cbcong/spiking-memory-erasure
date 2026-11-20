"""Reproduce the three-module sharpness witness with both update kernels."""

import json

from experiments.run_core_endpoint_witnesses import COUNTEREXAMPLE, MODEL, infinite_witness
from integer_snn.core_erasure import core_erasure_bounds
from integer_snn.model import zero_state
from integer_snn.network import NetworkSpec, network_reference_step, network_step


def checked_step(spec, state, **controls):
    exact = network_reference_step(spec, state, **controls)
    if exact != network_step(spec, state, **controls):
        raise AssertionError("Integer and rational implementations disagree")
    return exact


def trace(spec, state, peak):
    states = []
    first_visit = {}
    while state not in first_visit:
        first_visit[state] = len(states)
        states.append(state)
        state = checked_step(spec, state, erase=peak)
    return states, first_visit[state]


def written_state(spec):
    state = (zero_state(),) * 3
    for tick in range(1, 10):
        writes = (tick in (1, 2), tick in (2, 3), tick <= 5)
        state = checked_step(spec, state, inputs=tuple((14 * on, 0) for on in writes))
    return state


def verify():
    spec = NetworkSpec(MODEL, COUNTEREXAMPLE)
    zero = (zero_state(),) * 3
    state = written_state(spec)
    held, entry = trace(spec, state, 3)
    phases = []
    for duration, phase in enumerate(held):
        released, first = trace(spec, phase, 0)
        assert zero not in released
        assert any(any(any(cell.spike) for cell in network) for network in released[first:])
        phases.append(dict(duration=duration, transient=first, period=len(released) - first))
    witness = dict(written_from_zero=True, pressure_transient=entry,
                   pressure_period=len(held) - entry, release_all_phases=phases)
    assert witness == infinite_witness(spec)
    bounds = core_erasure_bounds(spec)
    return dict(weights=spec.weights, bounds=bounds, witness=witness)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
