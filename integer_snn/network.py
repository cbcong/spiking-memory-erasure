"""Synchronous integer networks of identical two-neuron memory modules.

W[target][source] supplies nonnegative, same-neuron-index spike input from the
PREVIOUS full network state.  Self-loops are permitted and explicitly stored.
The implementation guard is 64 modules; no Cartesian state enumeration occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Params, State, step, validate_state
from .reference import reference_step


@dataclass(frozen=True)
class NetworkSpec:
    params: Params
    weights: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if not isinstance(self.params, Params):
            raise TypeError("params must be Params")
        if not isinstance(self.weights, tuple) or not 1 <= len(self.weights) <= 64:
            raise ValueError("weights must be a tuple matrix with 1..64 modules")
        for row in self.weights:
            if not isinstance(row, tuple) or len(row) != len(self.weights):
                raise ValueError("weights must be a square tuple matrix")
            if any(type(value) is not int or value < 0 for value in row):
                raise ValueError("weights must be nonnegative integers, excluding bool")

    @property
    def module_count(self):
        return len(self.weights)

    @property
    def row_sums(self):
        return tuple(sum(row) for row in self.weights)

    @property
    def max_row_sum(self):
        return max(self.row_sums)

    @property
    def persisted_state_bits(self):
        return self.module_count * self.params.state_bits


def validate_network_state(spec: NetworkSpec, states: tuple[State, ...]) -> None:
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    if not isinstance(states, tuple) or len(states) != spec.module_count:
        raise ValueError("states must be a tuple with one State per module")
    for state in states:
        validate_state(spec.params, state)


def coupling_inputs(spec: NetworkSpec, states: tuple[State, ...]) -> tuple[tuple[int, int], ...]:
    validate_network_state(spec, states)
    return tuple(tuple(sum(weight * source.spike[cell]
                           for weight, source in zip(row, states)) for cell in (0, 1))
                 for row in spec.weights)


def _controls(spec, values, name):
    if values is None:
        return ((0, 0),) * spec.module_count
    if not isinstance(values, tuple) or len(values) != spec.module_count:
        raise ValueError(f"{name} must be a tuple with one integer pair per module")
    if any(not isinstance(pair, tuple) or len(pair) != 2
           or any(type(x) is not int for x in pair) for pair in values):
        raise ValueError(f"{name} must contain integer pairs, excluding bool")
    return values


def network_step(spec: NetworkSpec, states: tuple[State, ...], erase: int = 0,
                 inputs: tuple[tuple[int, int], ...] | None = None,
                 noise: tuple[tuple[int, int], ...] | None = None) -> tuple[State, ...]:
    """One simultaneous update; optional writes/noise are explicit controls.

    The erasure experiments set inputs/noise to None throughout.  No state
    register is directly cleared, clamped, or rewritten by this wrapper.
    """
    validate_network_state(spec, states)
    if type(erase) is not int or erase < 0:
        raise ValueError("erase must be a nonnegative integer")
    external = _controls(spec, inputs, "inputs")
    disturbance = _controls(spec, noise, "noise")
    incoming = coupling_inputs(spec, states)
    return tuple(step(spec.params, state,
                      inputs=tuple(incoming[m][i] + external[m][i] for i in (0, 1)),
                      erase=erase, noise=disturbance[m]) for m, state in enumerate(states))


def network_reference_step(spec: NetworkSpec, states: tuple[State, ...], erase: int = 0,
                           inputs: tuple[tuple[int, int], ...] | None = None,
                           noise: tuple[tuple[int, int], ...] | None = None) -> tuple[State, ...]:
    """Independent global input assembly and Fraction local transition.

    Does not call network_step, coupling_inputs, or the production local step.
    """
    validate_network_state(spec, states)
    if type(erase) is not int or erase < 0:
        raise ValueError("erase must be a nonnegative integer")
    external = _controls(spec, inputs, "inputs")
    disturbance = _controls(spec, noise, "noise")
    result = []
    for target in range(spec.module_count):
        signals = [external[target][0], external[target][1]]
        for source in range(spec.module_count - 1, -1, -1):
            for cell in (1, 0):
                if states[source].spike[cell] == 1:
                    signals[cell] += spec.weights[target][source]
        result.append(reference_step(spec.params, states[target], inputs=tuple(signals),
                                     erase=erase, noise=disturbance[target]))
    return tuple(result)


def current_spike_contract(spec: NetworkSpec, states: tuple[State, ...],
                           spiking_bound: int = 9, silent_bound: int = 12) -> bool:
    """A separate theorem-domain check, not an assumption enforced by the kernel."""
    validate_network_state(spec, states)
    if (type(spiking_bound) is not int or type(silent_bound) is not int
            or min(spiking_bound, silent_bound) < 0):
        raise ValueError("current bounds must be nonnegative integers")
    return all(state.current[i] <= (spiking_bound if state.spike[i] else silent_bound)
               for state in states for i in (0, 1))


def transition_contract(spec: NetworkSpec, previous: tuple[State, ...],
                        current: tuple[State, ...]) -> dict:
    """Independently inspect no-consecutive spikes and weighted two-tick inputs.

    This checks observed states; it does not assert the contract for arbitrary
    parameters or controls.  In the declared theorem, the contract is proved.
    """
    validate_network_state(spec, previous)
    validate_network_state(spec, current)
    consecutive = [(m, i) for m in range(spec.module_count) for i in (0, 1)
                   if previous[m].spike[i] and current[m].spike[i]]
    violations = []
    for target, row in enumerate(spec.weights):
        limit = sum(row)
        for cell in (0, 1):
            two_tick_input = sum(row[source] * (previous[source].spike[cell]
                                               + current[source].spike[cell])
                                 for source in range(spec.module_count))
            if two_tick_input > limit:
                violations.append((target, cell, two_tick_input, limit))
    return {"no_consecutive_spikes": not consecutive,
            "weighted_window_bound": not violations,
            "consecutive_spike_locations": consecutive,
            "weighted_window_violations": violations}
