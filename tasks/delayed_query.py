"""Physical write/hold/query/erase/reuse task, with no hidden state reset.

Targets enter only via the explicit two-tick write inputs.  Query preparation
does not accept an address or target; the address arrives afterwards and only
selects a two-tick spike readout.  Whole-pattern readout is offline instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from integer_snn.model import State, zero_state
from integer_snn.network import (NetworkSpec, validate_network_state,
                                 network_step, network_reference_step)


def _nonnegative(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _pattern(spec, pattern):
    if not isinstance(pattern, tuple) or len(pattern) != spec.module_count:
        raise ValueError("pattern must be a tuple with one bit per module")
    if any(type(bit) is not int or bit not in (0, 1) for bit in pattern):
        raise ValueError("pattern values must be integer bits, not bool")
    return pattern


def fresh_network(spec: NetworkSpec) -> tuple[State, ...]:
    """Initial experiment power-on only; reuse uses erase_and_recover instead."""
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    return (zero_state(),) * spec.module_count


def advance(spec: NetworkSpec, states: tuple[State, ...], ticks: int,
            *, inputs=None, erase=0, reference=False, audit: dict[str, int] | None = None):
    _nonnegative("ticks", ticks)
    _nonnegative("erase", erase)
    validate_network_state(spec, states)
    for _ in range(ticks):
        successor = network_step(spec, states, erase=erase, inputs=inputs)
        if reference:
            if successor != network_reference_step(spec, states, erase=erase, inputs=inputs):
                raise AssertionError("Independent Fraction network reference mismatch")
            if audit is not None:
                audit["reference_steps"] = audit.get("reference_steps", 0) + 1
        if audit is not None:
            audit["network_steps"] = audit.get("network_steps", 0) + 1
        states = successor
    return states


def write_pattern(spec: NetworkSpec, states: tuple[State, ...], pattern: tuple[int, ...],
                  *, reference=False, audit=None):
    """Exactly 14x2 external write followed by six unforced updates.

    No clean-state precondition is enforced here: the E2 counterfactual must
    physically attempt the SAME write on its actual residual state.
    """
    _pattern(spec, pattern)
    inputs = tuple((14, 0) if bit == 0 else (0, 14) for bit in pattern)
    states = advance(spec, states, 2, inputs=inputs, reference=reference, audit=audit)
    return advance(spec, states, 6, reference=reference, audit=audit)


@dataclass(frozen=True)
class QueryReady:
    spec: NetworkSpec
    states: tuple[State, ...]
    ready_tick: int
    delay: int


@dataclass(frozen=True)
class QueryReadout:
    index: int
    prediction: int | None
    selected_counts: tuple[int, int]
    next_states: tuple[State, ...]
    end_tick: int
    all_predictions: tuple[int | None, ...]
    all_counts: tuple[tuple[int, int], ...]

    @property
    def phase_valid(self):
        return self.selected_counts in ((1, 0), (0, 1))

    @property
    def all_phase_valid(self):
        return tuple(pair in ((1, 0), (0, 1)) for pair in self.all_counts)


def prepare_query(spec: NetworkSpec, states: tuple[State, ...], delay: int,
                  *, start_tick: int = 0, reference=False, audit=None) -> QueryReady:
    """Advance the entire delay WITHOUT an address, label, or future query."""
    _nonnegative("delay", delay)
    _nonnegative("start_tick", start_tick)
    states = advance(spec, states, delay, reference=reference, audit=audit)
    return QueryReady(spec, states, start_tick + delay, delay)


def query_after_delay(ready: QueryReady, index: int, *, reference=False, audit=None) -> QueryReadout:
    """Observe only the next two updates; current state is not a label oracle.

    Two counters record the queried cell's spike counts.  A unique larger count
    selects the bit; a tie is invalid.  phase_valid separately checks the exact
    (1,0)/(0,1) two-phase contract, so extra spikes are not falsely equated with
    content errors. all_counts/all_predictions are offline whole-pattern metrics.
    """
    if not isinstance(ready, QueryReady):
        raise TypeError("query requires the result of prepare_query")
    if type(index) is not int or not 0 <= index < ready.spec.module_count:
        raise ValueError("query index must be a valid integer address")
    states = ready.states
    counts = [[0, 0] for _ in range(ready.spec.module_count)]
    for _ in range(2):
        states = advance(ready.spec, states, 1, reference=reference, audit=audit)
        for module, state in enumerate(states):
            counts[module][0] += state.spike[0]
            counts[module][1] += state.spike[1]
    predictions = tuple(0 if pair[0] > pair[1] else 1 if pair[1] > pair[0] else None for pair in counts)
    return QueryReadout(index, predictions[index], tuple(counts[index]), states,
                        ready.ready_tick + 2, predictions, tuple(tuple(x) for x in counts))


@dataclass(frozen=True)
class EraseResult:
    states: tuple[State, ...]
    zero: bool
    amplitude: int
    duration: int
    recovery: int


def erase_and_recover(spec: NetworkSpec, states: tuple[State, ...], amplitude: int = 3,
                      duration: int = 4, recovery: int = 10, *, reference=False, audit=None):
    _nonnegative("duration", duration)
    _nonnegative("recovery", recovery)
    states = advance(spec, states, duration, erase=amplitude, reference=reference, audit=audit)
    states = advance(spec, states, recovery, reference=reference, audit=audit)
    return EraseResult(states, all(s == zero_state() for s in states), amplitude, duration, recovery)


@dataclass(frozen=True)
class IdealBitRegister:
    """Ideal storage reference with explicit direct write and clear permissions."""
    bits: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.bits, tuple) or not self.bits or any(type(x) is not int or x not in (0, 1) for x in self.bits):
            raise ValueError("bits must be a nonempty tuple of integer bits")

    def write(self, pattern):
        if len(pattern) != len(self.bits):
            raise ValueError("register width mismatch")
        return IdealBitRegister(pattern)

    def clear(self):
        return IdealBitRegister((0,) * len(self.bits))

    def read(self, index):
        if type(index) is not int or not 0 <= index < len(self.bits):
            raise ValueError("invalid address")
        return self.bits[index]


def logical_storage_ledger(spec: NetworkSpec, delay: int) -> dict[str, Any]:
    """Conservative logical accounting, not a hardware energy or RAM comparison."""
    _nonnegative("delay", delay)
    address = (spec.module_count - 1).bit_length()
    schedule = max(delay, 14, 8).bit_length()
    control = 4 + 2 + address + 1 + schedule + 3
    return {
        "snn_dynamic_state_bits": spec.persisted_state_bits,
        "query_spike_counters": {"count": 2, "bits_each": 2, "total_bits": 4},
        "query_window_counter_bits": 2, "query_address_bits": address,
        "query_output_bits": 1, "schedule_counter_bits": schedule, "schedule_phase_bits": 3,
        "controller_bits_without_register_sharing": control,
        "snn_dynamic_plus_controller_bits": spec.persisted_state_bits + control,
        "offline_full_pattern_window_bits": 4 * spec.module_count,
        "external_pattern_input_buffer_bits": spec.module_count,
        "ideal_register_content_bits": spec.module_count,
        "ideal_register_address_bits": address, "ideal_register_output_bits": 1,
        "not_counted_as_free": ["W and model parameter storage", "input driver", "host-side task labels and RNG",
                                "Python objects", "communication circuitry", "physical energy"],
        "baseline_permissions": "ideal register has direct bit assignment and clear; SNN only uses physical model inputs",
        "claim_limit": "accuracy/functional reference only; no fair energy or RAM superiority claim",
    }
