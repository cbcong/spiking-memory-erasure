"""Exact integer semantics for the two-neuron module in the proposal.

All persisted state is integer valued.  Current is updated first; both neurons
then use the *previous* spike tuple.  Membrane decay truncates towards zero,
voltage saturates symmetrically, and a threshold equality emits a spike and
hard-resets voltage to zero.  Python's arbitrary-precision intermediate
integers are a mathematical reference, not a claim about hardware accumulators.
"""

from dataclasses import dataclass
from typing import Iterator


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (bool is not accepted)")
    return value


def _integer_pair(name: str, value: object) -> tuple[int, int]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of two integers")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two entries")
    for index, entry in enumerate(value):
        _require_int(f"{name}[{index}]", entry)
    return value


def clip(x: int, lo: int, hi: int) -> int:
    """Saturate an integer to a closed interval; never wrap around."""
    _require_int("x", x)
    _require_int("lo", lo)
    _require_int("hi", hi)
    if lo > hi:
        raise ValueError("lo must be no greater than hi")
    return min(hi, max(lo, x))


def trunc_div(num: int, den: int) -> int:
    """Divide integers with truncation towards zero, without using floats."""
    _require_int("num", num)
    _require_int("den", den)
    if den == 0:
        raise ZeroDivisionError("integer division by zero")
    quotient = abs(num) // abs(den)
    return -quotient if (num < 0) != (den < 0) else quotient


@dataclass(frozen=True)
class Params:
    """Shared parameters of the two mutually inhibiting neurons.

    Voltage uses the symmetric range [-qv, qv], excluding the most negative
    two's-complement code.  Current uses [0, qi].  Weights and inputs have no
    invented bit-width restriction; a hardware deployment must account for
    their representation and accumulator ranges separately.
    """

    bits_v: int
    bits_i: int
    alpha_num: int
    alpha_shift: int
    beta_num: int
    beta_shift: int
    weight: int
    inhibition: int
    threshold: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_int(name, getattr(self, name))
        if self.bits_v < 2:
            raise ValueError("bits_v must be at least 2")
        if self.bits_i < 1:
            raise ValueError("bits_i must be at least 1")
        for name in ("alpha", "beta"):
            shift = getattr(self, f"{name}_shift")
            numerator = getattr(self, f"{name}_num")
            if shift < 1:
                raise ValueError(f"{name}_shift must be at least 1")
            if not 0 < numerator < (1 << shift):
                raise ValueError(f"{name} must satisfy 0 < numerator / 2**shift < 1")
        if self.weight < 0 or self.inhibition < 0:
            raise ValueError("weight and inhibition must be nonnegative")
        if not 1 <= self.threshold <= self.qv:
            raise ValueError("threshold must lie in [1, qv]")

    @property
    def qv(self) -> int:
        return (1 << (self.bits_v - 1)) - 1

    @property
    def qi(self) -> int:
        return (1 << self.bits_i) - 1

    @property
    def state_bits(self) -> int:
        """Logical persisted module bits, including two previous spikes."""
        return 2 * self.bits_v + 2 * self.bits_i + 2


@dataclass(frozen=True)
class State:
    """A post-update state; parameter-dependent validity is checked separately.

    Legal states need not be reachable from zero.  This deliberately allows
    exhaustive analysis over all states satisfying the representation and
    post-reset invariants.
    """

    v: tuple[int, int]
    current: tuple[int, int]
    spike: tuple[int, int]

    def __post_init__(self) -> None:
        _integer_pair("v", self.v)
        _integer_pair("current", self.current)
        _integer_pair("spike", self.spike)
        if any(value not in (0, 1) for value in self.spike):
            raise ValueError("spike entries must be 0 or 1")


def zero_state() -> State:
    return State(v=(0, 0), current=(0, 0), spike=(0, 0))


def validate_state(params: Params, state: State) -> None:
    """Raise on malformed/range-invalid states or violated reset invariants.

    A spike implies zero post-reset voltage; a nonspike implies voltage below
    threshold.  These necessary invariants do not establish reachability.
    """
    if not isinstance(params, Params):
        raise TypeError("params must be Params")
    if not isinstance(state, State):
        raise TypeError("state must be State")
    for index in range(2):
        voltage = state.v[index]
        current = state.current[index]
        spike = state.spike[index]
        if not -params.qv <= voltage <= params.qv:
            raise ValueError(f"v[{index}] lies outside [-qv, qv]")
        if not 0 <= current <= params.qi:
            raise ValueError(f"current[{index}] lies outside [0, qi]")
        if spike == 1 and voltage != 0:
            raise ValueError(f"spike[{index}]=1 requires v[{index}]=0")
        if spike == 0 and voltage >= params.threshold:
            raise ValueError(f"spike[{index}]=0 requires v[{index}] < threshold")


def step(
    params: Params,
    state: State,
    inputs: tuple[int, int] = (0, 0),
    erase: int = 0,
    noise: tuple[int, int] = (0, 0),
) -> State:
    """Apply equations (1)-(3) exactly, using old spikes synchronously.

    ``inputs`` and ``noise`` are signed integer additions in membrane units.
    ``erase`` is a common nonnegative external input, not register clearing.
    """
    validate_state(params, state)
    _integer_pair("inputs", inputs)
    _integer_pair("noise", noise)
    _require_int("erase", erase)
    if erase < 0:
        raise ValueError("erase must be nonnegative")

    currents = tuple(
        clip(
            (params.beta_num * state.current[index]) // (1 << params.beta_shift)
            + params.weight * state.spike[index],
            0,
            params.qi,
        )
        for index in range(2)
    )
    pre_reset = tuple(
        clip(
            trunc_div(params.alpha_num * state.v[index], 1 << params.alpha_shift)
            + currents[index]
            - params.inhibition * state.spike[1 - index]
            + inputs[index]
            - erase
            + noise[index],
            -params.qv,
            params.qv,
        )
        for index in range(2)
    )
    spikes = tuple(int(value >= params.threshold) for value in pre_reset)
    voltages = tuple(0 if spikes[index] else pre_reset[index] for index in range(2))
    return State(v=voltages, current=currents, spike=spikes)


def legal_state_count(params: Params) -> int:
    """Count legal post-update states, including potentially unreachable ones."""
    if not isinstance(params, Params):
        raise TypeError("params must be Params")
    per_neuron = (params.qi + 1) * (params.qv + params.threshold + 1)
    return per_neuron * per_neuron


def iter_legal_states(
    params: Params, *, max_states: int | None = 1 << 20
) -> Iterator[State]:
    """Yield every legal state once, with an explicit small-system size guard.

    Set ``max_states=None`` only after independently checking feasibility.
    The guard is checked when iteration starts, as with any generator.
    """
    count = legal_state_count(params)
    if max_states is not None:
        _require_int("max_states", max_states)
        if max_states < 1:
            raise ValueError("max_states must be positive or None")
        if count > max_states:
            raise ValueError(f"{count} legal states exceed max_states={max_states}")
    for spike0 in (0, 1):
        voltages0 = (0,) if spike0 else range(-params.qv, params.threshold)
        for voltage0 in voltages0:
            for current0 in range(params.qi + 1):
                for spike1 in (0, 1):
                    voltages1 = (0,) if spike1 else range(-params.qv, params.threshold)
                    for voltage1 in voltages1:
                        for current1 in range(params.qi + 1):
                            yield State(
                                v=(voltage0, voltage1),
                                current=(current0, current1),
                                spike=(spike0, spike1),
                            )
