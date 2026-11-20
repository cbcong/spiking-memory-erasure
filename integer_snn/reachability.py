"""Bounded, exact reachability checks for the integer two-cell model.

These helpers establish finite-set invariance for the supplied transition
semantics. They do not establish that arbitrary seeds can be written from
zero, or that an invariant set carries a particular task label.
"""

from collections import deque
from collections.abc import Iterable
from itertools import product

from .model import Params, State, step, validate_state, zero_state


# Refuse unexpectedly large exhaustive jobs instead of reporting a partial
# enumeration as a certificate. The caller can split a larger investigation.
_MAX_NOISE_PAIRS = 4096
_MAX_PRODUCT_PAIRS = 65536
_MAX_NETWORK_UPDATES = 2_000_000


def _nonnegative_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (bool is not accepted)")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def reachable_closure(
    params: Params,
    seeds: Iterable[State],
    noise_bound: int = 1,
    max_states: int = 4096,
) -> dict:
    """Find every state reachable under arbitrary bounded integer noise.

    At each tick both noise entries independently range over all integers in
    ``[-noise_bound, noise_bound]``. External inputs and erasure are zero.
    ``closed=True`` means the entire reachable set has been exhausted, with
    every outgoing transition checked. A state-limit exit returns an
    *incomplete* visited set, ``closed=False``, and ``reason='max_states'``.
    ``transition_checks`` counts actual calls to the model transition.

    Empty seeds are rejected to prevent vacuous certificates. Seed validity
    is checked, but physical reachability of the seeds is the caller's duty.
    """
    if not isinstance(params, Params):
        raise TypeError("params must be Params")
    _nonnegative_integer("noise_bound", noise_bound)
    _nonnegative_integer("max_states", max_states)
    if max_states == 0:
        raise ValueError("max_states must be positive")
    noise_pair_count = (2 * noise_bound + 1) ** 2
    if noise_pair_count > _MAX_NOISE_PAIRS:
        raise ValueError(f"{noise_pair_count} noise pairs exceed the bounded-check guard")

    visited: set[State] = set()
    frontier: deque[State] = deque()
    for seed in seeds:
        validate_state(params, seed)
        if seed not in visited:
            if len(visited) >= max_states:
                return {"closed": False, "states": visited,
                        "transition_checks": 0, "reason": "max_states"}
            visited.add(seed)
            frontier.append(seed)
    if not visited:
        raise ValueError("seeds must contain at least one state")

    noise_pairs = tuple(product(range(-noise_bound, noise_bound + 1), repeat=2))
    transition_checks = 0
    while frontier:
        source = frontier.popleft()
        for noise in noise_pairs:
            successor = step(params, source, noise=noise)
            transition_checks += 1
            if successor not in visited:
                if len(visited) >= max_states:
                    return {"closed": False, "states": visited,
                            "transition_checks": transition_checks,
                            "reason": "max_states"}
                visited.add(successor)
                frontier.append(successor)
    return {"closed": True, "states": visited,
            "transition_checks": transition_checks, "reason": "exhausted"}


def _coupled_step(
    params: Params,
    states: tuple[State, State],
    coupling: int,
    erase: int = 0,
) -> tuple[State, State]:
    """Simultaneous same-index spike coupling, always from both old states."""
    first, second = states
    inputs_first = tuple(coupling * spike for spike in second.spike)
    inputs_second = tuple(coupling * spike for spike in first.spike)
    return (
        step(params, first, inputs=inputs_first, erase=erase),
        step(params, second, inputs=inputs_second, erase=erase),
    )


def verify_coupled_product(
    params: Params,
    states_by_symbol: dict[int, set[State]],
    coupling: int = 1,
    erase_amplitude: int = 5,
    erase_duration: int = 16,
    recovery_steps: int = 16,
) -> dict:
    """Exhaust all four symbol-pair products for two interacting modules.

    For each initial pair, test one unforced coupled transition against its
    respective product set. Zero closure failures proves product invariance
    by induction, provided the supplied sets have the intended task labels.
    Separately, apply the identical erase amplitude and duration to both
    modules, then release for the fixed ``recovery_steps`` with coupling
    still active. No external noise is added in either phase.

    No register is cleared. ``max_recovery`` counts ticks *after release*,
    over successful pairs only; it is None if every pair fails to reach zero.
    Failures mean this particular set/protocol check failed, not that no
    other certificate or erasure protocol exists. Oversize jobs raise before
    running, rather than returning success on partial enumeration.
    """
    if not isinstance(params, Params):
        raise TypeError("params must be Params")
    if not isinstance(states_by_symbol, dict):
        raise TypeError("states_by_symbol must be a dictionary")
    if set(states_by_symbol) != {0, 1} or any(type(key) is not int for key in states_by_symbol):
        raise ValueError("states_by_symbol must have exactly integer keys 0 and 1")
    for name, value in (("coupling", coupling), ("erase_amplitude", erase_amplitude),
                        ("erase_duration", erase_duration), ("recovery_steps", recovery_steps)):
        _nonnegative_integer(name, value)
    for symbol, states in states_by_symbol.items():
        if not isinstance(states, (set, frozenset)):
            raise TypeError(f"states_by_symbol[{symbol}] must be a set")
        if not states:
            raise ValueError(f"states_by_symbol[{symbol}] must be nonempty")
        for state in states:
            validate_state(params, state)

    pair_count = sum(len(states) for states in states_by_symbol.values()) ** 2
    network_updates = pair_count * (1 + erase_duration + recovery_steps)
    if pair_count > _MAX_PRODUCT_PAIRS:
        raise ValueError(f"{pair_count} initial pairs exceed guard {_MAX_PRODUCT_PAIRS}")
    if network_updates > _MAX_NETWORK_UPDATES:
        raise ValueError(f"{network_updates} network updates exceed guard {_MAX_NETWORK_UPDATES}")

    zero_pair = (zero_state(), zero_state())
    initial_pairs = closure_failures = erase_failures = 0
    max_recovery = None
    # Sorting makes the execution order independent of set iteration order.
    ordered = {symbol: sorted(states, key=lambda s: (s.v, s.current, s.spike))
               for symbol, states in states_by_symbol.items()}
    for symbol_first, symbol_second in product((0, 1), repeat=2):
        for initial in product(ordered[symbol_first], ordered[symbol_second]):
            initial_pairs += 1
            successor = _coupled_step(params, initial, coupling)
            if (successor[0] not in states_by_symbol[symbol_first]
                    or successor[1] not in states_by_symbol[symbol_second]):
                closure_failures += 1

            states = initial
            for _ in range(erase_duration):
                states = _coupled_step(params, states, coupling, erase_amplitude)
            first_zero_tick = 0 if states == zero_pair else None
            for tick in range(1, recovery_steps + 1):
                states = _coupled_step(params, states, coupling)
                if states == zero_pair and first_zero_tick is None:
                    first_zero_tick = tick
            if states != zero_pair:
                erase_failures += 1
            else:
                # Zero is absorbing without external drive, even with spike
                # coupling. The observed first-zero time is therefore valid.
                assert first_zero_tick is not None
                max_recovery = (first_zero_tick if max_recovery is None
                                else max(max_recovery, first_zero_tick))

    return {"initial_pairs": initial_pairs, "closure_failures": closure_failures,
            "erase_failures": erase_failures, "max_recovery": max_recovery,
            "coupling": coupling, "erase_amplitude": erase_amplitude,
            "erase_duration": erase_duration, "recovery_steps": recovery_steps,
            "external_noise_bound": 0}
