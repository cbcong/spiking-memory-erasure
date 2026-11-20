"""Small-test reference using exact rational arithmetic independently of step.

Only the immutable data classes and input-state contract are shared with the
production model.  Decay, saturation, thresholding and update arithmetic are
implemented here without calling its arithmetic helpers or transition kernel.
"""

from fractions import Fraction

from .model import Params, State, validate_state


def reference_step(
    params: Params,
    state: State,
    inputs: tuple[int, int] = (0, 0),
    erase: int = 0,
    noise: tuple[int, int] = (0, 0),
) -> State:
    """Evaluate the same declared model through exact Fraction arithmetic."""
    validate_state(params, state)
    for name, pair in (("inputs", inputs), ("noise", noise)):
        if not isinstance(pair, tuple):
            raise TypeError(f"{name} must be a tuple of two integers")
        if len(pair) != 2:
            raise ValueError(f"{name} must contain exactly two entries")
        if any(type(value) is not int for value in pair):
            raise TypeError(f"{name} must contain integers, not bool or float")
    if type(erase) is not int:
        raise TypeError("erase must be an integer")
    if erase < 0:
        raise ValueError("erase must be nonnegative")

    alpha = Fraction(params.alpha_num, 2**params.alpha_shift)
    beta = Fraction(params.beta_num, 2**params.beta_shift)
    currents = []
    voltages = []
    spikes = []
    for index in (0, 1):
        decayed_current = int(beta * state.current[index])
        new_current = min(
            params.qi, max(0, decayed_current + params.weight * state.spike[index])
        )
        currents.append(new_current)
        potential = (
            int(alpha * state.v[index])
            + new_current
            - params.inhibition * state.spike[1 - index]
            + inputs[index]
            - erase
            + noise[index]
        )
        potential = min(params.qv, max(-params.qv, potential))
        if potential >= params.threshold:
            spikes.append(1)
            voltages.append(0)
        else:
            spikes.append(0)
            voltages.append(potential)
    return State(v=tuple(voltages), current=tuple(currents), spike=tuple(spikes))
