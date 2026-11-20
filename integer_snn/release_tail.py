"""Certify a complete reset after the last spike under constant inhibition."""

from functools import lru_cache

from .model import trunc_div
from .topology_certificate import MODEL


@lru_cache(maxsize=None)
def release_tail(amplitude: int) -> tuple[int, int]:
    """Choose extra pressure and recovery steps after a proved last spike L.

    The proof requires the fixed MODEL and a separate bound that continued
    pressure prevents all spikes after L. At L+1 no spikes remain, although
    that step could still have received spikes from L. Starting from the full
    legal silent box at L+1, advance the voltage endpoints and current upper
    bound through further pressure and recovery. During pressure we may
    intersect the voltage box with the legal silent range. After release,
    the upper endpoint must certify silence before that intersection is valid.

    Return (pressure_after_L, recovery), minimizing their sum within this
    certificate and breaking ties in favor of less pressure. The first
    component includes the step from L to L+1.
    """
    if type(amplitude) is not int or amplitude < 3:
        raise ValueError("release tail requires an integer amplitude of at least 3")

    voltage_floor = -MODEL.qv
    silent_ceiling = MODEL.threshold - 1
    alpha_denominator = 1 << MODEL.alpha_shift
    beta_denominator = 1 << MODEL.beta_shift
    pressure_lower = voltage_floor
    pressure_upper = silent_ceiling
    pressure_current = MODEL.qi
    best = None
    for pressure_steps in range(1, 15):
        lower = pressure_lower
        upper = pressure_upper
        current = pressure_current
        for recovery_steps in range(1, 15 - pressure_steps):
            current = MODEL.beta_num * current // beta_denominator
            lower = trunc_div(MODEL.alpha_num * lower, alpha_denominator)
            upper = trunc_div(MODEL.alpha_num * upper, alpha_denominator) + current
            if upper >= MODEL.threshold:
                break
            if lower == upper == current == 0:
                candidate = (pressure_steps + recovery_steps, pressure_steps, recovery_steps)
                best = candidate if best is None else min(best, candidate)
                break

        pressure_current = MODEL.beta_num * pressure_current // beta_denominator
        pressure_lower = max(
            voltage_floor,
            trunc_div(MODEL.alpha_num * pressure_lower, alpha_denominator) - amplitude,
        )
        pressure_upper = max(
            voltage_floor,
            min(
                silent_ceiling,
                trunc_div(MODEL.alpha_num * pressure_upper, alpha_denominator)
                + pressure_current - amplitude,
            ),
        )

    if best is None:
        raise AssertionError("the fixed model admits the certified 14-step tail")
    return best[1], best[2]
