"""Sound sufficient certificates for the two-neuron integer memory module.

These routines certify the fixed event branch described in the proposal (T1)
and a conditional input-driven erasure bound (T2).  They do not implement or
claim a general compositional/network-level T3 certificate.  The retention
calculation propagates complete integer intervals, never sampled trajectories.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .model import Params, clip, trunc_div


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _interval(name: str, value: Any, minimum: int, maximum: int) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two integer endpoints")
    lo, hi = value
    if type(lo) is not int or type(hi) is not int:
        raise ValueError(f"{name} endpoints must be integers")
    if not minimum <= lo <= hi <= maximum:
        raise ValueError(f"{name} must satisfy {minimum} <= lower <= upper <= {maximum}")
    return lo, hi


def _post_reset_interval(pre: list[int], threshold: int) -> tuple[list[int], list[int]]:
    """An enclosing interval for the *actual* threshold/reset operation.

    The mixed branch is used only to describe a failed certificate; it is never
    propagated as if it satisfied the requested event sequence.
    """
    lo, hi = pre
    if hi < threshold:
        return [lo, hi], [0, 0]
    if lo >= threshold:
        return [0, 0], [1, 1]
    return [min(lo, 0), max(threshold - 1, 0)], [0, 1]


def certify_retention(
    params: Params,
    period: int,
    current_interval: tuple[int, int] | list[int],
    inactive_v_interval: tuple[int, int] | list[int] | None = None,
    noise_bound: int = 0,
    active: int = 0,
) -> dict[str, Any]:
    """Certify a fixed-period memory branch using exact integer endpoints.

    At the section, the active neuron has v=0, spike=1 and current in the given
    interval.  The other neuron has current=0, spike=0 and its specified voltage
    interval.  There is no input or erase pulse.  Independently at every step,
    each neuron may receive any integer voltage noise in [-noise_bound, bound].

    A true result certifies all such states/noise sequences.  A false result
    means these sufficient conditions failed, not that retention is impossible.
    Initial write reachability is a separate obligation of the caller.
    """
    _integer("period", period, minimum=1)
    _integer("noise_bound", noise_bound)
    if type(active) is not int or active not in (0, 1):
        raise ValueError("active must be 0 or 1")
    initial_current = _interval("current_interval", current_interval, 0, params.qi)
    if inactive_v_interval is None:
        inactive_v_interval = (-params.qv, params.threshold - 1)
    initial_inactive_v = _interval(
        "inactive_v_interval", inactive_v_interval, -params.qv, params.threshold - 1
    )
    inactive = 1 - active
    currents = [[0, 0], [0, 0]]
    currents[active] = list(initial_current)
    voltages = [[0, 0], [0, 0]]
    voltages[inactive] = list(initial_inactive_v)
    spikes = [0, 0]
    spikes[active] = 1
    initial_phase = {
        "time": 0,
        "current": [x[:] for x in currents],
        "voltage_pre": None,
        "voltage_post": [x[:] for x in voltages],
        "spike": [[s, s] for s in spikes],
        "expected_spikes": spikes[:],
        "branch_valid": True,
    }
    result: dict[str, Any] = {
        "certified": False,
        "reason": "certificate calculation not completed",
        "scope": "T1_two_neuron_fixed_branch",
        "params": asdict(params),
        "period": period,
        "active": active,
        "current_interval": list(initial_current),
        "inactive_v_interval": list(initial_inactive_v),
        "noise_bound": noise_bound,
        "phases": [initial_phase],
        "imax": initial_current[1],
        "threshold_margins": [],
        "return_interval": None,
        "inactive_return_interval": None,
        "write_reachability_verified": False,
        "assumptions": [
            "integer voltage noise only; no current noise",
            "zero normal input and zero erase input while holding",
            "initial inactive current and spike are zero",
            "toward-zero voltage leak, nonnegative current floor, saturation, hard reset",
            "no cross-module coupling; this is not a T3 certificate",
        ],
    }
    den_v = 1 << params.alpha_shift
    den_i = 1 << params.beta_shift

    for time in range(1, period + 1):
        new_currents: list[list[int]] = []
        pre: list[list[int]] = []
        post: list[list[int]] = []
        observed_spikes: list[list[int]] = []
        expected = [0, 0]
        expected[active] = int(time == period)
        margins = [0, 0]
        failures: list[str] = []

        for neuron in (0, 1):
            current_bounds = [
                clip(
                    (params.beta_num * endpoint) // den_i + params.weight * spikes[neuron],
                    0,
                    params.qi,
                )
                for endpoint in currents[neuron]
            ]
            new_currents.append(current_bounds)
            voltage_bounds = [
                clip(
                    trunc_div(params.alpha_num * voltages[neuron][side], den_v)
                    + current_bounds[side]
                    - params.inhibition * spikes[1 - neuron]
                    + (-noise_bound if side == 0 else noise_bound),
                    -params.qv,
                    params.qv,
                )
                for side in (0, 1)
            ]
            pre.append(voltage_bounds)
            actual_post, actual_spikes = _post_reset_interval(voltage_bounds, params.threshold)
            post.append(actual_post)
            observed_spikes.append(actual_spikes)
            if expected[neuron]:
                margins[neuron] = voltage_bounds[0] - params.threshold
                if margins[neuron] < 0:
                    failures.append(f"active neuron is not guaranteed to fire at step {time}")
            else:
                margins[neuron] = params.threshold - 1 - voltage_bounds[1]
                if margins[neuron] < 0:
                    label = "active neuron fires before the target period" if neuron == active else "inactive neuron may fire"
                    failures.append(f"{label} at step {time}")

        result["imax"] = max(result["imax"], *(x[1] for x in new_currents))
        result["threshold_margins"].append(
            {"time": time, "by_neuron": margins, "expected_spikes": expected[:]}
        )
        result["phases"].append(
            {
                "time": time,
                "current": [x[:] for x in new_currents],
                "voltage_pre": pre,
                "voltage_post": post,
                "spike": observed_spikes,
                "expected_spikes": expected[:],
                "branch_valid": not failures,
            }
        )
        if failures:
            result["reason"] = "; ".join(failures)
            return result
        currents, voltages, spikes = new_currents, post, expected

    result["return_interval"] = currents[active][:]
    result["inactive_return_interval"] = voltages[inactive][:]
    result["minimum_threshold_margin"] = min(
        min(item["by_neuron"]) for item in result["threshold_margins"]
    )
    if currents[active][0] < initial_current[0] or currents[active][1] > initial_current[1]:
        result["reason"] = "active return-current interval is not contained in the initial interval"
        return result
    if voltages[inactive][0] < initial_inactive_v[0] or voltages[inactive][1] > initial_inactive_v[1]:
        result["reason"] = "inactive return-voltage interval is not contained in the initial interval"
        return result
    result["certified"] = True
    result["reason"] = "all integer event branches and return-interval conditions are certified"
    result["readout_window_steps"] = period
    return result


def _ceil_log_decay(target: int, numerator: int, denominator: int) -> int:
    """Smallest k with target * numerator**k <= denominator**k, exactly.

    This is ceil(log(target) / -log(numerator/denominator)), with no floating
    logarithm or rounding at a power boundary.  Doubling and binary search avoid
    an iteration for every decay step.
    """
    if target <= 1:
        return 0
    if not 0 < numerator < denominator:
        raise ValueError("decay ratio must be strictly between zero and one")

    def enough(exponent: int) -> bool:
        return target * pow(numerator, exponent) <= pow(denominator, exponent)

    lo, hi = 0, 1
    while not enough(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if enough(mid):
            hi = mid
        else:
            lo = mid
    return hi


def erasure_certificate(
    params: Params,
    imax: int,
    positive_input_bound: int = 0,
) -> dict[str, Any]:
    """Return a conditional, input-driven T2 erasure bound.

    imax must cover ALL allowed erase-start phases, not only the return section.
    Positive external/coupling voltage input is bounded only while the erase
    pulse is active.  During recovery all external inputs/noise must be absent
    (or their absence must be separately proved).  No state register is cleared
    directly.  The caller checks amplitude/duration/total-time budgets.
    """
    _integer("imax", imax)
    _integer("positive_input_bound", positive_input_bound)
    if imax > params.qi:
        raise ValueError("imax exceeds the current register range")
    den_i = 1 << params.beta_shift
    den_v = 1 << params.alpha_shift
    jmax = min(params.qi, (params.beta_num * imax) // den_i + params.weight)
    amplitude = (
        trunc_div(params.alpha_num * (params.threshold - 1), den_v)
        + jmax
        + positive_input_bound
    )
    duration = 1 + _ceil_log_decay(jmax + 1, params.beta_num, den_i)
    recovery = _ceil_log_decay(params.qv + 1, params.alpha_num, den_v)
    return {
        "certified": True,
        "scope": "T2_conditional_two_neuron_erasure_bound",
        "params": asdict(params),
        "imax": imax,
        "jmax": jmax,
        "positive_input_bound": positive_input_bound,
        "amplitude": amplitude,
        "duration": duration,
        "recovery": recovery,
        "total_steps": duration + recovery,
        "budget_checked": False,
        "assumptions": [
            "imax covers every permitted erase-start phase and neuron",
            "initial states are legal post-update states, so voltage < threshold",
            "positive_input_bound applies during the erase pulse only",
            "no sustained noise or external input during recovery",
            "negative bounded voltage inputs during the pulse are allowed",
            "toward-zero voltage leak and hard reset; no direct register clearing",
            "not a general network-level or T3 certificate",
        ],
    }
