"""Exact post-spike return maps for one uncoupled, externally silent module.

The active cell starts at voltage zero, old spike one and declared current k.
The other cell has zero current/spike and any legal subthreshold voltage.
Nonnegative inhibition and a constant nonnegative erase input keep that other
cell silent. This module does not cover coupling, time-varying inputs or noise.

A bottom return means no later spike UNDER THE SAME CONSTANT INPUT. It does
not mean zero state or permission to release the input. Finite-pulse schedules
are checked separately by replaying the full model until current/spike vanish.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .model import Params, State, clip, step, trunc_div


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, excluding bool")
    return value


def _domain(params: Params, current: int, amplitude: int) -> None:
    if not isinstance(params, Params):
        raise TypeError("params must be Params")
    _integer("current", current)
    _integer("amplitude", amplitude)
    if current > params.qi:
        raise ValueError("current exceeds the current register")


def first_return(params: Params, current: int, amplitude: int,
                 max_steps: int = 4096, include_trace: bool = False) -> dict:
    """Return 'returned', 'bottom', or 'unknown' with a checkable witness.

    The first update includes the old-spike self-excitation. Later updates
    before the first new spike only decay current. A bottom result is proved
    by zero current after a nonspike, or by v<=0 and I<=E after a nonspike.
    In the latter case current can still be positive. Budgets never mean bottom.
    """
    _domain(params, current, amplitude)
    _integer("max_steps", max_steps)
    if type(include_trace) is not bool:
        raise TypeError("include_trace must be bool")
    result = {"status": "unknown", "initial_current": current, "amplitude": amplitude,
              "return_current": None, "first_spike_step": None, "steps": 0,
              "reason": "first-return step budget exhausted",
              "bottom_is_zero_state": False,
              "scope": "uncoupled post-spike active section; inactive cell remains silent"}
    voltage, value, old_spike = 0, current, 1
    if include_trace:
        result["trace"] = [{"step": 0, "voltage": 0, "current": current, "spike": 1}]
    for tick in range(1, max_steps + 1):
        value = clip((params.beta_num * value) // (1 << params.beta_shift)
                     + params.weight * old_spike, 0, params.qi)
        voltage = clip(trunc_div(params.alpha_num * voltage, 1 << params.alpha_shift)
                       + value - amplitude, -params.qv, params.qv)
        spike = int(voltage >= params.threshold)
        post_voltage = 0 if spike else voltage
        result.update(steps=tick, terminal_active_state={
            "voltage": post_voltage, "current": value, "spike": spike})
        if include_trace:
            result["trace"].append({"step": tick, "voltage": post_voltage,
                                    "current": value, "spike": spike})
        if spike:
            result.update(status="returned", return_current=value, first_spike_step=tick,
                          reason="first next spike reached threshold and hard-reset voltage")
            return result
        if value == 0:
            result.update(status="bottom", reason="zero current after a nonspike",
                          bottom_proof={"kind": "zero_current", "current": 0,
                                        "voltage": voltage, "spike": 0,
                                        "future_input": amplitude})
            return result
        if voltage <= 0 and value <= amplitude:
            result.update(status="bottom", reason="nonpositive voltage and current at most erase input",
                          bottom_proof={"kind": "nonpositive_dominated", "current": value,
                                        "voltage": voltage, "spike": 0,
                                        "future_input": amplitude})
            return result
        old_spike = 0
    if "terminal_active_state" not in result:
        result["terminal_active_state"] = {"voltage": 0, "current": current, "spike": 1}
    return result


def certify_section_absorption(params: Params, upper_current: int, amplitude: int,
                               max_steps: int = 4096, max_sections: int = 65536,
                               include_map: bool = True) -> dict:
    """Check ALL sections k=0..U for R_E(k)<k, treating bottom as smaller.

    For this ordered scalar model a complete check is an exact decision about
    eventual absorption under constant E for the declared section interval.
    Unknown first returns prevent a complete decision, even if other rows
    contain a nonabsorbing witness. No value outside the interval is discarded
    as bottom. The order argument itself holds on the full register range.
    """
    _domain(params, upper_current, amplitude)
    _integer("max_steps", max_steps)
    _integer("max_sections", max_sections, 1)
    if type(include_map) is not bool:
        raise TypeError("include_map must be bool")
    result = {"status": "unknown", "complete": False, "exact_within_scope": False,
              "absorbs_all_sections": None, "params": asdict(params),
              "upper_current": upper_current, "amplitude": amplitude,
              "required_sections": upper_current + 1, "tested_sections": 0,
              "bottom_count": 0, "returned_count": 0, "unknown_count": 0,
              "return_step_updates": 0, "nondecreasing_returns": [],
              "scope": "every post-spike section k in [0,U], not all legal states",
              "release_schedule_proved": False,
              "complexity_note": "linear number of current sections; qi grows exponentially with bit width"}
    if include_map:
        result["map"] = []
    if upper_current + 1 > max_sections:
        result["reason"] = "section budget smaller than the complete required interval"
        return result
    for current in range(upper_current + 1):
        row = first_return(params, current, amplitude, max_steps)
        compact = {key: row[key] for key in ("initial_current", "status", "return_current",
                                             "first_spike_step", "steps", "reason")}
        if include_map:
            result["map"].append(compact)
        result["tested_sections"] += 1
        result["return_step_updates"] += row["steps"]
        if row["status"] == "unknown":
            result["unknown_count"] += 1
        elif row["status"] == "bottom":
            result["bottom_count"] += 1
        else:
            result["returned_count"] += 1
            if row["return_current"] >= current:
                result["nondecreasing_returns"].append({
                    "initial_current": current, "return_current": row["return_current"],
                    "first_spike_step": row["first_spike_step"]})
    if result["unknown_count"]:
        result["reason"] = "some first returns remain unknown; no complete absorption decision"
        return result
    absorbed = not result["nondecreasing_returns"]
    result.update(complete=True, exact_within_scope=True, absorbs_all_sections=absorbed,
                  status="absorbing" if absorbed else "nonabsorbing",
                  reason=("all complete return rows strictly decrease or reach bottom" if absorbed
                          else "a nondecreasing return gives a persistent current barrier"))
    return result


def minimum_erasure_amplitude(params: Params, upper_current: int,
                              max_amplitude: int | None = None,
                              max_steps: int = 4096, max_sections: int = 65536) -> dict:
    """Binary-search the exact amplitude for the declared complete interval.

    Every probe checks ALL k in [0,U]. The final amplitude and its predecessor
    (when positive) are explicitly checked. An unknown probe stops the search.
    Monotonicity is specific to this constant-input scalar return map; it is
    not a claim that full voltage trajectories remain ordered after resets.
    """
    _domain(params, upper_current, 0)
    _integer("max_steps", max_steps)
    _integer("max_sections", max_sections, 1)
    if max_amplitude is None:
        max_amplitude = params.qi
    _integer("max_amplitude", max_amplitude)
    cache: dict[int, dict] = {}
    result = {"status": "unknown", "exact": False, "minimal_amplitude": None,
              "upper_current": upper_current, "search_upper_amplitude": max_amplitude,
              "evaluations": [], "release_schedule_proved": False,
              "scope": "fixed nonnegative integer rectangular pulse, all k=0..U sections"}

    def evaluate(amplitude: int) -> dict:
        if amplitude not in cache:
            cert = certify_section_absorption(params, upper_current, amplitude,
                                               max_steps, max_sections, include_map=False)
            cache[amplitude] = cert
            result["evaluations"].append({key: cert[key] for key in (
                "amplitude", "status", "complete", "absorbs_all_sections",
                "tested_sections", "unknown_count", "return_step_updates")})
        return cache[amplitude]

    high = evaluate(max_amplitude)
    if not high["complete"]:
        result["reason"] = "upper-amplitude map is incomplete"
        return result
    if not high["absorbs_all_sections"]:
        result.update(status="unbracketed", reason="declared search upper amplitude is not absorbing")
        return result
    low, high_value = 0, max_amplitude
    while low < high_value:
        middle = (low + high_value) // 2
        cert = evaluate(middle)
        if not cert["complete"]:
            result["reason"] = "a binary-search map is incomplete"
            return result
        if cert["absorbs_all_sections"]:
            high_value = middle
        else:
            low = middle + 1
    at_minimum = evaluate(low)
    predecessor = evaluate(low - 1) if low else None
    if not at_minimum["complete"] or (predecessor is not None and not predecessor["complete"]):
        result["reason"] = "final boundary verification is incomplete"
        return result
    if not at_minimum["absorbs_all_sections"] or (predecessor is not None and predecessor["absorbs_all_sections"]):
        raise AssertionError("return-map amplitude order or boundary invariant failed")
    result.update(status="exact", exact=True, minimal_amplitude=low,
                  reason="complete maps certify absorption at the boundary and a barrier below it",
                  at_minimum=at_minimum, below_minimum=predecessor)
    return result


def certify_finite_section_pulse(params: Params, upper_current: int, amplitude: int,
                                 max_pulse_steps: int = 4096,
                                 max_recovery_steps: int = 4096,
                                 max_sections: int = 65536,
                                 max_total_updates: int = 2_000_000) -> dict:
    """Replay full model sections until every current/spike is zero, then release.

    This deliberately does not use a bottom return as a release condition.
    Representative inactive voltage zero is sufficient for active trajectories:
    any legal inactive voltage stays silent under nonnegative inhibition/erase.
    At release, BOTH cells' worst legal voltage magnitudes are covered by qv.
    The returned duration is sufficient, not a minimum-duration claim.
    """
    _domain(params, upper_current, amplitude)
    for name, value in (("max_pulse_steps", max_pulse_steps),
                        ("max_recovery_steps", max_recovery_steps),
                        ("max_total_updates", max_total_updates)):
        _integer(name, value)
    _integer("max_sections", max_sections, 1)
    result = {"certified": False, "status": "unknown", "amplitude": amplitude,
              "upper_current": upper_current, "duration": None, "recovery": None,
              "total_steps": None, "full_model_updates": 0,
              "scope": "all k=0..U post-spike sections and all legal inactive voltages",
              "release_rule": "all active current/spike zero; inactive current/spike are invariantly zero",
              "bottom_was_used_as_release": False,
              "assumptions": ["no coupling, normal input or noise",
                              "inactive current/spike initially zero; its voltage is legal",
                              "nonnegative inhibition and common constant erase input",
                              "toward-zero voltage leak, exact saturation and hard reset"]}
    if upper_current + 1 > max_sections:
        result["reason"] = "section budget smaller than the complete required interval"
        return result
    states = [State((0, 0), (current, 0), (1, 0)) for current in range(upper_current + 1)]
    duration = None
    for tick in range(1, max_pulse_steps + 1):
        if result["full_model_updates"] + len(states) > max_total_updates:
            result["reason"] = "full-model replay update budget exhausted"
            return result
        states = [step(params, state, erase=amplitude) for state in states]
        result["full_model_updates"] += len(states)
        if any(state.spike[1] or state.current[1] for state in states):
            raise AssertionError("inactive-silence assumption violated by the model")
        if all(state.current == (0, 0) and state.spike == (0, 0) for state in states):
            duration = tick
            break
    if duration is None:
        result["reason"] = "no common current-zero release within pulse budget"
        return result
    result["current_zero_at_pulse_end"] = True
    result["spike_zero_at_pulse_end"] = True
    result["candidate_duration"] = duration
    magnitude, recovery = params.qv, 0
    while magnitude:
        if recovery >= max_recovery_steps:
            result["reason"] = "voltage recovery budget exhausted"
            return result
        magnitude = (params.alpha_num * magnitude) // (1 << params.alpha_shift)
        recovery += 1
    result.update(certified=True, status="certified", duration=duration,
                  recovery=recovery, total_steps=duration + recovery,
                  voltage_abs_bound_at_release=params.qv,
                  reason="full-model section replay establishes current-zero release; integer tail covers both cells")
    return result
