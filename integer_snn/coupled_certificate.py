"""Scalar sufficient certificate for a whole coupled integer SNN.

The theorem is conditional on a declared event/current initial domain.
It proves the output window before using it as an input assumption, avoiding
circular assume-guarantee reasoning. No network trajectory is simulated here.
"""
from __future__ import annotations

from .model import clip, trunc_div
from .network import NetworkSpec


def certify_coupled_erasure(
    spec: NetworkSpec,
    amplitude: int,
    spiking_current_bound: int,
    silent_current_bound: int,
    max_duration: int = 64,
    max_recovery_steps: int = 4096,
) -> dict:
    """Return a conditional uniform certificate or unknown, never infeasible.

    Caller must establish initial legal post-states with I<=J when s=1 and
    I<=H when s=0. Coupling remains enabled. Normal inputs/noise are zero
    throughout pulse and recovery. Parameters are shared across all modules.

    Matrix processing uses the dense NetworkSpec representation, O(M**2);
    subsequent scalar checks do not enumerate the joint state space.
    """
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    for name, value in (("amplitude", amplitude), ("spiking_current_bound", spiking_current_bound),
                        ("silent_current_bound", silent_current_bound),
                        ("max_duration", max_duration), ("max_recovery_steps", max_recovery_steps)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer, excluding bool")
    p = spec.params
    E, J, H, R, K = amplitude, spiking_current_bound, silent_current_bound, spec.max_row_sum, p.threshold - 1
    if not 0 <= J <= H <= p.qi:
        raise ValueError("initial-domain bounds must satisfy 0 <= J <= H <= qi")
    F = lambda x: (p.beta_num * x) // (1 << p.beta_shift)
    T = lambda x: trunc_div(p.alpha_num * x, 1 << p.alpha_shift)
    Cv = lambda x: clip(x, -p.qv, p.qv)
    conditions = {
        "C0_silent_current": {"lhs": F(H), "rhs": J},
        "C0_spike_current": {"lhs": min(p.qi, F(J) + p.weight), "rhs": H},
        "C1_no_consecutive_spikes": {"lhs": H + R - E, "rhs": K},
        "C2_two_tick_window": {"lhs": T(Cv(H - E)) + F(H) + R - E, "rhs": K},
        "C3_future_silence": {"lhs": T(K) + F(J) + R - E, "rhs": K},
    }
    for condition in conditions.values():
        condition["holds"] = condition["lhs"] <= condition["rhs"]
    result = {
        "certified": False, "status": "conditions_unknown",
        "amplitude": E, "duration": None, "recovery": None, "total_steps": None,
        "module_count": spec.module_count, "max_row_sum": R,
        "initial_domain": {"spiking_current_upper": J, "silent_current_upper": H,
                           "spiking_voltage": 0, "silent_voltage": [-p.qv, K],
                           "all_legal_post_states_in_product_domain": True},
        "conditions": conditions,
        "assumptions": ["all initial cells belong to the explicitly stated domain",
                        "nonnegative same-index old-spike coupling; shared Params",
                        "common integer pulse; coupling stays enabled",
                        "zero normal input and noise during pulse and recovery",
                        "exact saturating current-first transition and toward-zero voltage leak"],
        "claim": "conditional sufficient certificate, not an optimal amplitude or infeasibility test",
        "proof_dependencies": [
            "C0+C1 establishes event/current invariant without window assumptions",
            "no consecutive output spikes implies weighted two-tick input bound",
            "C2+C3 limits all new network spikes to the first update",
            "release inequality makes global silence invariant with coupling enabled",
            "integer decay bounds yield finite exact zero"],
        "dense_matrix_entries_examined": spec.module_count ** 2,
        "joint_states_enumerated": 0,
    }
    failed = [name for name, cond in conditions.items() if not cond["holds"]]
    if failed:
        result["reason"] = "sufficient inequalities failed: " + ", ".join(failed)
        return result
    current_at_release = H
    duration = None
    for candidate in range(2, max_duration + 1):
        if T(K) + F(current_at_release) <= K:
            duration = candidate
            break
        current_at_release = F(current_at_release)
    if duration is None:
        result["status"] = "duration_unknown"
        result["reason"] = "no safe release duration within declared limit"
        return result
    result["release_condition"] = {
        "duration": duration, "current_upper": current_at_release,
        "lhs": T(K) + F(current_at_release), "rhs": K}
    lower, upper, current = -p.qv, K, current_at_release
    trace = [{"after_release_steps": 0, "current_upper": current, "voltage": [lower, upper]}]
    for recovery in range(max_recovery_steps + 1):
        if current == 0 and lower == upper == 0:
            result.update(certified=True, status="certified", duration=duration,
                          recovery=recovery, total_steps=duration + recovery,
                          reason="all scalar conditions and exact interval decay tail verified",
                          recovery_trace=trace, all_spikes_zero_from_update=2)
            return result
        if recovery == max_recovery_steps:
            break
        current = F(current)
        lower, upper = Cv(T(lower)), Cv(T(upper) + current)
        if upper >= p.threshold:
            raise AssertionError("release invariant violated by certificate implementation")
        trace.append({"after_release_steps": recovery + 1, "current_upper": current,
                      "voltage": [lower, upper]})
    result.update(status="recovery_unknown", reason="integer tail exceeds declared recovery limit",
                  recovery_trace=trace)
    return result
