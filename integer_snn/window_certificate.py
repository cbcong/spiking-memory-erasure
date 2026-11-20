"""A specialized interval checker using a proved two-tick input contract.

C0/C1 are checked independently before any temporal input restriction is used.
This is an abstract-interpretation specialization, not a new composition theory.
All cells of the network must initially belong to the supplied scalar union.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .model import clip, trunc_div
from .network import NetworkSpec
from .phase_erasure import from_retention_cert


@dataclass(frozen=True)
class _Box:
    v: tuple[int, int]
    current: tuple[int, int]
    spike: int
    last_p: int | None

    def serial(self):
        return {"v": list(self.v), "current": list(self.current),
                "spike": self.spike, "last_p": self.last_p}


class _Limit(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def scalar_boxes_from_retention(cert):
    """Project both cells of every certified T1 phase, including its return.

    Projection loses within-module correlations.  The resulting scalar product
    is an enclosing initial domain, not the exact original paired phase set.
    """
    projected = []
    for box in from_retention_cert(cert):
        for cell in (0, 1):
            value = {"v": box["v"][cell][:], "current": box["current"][cell][:],
                     "spike": box["spike"][cell]}
            if value not in projected:
                projected.append(value)
    return projected


def _read_boxes(params, values, J, H):
    if values is None:
        return [_Box((-params.qv, params.threshold - 1), (0, H), 0, None),
                _Box((0, 0), (0, J), 1, None)]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("initial_boxes must be a nonempty scalar box list")
    result = []
    for value in values:
        if not isinstance(value, dict) or not {"v", "current", "spike"} <= value.keys():
            raise ValueError("scalar boxes require v, current, spike")
        pairs = []
        for key in ("v", "current"):
            pair = value[key]
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(x) is not int for x in pair) or pair[0] > pair[1]):
                raise ValueError("box bounds must be ordered integer pairs")
            pairs.append(tuple(pair))
        v, current = pairs
        spike = value["spike"]
        if type(spike) is not int or spike not in (0, 1):
            raise ValueError("spike must be exact integer 0 or 1")
        if not -params.qv <= v[0] <= v[1] <= params.qv:
            raise ValueError("voltage is outside the register range")
        if (spike and v != (0, 0)) or (not spike and v[1] >= params.threshold):
            raise ValueError("box violates the legal post-reset invariant")
        if not 0 <= current[0] <= current[1] <= (J if spike else H):
            raise ValueError("box is outside the declared event/current initial domain")
        if value.get("last_p") is not None:
            raise ValueError("initial input history is unknown; last_p must be absent or None")
        result.append(_Box(v, current, spike, None))
    return list(dict.fromkeys(result))


def certify_window_erasure(
    spec: NetworkSpec, amplitude: int, spiking_current_bound: int, silent_current_bound: int,
    *, initial_boxes=None, mode="window", merge_mode="split", max_duration=32,
    max_boxes=512, max_operations=200000, max_recovery_steps=4096,
) -> dict[str, Any]:
    """Return a uniform common pulse certificate or explicitly budgeted unknown.

    `stepwise` disables only P[t]+P[t-1]<=R; it keeps identical scalar keys,
    initial sets and global-silence feedback.  Both modes first require C0/C1.
    Operation units are logical checker events, not hardware instructions.
    Limits are enforced before each charged event, including the recovery tail.
    """
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    for name, value in (("amplitude", amplitude), ("spiking_current_bound", spiking_current_bound),
                        ("silent_current_bound", silent_current_bound), ("max_duration", max_duration),
                        ("max_operations", max_operations), ("max_recovery_steps", max_recovery_steps)):
        _integer(name, value)
    _integer("max_boxes", max_boxes, 1)
    if mode not in ("window", "stepwise") or merge_mode not in ("key_hull", "split"):
        raise ValueError("invalid propagation mode or merge mode")
    p, E, J, H, R = spec.params, amplitude, spiking_current_bound, silent_current_bound, spec.max_row_sum
    if not 0 <= J <= H <= p.qi:
        raise ValueError("bounds must satisfy 0 <= J <= H <= qi")
    boxes = _read_boxes(p, initial_boxes, J, H)
    F = lambda x: (p.beta_num * x) // (1 << p.beta_shift)
    T = lambda x: trunc_div(p.alpha_num * x, 1 << p.alpha_shift)
    C = lambda x: clip(x, -p.qv, p.qv)
    counts = {key: 0 for key in ("prerequisite_checks", "input_branches", "endpoint_updates",
                                "reset_branches", "merge_visits", "safe_checks", "tail_iterations")}
    result = {
        "certified": False, "status": "unknown", "reason": "not yet evaluated",
        "params": asdict(p), "module_count": spec.module_count, "max_row_sum": R,
        "amplitude": E, "duration": None, "recovery": None, "total_steps": None,
        "initial_domain": {"spiking_current_bound": J, "silent_current_bound": H,
                           "scope": "every network cell lies in the scalar union", "boxes": [b.serial() for b in boxes]},
        "mode": mode, "merge_mode": merge_mode, "conditions": {}, "prefix_steps": 0,
        "history": [], "recovery_trace": [], "operation_counts": counts, "operation_count": 0,
        "max_operations": max_operations, "max_boxes": max_boxes, "max_boxes_seen": len(boxes),
        "joint_states_enumerated": 0, "dense_matrix_entries_examined": spec.module_count ** 2,
        "certificate_kind": "scalar_integer_interval_prefix_with_proved_input_window",
        "assumptions": [
            "shared Params, nonnegative integer same-index old-spike coupling with each row sum <= R",
            "all initial cells lie in the supplied scalar union within the declared J/H event domain",
            "common rectangular erase, no normal input or external noise during pulse and recovery",
            "C0/C1 are proved without temporal input assumptions before using any input window",
            "all abstract cells silent forces only the NEXT coupling input to zero; coupling is never disabled",
            "permanent zero coupling and zero mutual inhibition in the tail follow from the release invariant",
            "unknown is not infeasible; counts are abstract checker work events, excluding dense matrix scan",
        ],
    }

    def charge(kind):
        if result["operation_count"] >= max_operations:
            raise _Limit("operation_budget_unknown", "logical operation budget exhausted")
        counts[kind] += 1
        result["operation_count"] += 1

    def merge(values):
        collection = {}
        for box in values:
            charge("merge_visits")
            key = box if merge_mode == "split" else (box.spike, box.last_p)
            if key in collection and merge_mode == "key_hull":
                previous = collection[key]
                box = _Box((min(previous.v[0], box.v[0]), max(previous.v[1], box.v[1])),
                           (min(previous.current[0], box.current[0]), max(previous.current[1], box.current[1])),
                           box.spike, box.last_p)
            collection[key] = box
            result["max_boxes_seen"] = max(result["max_boxes_seen"], len(collection))
            if len(collection) > max_boxes:
                raise _Limit("box_budget_unknown", "active scalar-box budget exceeded")
        return list(collection.values())

    def propagate(current, tail=False):
        # Recomputed at every step.  One silent instant is NOT a permanent gate.
        globally_silent = all(box.spike == 0 for box in current)
        generated = []
        for box in current:
            if tail or globally_silent:
                inputs = (0,)
            else:
                upper = R - box.last_p if mode == "window" and box.last_p is not None else R
                inputs = range(upper + 1)
            for incoming in inputs:
                charge("input_branches")
                currents = tuple(min(p.qi, F(x) + p.weight * box.spike) for x in box.current)
                voltage = []
                for side in (0, 1):
                    charge("endpoint_updates")
                    inhibitory = 0 if tail or side == 1 else -p.inhibition
                    voltage.append(C(T(box.v[side]) + currents[side] + incoming
                                     + inhibitory - (0 if tail else E)))
                if voltage[0] < p.threshold:
                    charge("reset_branches")
                    generated.append(_Box((voltage[0], min(voltage[1], p.threshold - 1)),
                                          currents, 0, incoming))
                if voltage[1] >= p.threshold:
                    charge("reset_branches")
                    generated.append(_Box((0, 0), currents, 1, incoming))
        return merge(generated)

    def safe(current):
        if any(b.spike for b in current):
            return {"holds": False, "reason": "abstract union still contains spikes"}
        lower = max(0, max(b.v[1] for b in current))
        upper = p.threshold - 1
        current_max = max(b.current[1] for b in current)
        def good(k):
            charge("safe_checks")
            return T(k) + F(current_max) <= k
        if not good(upper):
            return {"holds": False, "reason": "no zero-input silent invariant upper bound",
                    "current_upper": current_max, "voltage_upper": lower}
        while lower < upper:
            middle = (lower + upper) // 2
            if good(middle):
                upper = middle
            else:
                lower = middle + 1
        return {"holds": True, "k": lower, "current_upper": current_max,
                "lhs": T(lower) + F(current_max), "rhs": lower}

    try:
        # Independent of C2/C3 and of all propagated trajectories.
        for name, left, right in (("C0_silent_current", F(H), J),
                                  ("C0_spike_current", min(p.qi, F(J) + p.weight), H),
                                  ("C1_no_consecutive_spikes", H + R - E, p.threshold - 1)):
            charge("prerequisite_checks")
            result["conditions"][name] = {"lhs": left, "rhs": right, "holds": left <= right}
        if not all(value["holds"] for value in result["conditions"].values()):
            result.update(status="conditions_unknown", reason="C0/C1 do not establish the event domain and output window")
            return result
        result["proved_window_contract"] = "P[t]+P[t+1]<=R from no consecutive source spikes under the pulse"
        boxes = merge(boxes)
        for time in range(max_duration + 1):
            result["prefix_steps"] = time
            release = safe(boxes)
            result["history"].append({"step": time, "boxes": [b.serial() for b in boxes],
                                      "all_silent": all(b.spike == 0 for b in boxes),
                                      "next_coupling_forced_zero": all(b.spike == 0 for b in boxes),
                                      "release_condition": release})
            if release["holds"]:
                result["release_condition"] = release
                result["release_boxes"] = [b.serial() for b in boxes]
                for recovery in range(max_recovery_steps + 1):
                    result["recovery_trace"].append({"step_after_release": recovery,
                                                     "boxes": [b.serial() for b in boxes]})
                    if all(b.current == (0, 0) and b.v == (0, 0) and b.spike == 0 for b in boxes):
                        result.update(certified=True, status="certified", duration=time, recovery=recovery,
                                      total_steps=time + recovery, reason="proved window prefix and zero-input silent tail reach exact zero")
                        return result
                    if recovery == max_recovery_steps:
                        raise _Limit("recovery_budget_unknown", "recovery iteration limit exhausted")
                    charge("tail_iterations")
                    boxes = propagate(boxes, tail=True)
                    if any(b.spike for b in boxes):
                        raise AssertionError("zero-input release invariant violated by interval propagation")
                raise AssertionError("unreachable recovery exit")
            if time < max_duration:
                boxes = propagate(boxes)
        result.update(status="duration_budget_unknown", reason="prefix duration limit exhausted without a safe release")
    except _Limit as limit:
        result.update(status=limit.status, reason=limit.reason)
    return result
