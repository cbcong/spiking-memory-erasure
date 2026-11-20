"""Symmetry-aware erasure search under a declared *accounting* work budget.

The budget is assessed after a complete certificate call.  It is not a CPU,
time, or preemption limit.  An over-budget call really executes, but its result
cannot be accepted.  Only the phase checker's documented abstract work events
are charged; preprocessing and Python overhead are measured separately in time.
"""

from __future__ import annotations

from typing import Any

from .model import Params, State, validate_state
from .phase_erasure import certify_phase_erasure


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def canonicalize_groups(
    params: Params,
    groups: dict[int, list[dict[str, Any]]],
    canonicalize: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map symbol 1 to symbol 0 orientation before either split or hull.

    Swapping neurons commutes with this shared-parameter two-neuron transition
    for a common erase amplitude and zero other inputs.  Certifying the UNION
    of mapped sets therefore gives ONE common pulse and recovery schedule for
    every original group.  We never combine group-specific durations by max.
    """
    if not isinstance(params, Params):
        raise TypeError("params must be Params")
    if type(canonicalize) is not bool:
        raise ValueError("canonicalize must be bool")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("groups must be a nonempty dictionary keyed by 0 or 1")
    if any(type(label) is not int or label not in (0, 1) for label in groups):
        raise ValueError("group labels must be integer 0 or 1")
    boxes = []
    group_counts = {}
    for label in sorted(groups):
        values = groups[label]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("each group must contain at least one box")
        group_counts[str(label)] = len(values)
        for value in values:
            if not isinstance(value, dict) or not {"v", "current", "spike"} <= value.keys():
                raise ValueError("every box needs v, current and spike")
            for field in ("v", "current", "spike"):
                if not isinstance(value[field], (list, tuple)) or len(value[field]) != 2:
                    raise ValueError("every field must have exactly two neuron entries")
            for field in ("v", "current"):
                for interval in value[field]:
                    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                            or any(type(x) is not int for x in interval)
                            or interval[0] > interval[1]):
                        raise ValueError("box intervals must be ordered integer pairs")
            for side in (0, 1):
                validate_state(params, State(tuple(x[side] for x in value["v"]),
                                             tuple(x[side] for x in value["current"]),
                                             tuple(value["spike"])))
            order = (1, 0) if canonicalize and label == 1 else (0, 1)
            boxes.append({"v": [list(value["v"][i]) for i in order],
                          "current": [list(value["current"][i]) for i in order],
                          "spike": [value["spike"][i] for i in order]})
    info = {
        "enabled": canonicalize,
        "group_box_counts": group_counts,
        "canonical_orientation": "symbol_0" if canonicalize else "original_mixed_symbols",
        "common_schedule_construction": "one certificate on union of transformed groups",
        "symmetry_assumptions": [
            "both neurons use the same Params fields, register ranges and reset rule",
            "coupling is symmetric mutual inhibition",
            "erase amplitude is identical at both neurons",
            "normal inputs, noise and inter-module coupling are absent",
            "coordinate swap commutes with every pulse and recovery step",
        ] if canonicalize else [],
        "scope": "shared-parameter symmetric two-neuron model only",
    }
    return boxes, info


def search_budgeted_erasure(
    params: Params,
    groups: dict[int, list[dict[str, Any]]],
    max_amplitude: int,
    work_budget: int,
    merge_mode: str = "split",
    max_prefix_steps: int = 24,
    max_boxes: int = 128,
    canonicalize: bool = True,
) -> dict[str, Any]:
    """Try E=0,... in fixed order, accepting only fully accounted certificates.

    A successful amplitude is minimal among certificates returned within this
    algorithm's search/box limits; it is not a true control optimum.  A rejected
    completed call contributes to computed_work, but not accounted_work.
    """
    _integer("max_amplitude", max_amplitude)
    _integer("work_budget", work_budget)
    _integer("max_prefix_steps", max_prefix_steps)
    _integer("max_boxes", max_boxes, 1)
    if merge_mode not in ("split", "hull"):
        raise ValueError("merge_mode must be split or hull")
    boxes, symmetry = canonicalize_groups(params, groups, canonicalize)
    result = {
        "certified": False, "status": "budget_unknown", "certificate": None,
        "minimum_certified_amplitude": None, "work_budget": work_budget,
        "accounted_work": 0, "computed_work": 0, "budget_overrun_work": 0,
        "calls_completed": 0, "accepted_calls": 0, "attempts": [],
        "merge_mode": merge_mode, "canonicalization": symmetry,
        "budget_kind": "post_call_abstract_work_accounting_not_hard_runtime",
        "work_unit": "phase checker abstract interval-work events",
        "preprocessing_charged": False,
    }
    for amplitude in range(max_amplitude + 1):
        remaining = work_budget - result["accounted_work"]
        if remaining == 0:
            result["reason"] = "no accounting budget remains for another full call"
            return result
        cert = certify_phase_erasure(params, boxes, amplitude,
                                     max_prefix_steps=max_prefix_steps, max_boxes=max_boxes,
                                     merge_mode=merge_mode, release_mode="safe")
        work = cert["operation_count"]
        result["computed_work"] += work
        result["calls_completed"] += 1
        accepted = work <= remaining
        result["attempts"].append({"amplitude": amplitude, "operation_count": work,
                                   "accepted_in_budget": accepted,
                                   "calculated_status": cert["status"]})
        if not accepted:
            result["budget_overrun_work"] = result["computed_work"] - work_budget
            result["reason"] = "completed call exceeded remaining accounting budget; result rejected"
            return result
        result["accounted_work"] += work
        result["accepted_calls"] += 1
        if cert["certified"]:
            result["certified"] = True
            result["status"] = "certified"
            result["certificate"] = cert
            result["minimum_certified_amplitude"] = amplitude
            result["reason"] = "first sufficient certificate in fixed integer amplitude order"
            return result
    result["status"] = "range_unknown"
    result["reason"] = "no certificate in the declared amplitude range and prefix/box limits"
    return result
