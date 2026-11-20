"""Phase-aware sufficient erasure certificates with exact interval arithmetic.

This is a set calculation, not a simulation of selected initial states.  Fixed
old-spike branches are propagated separately through threshold and hard reset.
The default policy releases the pulse as soon as a silent invariant enclosure
is valid with zero erase input.  The drain policy instead enters an invariant
enclosure under the pulse and waits for every current register to decay to zero.
Neither policy permits external inputs, noise, or coupling during erasure or
recovery.  Failure to certify is always reported as unknown, never impossibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from .model import Params, clip, trunc_div


# Resource protection, not a theorem assumption.  A hit returns unknown.
_MAX_DECAY_STEPS = 1_000_000


@dataclass(frozen=True)
class _Box:
    v: tuple[tuple[int, int], tuple[int, int]]
    current: tuple[tuple[int, int], tuple[int, int]]
    spike: tuple[int, int]

    def serial(self) -> dict[str, Any]:
        return {"v": [list(x) for x in self.v],
                "current": [list(x) for x in self.current],
                "spike": list(self.spike)}


def _nonnegative_integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _pair(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integer endpoints")
    if any(type(x) is not int for x in value) or value[0] > value[1]:
        raise ValueError(f"{name} must be an ordered integer interval")
    return tuple(value)


def _read_boxes(params: Params, values: Any) -> list[_Box]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("initial_boxes must be a nonempty list of boxes")
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not {"v", "current", "spike"} <= value.keys():
            raise ValueError(f"box {index} requires v, current, and spike")
        for key in ("v", "current", "spike"):
            if not isinstance(value[key], (list, tuple)) or len(value[key]) != 2:
                raise ValueError(f"box {index}: {key} must have two neuron entries")
        v = tuple(_pair("v", item) for item in value["v"])
        current = tuple(_pair("current", item) for item in value["current"])
        spike = tuple(value["spike"])
        if any(type(s) is not int or s not in (0, 1) for s in spike):
            raise ValueError("each box must have a fixed binary spike tuple")
        for neuron in (0, 1):
            if not -params.qv <= v[neuron][0] <= v[neuron][1] <= params.qv:
                raise ValueError("voltage interval exceeds the register range")
            if not 0 <= current[neuron][0] <= current[neuron][1] <= params.qi:
                raise ValueError("current interval exceeds the register range")
            if spike[neuron] and v[neuron] != (0, 0):
                raise ValueError("a spike requires the singleton reset voltage zero")
            if not spike[neuron] and v[neuron][1] >= params.threshold:
                raise ValueError("a silent post-update voltage must be below threshold")
        result.append(_Box(v, current, spike))
    return result


def from_retention_cert(cert: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy every certified T1 phase, including t=0 and the return phase t=p.

    The initial scope is the union of complete phase boxes, not just a measured
    orbit.  The return phase may be redundant but is intentionally preserved.
    Hold-noise allowances do not carry into this erasure certificate.
    """
    if not isinstance(cert, dict) or cert.get("certified") is not True:
        raise ValueError("a successful retention certificate is required")
    try:
        params = Params(**cert["params"])
        phases = cert["phases"]
    except (KeyError, TypeError) as error:
        raise ValueError("retention certificate lacks valid params or phases") from error
    if not isinstance(phases, list) or not phases:
        raise ValueError("retention certificate must have nonempty phases")
    boxes = []
    for phase in phases:
        if not isinstance(phase, dict) or phase.get("branch_valid") is not True:
            raise ValueError("every retention phase must certify a fixed branch")
        try:
            spikes = [_pair("phase spike", x) for x in phase["spike"]]
            if len(spikes) != 2 or any(lo != hi for lo, hi in spikes):
                raise ValueError("retention phases must have fixed binary spikes")
            boxes.append({"v": phase["voltage_post"], "current": phase["current"],
                          "spike": [x[0] for x in spikes]})
        except (KeyError, TypeError) as error:
            raise ValueError("retention phase lacks valid interval fields") from error
    return [box.serial() for box in _read_boxes(params, boxes)]


def _merge(boxes: list[_Box], mode: str, counts: dict[str, int]) -> list[_Box]:
    counts["box_merge_visits"] += len(boxes)
    unique = list(dict.fromkeys(boxes))
    if mode == "split":
        return unique
    # A sound but deliberately less precise baseline: union bounding boxes
    # with the same spike tuple.  Never invent a single mixed-spike interval.
    groups: dict[tuple[int, int], list[_Box]] = {}
    for box in unique:
        groups.setdefault(box.spike, []).append(box)
    result = []
    for spikes, group in groups.items():
        def hull(field: str) -> tuple[tuple[int, int], tuple[int, int]]:
            return tuple((min(getattr(b, field)[i][0] for b in group),
                          max(getattr(b, field)[i][1] for b in group)) for i in (0, 1))
        result.append(_Box(hull("v"), hull("current"), spikes))
    return result


def _advance(params: Params, boxes: list[_Box], amplitude: int,
             counts: dict[str, int]) -> list[_Box]:
    result = []
    for box in boxes:
        currents = []
        branches = []
        for neuron in (0, 1):
            counts["interval_neuron_updates"] += 1
            new_current = tuple(clip((params.beta_num * x) // (1 << params.beta_shift)
                                     + params.weight * box.spike[neuron], 0, params.qi)
                                for x in box.current[neuron])
            currents.append(new_current)
            z = tuple(clip(trunc_div(params.alpha_num * box.v[neuron][side],
                                    1 << params.alpha_shift) + new_current[side]
                           - params.inhibition * box.spike[1 - neuron] - amplitude,
                           -params.qv, params.qv) for side in (0, 1))
            choices = []
            if z[0] < params.threshold:
                choices.append(((z[0], min(z[1], params.threshold - 1)), 0))
            if z[1] >= params.threshold:
                choices.append(((0, 0), 1))
            branches.append(choices)
        for choice0, choice1 in product(*branches):
            result.append(_Box((choice0[0], choice1[0]), tuple(currents),
                               (choice0[1], choice1[1])))
    return result


def _safe_condition(params: Params, boxes: list[_Box], erase: int,
                    counts: dict[str, int]) -> dict[str, Any]:
    initial_k = [max(0, max(box.v[i][1] for box in boxes)) for i in (0, 1)]
    j = [max(box.current[i][1] for box in boxes) for i in (0, 1)]
    result = {"satisfied": False, "erase": erase, "initial_voltage_upper": initial_k,
              "current_upper": j, "k": [], "next_voltage_upper": []}
    if any(box.spike != (0, 0) for box in boxes):
        result["reason"] = "some branches still carry a spike"
        return result
    for neuron in (0, 1):
        current_next = (params.beta_num * j[neuron]) // (1 << params.beta_shift)

        def next_upper(k: int) -> int:
            counts["safe_inequality_checks"] += 1
            return trunc_div(params.alpha_num * k, 1 << params.alpha_shift) + current_next - erase

        low, high = initial_k[neuron], params.threshold - 1
        if low > high or next_upper(high) > high:
            result["reason"] = "no silent invariant upper voltage fits below threshold"
            return result
        # floor(alpha*k)-k is nonincreasing for 0<alpha<1.  The invariant
        # enclosure may need K larger than the observed upper voltage.
        while low < high:
            mid = (low + high) // 2
            if next_upper(mid) <= mid:
                high = mid
            else:
                low = mid + 1
        result["k"].append(low)
        result["next_voltage_upper"].append(next_upper(low))
    result["satisfied"] = True
    result["reason"] = "common silent invariant enclosure exists for both neurons"
    return result


def certify_phase_erasure(
    params: Params,
    initial_boxes: list[dict[str, Any]],
    amplitude: int,
    max_prefix_steps: int = 16,
    max_boxes: int = 256,
    merge_mode: str = "split",
    release_mode: str = "safe",
) -> dict[str, Any]:
    """Certify one common pulse/recovery schedule for every supplied box state.

    ``safe`` releases at the first prefix with a zero-input silent invariant
    enclosure.  ``drain`` certifies silence under the pulse, then holds it until
    current is zero.  This is a sufficient policy, not a minimum-duration solver.
    ``hull`` merges boxes only within a fixed spike pattern; ``split`` retains
    reset branches and phase correlations.  Both are conservative.

    operation_count counts the documented abstract interval-work events.  It
    is neither CPU instructions nor a hardware/energy cost.  Decay iteration
    protection returns unknown after one million iterations in either stage.
    """
    if not isinstance(params, Params):
        raise TypeError("params must be Params")
    _nonnegative_integer("amplitude", amplitude)
    _nonnegative_integer("max_prefix_steps", max_prefix_steps)
    _nonnegative_integer("max_boxes", max_boxes, 1)
    if merge_mode not in ("split", "hull"):
        raise ValueError("merge_mode must be 'split' or 'hull'")
    if release_mode not in ("safe", "drain"):
        raise ValueError("release_mode must be 'safe' or 'drain'")
    counts = dict.fromkeys(("box_merge_visits", "interval_neuron_updates",
                           "safe_inequality_checks", "current_decay_iterations",
                           "voltage_decay_iterations"), 0)
    boxes = _merge(_read_boxes(params, initial_boxes), merge_mode, counts)
    result: dict[str, Any] = {
        "certified": False, "status": "unknown", "reason": "not yet evaluated",
        "params": asdict(params), "amplitude": amplitude, "prefix_steps": 0,
        "duration": None, "recovery": None, "total_steps": None,
        "current_decay_steps": None, "voltage_decay_steps": None,
        "max_boxes_seen": len(boxes), "operation_count": 0,
        "operation_counts": counts, "operation_count_unit": "abstract interval-work events",
        "merge_mode": merge_mode, "release_mode": release_mode,
        "safe_condition_erase": 0 if release_mode == "safe" else amplitude,
        "terminal_safe_boxes": [], "pulse_end_boxes": [], "current_zero_boxes": [],
        "history": [], "suffix_history": [],
        "certificate_kind": "T2_phase_prefix_silent_invariant_erasure",
        "assumptions": [
            "initial set is the union of supplied legal post-update boxes",
            "every old spike is fixed inside each box; threshold equality fires and hard-resets",
            "one common nonnegative erase input; no direct register clearing",
            "normal input, voltage noise, current noise and cross-module coupling are zero throughout pulse and recovery",
            "integer toward-zero voltage leak, nonnegative floor current decay and saturation",
            "successful result is sufficient only; unknown does not imply erasure is impossible",
            "abstract work count excludes serialization and is not hardware energy or CPU instructions",
        ],
    }

    def finish(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        result["operation_count"] = sum(counts.values())
        return result

    for prefix in range(max_prefix_steps + 1):
        result["prefix_steps"] = prefix
        result["max_boxes_seen"] = max(result["max_boxes_seen"], len(boxes))
        if len(boxes) > max_boxes:
            return finish("box budget exceeded; sufficient certificate remains unknown")
        condition = _safe_condition(params, boxes, result["safe_condition_erase"], counts)
        result["history"].append({"step": prefix, "box_count": len(boxes),
                                  "boxes": [b.serial() for b in boxes],
                                  "safe_condition": condition})
        if condition["satisfied"]:
            result["terminal_safe_boxes"] = [b.serial() for b in boxes]
            result["safe_condition"] = condition
            suffix_amplitude = 0 if release_mode == "safe" else amplitude
            if release_mode == "safe":
                result["pulse_end_boxes"] = [b.serial() for b in boxes]
            current_steps = 0
            while any(b.current[i][1] for b in boxes for i in (0, 1)):
                if current_steps >= _MAX_DECAY_STEPS:
                    return finish("current-decay iteration budget exceeded")
                # Safety already proves that these transitions stay silent.
                boxes = _merge(_advance(params, boxes, suffix_amplitude, counts),
                               merge_mode, counts)
                current_steps += 1
                counts["current_decay_iterations"] += 1
                if any(b.spike != (0, 0) for b in boxes):
                    raise AssertionError("silent invariant violated by interval propagation")
                result["suffix_history"].append({"step": prefix + current_steps,
                                                  "erase": suffix_amplitude,
                                                  "boxes": [b.serial() for b in boxes]})
            result["current_decay_steps"] = current_steps
            result["current_zero_boxes"] = [b.serial() for b in boxes]
            if release_mode == "drain":
                result["pulse_end_boxes"] = [b.serial() for b in boxes]
            magnitude = max(abs(x) for b in boxes for interval in b.v for x in interval)
            result["voltage_magnitude_at_current_zero"] = magnitude
            voltage_steps = 0
            while magnitude:
                if voltage_steps >= _MAX_DECAY_STEPS:
                    return finish("voltage-decay iteration budget exceeded")
                magnitude = (params.alpha_num * magnitude) // (1 << params.alpha_shift)
                voltage_steps += 1
                counts["voltage_decay_iterations"] += 1
            result["voltage_decay_steps"] = voltage_steps
            result["duration"] = prefix + (current_steps if release_mode == "drain" else 0)
            result["recovery"] = voltage_steps + (current_steps if release_mode == "safe" else 0)
            result["total_steps"] = result["duration"] + result["recovery"]
            result["certified"] = True
            result["status"] = "certified"
            return finish("every initial box reaches zero under the certified common pulse and recovery schedule")
        if prefix < max_prefix_steps:
            boxes = _merge(_advance(params, boxes, amplitude, counts), merge_mode, counts)
    return finish("prefix budget exhausted before a common silent invariant enclosure was certified")
