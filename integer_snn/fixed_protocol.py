"""Fixed rectangular protocols and wall-accounted selection on one state domain.

Every accepted candidate really receives exactly D erase transitions before
recovery.  Caches preserve transition images, never extend an early-release
certificate.  Budget accounting happens after complete preprocessing/candidate
calls; it is deliberately not a hard runtime limit.  Native interval and
concrete event counts are distinct descriptive quantities, not equal costs.
"""

from __future__ import annotations

from collections import Counter
from itertools import product
import random
import time

from .budgeted_erasure import canonicalize_groups
from .certificates import certify_retention
from .model import Params, State, step, validate_state, zero_state
from .phase_erasure import _advance, _merge, _read_boxes, from_retention_cert


METHODS = ("certified", "empirical", "exact")


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def state_key(state):
    return state.v, state.current, state.spike


def enumerate_domain(boxes):
    """Materialize the finite union, with duplicates counted once."""
    result = set()
    for box in boxes:
        ranges = [range(lo, hi + 1) for lo, hi in box["v"] + box["current"]]
        for v0, v1, i0, i1 in product(*ranges):
            result.add(State((v0, v1), (i0, i1), tuple(box["spike"])))
    return tuple(sorted(result, key=state_key))


def stratified_sample(boxes, seed, target=128):
    """All phase corners plus phase-balanced random states without replacement.

    Enumerating the domain for sampling is preprocessing and is charged.  Small
    domains are exhaustively simulated and explicitly identified as such.
    The target bounds the total sample except if all mandatory corners exceed
    it; in that case use the full domain, never silently drop a phase corner.
    """
    _integer("target", target, 16)
    if target > 128:
        raise ValueError("target must not exceed 128")
    domain = enumerate_domain(boxes)
    rng = random.Random(seed)
    corners = set()
    phase_states = []
    for box in boxes:
        for v0, v1, i0, i1 in product(*(sorted(set(pair)) for pair in box["v"] + box["current"])):
            corners.add(State((v0, v1), (i0, i1), tuple(box["spike"])))
        members = list(enumerate_domain([box]))
        rng.shuffle(members)
        phase_states.append(members)
    if len(domain) <= target or len(corners) > target:
        selected = set(domain)
    else:
        selected = set(corners)
        # Repeated phase boxes remain separate strata, as in the input contract.
        while len(selected) < target:
            progress = False
            for members in phase_states:
                while members and members[-1] in selected:
                    members.pop()
                if members:
                    selected.add(members.pop())
                    progress = True
                if len(selected) >= target:
                    break
            if not progress:
                raise AssertionError("stratified sampler failed to cover its finite domain")
    return tuple(sorted(selected, key=state_key)), {
        "domain_states": len(domain), "sample_states": len(selected),
        "corner_states": len(corners), "all_phase_corners_included": corners <= selected,
        "full_domain": len(selected) == len(domain), "sample_seed": seed,
        "sampling_rule": "all phase corners, round-robin shuffled phase states without replacement",
    }


class IntervalChecker:
    """Sound split interval image propagation for an explicitly fixed D."""

    def __init__(self, params, boxes, max_boxes=128):
        self.params = params
        self.max_boxes = _integer("max_boxes", max_boxes, 1)
        self.counts = Counter(box_merge_visits=0, interval_neuron_updates=0,
                              interval_pulse_steps=0, interval_recovery_steps=0,
                              interval_recovery_cache_hits=0)
        self.initial = _merge(_read_boxes(params, boxes), "split", self.counts)
        self.amplitude = None
        self.duration = 0
        self.image = self.initial
        self.recovery_cache = {}

    def check(self, amplitude, duration, recovery_limit=32):
        _integer("amplitude", amplitude)
        _integer("duration", duration, 1)
        _integer("recovery_limit", recovery_limit)
        if amplitude != self.amplitude or duration < self.duration:
            self.amplitude, self.duration, self.image = amplitude, 0, self.initial
        while self.duration < duration:
            if len(self.image) > self.max_boxes:
                return {"accepted": False, "status": "box_unknown", "recovery": None}
            self.image = _merge(_advance(self.params, self.image, amplitude, self.counts),
                                "split", self.counts)
            self.duration += 1
            self.counts["interval_pulse_steps"] += 1
        # Only the image AFTER the requested D is used for release.
        if len(self.image) > self.max_boxes:
            return {"accepted": False, "status": "box_unknown", "recovery": None}
        key = (frozenset(self.image), recovery_limit)
        if key in self.recovery_cache:
            self.counts["interval_recovery_cache_hits"] += 1
            return dict(self.recovery_cache[key])
        image, seen = self.image, set()
        answer = {"accepted": False, "status": "interval_unknown", "recovery": None}
        for recovery in range(recovery_limit + 1):
            if all(b.spike == (0, 0) and b.v == ((0, 0), (0, 0))
                   and b.current == ((0, 0), (0, 0)) for b in image):
                answer = {"accepted": True, "status": "certified", "recovery": recovery}
                break
            signature = frozenset(image)
            if signature in seen or len(image) > self.max_boxes:
                break
            seen.add(signature)
            if recovery < recovery_limit:
                image = _merge(_advance(self.params, image, 0, self.counts), "split", self.counts)
                self.counts["interval_recovery_steps"] += 1
        self.recovery_cache[key] = answer
        return dict(answer)


class ConcreteChecker:
    """Exact concrete trajectories on the supplied set, possibly a sample."""

    def __init__(self, params, states):
        self.params = params
        self.initial = tuple(sorted(set(states), key=state_key))
        if not self.initial:
            raise ValueError("states must not be empty")
        for state in self.initial:
            validate_state(params, state)
        self.counts = Counter(concrete_transitions=0, concrete_transition_cache_hits=0,
                              concrete_release_cache_hits=0, concrete_pulse_steps=0,
                              concrete_release_starts=0)
        self.amplitude, self.duration, self.image = None, 0, self.initial
        self.transition_cache = {}
        self.release_cache = {zero_state(): 0}

    def transition(self, state, amplitude):
        key = (state, amplitude)
        if key in self.transition_cache:
            self.counts["concrete_transition_cache_hits"] += 1
            return self.transition_cache[key]
        result = step(self.params, state, erase=amplitude)
        self.transition_cache[key] = result
        self.counts["concrete_transitions"] += 1
        return result

    def release(self, start, limit):
        self.counts["concrete_release_starts"] += 1
        path, visited, state = [], set(), start
        for _ in range(limit + 1):
            if state in self.release_cache:
                self.counts["concrete_release_cache_hits"] += 1
                distance = self.release_cache[state]
                for previous in reversed(path):
                    distance = None if distance is None else distance + 1
                    self.release_cache[previous] = distance
                return self.release_cache[start]
            if state in visited:
                for previous in path:
                    self.release_cache[previous] = None  # genuine nonzero cycle
                return None
            visited.add(state)
            if len(path) == limit:
                return None  # finite horizon failure, not a cycle assertion
            path.append(state)
            state = self.transition(state, 0)
        raise AssertionError("unreachable release loop")

    def check(self, amplitude, duration, recovery_limit=32):
        _integer("amplitude", amplitude)
        _integer("duration", duration, 1)
        _integer("recovery_limit", recovery_limit)
        if amplitude != self.amplitude or duration < self.duration:
            self.amplitude, self.duration, self.image = amplitude, 0, self.initial
        while self.duration < duration:
            self.image = tuple(sorted({self.transition(s, amplitude) for s in self.image}, key=state_key))
            self.duration += 1
            self.counts["concrete_pulse_steps"] += 1
        recovery = 0
        for state in self.image:
            distance = self.release(state, recovery_limit)
            if distance is None or distance > recovery_limit:
                return {"accepted": False, "status": "concrete_counterexample", "recovery": None}
            recovery = max(recovery, distance)
        return {"accepted": True, "status": "concrete_pass", "recovery": recovery}


def certify_fixed_protocol(params, boxes, amplitude, duration, recovery_limit=32, max_boxes=128):
    """Standalone fixed protocol certificate; no earlier-release substitution."""
    checker = IntervalChecker(params, boxes, max_boxes)
    result = checker.check(amplitude, duration, recovery_limit)
    return {**result, "amplitude": amplitude, "duration": duration,
            "operation_counts": dict(checker.counts)}


def reconstruct_case(case):
    params = Params(**case["params"])
    certs = {s: certify_retention(params, case["period"], case["current_interval"],
                                 noise_bound=case["retention_noise_bound"], active=s)
             for s in (0, 1)}
    if not all(cert["certified"] for cert in certs.values()):
        raise ValueError("source retention certificate does not reconstruct")
    groups = {s: from_retention_cert(cert) for s, cert in certs.items()}
    boxes, _ = canonicalize_groups(params, groups)
    return params, groups, boxes


def build_checker(case, method, sample_seed, max_boxes, sample_target):
    params, groups, boxes = reconstruct_case(case)
    metadata = {"sample_states": None, "domain_states": None, "full_domain": True,
                "corner_states": None, "e0_excluded_by_nonzero_retention": True}
    if method == "certified":
        checker = IntervalChecker(params, boxes, max_boxes)
    elif method == "empirical":
        states, sampling = stratified_sample(boxes, sample_seed, sample_target)
        metadata.update(sampling)
        checker = ConcreteChecker(params, states)
    elif method == "exact":
        states = enumerate_domain(boxes)
        metadata.update(domain_states=len(states), sample_states=len(states))
        checker = ConcreteChecker(params, states)
    else:
        raise ValueError("unknown method")
    # The reconstructed nonzero periodic T1 domain supplies a surviving orbit;
    # with E=0 the pulse is unforced, so no D can erase the entire domain.
    return checker, metadata


def select_protocol(case, method, budget_seconds, sample_seed, *, max_amplitude=64,
                    max_duration=32, recovery_limit=32, max_boxes=128,
                    sample_target=128, clock=None):
    """Lexicographic E,D selection with preprocessing and full-call wall charges.

    Source JSON reading is common setup.  Retention reconstruction,
    canonicalization, enumeration/sampling, caches and candidate calls all run
    afresh and are charged on every real repeat.  A completed over-budget
    candidate contributes events and actual wall time but cannot be accepted.
    """
    if method not in METHODS:
        raise ValueError("unknown method")
    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)) or budget_seconds < 0:
        raise ValueError("budget_seconds must be nonnegative")
    _integer("max_amplitude", max_amplitude)
    _integer("max_duration", max_duration, 1)
    _integer("recovery_limit", recovery_limit)
    timer = clock or time.perf_counter
    started, cpu_start = timer(), time.process_time()
    checker = None
    metadata = {}
    calls, in_budget_calls, accepted, last_candidate = 0, 0, None, None
    charged = 0.0
    if budget_seconds == 0:
        elapsed = 0.0
        preprocessing = 0.0
        status = "budget_unknown"
    else:
        checker, metadata = build_checker(case, method, sample_seed, max_boxes, sample_target)
        elapsed = preprocessing = timer() - started
        status = "range_unknown"
        if elapsed > budget_seconds:
            status = "budget_unknown"
        else:
            charged = elapsed
            for amplitude in range(1, max_amplitude + 1):
                for duration in range(1, max_duration + 1):
                    # Do not start a new call once the previous accounting
                    # boundary already exhausted the budget.
                    if elapsed >= budget_seconds:
                        status = "budget_unknown"
                        break
                    answer = checker.check(amplitude, duration, recovery_limit)
                    calls += 1
                    elapsed = timer() - started
                    last_candidate = {"amplitude": amplitude, "duration": duration,
                                      "candidate_accepted": answer["accepted"]}
                    if elapsed > budget_seconds:
                        status = "budget_unknown"
                        break
                    in_budget_calls += 1
                    charged = elapsed
                    if answer["accepted"]:
                        accepted = {"amplitude": amplitude, "duration": duration,
                                    "reported_recovery": answer["recovery"],
                                    "deploy_total": duration + recovery_limit,
                                    "recovery_kind": "observed_maximum" if method == "empirical" and not metadata["full_domain"]
                                                     else "full_finite_domain_exact" if method != "certified"
                                                     else "sound_interval_bound"}
                        status = "accepted"
                        break
                if accepted is not None or status == "budget_unknown":
                    break
    return {"status": status, "protocol": accepted, "method": method,
            "budget_seconds": budget_seconds, "wall_seconds": elapsed,
            "cpu_seconds": time.process_time() - cpu_start,
            "preprocessing_seconds": preprocessing, "accounted_seconds": charged,
            "overrun_seconds": max(0.0, elapsed - budget_seconds),
            "calls_completed": calls, "calls_in_budget": in_budget_calls,
            "last_candidate": last_candidate,
            "native_event_counts": dict(checker.counts) if checker else {}, **metadata}
