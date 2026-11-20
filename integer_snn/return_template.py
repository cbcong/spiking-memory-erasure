"""Parameterized uniform erasure from exact, labelled first-return relations.

The template is sufficient for the parameter class, and yields an EXACT
minimum common amplitude only after every parameter and topology premise has
passed. A failed premise or exhausted budget is unknown, never infeasibility.
All legal post-reset states and arbitrary nonnegative internal inhibition are
covered. Coupling remains enabled through the finite pulse and recovery.

Proof: notes/topology_generalization_20261113.md. No external certificate dict
is accepted: every public network call reconstructs its parameter evidence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

from .model import Params
from .network import NetworkSpec
from .topology_certificate import analyze_topology


@dataclass(frozen=True)
class TemplateLimits:
    """Deterministic work/storage guards; not hard wall-time guarantees.

    Counts accumulate across ALL amplitude probes, first-event relations,
    write analysis and decay-tail construction in one public call. Graph work
    is a defined unit count, not a count of all graph-loop/CPU operations.
    """
    max_sections: int = 256
    max_relation_steps: int = 4096
    max_base_candidates: int = 64
    max_total_updates: int = 250_000
    max_edges: int = 100_000
    max_graph_work: int = 2_000_000
    max_write_steps: int = 4096
    max_decay_steps: int = 4096

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer, excluding bool")


class _Limit(Exception):
    pass


class _Budget:
    def __init__(self, limits):
        if not isinstance(limits, TemplateLimits):
            raise TypeError("limits must be TemplateLimits")
        self.limits = limits
        self.used = {"total_updates": 0, "edges": 0, "graph_work": 0,
                     "base_candidates": 0, "relations": 0}

    def add(self, kind, amount=1):
        limit = getattr(self.limits, "max_" + kind)
        if self.used[kind] + amount > limit:
            raise _Limit(f"max_{kind} exhausted before the next operation")
        self.used[kind] += amount

    def sections(self, p):
        if p.qi + 1 > self.limits.max_sections:
            raise _Limit("max_sections smaller than the full current-register domain")

    def report(self):
        return {"limits": asdict(self.limits), "used": dict(self.used),
                "semantics": "deterministic cumulative work/storage guards; no hard wall-time claim",
                "unit_definitions": {
                    "total_updates": "scalar envelope iterations, J write iterations and decay-tail leak updates",
                    "edges": "stored timed first-event rows across all complete and partial relations",
                    "graph_work": "charged return-graph record/node visits plus physical dense matrix entries; not all primitive operations",
                    "base_candidates": "candidate e0 values tested, including cached relations",
                    "relations": "first-event relation constructions started"}}


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, excluding bool")


def _params(p):
    if not isinstance(p, Params):
        raise TypeError("params must be Params")


def _beta(p, value):
    return p.beta_num * value // (1 << p.beta_shift)


def _alpha(p, value):
    magnitude = p.alpha_num * abs(value) // (1 << p.alpha_shift)
    return magnitude if value >= 0 else -magnitude


def _relation(p, amplitude, current, initial_spike, voltage, budget):
    """Exact maximal voltages for clean and ever-deficit first-event paths."""
    good, bad, old_spike = voltage, None, initial_spike
    rows = []
    budget.used["relations"] += 1
    for delay in range(1, budget.limits.max_relation_steps + 1):
        budget.add("total_updates")
        current = min(p.qi, _beta(p, current) + p.weight * old_spike)
        old_spike = 0
        clean = None if good is None else _alpha(p, good) + current - amplitude
        candidates = []
        if bad is not None:
            candidates.append(_alpha(p, bad) + current - amplitude)
        if good is not None:
            candidates.append(_alpha(p, good) + current - amplitude - 1)
        dirty = max(candidates) if candidates else None
        clean_spike = clean is not None and clean >= p.threshold
        dirty_spike = dirty is not None and dirty >= p.threshold
        if clean_spike or dirty_spike:
            budget.add("edges")
            rows.append({"return_current": current, "delay": delay,
                         "mandatory_full": not dirty_spike,
                         "no_deficit_path_exists": clean_spike,
                         "deficit_path_exists": dirty_spike})
        good = None if clean is None or clean_spike else max(-p.qv, clean)
        # Dirty nonspiking voltages form a complete integer lower interval.
        bad = None if dirty is None else max(-p.qv, min(p.threshold - 1, dirty))
        if current == 0:
            return rows
    raise _Limit("max_relation_steps exhausted before current-zero completeness proof")


def first_return_relation(params, amplitude, initial_current, *, initial_spike=1,
                          initial_voltage=None, limits=TemplateLimits()):
    """Complete first-event relation under membrane input -amplitude-h[t].

    Every h[t] is an arbitrary nonnegative integer. For initial_spike=1 this
    is a first RETURN; for 0 it is the first new spike from a legal silent
    state. Default silent voltage is threshold-1; an explicit legal voltage
    may be supplied. Full is universal over every path to that timed return.
    """
    _params(params)
    _integer("amplitude", amplitude)
    _integer("initial_current", initial_current)
    _integer("initial_spike", initial_spike)
    if initial_current > params.qi or initial_spike not in (0, 1):
        raise ValueError("initial current/spike outside its legal register domain")
    voltage = (0 if initial_spike else params.threshold - 1) if initial_voltage is None else initial_voltage
    if type(voltage) is not int or not -params.qv <= voltage < params.threshold:
        raise ValueError("initial_voltage must be a legal integer post-reset voltage")
    if initial_spike and voltage != 0:
        raise ValueError("initial_spike=1 requires initial_voltage=0")
    budget = _Budget(limits)
    result = {"complete": False, "status": "unknown", "relations": None,
              "params": asdict(params), "amplitude": amplitude,
              "initial": {"voltage": voltage, "current": initial_current, "spike": initial_spike},
              "full_quantifier": "all first-event paths have h=0 at every step",
              "scope": "arbitrary nonnegative integer additional membrane inhibition"}
    try:
        budget.sections(params)
        rows = _relation(params, amplitude, initial_current, initial_spike, voltage, budget)
        result.update(complete=True, status="complete", relations=rows,
                      reason="current became zero after all possible first events were recorded")
    except _Limit as exc:
        result.update(status="budget_unknown", reason=str(exc))
    result["budget"] = budget.report()
    return result


def _profile(relations, budget):
    """Iterative Kosaraju; no Python recursion-depth assumption."""
    count = len(relations)
    adjacent, reverse = [[] for _ in range(count)], [[] for _ in range(count)]
    for source, rows in enumerate(relations):
        for row in rows:
            budget.add("graph_work")
            target = row["return_current"]
            adjacent[source].append(target)
            reverse[target].append(source)
    seen, order = set(), []
    for node in range(count):
        if node in seen:
            continue
        seen.add(node)
        stack = [(node, 0)]
        while stack:
            budget.add("graph_work")
            vertex, next_edge = stack[-1]
            if next_edge == len(adjacent[vertex]):
                order.append(vertex)
                stack.pop()
                continue
            target = adjacent[vertex][next_edge]
            stack[-1] = (vertex, next_edge + 1)
            if target not in seen:
                seen.add(target)
                stack.append((target, 0))
    groups, membership = [], [-1] * count
    for node in reversed(order):
        if membership[node] >= 0:
            continue
        index, group, stack = len(groups), [], [node]
        membership[node] = index
        while stack:
            budget.add("graph_work")
            vertex = stack.pop()
            group.append(vertex)
            for source in reverse[vertex]:
                if membership[source] < 0:
                    membership[source] = index
                    stack.append(source)
        groups.append(sorted(group))
    successors, indegrees = [set() for _ in groups], [0] * len(groups)
    cyclic, bad = set(), []
    for source, rows in enumerate(relations):
        for row in rows:
            budget.add("graph_work")
            target = row["return_current"]
            a, b = membership[source], membership[target]
            if a == b:
                cyclic.add(a)
                if not row["mandatory_full"]:
                    bad.append({"source_current": source, **row})
            elif b not in successors[a]:
                successors[a].add(b)
                indegrees[b] += 1
    ready = deque(i for i, degree in enumerate(indegrees) if not degree)
    topological = []
    while ready:
        budget.add("graph_work")
        component = ready.popleft()
        topological.append(component)
        for child in sorted(successors[component]):
            indegrees[child] -= 1
            if indegrees[child] == 0:
                ready.append(child)
    if len(topological) != len(groups):
        raise AssertionError("return SCC condensation must be acyclic")
    return {"components": [{"id": i, "currents": group, "cyclic": i in cyclic}
                            for i, group in enumerate(groups)],
            "membership": membership, "topological_order": topological,
            "has_cycle": bool(cyclic), "internal_nonfull_edges": bad,
            "edge_count": sum(map(len, relations))}


def _all_state_bound(graph, initial_events, physical_cycle_length, budget):
    relations, profile = graph["relations"], graph["profile"]
    member = profile["membership"]
    costs = [0] * len(profile["components"])
    for component in reversed(profile["topological_order"]):
        record = profile["components"][component]
        exit_cost = 0
        for source in record["currents"]:
            for row in relations[source]:
                budget.add("graph_work")
                other = member[row["return_current"]]
                if other != component:
                    exit_cost = max(exit_cost, row["delay"] + costs[other])
        costs[component] = exit_cost + (physical_cycle_length if record["cyclic"] else 0)
    latest = max(costs, default=0)
    for rows in initial_events:
        for row in rows:
            budget.add("graph_work")
            latest = max(latest, row["delay"] + costs[member[row["return_current"]]])
    return latest


def _check(status="not_evaluated", reason="earlier prerequisite not established", **extra):
    return {"status": status, "reason": reason, **extra}


def _write_protocol(p, budget):
    """Continuous positive write; failure here is only protocol nonreachability."""
    current, trace = 0, [0]  # first positive write emits with current zero
    for _ in range(budget.limits.max_write_steps):
        if current == p.qi:
            return {"complete": True, "protocol_reaches_saturation": True,
                    "duration": len(trace), "activity_current_trace": trace,
                    "amplitude": p.threshold, "target_cell": 0,
                    "controls": "same target cell in all modules; other controls zero"}
        budget.add("total_updates")
        next_current = min(p.qi, _beta(p, current) + p.weight)
        if next_current == current:
            return {"complete": True, "protocol_reaches_saturation": False,
                    "duration": None, "activity_current_trace": trace,
                    "fixed_current": current, "amplitude": p.threshold, "target_cell": 0,
                    "scope": "this continuous positive-write protocol only; other reachability not decided"}
        if next_current < current:
            raise AssertionError("J iteration from zero must be nondecreasing")
        current = next_current
        trace.append(current)
    raise _Limit("max_write_steps exhausted before protocol saturation or fixed-point proof")


def _decay_count(p, current, voltage, budget):
    output = []
    for value, decay, name in ((current, _beta, "current"), (voltage, _alpha, "voltage")):
        steps = 0
        while value:
            if steps >= budget.limits.max_decay_steps:
                raise _Limit(f"max_decay_steps exhausted for {name} recovery tail")
            budget.add("total_updates")
            value = decay(p, value)
            steps += 1
        output.append(steps)
    return output


def _template(p, base_amplitude, budget):
    result = {"certified": False, "complete": False, "status": "template_unknown",
              "params": asdict(p), "c": p.qi - p.threshold + 1,
              "base_amplitude": None, "local_last_spike_upper": None,
              "checks": {name: _check() for name in
                         ("domain", "P1_saturated_current", "P2_local_absorption",
                          "P3_positive_input_barrier", "P4_full_cyclic_returns")},
              "scope": "sufficient parameter template; all legal scalar post-reset states",
              "failure_meaning": "unknown, not a proof that uniform erasure is impossible"}
    checks, c = result["checks"], result["c"]
    if c < 1:
        checks["domain"] = _check("failed", "template requires threshold <= current-register maximum")
        result["reason"] = "parameter domain not covered"
        return result
    checks["domain"] = _check("passed", "c>=1; model validates integer leaks, reset domain and nonnegative inhibition")
    if min(p.qi, _beta(p, p.qi) + p.weight) != p.qi:
        checks["P1_saturated_current"] = _check("failed", "J(qI) differs from qI")
        result["reason"] = "saturated-current tonic premise not established"
        return result
    checks["P1_saturated_current"] = _check("passed", "J(qI)=qI; this does not itself prove write reachability")
    budget.sections(p)
    cache = {}

    def graph(amplitude):
        if amplitude not in cache:
            rows = [_relation(p, amplitude, k, 1, 0, budget) for k in range(p.qi + 1)]
            cache[amplitude] = {"amplitude": amplitude, "relations": rows,
                                "profile": _profile(rows, budget)}
        return cache[amplitude]

    weak = graph(c)
    result["weak_return_graph"] = weak
    bad = weak["profile"]["internal_nonfull_edges"]
    if bad:
        checks["P4_full_cyclic_returns"] = _check("failed", "a cyclic return SCC has a nonfull internal edge",
                                                 witness=bad[0])
        result["reason"] = "continuous predecessor input is not necessary for every cyclic return"
        return result
    checks["P4_full_cyclic_returns"] = _check("passed", "every internal edge of every cyclic return SCC is universally full")
    if base_amplitude is not None and base_amplitude < c:
        checks["P2_local_absorption"] = _check("failed", "declared e0 must be at least c")
        result["reason"] = "base amplitude outside template domain"
        return result
    bases = range(c, p.qi + 1) if base_amplitude is None else (base_amplitude,)
    chosen = None
    tested = []
    result["tested_base_amplitudes"] = tested
    for amplitude in bases:
        budget.add("base_candidates")
        candidate = graph(amplitude)
        tested.append({"amplitude": amplitude, "acyclic": not candidate["profile"]["has_cycle"]})
        if not candidate["profile"]["has_cycle"]:
            chosen = candidate
            break
    if chosen is None:
        checks["P2_local_absorption"] = _check("failed", "declared/searched base graphs still have a return cycle")
        result["reason"] = "robust local absorption premise not established"
        return result
    base = chosen["amplitude"]
    checks["P2_local_absorption"] = _check("passed", "complete e0 first-return graph is acyclic", amplitude=base)
    result["base_return_graph"] = chosen
    predecessor = graph(base - 1)
    deterministic = []
    for rows in predecessor["relations"]:
        budget.add("graph_work")
        clean = [row for row in rows if row["no_deficit_path_exists"]]
        if len(clean) > 1:
            raise AssertionError("clean first return is deterministic")
        deterministic.append(clean[0]["return_current"] if clean else None)
    lower, suffix_min = None, p.qi
    for k in range(p.qi, -1, -1):
        budget.add("graph_work")
        if deterministic[k] is None:
            break
        suffix_min = min(suffix_min, deterministic[k])
        if suffix_min >= k:
            lower = k
    if lower is None:
        checks["P3_positive_input_barrier"] = _check("failed", "no upward-closed first-return barrier at e0-1",
                                                    deterministic_returns=deterministic)
        result["reason"] = "exact amplitude lower-bound premise not established"
        return result
    checks["P3_positive_input_barrier"] = _check("passed", "all k>=lower return to current>=lower at e0-1",
                                                lower_current=lower, amplitude=base - 1,
                                                deterministic_returns=deterministic)
    weak_initial = [_relation(p, c, k, 0, p.threshold - 1, budget) for k in range(p.qi + 1)]
    base_initial = (weak_initial if base == c else
                    [_relation(p, base, k, 0, p.threshold - 1, budget) for k in range(p.qi + 1)])
    local = _all_state_bound(chosen, base_initial, 0, budget)
    write = _write_protocol(p, budget)
    barrier_write = next((i + 1 for i, value in enumerate(write["activity_current_trace"])
                          if value >= lower), None)
    write["protocol_reaches_barrier"] = barrier_write is not None
    write["barrier_entry_duration"] = barrier_write
    drain, recovery = _decay_count(p, p.qi, p.qv, budget)
    result.update(certified=True, complete=True, status="parameter_certified", reason="P1-P4 all established",
                  base_amplitude=base, local_last_spike_upper=local,
                  positive_barrier_lower_current=lower, write_protocol=write,
                  drain_after_last_spike=1 + drain, recovery=recovery,
                  weak_initial_events=weak_initial,
                  current_sections=p.qi + 1, network_joint_states_enumerated=0,
                  time_bound_mode="conservative n steps per cyclic return SCC; not minimum duration",
                  complexity="finite current sections; qI grows exponentially with current bit width")
    return result


def certify_return_template(params, base_amplitude=None, *, limits=TemplateLimits()):
    """Check P1-P4; optionally require a specific e0 instead of bounded search."""
    _params(params)
    if base_amplitude is not None:
        _integer("base_amplitude", base_amplitude)
    budget = _Budget(limits)
    try:
        result = _template(params, base_amplitude, budget)
    except _Limit as exc:
        result = {"certified": False, "complete": False, "status": "budget_unknown",
                  "params": asdict(params), "base_amplitude": None,
                  "reason": str(exc), "failure_meaning": "incomplete, not non-erasure"}
    result["budget"] = budget.report()
    return result


def certify_template_erasure(spec, amplitude=None, *, base_amplitude=None, limits=TemplateLimits()):
    """Rebuild evidence and certify E*=max(e0,c+b) on simple-cycle SCC graphs.

    All returned durations are sufficient common bounds. A below-threshold
    counterexample concerns all legal states; write-protocol reachability is
    reported separately. No externally supplied certificate is trusted.
    """
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    if amplitude is not None:
        _integer("amplitude", amplitude)
    if base_amplitude is not None:
        _integer("base_amplitude", base_amplitude)
    budget = _Budget(limits)
    result = {"certified": False, "complete": False, "status": "unknown",
              "amplitude": amplitude, "minimum_amplitude": None,
              "duration": None, "recovery": None, "total_steps": None,
              "params": asdict(spec.params),
              "scope": "all legal complete two-cell network states; fixed common rectangular pulse",
              "assumptions": ["nonnegative integer synchronous old-spike same-index coupling",
                              "coupling stays enabled; no normal inputs or noise during pulse/recovery",
                              "P1-P4 parameter premises and P5 simple-cycle SCC topology"],
              "minimum_amplitude_scope": "exact only within the fully established parameter/topology template",
              "duration_optimality": "not claimed", "network_joint_states_enumerated": 0}
    try:
        budget.add("graph_work", spec.module_count * spec.module_count)
        topology = analyze_topology(spec)
        result["topology"] = topology
        if not topology["supported"]:
            result.update(status="graph_unknown", reason="P5 unsupported SCC with additional internal feedback",
                          P5=_check("failed", "SCC is neither an acyclic singleton nor a simple directed cycle"))
        else:
            result["P5"] = _check("passed", "arbitrary external SCC inputs allowed; each internal SCC is singleton/cycle")
            template = _template(spec.params, base_amplitude, budget)
            result["parameter_certificate"] = template
            if not template["certified"]:
                result.update(status="template_unknown", reason=template["reason"])
            else:
                c, base = template["c"], template["base_amplitude"]
                bottleneck = max((record["cycle_min_weight"] for record in topology["components"]
                                  if record["kind"] == "simple_cycle"), default=0)
                minimum = max(base, c + bottleneck)
                effective = minimum if amplitude is None else amplitude
                result.update(amplitude=effective, minimum_amplitude=minimum, cycle_bottleneck=bottleneck)
                if effective < minimum:
                    barrier = effective < base
                    write = template["write_protocol"]
                    reachable = write["protocol_reaches_barrier"] if barrier else write["protocol_reaches_saturation"]
                    duration = write["barrier_entry_duration"] if barrier else write["duration"]
                    result.update(complete=True, status="counterexample",
                                  reason="a legal persistent-firing state defeats every finite pulse duration",
                                  counterexample={
                                      "kind": "global_current_barrier" if barrier else "strong_cycle_tonic",
                                      "all_legal_state_witness": True,
                                      "initial_active_section_current": template["positive_barrier_lower_current"] if barrier else spec.params.qi,
                                      "inactive_channel_current_and_spike": 0,
                                      "critical_sccs": [] if barrier else
                                      [r["id"] for r in topology["components"] if r["kind"] == "simple_cycle"
                                       and r["cycle_min_weight"] >= effective - c + 1],
                                      "any_finite_duration_fails": True,
                                      "positive_write_protocol_reaches_failure_domain": reachable,
                                      "positive_write_protocol_duration": duration,
                                      "positive_write_amplitude": spec.params.threshold,
                                      "reachability_scope": "specified continuous positive-write protocol; not all possible writing strategies"})
                else:
                    local, bounds, weak_bounds = template["local_last_spike_upper"], {}, {}
                    for index in topology["topological_order"]:
                        budget.add("graph_work")
                        record = topology["components"][index]
                        parents = record["parent_sccs"]
                        activation = 0 if not parents else 1 + max(bounds[parent]["last_spike_upper"] for parent in parents)
                        n = record["cycle_length"]
                        weak_edge, weak_time = None, None
                        if n:
                            if n not in weak_bounds:
                                weak_bounds[n] = _all_state_bound(template["weak_return_graph"], template["weak_initial_events"], n, budget)
                            weak_time = weak_bounds[n]
                            within = weak_time + (n - 1) * (local + 1)
                            eligible = [edge for edge in record["internal_edges"] if edge["weight"] <= effective - c]
                            if not eligible:
                                raise AssertionError("amplitude formula must supply a weak cycle edge")
                            weak_edge = min(eligible, key=lambda edge: (edge["weight"], edge["source"], edge["target"]))
                        else:
                            within = local
                        bounds[index] = {"scc": index, "activation_after_last_parent_input": activation,
                                         "local_last_spike_upper": within,
                                         "last_spike_upper": activation + within,
                                         "weak_node_last_spike_upper": weak_time, "weak_edge": weak_edge}
                    last = max(row["last_spike_upper"] for row in bounds.values())
                    duration, recovery = last + template["drain_after_last_spike"], template["recovery"]
                    result.update(certified=True, complete=True, status="certified",
                                  last_spike_upper=last, duration=duration, recovery=recovery,
                                  total_steps=duration + recovery,
                                  scc_time_bounds=[bounds[i] for i in topology["topological_order"]],
                                  release_domain={"spike": 0, "current": 0,
                                                  "voltage": [-spec.params.qv, spec.params.threshold - 1]},
                                  reason="P1-P5 established; all-state finite pulse and release proved")
    except _Limit as exc:
        # Never retain a previously computed exact threshold as an accepted
        # certificate if constructing its deployment bound ran out of budget.
        result.update(certified=False, complete=False, status="budget_unknown",
                      minimum_amplitude=None, duration=None, recovery=None, total_steps=None, reason=str(exc))
    result["budget"] = budget.report()
    return result
