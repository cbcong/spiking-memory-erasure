"""Topology-dependent uniform erasure for one explicitly fixed integer model.

Supported SCCs are acyclic singletons or a single directed cycle, including
one-node self-loops. Between SCCs, arbitrary nonnegative integer edges are
allowed. The certificate covers ALL legal post-update states, not only a
holding orbit. It is a mathematical model certificate, not a novelty claim.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict

from .model import Params
from .network import NetworkSpec


MODEL = Params(5, 4, 3, 2, 3, 2, 6, 4, 14)


def _scc_profile(spec: NetworkSpec) -> dict:
    """Deterministic Tarjan partition; edges run source -> target."""
    n = spec.module_count
    children = [[target for target in range(n) if spec.weights[target][source] > 0]
                for source in range(n)]
    indices = [-1] * n
    low = [-1] * n
    stack, on_stack, components = [], set(), []
    sequence = 0

    def visit(node):
        nonlocal sequence
        indices[node] = low[node] = sequence
        sequence += 1
        stack.append(node)
        on_stack.add(node)
        for child in children[node]:
            if indices[child] < 0:
                visit(child)
                low[node] = min(low[node], low[child])
            elif child in on_stack:
                low[node] = min(low[node], indices[child])
        if low[node] == indices[node]:
            component = []
            while True:
                child = stack.pop()
                on_stack.remove(child)
                component.append(child)
                if child == node:
                    break
            components.append(sorted(component))

    for node in range(n):
        if indices[node] < 0:
            visit(node)
    components.sort(key=lambda nodes: nodes[0])
    membership = {node: c for c, nodes in enumerate(components) for node in nodes}
    parents = [set() for _ in components]
    successors = [set() for _ in components]
    records = []
    for c, nodes in enumerate(components):
        internal_edges = [(source, target, spec.weights[target][source])
                          for target in nodes for source in nodes if spec.weights[target][source] > 0]
        trivial = len(nodes) == 1 and not internal_edges
        incoming_counts = [sum(target == node for _, target, _ in internal_edges) for node in nodes]
        outgoing_counts = [sum(source == node for source, _, _ in internal_edges) for node in nodes]
        cycle = bool(internal_edges) and all(v == 1 for v in incoming_counts + outgoing_counts)
        records.append({
            "id": c, "nodes": nodes, "kind": "acyclic_singleton" if trivial else
            "simple_cycle" if cycle else "unsupported_scc",
            "internal_edges": [{"source": source, "target": target, "weight": weight}
                               for source, target, weight in internal_edges],
            "cycle_length": len(nodes) if cycle else 0,
            "cycle_min_weight": min((w for _, _, w in internal_edges), default=0) if cycle else None,
        })
    for source in range(n):
        for target in children[source]:
            a, b = membership[source], membership[target]
            if a != b:
                successors[a].add(b)
                parents[b].add(a)
    remaining = [len(p) for p in parents]
    ready = deque(c for c, degree in enumerate(remaining) if degree == 0)
    order = []
    while ready:
        c = ready.popleft()
        order.append(c)
        for child in sorted(successors[c]):
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if len(order) != len(components):
        raise AssertionError("SCC condensation must be acyclic")
    for c, record in enumerate(records):
        record["parent_sccs"] = sorted(parents[c])
    return {"components": records, "topological_order": order,
            "node_to_scc": [membership[node] for node in range(n)],
            "supported": all(r["kind"] != "unsupported_scc" for r in records),
            "positive_edges": sum(len(row) for row in children),
            "dense_matrix_entries": n * n}


def analyze_topology(spec: NetworkSpec) -> dict:
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    return _scc_profile(spec)


def certify_row_sum_erasure(spec: NetworkSpec, *, shorter_pulse: bool = False) -> dict:
    """Full-state sufficient baseline for ANY nonnegative graph, same model.

    R>0: E=R+2 gives global no-adjacent spikes, hence P(t)+P(t+1)<=R.
    The local effective-E2 return needs two consecutive full row inputs;
    the low-current recurrent return is therefore impossible. Last spike<=6.
    R=0 uses E=3 and last spike<=4. Alternatively E=R+3 gives last<=4
    without the window argument. Both release with I<=6 and recover in 10.
    These are sufficient protocols, not per-graph minimum amplitudes.
    """
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    if type(shorter_pulse) is not bool:
        raise ValueError("shorter_pulse must be bool")
    if spec.params != MODEL:
        return {"certified": False, "status": "model_unknown"}
    row = spec.max_row_sum
    amplitude = row + 3 if shorter_pulse else max(3, row + 2)
    last = 4 if shorter_pulse or row == 0 else 6
    return {"certified": True, "status": "certified", "amplitude": amplitude,
            "duration": last + 4, "recovery": 10, "total_steps": last + 14,
            "last_spike_upper": last, "row_sum": row,
            "scope": "all legal post-update states; arbitrary nonnegative graph; explicit MODEL",
            "variant": "higher_amplitude_shorter_pulse" if shorter_pulse else "lower_amplitude",
            "assumptions": ["coupling remains enabled", "no normal input or noise during pulse and recovery"],
            "amplitude_optimality": "not claimed for this individual graph"}


def certify_topology_erasure(spec: NetworkSpec, amplitude: int | None = None) -> dict:
    """Exact amplitude threshold on the supported graph class; conservative D.

    Under E>=E_min:
      local SCC activation A is zero for sources, otherwise 1+max(parent T);
      local last-spike bound L is 4 for acyclic singleton, or
      6+2*floor(n/2)+5*(n-1) for a directed n-cycle;
      T=A+L, common D=max(T)+4, common recovery=10.
    Coupling is never switched off. Other inputs/noise must be zero.
    Unsupported structures return unknown, never a positive certificate.
    """
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    if amplitude is not None and (type(amplitude) is not int or amplitude < 0):
        raise ValueError("amplitude must be a nonnegative integer, excluding bool")
    result = {
        "certified": False, "status": "model_unknown", "amplitude": amplitude,
        "minimum_amplitude": None, "duration": None, "recovery": None, "total_steps": None,
        "scope": "fixed-model simple-cycle SCC graph class; all legal post-update network states",
        "assumptions": ["Params equals the explicit model point",
                        "each SCC is an acyclic singleton or one simple directed cycle",
                        "arbitrary nonnegative integer edges between SCCs",
                        "same-index simultaneous old-spike coupling remains enabled",
                        "no normal input or noise during pulse and recovery",
                        "constant common integer amplitude; no direct state assignment"],
        "initial_domain": {"scalar_legal_states": 480, "network_state_count_base": 480,
                           "network_state_count_exponent": 2 * spec.module_count},
        "joint_states_enumerated_by_checker": 0,
        "amplitude_optimality": "within stated graph/model class and uniform finite rectangular protocols",
        "duration_optimality": "not claimed; conservative constructive upper bound",
        "params": asdict(spec.params),
    }
    if spec.params != MODEL:
        result["reason"] = "fixed-model theorem does not cover these parameters"
        return result
    graph = analyze_topology(spec)
    result["graph"] = graph
    if not graph["supported"]:
        result["status"] = "graph_unknown"
        result["reason"] = "a strongly connected component has more than a simple cycle"
        return result
    bottleneck = max((r["cycle_min_weight"] for r in graph["components"]
                      if r["kind"] == "simple_cycle"), default=0)
    minimum = max(3, 2 + bottleneck)
    E = minimum if amplitude is None else amplitude
    baseline = certify_row_sum_erasure(spec)
    result.update(amplitude=E, minimum_amplitude=minimum, cycle_bottleneck=bottleneck,
                  row_sum_sufficient_amplitude=baseline["amplitude"],
                  row_sum_sufficient_duration=baseline["duration"], row_sum_sufficient_recovery=10,
                  row_sum_baseline=baseline,
                  row_sum_shorter_pulse_baseline=certify_row_sum_erasure(spec, shorter_pulse=True))
    if E < minimum:
        result["status"] = "counterexample"
        result["counterexample"] = {
            "kind": "global_return_current_barrier" if E <= 2 else "strong_cycle_tonic",
            "critical_sccs": [] if E <= 2 else
            [r["id"] for r in graph["components"] if r["kind"] == "simple_cycle"
             and r["cycle_min_weight"] >= E - 1],
            "reachable_write": {"target_cell": 0, "amplitude": 14, "duration": 5,
                                "modules": "all", "start": "all registers zero"},
            "post_write_active_current": 15,
            "off_cells_current_and_spike": 0,
            "any_finite_duration_fails": True,
            "reason": ("post-spike current >=9 returns within two steps and remains >=9"
                       if E <= 2 else "critical cycle keeps current15 and emits every step"),
        }
        result["reason"] = "reachable persistent-firing counterexample for every finite pulse duration"
        return result
    bounds = {}
    for c in graph["topological_order"]:
        record = graph["components"][c]
        parents = record["parent_sccs"]
        activation = 0 if not parents else 1 + max(bounds[p]["last_spike_upper"] for p in parents)
        n = record["cycle_length"]
        local = 4 if record["kind"] == "acyclic_singleton" else 6 + 2 * (n // 2) + 5 * (n - 1)
        weak = None
        if n:
            eligible = [edge for edge in record["internal_edges"] if edge["weight"] <= E - 2]
            if not eligible:
                raise AssertionError("amplitude threshold must supply a weak edge in every cycle")
            weak = min(eligible, key=lambda edge: (edge["weight"], edge["target"], edge["source"]))
        bounds[c] = {"scc": c, "activation_after_last_parent_input": activation,
                     "local_last_spike_upper": local, "last_spike_upper": activation + local,
                     "weak_edge": weak}
    last = max(r["last_spike_upper"] for r in bounds.values())
    result.update(certified=True, status="certified", last_spike_upper=last,
                  duration=last + 4, recovery=10, total_steps=last + 14,
                  scc_time_bounds=[bounds[c] for c in graph["topological_order"]],
                  release_domain={"spike": 0, "current": [0, 6], "voltage": [-15, 13]},
                  reason="all supported SCCs have a weak edge or are acyclic; finite release bound constructed")
    return result
