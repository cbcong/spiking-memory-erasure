"""One-integer bracket for the fixed model on arbitrary nonnegative graphs."""
from .network import NetworkSpec
from .topology_certificate import MODEL, certify_row_sum_erasure


def core_erasure_bounds(spec: NetworkSpec) -> dict:
    if spec.params != MODEL:
        raise ValueError("core bounds require the fixed topology model")
    remaining = list(range(spec.module_count))
    incoming = list(spec.row_sums)
    k = -1
    core = []
    while remaining:
        node = min(remaining, key=incoming.__getitem__)
        if incoming[node] > k:
            k = incoming[node]
            core = remaining.copy()
        remaining.remove(node)
        for target in remaining:
            incoming[target] -= spec.weights[target][node]

    # At a fixed k whole layers can be removed together, charging one lag per layer.
    remaining = list(range(spec.module_count))
    incoming = list(spec.row_sums)
    layers = []
    while remaining:
        layer = [node for node in remaining if incoming[node] <= k]
        if not layer:
            raise AssertionError("peeling stalled at the certified degeneracy")
        layers.append(layer)
        remaining = [node for node in remaining if incoming[node] > k]
        for target in remaining:
            incoming[target] -= sum(spec.weights[target][source] for source in layer)
    baseline = certify_row_sum_erasure(spec)
    amplitude, duration = min((k + 3, 5*len(layers) + 3),
                              (baseline["amplitude"], baseline["duration"]))
    return dict(k=k, core=core, layers=layers, lower=max(3, k+2), upper=amplitude,
                duration_at_upper=duration, recovery_at_upper=10)
