"""Deadline-constrained uniform erasure from bounded rounds of weighted peeling."""
from .topology_certificate import MODEL, certify_row_sum_erasure


def deadline_peeling(spec, deadlines):
    """Choose the least sufficient amplitude among the row and peeling certificates.

    For E>=3 a layer contains nodes with residual incoming weight <= E-3.
    The fixed-model local lemma bounds its last spike by step 4 from restart.
    Each successive layer restarts one step after the previous last-spike bound.
    Thus m layers suffice with pressure 5m+3 and recovery 10. At larger E the
    remaining vertex set is a subset after every round (nonnegative weights).
    Binary search therefore finds the minimum feasible E for THIS certificate.
    """
    if spec.params != MODEL:
        raise ValueError("deadline bounds require the fixed erasure model")
    deadlines = tuple(deadlines)
    if any(type(t) is not int or t < 0 for t in deadlines):
        raise ValueError("deadlines must be nonnegative integer network steps")
    rows = spec.row_sums
    n = len(rows)
    edges = [[] for _ in rows]
    for i, row in enumerate(spec.weights):
        for j, weight in enumerate(row):
            if weight:
                edges[j].append((i, weight))

    def layers(amplitude, max_rounds):
        threshold = amplitude - 3
        remaining = list(rows)
        current = [i for i, total in enumerate(rows) if total <= threshold]
        # Each vertex enters the frontier once, when its residual crosses the threshold.
        removed = 0
        for step in range(1, max_rounds + 1):
            if not current:
                break
            removed += len(current)
            if removed == n:
                return step
            following = []
            for source in current:
                for target, weight in edges[source]:
                    old = remaining[target]
                    remaining[target] -= weight
                    if remaining[target] <= threshold < old:
                        following.append(target)
            current = following
        return None

    baseline = certify_row_sum_erasure(spec)
    result = {}
    for deadline in deadlines:
        best = None
        if baseline["total_steps"] <= deadline:
            best = (baseline["amplitude"], baseline["duration"], baseline["recovery"])
        rounds = min(n, (deadline - 13) // 5)
        if rounds >= 1:
            lo = 3
            hi = min(max(rows) + 3, best[0] if best else max(rows) + 3)
            while lo <= hi:
                mid = (lo + hi) // 2
                count = layers(mid, rounds)
                if count is None:
                    lo = mid + 1
                else:
                    candidate = (mid, 5 * count + 3, 10)
                    best = candidate if best is None else min(best, candidate)
                    hi = mid - 1
        result[deadline] = best
    return result
