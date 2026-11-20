"""Deadline-constrained erasure using margin-dependent stopping times."""

from functools import lru_cache
from heapq import heappop, heappush

from .model import trunc_div
from .release_tail import release_tail
from .topology_certificate import MODEL


@lru_cache(maxsize=5)
def _local_last_spike(margin: int) -> int:
    """Last possible firing tick under net pressure at least margin (3..7).

    At each transition, arbitrary extra inhibition can lower the raw voltage
    to any smaller integer. The resulting finite reachable-state sets are
    nested, since the initial set is the entire legal domain. The first
    set containing no firing state therefore implies perpetual silence.
    """
    states = {(v, i, 0) for v in range(-MODEL.qv, MODEL.threshold)
              for i in range(MODEL.qi + 1)}
    states.update((0, i, 1) for i in range(MODEL.qi + 1))
    tick = 0
    while any(spike for _, _, spike in states):
        following = set()
        for voltage, current, spike in states:
            new_current = min(MODEL.qi, MODEL.beta_num * current // (1 << MODEL.beta_shift)
                              + MODEL.weight * spike)
            upper = trunc_div(MODEL.alpha_num * voltage, 1 << MODEL.alpha_shift)
            upper += new_current - margin
            silent_upper = max(-MODEL.qv, min(MODEL.threshold - 1, upper))
            following.update((v, new_current, 0) for v in range(-MODEL.qv, silent_upper + 1))
            if upper >= MODEL.threshold:
                following.add((0, new_current, 1))
        if following == states:
            raise AssertionError("the fixed model must settle under this margin")
        states = following
        tick += 1
    return tick - 1


def timed_erasure(spec, deadline: int) -> tuple[int, int, int] | None:
    """Select the lowest amplitude in the margin-dependent certificate.

    A removal at time t means a node has no spikes at t or later, so the
    transition from t to t+1 cannot relay one of its old spikes. The queue
    retains the earliest bound for each node, including bounds established
    before each reduction of its possible incoming excitation.
    """
    if spec.params != MODEL:
        raise ValueError("deadline bounds require the fixed erasure model")
    if type(deadline) is not int or deadline < 0:
        raise ValueError("deadline must be a nonnegative integer")
    rows = spec.row_sums
    outgoing = [[] for _ in rows]
    for target, row in enumerate(spec.weights):
        for source, weight in enumerate(row):
            if weight:
                outgoing[source].append((target, weight))

    def last_spike(amplitude, limit):
        residual = list(rows)
        scheduled = [limit + 2] * len(rows)
        queue = []

        def schedule(node, restart):
            margin = amplitude - residual[node]
            if margin >= 3:
                stop = restart + _local_last_spike(min(margin, 7)) + 1
                if stop < scheduled[node]:
                    scheduled[node] = stop
                    heappush(queue, (stop, node))

        for node in range(len(rows)):
            schedule(node, 0)
        left = len(rows)
        while queue:
            stop, source = heappop(queue)
            if stop != scheduled[source]:
                continue
            left -= 1
            if not left:
                return stop - 1
            for target, weight in outgoing[source]:
                if scheduled[target] > stop:
                    residual[target] -= weight
                    schedule(target, stop)
        return None

    # Higher E can only advance a removal: once each predecessor's earlier
    # bound is available, the residual row is no greater and the local
    # response no slower. The release total is also nonincreasing at this
    # fixed model (computed for 3..35, after which the first held step
    # from the silent box saturates every voltage to -qv).
    low = 3
    # At 9 the tail reaches its model-wide minimum of 12 steps.
    high = max(9, max(rows) + 7)
    best = None
    while low <= high:
        amplitude = (low + high) // 2
        extra, recovery = release_tail(amplitude)
        limit = deadline - extra - recovery
        last = last_spike(amplitude, limit)
        if last is None:
            low = amplitude + 1
        else:
            best = (amplitude, last + extra, recovery)
            high = amplitude - 1
    return best
