import itertools
import unittest

from integer_snn.network import NetworkSpec
from integer_snn.release_tail import release_tail
from integer_snn.timed_erasure import _local_last_spike, timed_erasure
from integer_snn.topology_certificate import MODEL


class TimedErasureTests(unittest.TestCase):
    def test_local_bounds_and_tail_monotonicity(self):
        self.assertEqual([4, 2, 1, 1, 0], [_local_last_spike(m) for m in range(3, 8)])
        totals = [sum(release_tail(e)) for e in range(3, 37)]
        self.assertEqual(sorted(totals, reverse=True), totals)
        self.assertEqual(release_tail(35), release_tail(100))

    def test_matches_amplitude_scan_and_direct_time_updates(self):
        deadlines = (11, 12, 13, 15, 17, 18, 20, 24)
        local_delays = (5, 3, 2, 2, 1)
        for a, b, c, d in itertools.product(range(3), repeat=4):
            matrix = ((a, b), (c, d))
            spec = NetworkSpec(MODEL, matrix)
            candidates = []
            for amplitude in range(3, 11 + spec.max_row_sum):
                stop = [float("inf"), float("inf")]
                pressure, recovery = release_tail(amplitude)
                for tick in range(max(deadlines) - pressure - recovery + 1):
                    for target, row in enumerate(matrix):
                        margin = amplitude - sum(weight for src, weight in enumerate(row) if stop[src] > tick)
                        if margin >= 3:
                            stop[target] = min(stop[target], tick + local_delays[min(margin, 7) - 3])
                    if max(stop) <= tick + 1:
                        candidates.append((amplitude, max(stop) - 1 + pressure, recovery))
                        break
            for deadline in deadlines:
                feasible = [plan for plan in candidates if sum(plan[1:]) <= deadline]
                self.assertEqual(timed_erasure(spec, deadline), min(feasible, default=None), (matrix, deadline))

    def test_half_of_a_directed_pair_already_settles(self):
        spec = NetworkSpec(MODEL, ((0, 0), (3, 0)))
        self.assertEqual(timed_erasure(spec, 18), (4, 10, 8))

    def test_zero_weight_graph_can_use_the_shortest_release_tail(self):
        spec = NetworkSpec(MODEL, ((0,),))
        self.assertEqual(timed_erasure(spec, 12), (9, 4, 8))
