import itertools
import unittest

from integer_snn.deadline_erasure import deadline_peeling
from integer_snn.network import NetworkSpec
from integer_snn.topology_certificate import MODEL, certify_row_sum_erasure


class DeadlinePeelingTests(unittest.TestCase):
    def test_against_all_amplitudes_including_loops(self):
        deadlines = (0, 17, 18, 19, 20, 23, 35)
        for a, b, c, d in itertools.product(range(3), repeat=4):
            matrix = ((a, b), (c, d))
            spec = NetworkSpec(MODEL, matrix)
            base = certify_row_sum_erasure(spec)
            candidates = [(base["amplitude"], base["duration"], base["recovery"])]
            for amplitude in range(3, max(spec.row_sums) + 4):
                residual = {0, 1}
                rounds = 0
                while residual:
                    kept = {i for i in residual if sum(matrix[i][j] for j in residual) > amplitude - 3}
                    if kept == residual:
                        break
                    residual = kept
                    rounds += 1
                if not residual:
                    candidates.append((amplitude, 5 * rounds + 3, 10))
            expected = {t: min((p for p in candidates if sum(p[1:]) <= t), default=None) for t in deadlines}
            self.assertEqual(expected, deadline_peeling(spec, deadlines), matrix)
