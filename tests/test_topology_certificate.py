"""Graph orientation, supported-class boundaries and independent dynamics."""
from dataclasses import replace
import itertools
import random
import unittest

from integer_snn.model import State, zero_state
from integer_snn.network import NetworkSpec, network_step, network_reference_step
from integer_snn.topology_certificate import (MODEL, analyze_topology, certify_topology_erasure,
                                              certify_row_sum_erasure)


def brute_scc(weights):
    n = len(weights)
    reachable = [[i == j or weights[j][i] > 0 for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reachable[i][j] |= reachable[i][k] and reachable[k][j]
    return sorted({tuple(j for j in range(n) if reachable[i][j] and reachable[j][i]) for i in range(n)})


class TopologyCertificateTests(unittest.TestCase):
    def test_scc_partition_matches_independent_transitive_closure(self):
        for flat in itertools.product((0, 1), repeat=9):
            w = tuple(tuple(flat[3 * i:3 * i + 3]) for i in range(3))
            profile = analyze_topology(NetworkSpec(MODEL, w))
            self.assertEqual([tuple(c["nodes"]) for c in profile["components"]], brute_scc(w))

    def test_equal_max_row_sum_different_exact_amplitude(self):
        for weights, expected in ((((0, 2), (0, 0)), 3),
                                  (((0, 2), (1, 0)), 3),
                                  (((0, 2), (2, 0)), 4)):
            cert = certify_topology_erasure(NetworkSpec(MODEL, weights))
            self.assertTrue(cert["certified"])
            self.assertEqual(cert["minimum_amplitude"], expected)

    def test_large_weight_dag_with_multiple_inputs_is_supported(self):
        w = ((0, 0, 0, 0), (1000, 0, 0, 0), (7, 900, 0, 0), (500, 30, 400, 0))
        c = certify_topology_erasure(NetworkSpec(MODEL, w))
        self.assertEqual(c["minimum_amplitude"], 3)
        self.assertEqual((c["last_spike_upper"], c["duration"], c["total_steps"]), (19, 23, 33))
        self.assertEqual(c["row_sum_sufficient_amplitude"], 1002)
        self.assertEqual(c["row_sum_sufficient_duration"], 10)
        self.assertEqual(c["row_sum_shorter_pulse_baseline"]["amplitude"], 1003)

    def test_cycle_scc_with_arbitrary_external_inputs_and_clock_offset(self):
        # source0 -> self-loop1 (weight1) -> cycle(2,3); source0 also enters node3.
        w = ((0, 0, 0, 0), (9, 1, 0, 0), (0, 5, 0, 4), (7, 0, 2, 0))
        c = certify_topology_erasure(NetworkSpec(MODEL, w))
        self.assertTrue(c["certified"])
        self.assertEqual(c["minimum_amplitude"], 4)
        b = c["scc_time_bounds"]
        self.assertEqual([(r["activation_after_last_parent_input"], r["local_last_spike_upper"],
                           r["last_spike_upper"]) for r in b], [(0, 4, 4), (5, 6, 11), (12, 13, 25)])
        self.assertEqual(c["duration"], 29)

    def test_multi_input_feedback_counterexample_is_not_certified(self):
        spec = NetworkSpec(MODEL, ((0, 0, 1), (0, 0, 1), (2, 2, 0)))
        c = certify_topology_erasure(spec, 3)
        self.assertEqual(c["status"], "graph_unknown")
        self.assertIsNone(c["minimum_amplitude"])
        a = State((0, -14), (9, 0), (1, 0))
        b = State((10, -15), (12, 0), (0, 0))
        tonic = State((0, -15), (15, 0), (1, 0))
        self.assertEqual(network_reference_step(spec, (a, b, tonic), erase=3), (b, a, tonic))
        self.assertEqual(network_reference_step(spec, (b, a, tonic), erase=3), (a, b, tonic))
        # The stronger row bound remains valid even outside the topology class.
        baseline = certify_row_sum_erasure(spec)
        self.assertEqual((baseline["amplitude"], baseline["duration"]), (6, 10))
        states = (a, b, tonic)
        for tick in range(baseline["total_steps"]):
            states = network_reference_step(spec, states, erase=6 if tick < 10 else 0)
        self.assertEqual(states, (zero_state(),) * 3)

    def test_supported_random_full_legal_states_hit_certified_zero(self):
        rng = random.Random(36113)
        graphs = (((1,),), ((0, 2), (1, 0)),
                  ((0, 0, 0), (8, 0, 0), (3, 7, 0)),
                  ((1, 0, 0, 0), (5, 0, 0, 2), (7, 0, 0, 0), (0, 3, 6, 0)))
        for w in graphs:
            spec = NetworkSpec(MODEL, w)
            cert = certify_topology_erasure(spec)
            self.assertTrue(cert["certified"])
            for _ in range(12):
                states = []
                for _ in w:
                    s = (rng.randrange(2), rng.randrange(2))
                    states.append(State(tuple(0 if spike else rng.randrange(-15, 14) for spike in s),
                                        (rng.randrange(16), rng.randrange(16)), s))
                states = tuple(states)
                for t in range(cert["total_steps"]):
                    e = cert["amplitude"] if t < cert["duration"] else 0
                    reference = network_reference_step(spec, states, erase=e)
                    states = network_step(spec, states, erase=e)
                    self.assertEqual(states, reference)
                    if cert["last_spike_upper"] < t + 1 <= cert["duration"]:
                        self.assertFalse(any(any(s.spike) for s in states))
                self.assertEqual(states, (zero_state(),) * len(w))

    def test_below_bound_has_reachable_persistent_state_not_unknown(self):
        for w, e in ((((0, 2), (2, 0)), 3), (((0,),), 2), (((5,),), 6)):
            spec = NetworkSpec(MODEL, w)
            c = certify_topology_erasure(spec, e)
            self.assertEqual(c["status"], "counterexample")
            states = (zero_state(),) * len(w)
            for _ in range(5):
                states = network_reference_step(spec, states, inputs=((14, 0),) * len(w))
            self.assertTrue(all(s.current[0] == 15 and s.spike[0] == 1 for s in states))
            for _ in range(40):
                states = network_reference_step(spec, states, erase=e)
                self.assertTrue(any(s.current[0] >= 9 for s in states))
            for _ in range(20):
                states = network_reference_step(spec, states)
            self.assertNotEqual(states, (zero_state(),) * len(w))

    def test_model_and_amplitude_guards(self):
        spec = NetworkSpec(replace(MODEL, bits_i=5), ((0,),))
        self.assertEqual(certify_topology_erasure(spec)["status"], "model_unknown")
        for e in (True, -1, 2.5):
            with self.assertRaises(ValueError):
                certify_topology_erasure(NetworkSpec(MODEL, ((0,),)), e)


if __name__ == "__main__":
    unittest.main()
