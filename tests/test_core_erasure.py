"""Regressions for the full-state, fixed-model amplitude bracket."""
from dataclasses import replace
import unittest

from integer_snn.core_erasure import core_erasure_bounds
from integer_snn.model import State, zero_state
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL


class CoreErasureTests(unittest.TestCase):
    def test_weighted_self_loop_caps_at_row_bound(self):
        result = core_erasure_bounds(NetworkSpec(MODEL, ((4,),)))
        self.assertEqual((result["k"], result["lower"], result["upper"]), (4, 6, 6))
        self.assertEqual(result["duration_at_upper"], 10)

    def test_dense_incoming_dag_has_source_first_layers(self):
        graph = ((0, 0, 0), (100, 0, 0), (25, 75, 0))
        result = core_erasure_bounds(NetworkSpec(MODEL, graph))
        self.assertEqual(result["layers"], [[0], [1], [2]])
        self.assertEqual((result["lower"], result["upper"], result["duration_at_upper"]),
                         (3, 3, 18))

    def test_two_phase_feedback_orbit_and_upper_protocol(self):
        spec = NetworkSpec(MODEL, ((0, 0, 1), (0, 0, 1), (2, 2, 0)))
        result = core_erasure_bounds(spec)
        self.assertEqual((result["k"], result["lower"], result["upper"]), (1, 3, 4))
        self.assertEqual(result["layers"], [[0, 1], [2]])
        a = State((0, -14), (9, 0), (1, 0))
        b = State((10, -15), (12, 0), (0, 0))
        c = State((0, -15), (15, 0), (1, 0))
        state = (a, b, c)
        self.assertEqual(network_reference_step(spec, state, erase=3), (b, a, c))
        self.assertEqual(network_reference_step(spec, (b, a, c), erase=3), state)
        for _ in range(result["duration_at_upper"]):
            state = network_reference_step(spec, state, erase=4)
        for _ in range(result["recovery_at_upper"]):
            state = network_reference_step(spec, state)
        self.assertEqual(state, (zero_state(),) * 3)

    def test_alternate_model_rejected(self):
        spec = NetworkSpec(replace(MODEL, threshold=13), ((0,),))
        with self.assertRaises(ValueError):
            core_erasure_bounds(spec)
