"""Synchronous network semantics and event-window theorem domain checks."""

from dataclasses import FrozenInstanceError
import random
import unittest
from unittest.mock import patch

from integer_snn.model import Params, State, zero_state
from integer_snn.network import (NetworkSpec, network_step, network_reference_step,
                                 coupling_inputs, validate_network_state,
                                 current_spike_contract, transition_contract)
from experiments.run_coupled_erasure import (witness_params, local_branch_audit,
                                             random_initial, weight_two_counterexample,
                                             topologies, sampled_network_case)


class NetworkKernelTests(unittest.TestCase):
    def test_weight_orientation_and_same_index_inputs(self):
        spec = NetworkSpec(witness_params(), ((0, 2, 0), (0, 0, 3), (4, 0, 0)))
        states = (State((0, 0), (0, 0), (1, 0)),
                  State((0, 0), (0, 0), (0, 1)), State((0, 0), (0, 0), (1, 1)))
        self.assertEqual(coupling_inputs(spec, states), ((0, 2), (3, 3), (4, 0)))
        self.assertEqual(spec.row_sums, (2, 3, 4))
        self.assertEqual(spec.max_row_sum, 4)
        self.assertEqual(spec.persisted_state_bits, 60)
        self.assertEqual(network_step(spec, states), network_reference_step(spec, states))

    def test_new_spike_cannot_affect_another_module_in_same_tick(self):
        params = Params(3, 3, 1, 1, 1, 1, 1, 0, 2)
        spec = NetworkSpec(params, ((0, 0), (1, 0)))
        states = (State((0, 0), (4, 0), (0, 0)), zero_state())
        successor = network_step(spec, states)
        self.assertEqual(successor[0].spike, (1, 0))
        self.assertEqual(successor[1], zero_state())
        next_state = network_step(spec, successor)
        self.assertEqual(next_state[1].v, (1, 0))

    def test_erase_is_not_a_hidden_register_clear(self):
        spec = NetworkSpec(witness_params(), ((0,),))
        states = (State((0, 0), (9, 0), (1, 0)),)
        successor = network_step(spec, states, erase=100)
        self.assertEqual(successor[0].current, (12, 0))
        self.assertEqual(successor[0].v, (-15, -15))
        self.assertNotEqual(successor, (zero_state(),))

    def test_explicit_inputs_noise_and_autapse_agree_with_reference(self):
        spec = NetworkSpec(witness_params(), ((2,),))
        states = (State((0, -3), (9, 2), (1, 0)),)
        actual = network_step(spec, states, erase=4, inputs=((1, 7),), noise=((-1, 1),))
        self.assertEqual(actual, network_reference_step(spec, states, erase=4,
                                                       inputs=((1, 7),), noise=((-1, 1),)))

    def test_theorem_domain_is_checked_separately_from_general_legal_kernel(self):
        spec = NetworkSpec(witness_params(), ((0,),))
        outside = (State((0, 0), (15, 15), (1, 0)),)
        validate_network_state(spec, outside)
        self.assertFalse(current_spike_contract(spec, outside))
        self.assertIsInstance(network_step(spec, outside)[0], State)
        bad = (State((1, 0), (0, 0), (1, 0)),)
        with self.assertRaises(ValueError):
            network_step(spec, bad)

    def test_window_contract_does_not_assume_no_consecutive_firing(self):
        spec = NetworkSpec(witness_params(), ((0, 2), (2, 0)))
        old = (State((0, 0), (9, 0), (1, 0)),) * 2
        observed = transition_contract(spec, old, old)
        self.assertFalse(observed["no_consecutive_spikes"])
        self.assertFalse(observed["weighted_window_bound"])
        silent = (zero_state(),) * 2
        observed = transition_contract(spec, old, silent)
        self.assertTrue(observed["no_consecutive_spikes"])
        self.assertTrue(observed["weighted_window_bound"])

    def test_random_network_reference_and_window_checks(self):
        rng = random.Random(9127)
        for modules in (1, 2, 4):
            for R in (0, 1, 2, 4):
                matrix = topologies(modules, R, 123)[-1][1]
                spec = NetworkSpec(witness_params(16), matrix)
                for sample in range(12):
                    states = random_initial(spec, rng, sample)
                    for tick in range(14):
                        successor = network_step(spec, states, erase=max(3, R + 2) if tick < 4 else 0)
                        self.assertEqual(successor, network_reference_step(spec, states,
                                         erase=max(3, R + 2) if tick < 4 else 0))
                        observed = transition_contract(spec, states, successor)
                        self.assertTrue(observed["no_consecutive_spikes"])
                        self.assertTrue(observed["weighted_window_bound"])
                        states = successor
                    self.assertEqual(states, (zero_state(),) * modules)

    def test_complete_local_input_branch_bounds_include_voltage_thirteen(self):
        for R in (0, 1, 2, 4):
            for inhibition in (0, 16):
                audit = local_branch_audit(R, inhibition)
                self.assertEqual(audit["initial_scalar_states"], 387)
                self.assertTrue(audit["zero_at_14"])
                self.assertTrue(audit["no_consecutive_spikes"])
                self.assertEqual(audit["history"][3]["max_current"], 6)
                if R in (1, 2):
                    self.assertEqual(audit["history"][3]["max_voltage"], 13)

    def test_weight_two_e3_failure_is_an_actual_nonzero_cycle(self):
        result = weight_two_counterexample()
        self.assertGreater(result["nonzero_cycle_period"], 0)
        spec = NetworkSpec(witness_params(), ((0, 2), (2, 0)))
        cycle = [tuple(State(tuple(s["v"]), tuple(s["current"]), tuple(s["spike"])) for s in pair)
                 for pair in result["cycle"]]
        for index, current in enumerate(cycle):
            self.assertNotEqual(current, (zero_state(), zero_state()))
            self.assertEqual(network_reference_step(spec, current), cycle[(index + 1) % len(cycle)])

    def test_zero_before_pulse_is_not_counted_as_completed_erasure(self):
        spec = NetworkSpec(witness_params(), ((0,),))
        with patch("experiments.run_coupled_erasure.random_initial", return_value=(zero_state(),)):
            result = sampled_network_case(spec, samples=1, seed=1)
        self.assertEqual(result["latest_sampled_zero_tick"], 9)

    def test_shape_type_size_and_frozen_contracts(self):
        params = witness_params()
        for bad in ((), ((0, 1),), ((False,),), ((-1,),), [[0]], ((0.5,),), ((0,),) * 65):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                NetworkSpec(params, bad)
        spec = NetworkSpec(params, ((0,),))
        with self.assertRaises(FrozenInstanceError):
            spec.weights = ((1,),)
        for kwargs in ({"erase": True}, {"erase": -1}, {"inputs": ((True, 0),)},
                       {"inputs": ()}, {"noise": ((1.5, 0),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                network_step(spec, (zero_state(),), **kwargs)
        with self.assertRaises(ValueError):
            network_step(spec, [zero_state()])
        maximum = NetworkSpec(params, tuple((0,) * 64 for _ in range(64)))
        self.assertEqual(network_step(maximum, (zero_state(),) * 64), (zero_state(),) * 64)


if __name__ == "__main__":
    unittest.main()
