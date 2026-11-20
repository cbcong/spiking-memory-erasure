"""Physical reachability/reuse and an address-free query preparation API."""

from dataclasses import fields
from itertools import product
import unittest
from unittest.mock import patch

from integer_snn.model import State, zero_state
from integer_snn.network import network_step
from tasks.delayed_query import (QueryReady, fresh_network, write_pattern, prepare_query,
                                 query_after_delay, erase_and_recover, IdealBitRegister,
                                 logical_storage_ledger)
from experiments.run_delayed_query import make_spec


class DelayedQueryTests(unittest.TestCase):
    def test_actual_two_tick_write_and_six_tick_warmup_reach_declared_states(self):
        for modules, topology in product((2, 4), ("uncoupled", "ring")):
            spec = make_spec(modules, topology)
            for pattern in product((0, 1), repeat=modules):
                audit = {}
                states = write_pattern(spec, fresh_network(spec), pattern, reference=True, audit=audit)
                self.assertEqual(audit, {"reference_steps": 8, "network_steps": 8})
                for module, bit in enumerate(pattern):
                    other = 1 - bit
                    nu = sum(spec.weights[module][source] for source in range(modules)
                             if pattern[source] != bit)
                    self.assertEqual((states[module].v[bit], states[module].current[bit], states[module].spike[bit]),
                                     (0, 9, 1))
                    self.assertEqual((states[module].v[other], states[module].current[other], states[module].spike[other]),
                                     (-5 + 2 * nu, 0, 0))

    def test_write_uses_inputs_and_never_calls_fresh_initialization(self):
        spec = make_spec(2, "ring")
        initial = fresh_network(spec)
        with patch("tasks.delayed_query.fresh_network", side_effect=AssertionError("hidden clear")), \
             patch("tasks.delayed_query.network_step", wraps=network_step) as physical:
            states = write_pattern(spec, initial, (0, 1))
        self.assertEqual(physical.call_count, 8)
        self.assertEqual([c.kwargs["inputs"] for c in physical.call_args_list[:2]],
                         [((14, 0), (0, 14))] * 2)
        self.assertTrue(all(c.kwargs["inputs"] is None for c in physical.call_args_list[2:]))
        self.assertNotEqual(states, initial)

    def test_queries_work_in_both_phases_and_after_long_delay(self):
        for topology, delay in product(("uncoupled", "ring"), (0, 1, 16, 64, 256)):
            spec = make_spec(4, topology)
            pattern = (0, 1, 1, 0)
            states = write_pattern(spec, fresh_network(spec), pattern)
            ready = prepare_query(spec, states, delay, start_tick=8, reference=True)
            for index in range(4):
                read = query_after_delay(ready, index, reference=True)
                self.assertEqual(read.prediction, pattern[index])
                self.assertEqual(read.all_predictions, pattern)
                self.assertEqual(read.end_tick, 10 + delay)

    def test_address_cannot_enter_preparation_and_does_not_change_latent_transition(self):
        spec = make_spec(2, "ring")
        states = write_pattern(spec, fresh_network(spec), (0, 1))
        with self.assertRaises(TypeError):
            prepare_query(spec, states, 3, index=0)
        with self.assertRaises(TypeError):
            query_after_delay(states, 0)
        ready = prepare_query(spec, states, 3)
        self.assertEqual({f.name for f in fields(QueryReady)}, {"spec", "states", "ready_tick", "delay"})
        read0, read1 = query_after_delay(ready, 0), query_after_delay(ready, 1)
        self.assertEqual(read0.next_states, read1.next_states)
        self.assertEqual(read0.all_counts, read1.all_counts)
        self.assertEqual((read0.prediction, read1.prediction), (0, 1))

    def test_e3_reuse_and_e2_old_content_intrusion_are_physical(self):
        for topology in ("uncoupled", "ring"):
            spec = make_spec(2, topology)
            old = write_pattern(spec, fresh_network(spec), (0, 0))
            old_read = query_after_delay(prepare_query(spec, old, 0), 0)
            for amplitude in (2, 3):
                with patch("tasks.delayed_query.fresh_network", side_effect=AssertionError("hidden clear")):
                    erased = erase_and_recover(spec, old_read.next_states, amplitude, reference=True)
                    rewritten = write_pattern(spec, erased.states, (1, 1), reference=True)
                    read = query_after_delay(prepare_query(spec, rewritten, 0), 0, reference=True)
                self.assertEqual(erased.zero, amplitude == 3)
                self.assertEqual(read.prediction, 1 if amplitude == 3 else 0)
                if amplitude == 2:
                    self.assertNotEqual(erased.states, (zero_state(), zero_state()))

    def test_failed_or_ambiguous_spike_counts_are_not_silently_mapped_to_a_bit(self):
        spec = make_spec(2, "uncoupled")
        silent = query_after_delay(prepare_query(spec, fresh_network(spec), 0), 0)
        self.assertIsNone(silent.prediction)
        self.assertEqual(silent.selected_counts, (0, 0))
        both = (State((0, 0), (9, 9), (1, 1)),) * 2
        ambiguous = query_after_delay(prepare_query(spec, both, 0), 0)
        self.assertEqual(ambiguous.selected_counts, (1, 1))
        self.assertIsNone(ambiguous.prediction)

    def test_noncanonical_extra_spike_is_diagnosed_without_fabricating_content_error(self):
        spec = make_spec(2, "ring")
        old = write_pattern(spec, fresh_network(spec), (0, 0))
        read = query_after_delay(prepare_query(spec, old, 0), 0)
        residual = erase_and_recover(spec, read.next_states, amplitude=2)
        rewritten = write_pattern(spec, residual.states, (0, 0))
        read = query_after_delay(prepare_query(spec, rewritten, 0), 0)
        self.assertEqual(read.selected_counts, (2, 0))
        self.assertEqual(read.prediction, 0)
        self.assertFalse(read.phase_valid)

    def test_ideal_register_declares_different_direct_clear_and_write_permissions(self):
        register = IdealBitRegister((0, 0)).write((1, 0))
        self.assertEqual(register.read(0), 1)
        cleared = register.clear()
        self.assertEqual(cleared.bits, (0, 0))
        self.assertEqual(register.bits, (1, 0))
        spec = make_spec(16, "ring")
        ledger = logical_storage_ledger(spec, 256)
        self.assertEqual(ledger["snn_dynamic_state_bits"], 320)
        self.assertEqual(ledger["query_spike_counters"]["total_bits"], 4)
        self.assertEqual(ledger["query_window_counter_bits"], 2)
        self.assertEqual(ledger["query_address_bits"], 4)
        self.assertEqual(ledger["schedule_counter_bits"], 9)
        self.assertEqual(ledger["ideal_register_content_bits"], 16)

    def test_parameter_and_address_validation(self):
        spec = make_spec(2, "ring")
        states = fresh_network(spec)
        for pattern in ((0,), (0, True), (0, 2), [0, 1]):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                write_pattern(spec, states, pattern)
        for delay in (-1, True, 1.5):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                prepare_query(spec, states, delay)
        ready = prepare_query(spec, states, 0)
        for index in (-1, 2, True, 0.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                query_after_delay(ready, index)


if __name__ == "__main__":
    unittest.main()
