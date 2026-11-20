import unittest
from integer_snn.model import Params
from integer_snn.network import NetworkSpec
from tasks.delayed_query import fresh_network, write_pattern, prepare_query, erase_and_recover
from tasks.shd_query import encode_classes, query_class_after_delay


class SHDQueryTests(unittest.TestCase):
    def setUp(self):
        self.spec = NetworkSpec(Params(5, 4, 3, 2, 3, 2, 6, 4, 14),
                                tuple(tuple(int(j == (i - 1) % 20) for j in range(20)) for i in range(20)))

    def test_every_slot_reads_only_its_stored_class_after_delay(self):
        classes = (0, 7, 12, 19)
        states = write_pattern(self.spec, fresh_network(self.spec), encode_classes(classes))
        ready = prepare_query(self.spec, states, 17)
        returned = [query_class_after_delay(ready, slot, reference=True) for slot in range(4)]
        self.assertEqual([r.prediction for r in returned], list(classes))
        self.assertTrue(all(r.next_states == returned[0].next_states for r in returned))
        self.assertTrue(all(len(r.counts) == 5 for r in returned))

    def test_zero_state_is_invalid_not_class_zero(self):
        ready = prepare_query(self.spec, fresh_network(self.spec), 0)
        self.assertIsNone(query_class_after_delay(ready, 0).prediction)

    def test_noncanonical_spike_counts_can_still_read_old_content(self):
        states = write_pattern(self.spec, fresh_network(self.spec), encode_classes((0, 0, 0, 0)))
        residual = erase_and_recover(self.spec, states, amplitude=2).states
        states = write_pattern(self.spec, residual, encode_classes((0, 0, 0, 0)))
        read = query_class_after_delay(prepare_query(self.spec, states, 1), 0)
        self.assertEqual(read.prediction, 0)
        self.assertFalse(read.phase_valid)

    def test_invalid_class_and_query_are_rejected(self):
        for values in ((0, 1, 2), (0, 1, 2, 20), (False, 1, 2, 3)):
            with self.assertRaises(ValueError):
                encode_classes(values)
        ready = prepare_query(self.spec, fresh_network(self.spec), 0)
        for slot in (True, -1, 4):
            with self.assertRaises(ValueError):
                query_class_after_delay(ready, slot)


if __name__ == "__main__":
    unittest.main()
