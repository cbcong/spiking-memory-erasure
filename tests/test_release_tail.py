import itertools
import unittest

from integer_snn.model import State, zero_state
from integer_snn.reference import reference_step
from integer_snn.release_tail import release_tail
from integer_snn.topology_certificate import MODEL


class ReleaseTailTests(unittest.TestCase):
    def test_representative_tails(self):
        self.assertEqual((5, 8), release_tail(3))
        self.assertEqual((4, 9), release_tail(6))
        self.assertEqual((4, 8), release_tail(9))
        self.assertEqual((3, 9), release_tail(100))

    def test_replay_every_silent_start(self):
        zero = zero_state()
        for amplitude in (*range(3, 36), 100):
            pressure, recovery = release_tail(amplitude)
            for v, current in itertools.product(range(-MODEL.qv, MODEL.threshold), range(MODEL.qi + 1)):
                start = State((v, 0), (current, 0), (0, 0))
                state = start
                for _ in range(pressure - 1):
                    state = reference_step(MODEL, state, erase=amplitude)
                    if any(state.spike):
                        break
                if any(state.spike):
                    # This box trajectory violates the separate last-spike premise.
                    continue
                for _ in range(recovery):
                    state = reference_step(MODEL, state)
                    self.assertFalse(any(state.spike), (amplitude, start, state))
                self.assertEqual(zero, state, (amplitude, start))

    def test_rejects_unsupported_amplitudes(self):
        for value in (2, 3.0, True):
            with self.assertRaises(ValueError):
                release_tail(value)
