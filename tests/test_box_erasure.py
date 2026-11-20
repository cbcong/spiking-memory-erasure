import random
import unittest

from integer_snn.box_erasure import box_step, start_boxes
from integer_snn.model import State
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL


class BoxErasureTests(unittest.TestCase):
    def test_box_encloses_independent_fraction_trajectories(self):
        rng = random.Random(20261119)
        spec = NetworkSpec(MODEL, ((3, 2), (5, 0)))
        states = [
            tuple(self.legal_module_state(rng) for _ in range(spec.module_count))
            for _ in range(128)
        ]
        for amplitude, duration in ((0, 0), (3, 5), (5, 15), (50, 3)):
            boxes = start_boxes(spec)
            active = states
            for tick in range(24):
                level = amplitude if tick < duration else 0
                active = [network_reference_step(spec, state, erase=level) for state in active]
                boxes = box_step(spec, boxes, level)
                for state in active:
                    spikes = tuple(s for module in state for s in module.spike)
                    voltage = tuple(v for module in state for v in module.v)
                    current = tuple(i for module in state for i in module.current)
                    for v, i, (v_lo, v_hi, i_lo, i_hi) in zip(voltage, current, boxes[spikes]):
                        self.assertLessEqual(v_lo, v)
                        self.assertLessEqual(v, v_hi)
                        self.assertLessEqual(i_lo, i)
                        self.assertLessEqual(i, i_hi)

    @staticmethod
    def legal_module_state(rng):
        spikes = tuple(rng.randrange(2) for _ in range(2))
        voltages = tuple(0 if s else rng.randrange(-MODEL.qv, MODEL.threshold) for s in spikes)
        currents = tuple(rng.randint(0, MODEL.qi) for _ in range(2))
        return State(voltages, currents, spikes)
