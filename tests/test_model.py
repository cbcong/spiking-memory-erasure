"""Tests for scientific implementation semantics, not training performance."""

from dataclasses import FrozenInstanceError, replace
import random
import unittest

from integer_snn import (
    Params,
    State,
    clip,
    iter_legal_states,
    legal_state_count,
    reference_step,
    step,
    trunc_div,
    validate_state,
    zero_state,
)


def example_params(**changes: int) -> Params:
    baseline = Params(
        bits_v=4,
        bits_i=3,
        alpha_num=1,
        alpha_shift=1,
        beta_num=1,
        beta_shift=1,
        weight=4,
        inhibition=6,
        threshold=3,
    )
    return replace(baseline, **changes)


class IntegerArithmeticTests(unittest.TestCase):
    def test_truncation_signs_and_large_integer(self) -> None:
        for numerator, denominator, expected in (
            (-7, 2, -3), (7, -2, -3), (-7, -2, 3), (-1, 2, 0), (0, -2, 0)
        ):
            with self.subTest(numerator=numerator, denominator=denominator):
                self.assertEqual(trunc_div(numerator, denominator), expected)
        huge = 10**100 + 1
        self.assertEqual(trunc_div(-huge, 2), -(huge // 2))
        with self.assertRaises(ZeroDivisionError):
            trunc_div(1, 0)

    def test_saturation_and_type_rejection(self) -> None:
        self.assertEqual([clip(x, -7, 7) for x in (-100, -7, 0, 7, 100)], [-7, -7, 0, 7, 7])
        with self.assertRaises(ValueError):
            clip(0, 1, -1)
        for call in (lambda: clip(1.0, 0, 2), lambda: trunc_div(True, 2)):
            with self.assertRaises(TypeError):
                call()


class ContractTests(unittest.TestCase):
    def test_parameter_ranges_and_logical_bits(self) -> None:
        params = example_params()
        self.assertEqual((params.qv, params.qi, params.state_bits), (7, 7, 16))
        invalid = (
            {"bits_v": 1}, {"bits_i": 0}, {"alpha_num": 0},
            {"alpha_num": 2}, {"alpha_shift": 0}, {"beta_num": -1},
            {"beta_num": 2}, {"beta_shift": -1}, {"weight": -1},
            {"inhibition": -1}, {"threshold": 0}, {"threshold": 8},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                example_params(**changes)
        for changes in ({"weight": True}, {"alpha_num": 0.5}):
            with self.subTest(changes=changes), self.assertRaises(TypeError):
                example_params(**changes)
        with self.assertRaises(FrozenInstanceError):
            params.weight = 0

    def test_post_update_invariants_without_assuming_reachability(self) -> None:
        params = example_params()
        validate_state(params, State((0, 0), (0, 0), (1, 1)))
        for state in (
            State((-8, 0), (0, 0), (0, 0)),
            State((0, 0), (-1, 0), (0, 0)),
            State((0, 0), (8, 0), (0, 0)),
            State((1, 0), (0, 0), (1, 0)),
            State((3, 0), (0, 0), (0, 0)),
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                validate_state(params, state)
        with self.assertRaises(ValueError):
            State((0, 0), (0, 0), (2, 0))
        with self.assertRaises(ValueError):
            State((0,), (0, 0), (0, 0))
        with self.assertRaises(TypeError):
            State([0, 0], (0, 0), (0, 0))
        with self.assertRaises(TypeError):
            State((False, 0), (0, 0), (0, 0))

    def test_input_validation_matches_reference(self) -> None:
        params = example_params()
        for function in (step, reference_step):
            for kwargs, error in (
                ({"erase": -1}, ValueError),
                ({"erase": 0.5}, TypeError),
                ({"inputs": (1,)}, ValueError),
                ({"inputs": [1, 2]}, TypeError),
                ({"inputs": (True, 0)}, TypeError),
                ({"noise": (0, 0.5)}, TypeError),
            ):
                with self.subTest(function=function.__name__, kwargs=kwargs), self.assertRaises(error):
                    function(params, zero_state(), **kwargs)


class TransitionTests(unittest.TestCase):
    def test_zero_is_absorbing_without_input(self) -> None:
        self.assertEqual(step(example_params(), zero_state()), zero_state())

    def test_negative_decay_truncates_towards_zero(self) -> None:
        result = step(example_params(), State((-1, -3), (0, 0), (0, 0)))
        self.assertEqual(result, State((0, -1), (0, 0), (0, 0)))

    def test_current_is_updated_before_voltage(self) -> None:
        params = example_params(threshold=6)
        result = step(params, State((0, 0), (4, 0), (0, 0)))
        self.assertEqual(result, State((2, 0), (2, 0), (0, 0)))

    def test_threshold_equality_and_voltage_saturation(self) -> None:
        params = example_params()
        self.assertEqual(step(params, zero_state(), (3, 0)), State((0, 0), (0, 0), (1, 0)))
        self.assertEqual(step(params, zero_state(), (-100, 100)), State((-7, 0), (0, 0), (0, 1)))

    def test_current_saturates_instead_of_wrapping(self) -> None:
        params = example_params(weight=20)
        result = step(params, State((0, 0), (7, 0), (1, 0)))
        self.assertEqual(result.current, (7, 0))
        self.assertEqual(result.spike, (1, 0))

    def test_both_neurons_use_previous_spikes(self) -> None:
        params = example_params()
        result = step(params, State((0, 0), (0, 0), (1, 0)))
        self.assertEqual(result, State((0, -6), (4, 0), (1, 0)))
        new_spike_only = step(params, zero_state(), (3, 0))
        self.assertEqual(new_spike_only.v[1], 0)
        self.assertEqual(new_spike_only.current, (0, 0))
        swapped = step(params, State((0, 0), (0, 0), (0, 1)))
        self.assertEqual(swapped, State((-6, 0), (0, 4), (0, 1)))

    def test_erase_is_external_input_not_state_clearing(self) -> None:
        params = example_params()
        result = step(params, State((0, 0), (3, 0), (1, 0)), erase=5)
        self.assertEqual(result, State((0, -7), (5, 0), (0, 0)))
        self.assertNotEqual(result, zero_state())
        self.assertEqual(step(params, zero_state(), noise=(-1, 2)), State((-1, 2), (0, 0), (0, 0)))

    def test_fraction_reference_all_small_legal_states_and_controls(self) -> None:
        params = example_params(bits_v=2, bits_i=1, threshold=1, weight=2, inhibition=1)
        controls = (
            ((0, 0), 0, (0, 0)),
            ((3, -3), 0, (0, 0)),
            ((0, 0), 2, (0, 0)),
            ((0, 0), 1, (-1, 1)),
        )
        for state in iter_legal_states(params):
            for inputs, erase, noise in controls:
                with self.subTest(state=state, inputs=inputs, erase=erase, noise=noise):
                    result = step(params, state, inputs, erase, noise)
                    self.assertEqual(result, reference_step(params, state, inputs, erase, noise))
                    validate_state(params, result)

    def test_fraction_reference_random_multistep_trajectories(self) -> None:
        rng = random.Random(20261112)
        for _ in range(20):
            bits_v = rng.randrange(2, 7)
            alpha_shift = rng.randrange(1, 5)
            beta_shift = rng.randrange(1, 5)
            params = Params(
                bits_v=bits_v, bits_i=rng.randrange(1, 7),
                alpha_num=rng.randrange(1, 1 << alpha_shift), alpha_shift=alpha_shift,
                beta_num=rng.randrange(1, 1 << beta_shift), beta_shift=beta_shift,
                weight=rng.randrange(20), inhibition=rng.randrange(20),
                threshold=rng.randrange(1, 1 << (bits_v - 1)),
            )
            actual = expected = zero_state()
            for _ in range(50):
                inputs = (rng.randrange(-20, 21), rng.randrange(-20, 21))
                noise = (rng.randrange(-2, 3), rng.randrange(-2, 3))
                erase = rng.randrange(10)
                actual = step(params, actual, inputs, erase, noise)
                expected = reference_step(params, expected, inputs, erase, noise)
                self.assertEqual(actual, expected)
                validate_state(params, actual)


class EnumerationTests(unittest.TestCase):
    def test_count_uniqueness_and_invariance(self) -> None:
        params = example_params(bits_v=2, bits_i=1, threshold=1)
        states = list(iter_legal_states(params))
        self.assertEqual(legal_state_count(params), 36)
        self.assertEqual(len(states), 36)
        self.assertEqual(len(set(states)), 36)
        self.assertIn(zero_state(), states)
        for state in states:
            validate_state(params, state)
            self.assertIn(step(params, state), states)

    def test_enumeration_guard(self) -> None:
        params = example_params()
        with self.assertRaises(ValueError):
            next(iter_legal_states(params, max_states=1))
        with self.assertRaises(ValueError):
            next(iter_legal_states(params, max_states=0))
        with self.assertRaises(TypeError):
            next(iter_legal_states(params, max_states=True))


if __name__ == "__main__":
    unittest.main()
