import unittest
import numpy as np
from encoders.linear import loss_gradient, evaluate


class LinearEncoderTests(unittest.TestCase):
    def test_gradient_matches_independent_finite_differences(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(7, 4))
        y = rng.integers(0, 3, size=7)
        w, b = rng.normal(size=(4, 3)), rng.normal(size=3)
        loss, gw, gb = loss_gradient(x, y, w, b, .03)
        eps = 1e-6
        for array, analytic in ((w, gw), (b, gb)):
            for index in np.ndindex(array.shape):
                old = array[index]
                array[index] = old + eps
                plus = loss_gradient(x, y, w, b, .03)[0]
                array[index] = old - eps
                minus = loss_gradient(x, y, w, b, .03)[0]
                array[index] = old
                self.assertAlmostEqual((plus - minus) / (2 * eps), analytic[index], places=6)
        updated = loss_gradient(x, y, w - .01 * gw, b - .01 * gb, .03)[0]
        self.assertLess(updated, loss)

    def test_evaluation_stays_finite_for_large_logits(self):
        x = np.array([[1.0, 0], [0, 1.0]])
        w = np.array([[10000.0, -10000.0], [-10000.0, 10000.0]])
        result, prediction = evaluate(x, np.array([0, 1]), w, np.zeros(2))
        self.assertEqual(result["accuracy"], 1)
        self.assertTrue(np.isfinite(result["nll"]))
        self.assertEqual(prediction.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
