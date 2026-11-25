"""Independent numerical and mathematical checks of the educational classifier."""

import unittest

import numpy as np

from src.softmax import NumpySoftmaxClassifier, check_gradient, log_softmax, loss_and_grad


class SoftmaxMathTests(unittest.TestCase):
    def test_all_weight_and_intercept_gradients_match_finite_differences(self):
        result = check_gradient()
        self.assertEqual(result["dtype"], "float64")
        self.assertEqual(result["parameters_checked"], 15)
        self.assertLess(result["max_absolute_error"], 1e-7)

    def test_extreme_logits_have_finite_log_loss_and_normalized_probabilities(self):
        log_probabilities = log_softmax([[1000, -1000, 0], [-1000, 1000, 0]])
        self.assertTrue(np.isfinite(log_probabilities).all())
        np.testing.assert_allclose(np.exp(log_probabilities).sum(axis=1), 1)
        self.assertEqual(-log_probabilities[0, 1], 2000)

    def test_common_weight_and_intercept_shift_preserves_probabilities(self):
        rng = np.random.default_rng(42)
        X, W, b = rng.normal(size=(6, 4)), rng.normal(size=(4, 3)), rng.normal(size=3)
        shift = rng.normal(size=(4, 1))
        np.testing.assert_allclose(log_softmax(X @ W + b), log_softmax(X @ (W + shift) + b + 17), atol=1e-13)

    def test_penalty_applies_to_weights_and_not_intercepts(self):
        X = np.array([[1.0, 2.0], [-1.0, 0.0], [0.0, -2.0]])
        W = np.arange(6, dtype=float).reshape(2, 3) / 10
        b, y = np.array([2.0, -1.0, 0.5]), np.array([0, 1, 2])
        loss0, W0, b0 = loss_and_grad(X, y, W, b, 0)
        loss1, W1, b1 = loss_and_grad(X, y, W, b, 0.2)
        self.assertAlmostEqual(loss1 - loss0, 0.1 * np.sum(W * W))
        np.testing.assert_allclose(W1 - W0, 0.2 * W)
        np.testing.assert_allclose(b1, b0)

    def test_training_learns_a_simple_problem_with_monotone_objective(self):
        X = np.array([[-3, -3], [-2, -3], [3, -3], [2, -3], [0, 3], [0, 2]], dtype=float)
        y = np.repeat(np.arange(3), 2)
        model = NumpySoftmaxClassifier(regularization=0.02, max_iter=2000, tolerance=1e-6).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
        self.assertLess(model.objective_, np.log(3))
        self.assertTrue(all(b["objective"] <= a["objective"] + 1e-14 for a, b in zip(model.history_, model.history_[1:])))

    def test_invalid_inputs_and_prediction_before_fit_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "before prediction"):
            NumpySoftmaxClassifier().predict([[1, 2]])
        with self.assertRaisesRegex(ValueError, "integer class indices"):
            NumpySoftmaxClassifier().fit([[1, 2]], [0.5])
        with self.assertRaisesRegex(ValueError, "Invalid regularization"):
            NumpySoftmaxClassifier(regularization=-1).fit([[1, 2]], [0])


if __name__ == "__main__":
    unittest.main()
