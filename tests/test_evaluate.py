"""Guard metric class order, zero-probability handling, and missing classes."""

import unittest

import numpy as np
import pandas as pd

from src.evaluate import classification_metrics
from src.schema import CLASSES


class ClassificationMetricTests(unittest.TestCase):
    def test_log_loss_uses_project_order_instead_of_alphabetical_order(self):
        probabilities = pd.DataFrame([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], columns=CLASSES)
        result = classification_metrics(CLASSES, CLASSES, probabilities)
        self.assertAlmostEqual(result["log_loss"], -np.log(0.9))
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_classes_not_predicted_are_included_in_macro_f1_and_recall(self):
        result = classification_metrics(CLASSES, ["moderate_stress"] * 3, [[0, 1, 0]] * 3)
        self.assertAlmostEqual(result["accuracy"], 1/3)
        self.assertAlmostEqual(result["macro_f1"], 1/6)
        self.assertEqual(result["per_class"]["healthy"]["recall"], 0)
        self.assertEqual(result["zero_probability_true_labels"], 2)
        self.assertTrue(np.isfinite(result["log_loss"]))
        self.assertAlmostEqual(result["log_loss"], -2*np.log(np.finfo(float).eps)/3)

    def test_missing_true_classes_still_have_explicit_zero_support(self):
        result = classification_metrics(["healthy"], ["healthy"], [[1, 0, 0]])
        self.assertAlmostEqual(result["macro_f1"], 1/3)
        self.assertEqual(result["per_class"]["high_stress"]["support"], 0)
        self.assertEqual(len(result["confusion_matrix"]), 3)

    def test_incorrect_column_order_and_invalid_probabilities_are_rejected(self):
        wrong = pd.DataFrame([[0.8, 0.1, 0.1]], columns=sorted(CLASSES))
        with self.assertRaisesRegex(ValueError, "project class order"):
            classification_metrics(["healthy"], ["healthy"], wrong)
        with self.assertRaisesRegex(ValueError, "Invalid class probabilities"):
            classification_metrics(["healthy"], ["healthy"], [[0.8, 0.8, 0.1]])


if __name__ == "__main__":
    unittest.main()
