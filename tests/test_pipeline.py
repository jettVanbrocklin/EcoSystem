"""Verify learned preprocessing boundaries and probability-column alignment."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.pipeline import (
    build_model_pipeline, build_preprocessing_pipeline, predict_class_probabilities,
    preprocessing_state,
)
from src.schema import CATEGORICAL_FEATURES, CLASSES, FEATURES, NUMERIC_FEATURES


def inputs(count=6):
    frame = pd.DataFrame({name: np.arange(count, dtype=float) + index
                          for index, name in enumerate(NUMERIC_FEATURES)})
    for name in CATEGORICAL_FEATURES:
        frame[name] = pd.Series(["a", "b"] * ((count + 1) // 2), dtype="string").iloc[:count]
    return frame


class PreprocessingTests(unittest.TestCase):
    def test_training_median_and_scale_are_unchanged_by_extreme_validation(self):
        train = inputs()
        train.loc[0, NUMERIC_FEATURES[0]] = np.nan
        pipeline = build_preprocessing_pipeline().fit(train)
        state = preprocessing_state(pipeline)
        self.assertEqual(state["numeric_medians"][0], 3.0)
        self.assertAlmostEqual(state["numeric_mean"][0], 3.0)
        validation = inputs(1)
        validation.loc[:, list(NUMERIC_FEATURES)] = 1e9
        pipeline.transform(validation)
        self.assertEqual(state, preprocessing_state(pipeline))
        self.assertTrue(np.allclose(pipeline.transform(train)[:, :8].mean(axis=0), 0))

    def test_extra_labels_ids_and_unknown_fields_never_enter_features(self):
        raw = inputs()
        selected = raw.copy(deep=True)
        raw["health_score"] = 1000.0
        raw["health_class"] = "leak"
        raw["record_id"] = "identifier"
        raw["unexpected_label"] = 1
        pipeline = build_preprocessing_pipeline().fit(raw)
        names = pipeline.get_feature_names_out()
        self.assertFalse(any(term in name for name in names for term in ["health_score", "health_class", "record_id", "unexpected_label"]))
        np.testing.assert_allclose(pipeline.transform(raw), pipeline.transform(selected))
        raw["health_score"] = -1000.0
        np.testing.assert_allclose(pipeline.transform(raw), pipeline.transform(selected))

    def test_missing_values_unseen_categories_and_single_row(self):
        train = inputs()
        original = train.copy(deep=True)
        pipeline = build_preprocessing_pipeline().fit(train)
        missing = train.iloc[[0]].copy()
        missing.loc[:, list(NUMERIC_FEATURES)] = np.nan
        missing.loc[:, list(CATEGORICAL_FEATURES)] = pd.NA
        self.assertTrue(np.isfinite(pipeline.transform(missing)).all())
        unseen = train.iloc[[0]].copy()
        for name in CATEGORICAL_FEATURES:
            unseen[name] = "unseen_category"
        result = pipeline.transform(unseen)
        self.assertEqual(result.shape, (1, len(pipeline.get_feature_names_out())))
        self.assertTrue(np.all(result[:, 8:] == 0))
        pd.testing.assert_frame_equal(train, original)

    def test_all_missing_training_columns_keep_a_stable_output_width(self):
        train = inputs()
        train[NUMERIC_FEATURES[0]] = np.nan
        train[CATEGORICAL_FEATURES[0]] = pd.NA
        pipeline = build_preprocessing_pipeline().fit(train)
        result = pipeline.transform(inputs(1))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result.shape[1], len(pipeline.get_feature_names_out()))
        self.assertIn("numeric__" + NUMERIC_FEATURES[0], pipeline.get_feature_names_out())

    def test_bad_numeric_values_and_missing_required_columns_raise_clear_errors(self):
        pipeline = build_preprocessing_pipeline().fit(inputs())
        with self.assertRaisesRegex(ValueError, "Missing required predictors"):
            pipeline.transform(inputs().drop(columns=[FEATURES[0]]))
        bad = inputs()
        bad[FEATURES[0]] = "not numeric"
        with self.assertRaisesRegex(ValueError, "must be numeric or missing"):
            pipeline.transform(bad)
        bad = inputs()
        bad.loc[0, FEATURES[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "contains infinity"):
            pipeline.transform(bad)

    def test_class_probabilities_follow_project_order_not_estimator_order(self):
        train = inputs()
        target = ["healthy", "high_stress", "high_stress", "moderate_stress", "moderate_stress", "moderate_stress"]
        pipeline = build_model_pipeline(DummyClassifier(strategy="prior")).fit(train, target)
        self.assertEqual(pipeline.classes_.tolist(), ["healthy", "high_stress", "moderate_stress"])
        probabilities = predict_class_probabilities(pipeline, train.iloc[[0]])
        self.assertEqual(tuple(probabilities.columns), CLASSES)
        np.testing.assert_allclose(probabilities.iloc[0], [1/6, 3/6, 2/6])

    def test_cross_validation_refits_whole_pipeline_inside_each_fold(self):
        train = inputs(12)
        target = np.asarray(list(CLASSES) * 4)
        model = build_model_pipeline(DummyClassifier(strategy="prior"))
        fresh = clone(model)
        self.assertFalse(hasattr(fresh.named_steps["preprocess"], "transformers_"))
        folds = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(train, target))
        result = cross_validate(model, train, target, cv=folds, return_estimator=True)
        for estimator, (fit_indices, _) in zip(result["estimator"], folds):
            scaler = estimator.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
            self.assertEqual(scaler.n_samples_seen_, 6)
            np.testing.assert_allclose(scaler.mean_, train.iloc[fit_indices][list(NUMERIC_FEATURES)].mean().to_numpy())
        self.assertFalse(hasattr(model.named_steps["preprocess"], "transformers_"))


if __name__ == "__main__":
    unittest.main()
