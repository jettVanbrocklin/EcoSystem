"""Check leakage boundaries and split independence on small synthetic tables."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.schema import CLASSES, FEATURES, NUMERIC_FEATURES, features_target_ids
from src.splits import audit_integrity, build_artifacts, load_development_split, validate_membership


def frame(count, prefix, offset=0):
    result = pd.DataFrame({
        name: [offset + i + 1.0 for i in range(count)] if name in NUMERIC_FEATURES else ["example"] * count
        for name in FEATURES
    })
    result["record_id"] = [f"{prefix}_{i}" for i in range(count)]
    result["health_class"] = [CLASSES[i % 3] for i in range(count)]
    result["health_score"] = 50.0
    return result


class DevelopmentBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = frame(30, "train"), frame(10, "test", offset=100)

    def test_predictor_allowlist_excludes_labels_id_and_unexpected_fields(self):
        self.train["future_leaking_column"] = self.train.health_class
        predictors, target, ids = features_target_ids(self.train)
        self.assertEqual(tuple(predictors.columns), FEATURES)
        self.assertEqual(target.name, "health_class")
        self.assertEqual(ids.name, "record_id")
        predictors.iloc[0, 0] = -1
        self.assertNotEqual(self.train.iloc[0][FEATURES[0]], -1)

    def test_test_labels_and_score_are_not_needed_for_integrity_checks(self):
        test = self.test.drop(columns=["health_class", "health_score"])
        self.assertFalse(any(audit_integrity(self.train, test).values()))

    def test_cross_source_predictor_overlap_is_rejected(self):
        self.test.loc[0, list(FEATURES)] = self.train.loc[0, list(FEATURES)]
        with self.assertRaisesRegex(ValueError, "Integrity audit failed"):
            audit_integrity(self.train, self.test)

    def test_conflicting_training_duplicates_are_rejected(self):
        self.train.loc[1, list(FEATURES)] = self.train.loc[0, list(FEATURES)]
        with self.assertRaisesRegex(ValueError, "Integrity audit failed"):
            audit_integrity(self.train, self.test)

    def test_duplicate_ids_are_rejected(self):
        self.test.loc[0, "record_id"] = self.train.loc[0, "record_id"]
        with self.assertRaisesRegex(ValueError, "Integrity audit failed"):
            audit_integrity(self.train, self.test)

    def test_persistence_reproducibility_and_development_only_reload(self):
        manifest = {"revision": "fixture", "files": {
            "train.csv": {"sha256": "fixture_train"},
            "test.csv": {"sha256": "fixture_test", "rows": 10},
        }}
        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.splits.verify_snapshot", return_value=manifest
        ), patch("src.splits.load_split", side_effect=lambda split, raw: self.train if split == "train" else self.test) as loader:
            directory = Path(temporary)
            build_artifacts(processed_dir=directory)
            original = (directory / "split_membership.csv").read_bytes()
            build_artifacts(processed_dir=directory)
            self.assertEqual(original, (directory / "split_membership.csv").read_bytes())
            loader.reset_mock()
            fit = load_development_split(processed_dir=directory)
            validation = load_development_split("validation", processed_dir=directory)
            self.assertEqual((len(fit), len(validation)), (24, 6))
            self.assertFalse(set(fit.record_id) & set(validation.record_id))
            self.assertEqual(set(fit.health_class), set(CLASSES))
            self.assertEqual(set(validation.health_class), set(CLASSES))
            self.assertTrue(all(call.args[0] == "train" for call in loader.call_args_list))
            membership = pd.read_csv(directory / "split_membership.csv")
            membership.loc[membership.source_split == "test", "partition"] = "validation"
            with self.assertRaisesRegex(ValueError, "development/test boundary"):
                validate_membership(membership, self.train, len(self.test), self.test)
            (directory / "split_membership.csv").write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "membership hash mismatch"):
                load_development_split(processed_dir=directory)


if __name__ == "__main__":
    unittest.main()
