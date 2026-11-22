"""Leakage-safe preprocessing and composable model pipelines."""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.config import PROJECT_ROOT
from src.schema import CATEGORICAL_FEATURES, CLASSES, FEATURES, NUMERIC_FEATURES
from src.splits import PROCESSED_DIR, load_development_split

PROBABILITY_CLASSES = CLASSES
CLASS_TO_PROBABILITY_COLUMN = {label: index for index, label in enumerate(PROBABILITY_CLASSES)}


class FeatureSelector(TransformerMixin, BaseEstimator):
    """Select documented inputs and normalize pandas missing-value representations.

    Extra fields (including targets/IDs) are dropped. Numeric parsing errors and
    infinities are rejected; NaN/None/pd.NA are left for training-fitted imputers.
    No measurement units are converted.
    """

    def _prepare(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Pipeline input must be a pandas DataFrame with named predictor columns.")
        if X.columns.duplicated().any():
            raise ValueError("Pipeline input contains duplicate column names.")
        missing = set(FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required predictors: {sorted(missing)}")
        selected = X.loc[:, list(FEATURES)].copy()
        for name in NUMERIC_FEATURES:
            try:
                values = pd.to_numeric(selected[name], errors="raise").to_numpy(dtype=float, na_value=np.nan)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Predictor {name!r} must be numeric or missing.") from error
            if np.isinf(values).any():
                raise ValueError(f"Predictor {name!r} contains infinity; use finite values or missing values.")
            selected[name] = values
        for name in CATEGORICAL_FEATURES:
            values = selected[name].astype("string").astype(object)
            selected[name] = values.where(pd.notna(values), np.nan)
        return selected

    def fit(self, X, y=None):
        self._prepare(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        return self._prepare(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(FEATURES, dtype=object)


def build_preprocessing_pipeline():
    """Return a fresh, unfitted pipeline safe to clone inside cross-validation."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([("features", FeatureSelector()), ("preprocess", columns)])


def build_model_pipeline(classifier):
    """Append a cloned classifier; pass this entire object into any CV search."""
    preprocessing = build_preprocessing_pipeline()
    return Pipeline([*preprocessing.steps, ("classifier", clone(classifier))])


def predict_class_probabilities(fitted_pipeline, rows):
    """Return probabilities aligned to the explicit project class order."""
    actual = list(fitted_pipeline.classes_)
    if len(actual) != len(PROBABILITY_CLASSES) or set(actual) != set(PROBABILITY_CLASSES):
        raise ValueError("Classifier classes differ from the three expected project classes.")
    probabilities = np.asarray(fitted_pipeline.predict_proba(rows), dtype=float)
    if probabilities.shape != (len(rows), len(actual)):
        raise ValueError("Unexpected predicted-probability shape.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Predicted probabilities must be finite and between zero and one.")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Predicted probabilities must sum to one per record.")
    order = [actual.index(label) for label in PROBABILITY_CLASSES]
    return pd.DataFrame(probabilities[:, order], columns=PROBABILITY_CLASSES, index=rows.index)


def preprocessing_state(fitted_pipeline):
    """Capture learned statistics and categories for reproducibility checks."""
    columns = fitted_pipeline.named_steps["preprocess"]
    numeric = columns.named_transformers_["numeric"]
    categorical = columns.named_transformers_["categorical"]
    return {
        "numeric_medians": numeric.named_steps["impute"].statistics_.tolist(),
        "numeric_mean": numeric.named_steps["scale"].mean_.tolist(),
        "numeric_scale": numeric.named_steps["scale"].scale_.tolist(),
        "numeric_variance": numeric.named_steps["scale"].var_.tolist(),
        "categorical_modes": categorical.named_steps["impute"].statistics_.tolist(),
        "categories": [values.tolist() for values in categorical.named_steps["encode"].categories_],
    }


def verify_preprocessing():
    """Fit on development-train and verify held-out transforms without modeling."""
    train = load_development_split("development_train").loc[:, list(FEATURES)]
    validation = load_development_split("validation").loc[:, list(FEATURES)]
    preprocessing = build_preprocessing_pipeline()
    transformed_train = preprocessing.fit_transform(train)
    state_before = preprocessing_state(preprocessing)
    transformed_validation = preprocessing.transform(validation)
    if state_before != preprocessing_state(preprocessing):
        raise AssertionError("Validation transformation altered fitted preprocessing state.")
    sample = train.iloc[[0]].copy()
    transformed_sample = preprocessing.transform(sample)
    missing = sample.copy()
    missing.loc[:, list(NUMERIC_FEATURES)] = np.nan
    missing.loc[:, list(CATEGORICAL_FEATURES)] = pd.NA
    transformed_missing = preprocessing.transform(missing)
    unseen = sample.copy()
    for name in CATEGORICAL_FEATURES:
        unseen[name] = "__unseen_setup_check__"
    transformed_unseen = preprocessing.transform(unseen)
    if not np.all(transformed_unseen[:, len(NUMERIC_FEATURES):] == 0):
        raise AssertionError("Unseen categorical values were not ignored.")
    for result in [transformed_train, transformed_validation, transformed_sample, transformed_missing, transformed_unseen]:
        if not np.isfinite(result).all() or result.shape[1] != transformed_train.shape[1]:
            raise AssertionError("Transformation produced invalid values or an inconsistent width.")
    if state_before != preprocessing_state(preprocessing):
        raise AssertionError("Input smoke checks altered fitted preprocessing state.")
    names = preprocessing.get_feature_names_out().tolist()
    if any(any(excluded in name for excluded in ["health_score", "health_class", "record_id"]) for name in names):
        raise AssertionError("An excluded field entered transformed features.")
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / "development_preprocessor.joblib"
    joblib.dump(preprocessing, artifact_path)
    restored = joblib.load(artifact_path)
    np.testing.assert_allclose(restored.transform(validation), transformed_validation)
    split = json.loads((PROCESSED_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    summary = {
        "status": "preprocessing_verified; no project classifier fitted",
        "fit_partition": "development_train", "fit_rows": len(train),
        "transform_only_partition": "validation", "transform_only_rows": len(validation),
        "training_shape": list(transformed_train.shape),
        "validation_shape": list(transformed_validation.shape),
        "dataset_revision": split["revision"], "source_sha256": split["source_sha256"],
        "membership_sha256": split["membership_sha256"],
        "environment_metadata": "reports/environment.json",
        "feature_names": names, "input_features": list(FEATURES),
        "probability_classes": list(PROBABILITY_CLASSES),
        "class_to_probability_column": CLASS_TO_PROBABILITY_COLUMN,
        "state": state_before,
        "checks": {"validation_did_not_refit": True, "new_row": True,
                   "missing_values": True, "unseen_categories": True,
                   "finite_consistent_width": True, "excluded_fields_absent": True,
                   "serialization_roundtrip": True},
        "artifact": "models/development_preprocessor.joblib",
        "note": "Intermediate development preprocessor only; future CV must fit the complete model pipeline inside each fold.",
    }
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (PROJECT_ROOT / "reports" / "preprocessing_summary.json").write_text(payload, encoding="utf-8")
    (models_dir / "development_preprocessor.metadata.json").write_text(payload, encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = verify_preprocessing()
    print(json.dumps({key: result[key] for key in ["status", "training_shape", "validation_shape", "class_to_probability_column", "checks"]}, indent=2))
