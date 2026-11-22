"""Audit structural independence and persist leakage-safe development splits."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, SPLIT_SEED
from src.data import RAW_DIR, load_split, sha256, verify_snapshot
from src.schema import (
    CATEGORICAL_FEATURES, CLASSES, EXCLUDED_COLUMNS, FEATURES,
    NUMERIC_FEATURES, TARGET, features_target_ids,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def audit_integrity(train, test):
    """Compare test IDs/predictors only; never inspect test labels or scores."""
    features_target_ids(train)
    if set(train[TARGET].unique()) != set(CLASSES):
        raise ValueError("Provided training data must contain all three expected classes.")
    train_predictors = train.loc[:, list(FEATURES)]
    test_predictors = test.loc[:, list(FEATURES)]
    checks = {
        "train_missing_ids": int(train.record_id.isna().sum()),
        "test_missing_ids": int(test.record_id.isna().sum()),
        "train_duplicate_ids": int(train.record_id.duplicated().sum()),
        "test_duplicate_ids": int(test.record_id.duplicated().sum()),
        "train_duplicate_predictor_rows": int(train_predictors.duplicated().sum()),
        "test_duplicate_predictor_rows": int(test_predictors.duplicated().sum()),
        "train_conflicting_predictor_groups": int((
            train.groupby(list(FEATURES), dropna=False)[TARGET].nunique(dropna=False) > 1
        ).sum()),
        "cross_split_id_overlap": len(set(train.record_id) & set(test.record_id)),
        "cross_split_predictor_overlap": len(train_predictors.merge(
            test_predictors.drop_duplicates(), on=list(FEATURES), how="inner"
        )),
    }
    if any(checks.values()):
        raise ValueError(f"Integrity audit failed; resolve independence before splitting: {checks}")
    return checks


def validate_membership(membership, train, test_rows, test=None):
    """Require complete, nonoverlapping membership and matching source records."""
    required = {"source_split", "row_index", "record_id", "partition"}
    if set(membership.columns) != required:
        raise ValueError("Unexpected membership schema.")
    if membership.isna().any().any() or membership.record_id.duplicated().any():
        raise ValueError("Missing membership fields or overlapping record IDs.")
    if not membership.source_split.isin(["train", "test"]).all():
        raise ValueError("Unknown membership source split.")
    for source, count in [("train", len(train)), ("test", test_rows)]:
        rows = membership.loc[membership.source_split == source]
        if sorted(rows.row_index.tolist()) != list(range(count)):
            raise ValueError(f"Incomplete or repeated {source} row membership.")
        permitted = ["development_train", "validation"] if source == "train" else ["test"]
        if not rows.partition.isin(permitted).all():
            raise ValueError("A source record crossed the development/test boundary.")
        frame = train if source == "train" else test
        if frame is not None:
            expected = frame.iloc[rows.row_index.to_numpy()].record_id.tolist()
            if rows.record_id.tolist() != expected:
                raise ValueError("Membership record IDs do not match source row positions.")
    counts = membership.partition.value_counts().to_dict()
    if counts.get("development_train") != int(0.8 * len(train)) or counts.get("validation") != len(train) - int(0.8 * len(train)):
        raise ValueError("Membership does not implement the declared 80/20 split.")


def build_artifacts(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    """Create deterministic memberships and structural evidence without EDA."""
    manifest = verify_snapshot(raw_dir)
    train, test = load_split("train", raw_dir), load_split("test", raw_dir)
    checks = audit_integrity(train, test)
    fit_indices, validation_indices = train_test_split(
        train.index.to_numpy(), test_size=0.2, random_state=SPLIT_SEED,
        stratify=train[TARGET],
    )
    parts = []
    for partition, source, frame, indices in [
        ("development_train", "train", train, fit_indices),
        ("validation", "train", train, validation_indices),
        ("test", "test", test, test.index.to_numpy()),
    ]:
        parts.append(pd.DataFrame({
            "source_split": source, "row_index": indices,
            "record_id": frame.loc[indices, "record_id"].to_numpy(),
            "partition": partition,
        }))
    membership = pd.concat(parts, ignore_index=True).sort_values(
        ["source_split", "row_index"], ignore_index=True
    )
    validate_membership(membership, train, len(test), test)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    membership_path = processed_dir / "split_membership.csv"
    membership.to_csv(membership_path, index=False, lineterminator="\n")
    development = train.loc[fit_indices]
    schema = []
    for column in train.columns:
        entry = {
            "column": column, "loaded_dtype": str(development[column].dtype),
            "role": "numeric_predictor" if column in NUMERIC_FEATURES else (
                "categorical_predictor" if column in CATEGORICAL_FEATURES else "excluded"
            ),
            "development_train_missing": int(development[column].isna().sum()),
        }
        if column in CATEGORICAL_FEATURES:
            values = sorted(development[column].dropna().unique().tolist())
            entry.update({"observed_categories": values, "category_count": len(values)})
        if column in NUMERIC_FEATURES:
            entry["observed_range"] = [float(development[column].min()), float(development[column].max())]
        schema.append(entry)
    metadata = {
        "revision": manifest["revision"],
        "source_sha256": {name: manifest["files"][name]["sha256"] for name in ["train.csv", "test.csv"]},
        "seed": SPLIT_SEED, "validation_fraction": 0.2,
        "method": "stratified random split of provided train only; no plant-identity/time fields supplied",
        "membership_sha256": sha256(membership_path),
        "counts": {str(k): int(v) for k, v in membership.partition.value_counts().items()},
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "excluded_columns": list(EXCLUDED_COLUMNS), "target": TARGET, "classes": list(CLASSES),
        "integrity": checks, "schema_scope": "development_train only",
        "schema": schema,
        "test_access": "schema, row counts, IDs and exact predictor-overlap checks only; no label/score summaries or feature distributions",
    }
    (processed_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_development_split(partition="development_train", raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    """Load a saved development partition; never load the test table here."""
    if partition not in ("development_train", "validation"):
        raise ValueError("Choose 'development_train' or 'validation'; test access is separate.")
    processed_dir = Path(processed_dir)
    paths = [processed_dir / "split_metadata.json", processed_dir / "split_membership.csv"]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Development split artifacts missing; run: python -m src.splits")
    metadata = json.loads(paths[0].read_text(encoding="utf-8"))
    manifest = verify_snapshot(raw_dir)
    expected_hashes = {name: manifest["files"][name]["sha256"] for name in ["train.csv", "test.csv"]}
    if metadata["revision"] != manifest["revision"] or metadata["source_sha256"] != expected_hashes:
        raise ValueError("Saved split belongs to a different raw dataset snapshot.")
    if sha256(paths[1]) != metadata["membership_sha256"]:
        raise ValueError("Saved split membership hash mismatch.")
    membership = pd.read_csv(paths[1], dtype={"record_id": "string"})
    train = load_split("train", raw_dir)
    validate_membership(membership, train, manifest["files"]["test.csv"]["rows"])
    indices = membership.loc[membership.partition == partition, "row_index"].to_numpy()
    return train.iloc[indices].copy()


if __name__ == "__main__":
    result = build_artifacts()
    (PROJECT_ROOT / "reports" / "data_boundaries.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": result["counts"], "integrity": result["integrity"]}, indent=2))
