"""Read and validate the immutable local dataset; never access the network."""

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_ID = "soumikmahato/plant-health-points-synthetic"
DATASET_REVISION = "d7b9f23bce2f5cf02c3be85c38506cf386d9d0b2"
EXPECTED_ROWS = {"train": 2000, "test": 500}
TEXT_COLUMNS = ("record_id", "plant_type", "habitat", "season", "health_class")
NUMERIC_COLUMNS = (
    "air_humidity_pct", "soil_humidity_pct", "ethylene_ppm", "co2_ppm",
    "o2_ppm", "ldr_on_plant", "leaf_color_index", "temperature_c", "health_score",
)
EXPECTED_COLUMNS = (
    "record_id", "plant_type", "habitat", "season", *NUMERIC_COLUMNS, "health_class",
)
SNAPSHOT_FILES = ("README.md", "train.csv", "test.csv")


def sha256(path: Path) -> str:
    """Hash file bytes without changing the raw file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(raw_dir: Path = RAW_DIR) -> dict:
    """Require an acquired snapshot and its provenance manifest."""
    path = Path(raw_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Local dataset manifest missing: {path}. "
            "From the repository root, run: python -m src.download_data"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("revision") != DATASET_REVISION:
        raise ValueError("Dataset identity/revision differs from the pinned project snapshot.")
    return manifest


def verify_snapshot(raw_dir: Path = RAW_DIR) -> dict:
    """Check file hashes only; do not summarize targets or distributions."""
    raw_dir = Path(raw_dir)
    manifest = read_manifest(raw_dir)
    for filename in SNAPSHOT_FILES:
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Snapshot file missing: {path}. Restore the pinned snapshot; "
                "the loader never downloads files automatically."
            )
        entry = manifest.get("files", {}).get(filename, {})
        if sha256(path) != entry.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {path}; raw snapshot has changed.")
    return manifest


def load_split(split: str = "train", raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load one split offline, checking provenance, schema, types, and row count.

    Missing values are retained for the later audit/imputation pipeline. This
    structural check does not compute class counts or feature distributions.
    """
    if split not in EXPECTED_ROWS:
        raise ValueError(f"Unknown split {split!r}; choose 'train' or 'test'.")
    raw_dir = Path(raw_dir)
    verify_snapshot(raw_dir)
    path = raw_dir / f"{split}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)) or set(header) != set(EXPECTED_COLUMNS):
        raise ValueError(f"Unexpected or duplicate CSV columns in {path}: {header}")
    dtypes = {name: "string" for name in TEXT_COLUMNS}
    dtypes.update({name: "float64" for name in NUMERIC_COLUMNS})
    frame = pd.read_csv(path, dtype=dtypes)
    if len(frame) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"Expected {EXPECTED_ROWS[split]} {split} rows, found {len(frame)}."
        )
    return frame.loc[:, list(EXPECTED_COLUMNS)]


if __name__ == "__main__":
    for split_name in EXPECTED_ROWS:
        data = load_split(split_name)
        print(f"{split_name}: {len(data)} rows, {len(data.columns)} columns; schema/hash checks passed")
