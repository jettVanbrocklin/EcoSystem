"""Explicit one-time acquisition of the pinned public CSV snapshot."""

import argparse
import csv
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from src.config import PROJECT_ROOT
from src.data import (
    DATASET_ID, DATASET_REVISION, EXPECTED_COLUMNS, EXPECTED_ROWS,
    RAW_DIR, SNAPSHOT_FILES, load_split, sha256, verify_snapshot,
)


def download_snapshot(raw_dir: Path = RAW_DIR) -> dict:
    """Acquire once, preserving bytes; refuse to overwrite an existing snapshot."""
    raw_dir = Path(raw_dir)
    if (raw_dir / "manifest.json").exists():
        manifest = verify_snapshot(raw_dir)
        for split in EXPECTED_ROWS:
            load_split(split, raw_dir)
        print("Existing pinned snapshot verified; no network request or file rewrite.")
        return manifest
    raw_dir.mkdir(parents=True, exist_ok=True)
    occupied = [p.name for p in raw_dir.iterdir() if p.name != ".gitkeep"]
    if occupied:
        raise FileExistsError(
            f"Refusing to overwrite a partial/unrecognized snapshot in {raw_dir}: {occupied}. "
            "Restore its manifest or choose a new empty directory with --output-dir."
        )
    base = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}"
    with tempfile.TemporaryDirectory(prefix=".acquire-", dir=raw_dir) as temporary:
        staging = Path(temporary)
        files = {}
        for filename in SNAPSHOT_FILES:
            url = f"{base}/{filename}"
            request = Request(url, headers={"User-Agent": "EcoSystem-local-dataset-setup"})
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            path = staging / filename
            path.write_bytes(payload)
            entry = {"source_url": url, "sha256": sha256(path), "bytes": len(payload)}
            if filename.endswith(".csv"):
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    header = next(reader, [])
                    count = sum(1 for _ in reader)
                split = path.stem
                if len(header) != len(set(header)) or set(header) != set(EXPECTED_COLUMNS):
                    raise ValueError(f"Source schema changed for {filename}: {header}")
                if count != EXPECTED_ROWS[split]:
                    raise ValueError(f"Source row count changed for {filename}: {count}")
                entry.update({"rows": count, "columns": header})
            files[filename] = entry
        manifest = {
            "dataset_id": DATASET_ID,
            "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
            "revision": DATASET_REVISION,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "license": None,
            "license_status": "Not declared in dataset card or repository files at this revision; no reuse license inferred.",
            "use_restrictions": [
                "Educational synthetic-data modeling only.",
                "Not appropriate for scientific conclusions, production agronomy, or plant-care decisions.",
                "Do not use either label as a predictor of the other label.",
            ],
            "files": files,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for split in EXPECTED_ROWS:
            load_split(split, staging)
        for filename in (*SNAPSHOT_FILES, "manifest.json"):
            shutil.move(str(staging / filename), raw_dir / filename)
    print(f"Saved and validated pinned snapshot in {raw_dir}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--metadata-output", type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_snapshot.json",
        help="Save a separate, shareable provenance record outside the raw files.",
    )
    arguments = parser.parse_args()
    metadata = download_snapshot(arguments.output_dir)
    arguments.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
