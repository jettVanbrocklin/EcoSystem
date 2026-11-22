"""Snapshot integrity and offline-loading checks using synthetic file fixtures."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data import (
    DATASET_ID, DATASET_REVISION, EXPECTED_COLUMNS, EXPECTED_ROWS,
    SNAPSHOT_FILES, TEXT_COLUMNS, load_split,
)
from src.download_data import download_snapshot


class LocalSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.raw = Path(self.temporary.name)
        (self.raw / "README.md").write_text("Test fixture, not source data.\n", encoding="utf-8")
        row = {name: ("fixture" if name in TEXT_COLUMNS else 1.0) for name in EXPECTED_COLUMNS}
        row["health_class"] = "healthy"
        for split, count in EXPECTED_ROWS.items():
            with (self.raw / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS)
                writer.writeheader()
                writer.writerows(row for _ in range(count))
        self.refresh_manifest()

    def refresh_manifest(self):
        manifest = {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "files": {
                name: {"sha256": hashlib.sha256((self.raw / name).read_bytes()).hexdigest()}
                for name in SNAPSHOT_FILES
            },
        }
        (self.raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_load_and_repeated_acquisition_are_offline_and_preserve_bytes(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.raw.iterdir()}
        with patch("socket.create_connection", side_effect=AssertionError("Network forbidden")), patch(
            "src.download_data.urlopen", side_effect=AssertionError("Download forbidden")
        ):
            self.assertEqual(load_split("train", self.raw).shape, (2000, 14))
            self.assertEqual(load_split("test", self.raw).shape, (500, 14))
            download_snapshot(self.raw)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.raw.iterdir()}
        self.assertEqual(before, after)

    def test_missing_setup_explains_acquisition_command(self):
        with self.assertRaisesRegex(FileNotFoundError, "python -m src.download_data"):
            load_split("train", self.raw / "missing")

    def test_corruption_is_rejected(self):
        with (self.raw / "train.csv").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_split("train", self.raw)

    def test_schema_is_checked_even_when_hashes_match(self):
        path = self.raw / "train.csv"
        path.write_text(path.read_text().replace("health_class", "unexpected_target", 1), encoding="utf-8")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "CSV columns"):
            load_split("train", self.raw)

    def test_row_count_is_checked_even_when_hashes_match(self):
        path = self.raw / "train.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Expected 2000 train rows, found 1999"):
            load_split("train", self.raw)

    def test_partial_download_is_not_overwritten(self):
        (self.raw / "manifest.json").unlink()
        with patch("src.download_data.urlopen", side_effect=AssertionError("Download forbidden")):
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                download_snapshot(self.raw)


if __name__ == "__main__":
    unittest.main()
