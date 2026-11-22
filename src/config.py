"""Shared reproducibility settings for project experiments."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
SPLIT_SEED = RANDOM_SEED
MODEL_SEED = RANDOM_SEED
BOOTSTRAP_SEED = RANDOM_SEED
