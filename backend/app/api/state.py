"""Development-only in-memory state for uploaded datasets and training jobs."""

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS: dict[str, dict[str, Any]] = {}
JOBS: dict[str, dict[str, Any]] = {}
