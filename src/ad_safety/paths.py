from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TORCH_CACHE_DIR = PROJECT_ROOT / ".torch-cache"


def ensure_project_dirs() -> None:
    for path in (CONFIG_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR, TORCH_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
