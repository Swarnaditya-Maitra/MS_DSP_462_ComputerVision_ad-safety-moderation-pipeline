from __future__ import annotations

import json
from pathlib import Path

from .paths import MODEL_DIR, PROJECT_ROOT


MODEL_MANIFEST = MODEL_DIR / "pretrained_model_manifest.json"
VISUAL_BACKBONE_REPO_ID = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
DETECTOR_REPO_ID = "IDEA-Research/grounding-dino-tiny"


def file_ready(path: Path) -> bool:
    """Return true only for a readable, non-empty local artifact."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def model_snapshot_ready(
    repo_id: str,
    *,
    manifest_path: Path = MODEL_MANIFEST,
    project_root: Path = PROJECT_ROOT,
) -> bool:
    """Verify the manifest entry and minimum files required for offline loading."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    row = next(
        (item for item in manifest.get("models", []) if item.get("repo_id") == repo_id),
        None,
    )
    if row is None:
        return False
    snapshot = Path(str(row.get("local_snapshot", "")))
    if not snapshot.is_absolute():
        snapshot = project_root / snapshot
    required = {"model.safetensors", "config.json"}
    if repo_id == DETECTOR_REPO_ID:
        required.update(
            {
                "preprocessor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            }
        )
    return all(file_ready(snapshot / filename) for filename in required)
