from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .paths import MODEL_DIR, PROJECT_ROOT


MODEL_MANIFEST = MODEL_DIR / "pretrained_model_manifest.json"
TRAINED_HEAD_MANIFEST = MODEL_DIR / "trained_head_manifest.json"
VISUAL_BACKBONE_REPO_ID = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
DETECTOR_REPO_ID = "IDEA-Research/grounding-dino-tiny"


def file_ready(path: Path) -> bool:
    """Return true only for a readable, non-empty local artifact."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def trained_head_status(
    path: Path,
    *,
    manifest_path: Path = TRAINED_HEAD_MANIFEST,
    project_root: Path = PROJECT_ROOT,
) -> tuple[bool, str]:
    """Verify a trained head before any pickle-based deserialization occurs."""

    try:
        root = project_root.resolve()
        resolved_path = path.resolve()
        relative_path = resolved_path.relative_to(root).as_posix()
    except (OSError, ValueError):
        return False, "artifact path is outside the project root"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False, "trained-head manifest is missing or invalid"
    if manifest.get("schema_version") != "1.0.0":
        return False, "trained-head manifest schema is unsupported"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return False, "trained-head manifest has no artifact list"
    record = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict) and item.get("path") == relative_path
        ),
        None,
    )
    if record is None:
        return False, "artifact is not declared in the trained-head manifest"
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_sha)
    ):
        return False, "trained-head manifest metadata is invalid"
    try:
        if not resolved_path.is_file():
            return False, "artifact is missing"
        actual_bytes = resolved_path.stat().st_size
        digest = hashlib.sha256()
        with resolved_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False, "artifact is unreadable"
    if actual_bytes != expected_bytes:
        return False, "artifact byte count does not match the manifest"
    if digest.hexdigest().casefold() != expected_sha.casefold():
        return False, "artifact SHA-256 does not match the manifest"
    return True, "artifact byte count and SHA-256 verified"


def trained_head_ready(
    path: Path,
    *,
    manifest_path: Path = TRAINED_HEAD_MANIFEST,
    project_root: Path = PROJECT_ROOT,
) -> bool:
    """Return true only for a trained head that matches the tracked manifest."""

    ready, _ = trained_head_status(
        path, manifest_path=manifest_path, project_root=project_root
    )
    return ready


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
