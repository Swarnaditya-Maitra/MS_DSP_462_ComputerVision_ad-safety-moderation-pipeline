#!/usr/bin/env python3
"""Download pinned pretrained model snapshots into the Project cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
MANIFEST_PATH = PROJECT_ROOT / "models" / "pretrained_model_manifest.json"

CORE_MODEL_REPO_ID = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
GROUNDING_DINO_REPO_ID = "IDEA-Research/grounding-dino-tiny"

MODEL_SPECS = (
    {
        "repo_id": CORE_MODEL_REPO_ID,
        "revision": "063c6c38a5d8510b2e57df480445e94b231dad2c",
        "license": "apache-2.0",
    },
    {
        "repo_id": "timm/resnet50.a1_in1k",
        "revision": "767268603ca0cb0bfe326fa87277f19c419566ef",
        "license": "apache-2.0",
    },
    {
        "repo_id": GROUNDING_DINO_REPO_ID,
        "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "license": "apache-2.0",
    },
)


def select_model_specs(
    *, core_only: bool = False, skip_grounding_dino: bool = False
) -> tuple[dict[str, str], ...]:
    """Return the pinned snapshots selected by the command-line flags."""

    if core_only:
        return tuple(spec for spec in MODEL_SPECS if spec["repo_id"] == CORE_MODEL_REPO_ID)
    if skip_grounding_dino:
        return tuple(spec for spec in MODEL_SPECS if spec["repo_id"] != GROUNDING_DINO_REPO_ID)
    return MODEL_SPECS


def snapshot_fingerprint(snapshot_path: Path) -> tuple[str, int, int]:
    """Hash paths and contents so a local snapshot can be independently checked."""

    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(item for item in snapshot_path.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot_path).as_posix()
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        file_count += 1
    return digest.hexdigest(), total_bytes, file_count


def build_snapshot_record(
    spec: Mapping[str, str], snapshot_path: Path, *, project_root: Path
) -> dict[str, Any]:
    """Describe downloaded bytes in the same shape as the tracked manifest."""

    try:
        relative_snapshot = snapshot_path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Downloaded snapshot is outside the project cache: {snapshot_path}"
        ) from exc
    fingerprint, total_bytes, file_count = snapshot_fingerprint(snapshot_path)
    return {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "local_snapshot": relative_snapshot,
        "license": spec["license"],
        "snapshot_sha256": fingerprint,
        "snapshot_bytes": total_bytes,
        "snapshot_files": file_count,
    }


def validate_snapshot_record(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Reject any downloaded snapshot that differs from the reviewed record."""

    keys = (
        "repo_id",
        "revision",
        "local_snapshot",
        "license",
        "snapshot_sha256",
        "snapshot_bytes",
        "snapshot_files",
    )
    mismatches = [
        f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
        for key in keys
        if actual.get(key) != expected.get(key)
    ]
    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(f"Pinned snapshot verification failed. {details}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Download only the pinned ViT snapshot required by the app",
    )
    parser.add_argument("--skip-grounding-dino", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only the pinned local cache")
    args = parser.parse_args()

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Tracked model manifest is not readable: {MANIFEST_PATH}") from exc
    expected_by_repo = {
        str(row.get("repo_id")): row for row in manifest.get("models", [])
    }

    selected_specs = select_model_specs(
        core_only=args.core_only,
        skip_grounding_dino=args.skip_grounding_dino,
    )
    for spec in selected_specs:
        repo_id = str(spec["repo_id"])
        revision = str(spec["revision"])
        print(f"Resolving {repo_id} at pinned revision {revision}")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=HF_HOME / "hub",
            local_files_only=args.offline,
        )
        expected = expected_by_repo.get(repo_id)
        if expected is None:
            raise RuntimeError(f"Tracked model manifest has no record for {repo_id}.")
        actual = build_snapshot_record(
            spec,
            Path(snapshot_path),
            project_root=PROJECT_ROOT,
        )
        validate_snapshot_record(actual, expected)
        print(
            f"Verified {actual['snapshot_files']} files, "
            f"{actual['snapshot_bytes']} bytes, SHA-256 {actual['snapshot_sha256']}"
        )

    print(f"Verified against tracked model manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
