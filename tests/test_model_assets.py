from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ad_safety.model_assets import (
    DETECTOR_REPO_ID,
    VISUAL_BACKBONE_REPO_ID,
    model_snapshot_ready,
    trained_head_ready,
    trained_head_status,
)


def _write_manifest(root: Path, repo_id: str, snapshot: Path) -> Path:
    manifest = root / "models" / "pretrained_model_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "repo_id": repo_id,
                        "local_snapshot": str(snapshot.relative_to(root)),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_visual_backbone_requires_nonempty_weights_and_config(tmp_path: Path) -> None:
    snapshot = tmp_path / "models" / "snapshots" / "vit"
    snapshot.mkdir(parents=True)
    manifest = _write_manifest(tmp_path, VISUAL_BACKBONE_REPO_ID, snapshot)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert not model_snapshot_ready(
        VISUAL_BACKBONE_REPO_ID, manifest_path=manifest, project_root=tmp_path
    )

    (snapshot / "model.safetensors").write_bytes(b"weights")
    assert model_snapshot_ready(
        VISUAL_BACKBONE_REPO_ID, manifest_path=manifest, project_root=tmp_path
    )


def test_detector_requires_processor_and_tokenizer_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "models" / "snapshots" / "detector"
    snapshot.mkdir(parents=True)
    manifest = _write_manifest(tmp_path, DETECTOR_REPO_ID, snapshot)
    for filename in ("model.safetensors", "config.json"):
        (snapshot / filename).write_text("ready", encoding="utf-8")

    assert not model_snapshot_ready(DETECTOR_REPO_ID, manifest_path=manifest, project_root=tmp_path)

    for filename in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (snapshot / filename).write_text("{}", encoding="utf-8")
    assert model_snapshot_ready(DETECTOR_REPO_ID, manifest_path=manifest, project_root=tmp_path)


def test_trained_head_requires_manifest_size_and_sha256(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "vit_policy_head.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"reviewed head")
    manifest = tmp_path / "models" / "trained_head_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifacts": [
                    {
                        "role": "primary",
                        "path": "models/vit_policy_head.joblib",
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert trained_head_ready(
        artifact, manifest_path=manifest, project_root=tmp_path
    )
    artifact.write_bytes(b"substituted head")
    ready, detail = trained_head_status(
        artifact, manifest_path=manifest, project_root=tmp_path
    )
    assert not ready
    assert "manifest" in detail


def test_trained_head_rejects_paths_outside_project_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.joblib"
    outside.write_bytes(b"outside")

    ready, detail = trained_head_status(
        outside,
        manifest_path=tmp_path / "missing-manifest.json",
        project_root=tmp_path,
    )

    assert not ready
    assert "outside" in detail
