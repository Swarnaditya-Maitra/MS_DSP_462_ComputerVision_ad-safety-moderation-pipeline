from __future__ import annotations

import json
from pathlib import Path

from ad_safety.model_assets import (
    DETECTOR_REPO_ID,
    VISUAL_BACKBONE_REPO_ID,
    model_snapshot_ready,
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
