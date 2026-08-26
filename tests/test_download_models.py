from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from scripts import download_models


def _repo_ids(*, core_only: bool = False, skip_grounding_dino: bool = False) -> list[str]:
    return [
        spec["repo_id"]
        for spec in download_models.select_model_specs(
            core_only=core_only,
            skip_grounding_dino=skip_grounding_dino,
        )
    ]


def test_select_model_specs_preserves_default_and_skip_behavior() -> None:
    assert _repo_ids() == [spec["repo_id"] for spec in download_models.MODEL_SPECS]
    assert _repo_ids(skip_grounding_dino=True) == [
        download_models.CORE_MODEL_REPO_ID,
        "timm/resnet50.a1_in1k",
    ]


def test_select_model_specs_core_only_returns_only_app_backbone() -> None:
    expected = [download_models.CORE_MODEL_REPO_ID]

    assert _repo_ids(core_only=True) == expected
    assert _repo_ids(core_only=True, skip_grounding_dino=True) == expected


def test_main_core_only_uses_offline_cache_without_network(
    monkeypatch: Any, tmp_path: Path
) -> None:
    cache_root = tmp_path / ".cache" / "huggingface"
    manifest_path = tmp_path / "models" / "pretrained_model_manifest.json"
    snapshot = cache_root / "hub" / "snapshots" / "vit"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"pinned weights")
    spec = next(
        row
        for row in download_models.MODEL_SPECS
        if row["repo_id"] == download_models.CORE_MODEL_REPO_ID
    )
    expected = download_models.build_snapshot_record(
        spec, snapshot, project_root=tmp_path
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_text = json.dumps({"models": [expected]}, indent=2) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(download_models, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(download_models, "HF_HOME", cache_root)
    monkeypatch.setattr(download_models, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(download_models, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--core-only", "--offline"])
    monkeypatch.setenv("HF_HOME", "test-sentinel")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "test-sentinel")

    download_models.main()

    assert calls == [
        {
            "repo_id": download_models.CORE_MODEL_REPO_ID,
            "revision": "063c6c38a5d8510b2e57df480445e94b231dad2c",
            "cache_dir": cache_root / "hub",
            "local_files_only": True,
        }
    ]
    assert manifest_path.read_text(encoding="utf-8") == manifest_text


def test_validate_snapshot_record_rejects_modified_bytes() -> None:
    expected = {
        "repo_id": download_models.CORE_MODEL_REPO_ID,
        "revision": "abc",
        "local_snapshot": ".cache/model",
        "license": "apache-2.0",
        "snapshot_sha256": "1" * 64,
        "snapshot_bytes": 100,
        "snapshot_files": 2,
    }
    actual = {**expected, "snapshot_sha256": "2" * 64}

    try:
        download_models.validate_snapshot_record(actual, expected)
    except RuntimeError as exc:
        assert "snapshot_sha256" in str(exc)
    else:
        raise AssertionError("modified snapshot should fail verification")
