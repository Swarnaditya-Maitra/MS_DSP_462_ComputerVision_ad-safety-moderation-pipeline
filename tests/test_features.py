from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from ad_safety import features


def test_pinned_backbones_resolve_to_local_safetensors() -> None:
    missing = []
    for model_name in (features.VIT_MODEL_NAME, features.CNN_MODEL_NAME):
        try:
            features.resolve_local_pretrained_file(model_name)
        except features.BackboneArtifactError:
            missing.append(model_name)
    if missing:
        pytest.skip("requires pinned snapshots from scripts/download_models.py")

    for model_name in (features.VIT_MODEL_NAME, features.CNN_MODEL_NAME):
        weights = features.resolve_local_pretrained_file(model_name)
        assert weights.is_file()
        assert weights.name == "model.safetensors"
        assert "snapshots" in weights.parts


def test_feature_extractor_passes_exact_local_file_to_timm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    local_weights = tmp_path / "model.safetensors"
    local_weights.write_bytes(b"test sentinel")

    class FakeModel:
        num_features = 16

        def eval(self) -> "FakeModel":
            return self

        def to(self, device: torch.device) -> "FakeModel":
            captured["device"] = device
            return self

    def fake_create_model(model_name: str, **kwargs: Any) -> FakeModel:
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(features.timm, "create_model", fake_create_model)
    monkeypatch.setattr(features.timm.data, "resolve_model_data_config", lambda model: {})
    monkeypatch.setattr(features.timm.data, "create_transform", lambda **kwargs: "transform")
    monkeypatch.setattr(features, "resolve_local_pretrained_file", lambda model_name: local_weights)

    extractor = features.BackboneFeatureExtractor(features.VIT_MODEL_NAME, device="cpu")

    overlay_file = Path(captured["kwargs"]["pretrained_cfg_overlay"]["file"])
    assert captured["model_name"] == features.VIT_MODEL_NAME
    assert captured["kwargs"]["pretrained"] is True
    assert overlay_file == local_weights
    assert extractor.transform == "transform"


def test_unknown_backbone_requires_explicit_remote_opt_in() -> None:
    with pytest.raises(features.BackboneArtifactError, match="not pinned"):
        features.BackboneFeatureExtractor("unregistered_model", device="cpu")
