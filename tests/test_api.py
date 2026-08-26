from __future__ import annotations

import io
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api as api_module


@pytest.fixture(autouse=True)
def reset_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "_ENGINE", None)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(api_module.app)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _animated_webp_bytes() -> bytes:
    buffer = io.BytesIO()
    first = Image.new("RGB", (64, 48), "white")
    second = Image.new("RGB", (64, 48), "red")
    first.save(
        buffer,
        format="WEBP",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    return buffer.getvalue()


def _mark_model_snapshots_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "model_snapshot_ready", lambda repo_id: True)


def test_health_does_not_construct_heavy_engine(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called() -> Any:
        pytest.fail("health endpoint constructed the heavy moderation engine")

    monkeypatch.setattr(api_module, "_build_engine", fail_if_called)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["service"] == "ad-safety-moderation"
    assert payload["engine_loaded"] is False
    assert set(payload["components"]) == {
        "classifier_artifact",
        "visual_backbone_snapshot",
        "detector_snapshot",
        "detector_loaded",
        "tesseract",
    }


def test_corrupt_api_input_is_rejected_before_engine_load(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "_build_engine", lambda: pytest.fail("engine should not load"))

    response = client.post("/analyze", files={"file": ("bad.png", b"not an image", "image/png")})

    assert response.status_code == 422
    assert "readable" in response.json()["detail"]


def test_unsupported_content_type_returns_415_before_engine_load(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "_build_engine", lambda: pytest.fail("engine should not load"))

    response = client.post(
        "/analyze",
        files={"file": ("creative.txt", _png_bytes(), "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only JPEG, PNG, and WebP uploads are supported."


def test_oversized_api_input_is_rejected_before_decode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "MAX_IMAGE_BYTES", 8)
    monkeypatch.setattr(api_module, "_build_engine", lambda: pytest.fail("engine should not load"))

    response = client.post("/analyze", files={"file": ("large.png", b"123456789", "image/png")})

    assert response.status_code == 413


def test_animated_api_input_is_rejected_before_engine_load(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "_build_engine", lambda: pytest.fail("engine should not load"))

    response = client.post(
        "/analyze",
        files={"file": ("animated.webp", _animated_webp_bytes(), "image/webp")},
    )

    assert response.status_code == 422
    assert "Animated images" in response.json()["detail"]


def test_missing_classifier_head_returns_503_before_engine_load(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api_module, "CLASSIFIER_ARTIFACT", tmp_path / "missing.joblib")
    monkeypatch.setattr(api_module, "_build_engine", lambda: pytest.fail("engine should not load"))

    response = client.post(
        "/analyze?run_detector=false&run_ocr=false&explain=false",
        files={"file": ("creative.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "classifier artifact" in response.json()["detail"]


def test_missing_vit_snapshot_returns_503_before_engine_load(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "vit_policy_head.joblib"
    artifact.write_bytes(b"test sentinel")
    monkeypatch.setattr(api_module, "CLASSIFIER_ARTIFACT", artifact)
    monkeypatch.setattr(api_module, "model_snapshot_ready", lambda repo_id: False)
    monkeypatch.setattr(api_module, "_build_engine", lambda: pytest.fail("engine should not load"))

    response = client.post(
        "/analyze?run_detector=false&run_ocr=false&explain=false",
        files={"file": ("creative.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "The required ViT backbone snapshot is incomplete."
    assert api_module._ENGINE is None


def test_requested_detector_with_missing_snapshot_returns_503_before_engine_load(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "vit_policy_head.joblib"
    artifact.write_bytes(b"test sentinel")
    monkeypatch.setattr(api_module, "CLASSIFIER_ARTIFACT", artifact)
    monkeypatch.setattr(
        api_module,
        "model_snapshot_ready",
        lambda repo_id: repo_id == api_module.VISUAL_BACKBONE_REPO_ID,
    )
    monkeypatch.setattr(api_module, "_build_engine", lambda: pytest.fail("engine should not load"))

    response = client.post(
        "/analyze?run_detector=true&run_ocr=false&explain=false",
        files={"file": ("creative.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Object context was requested, but the detector snapshot is incomplete."
    )
    assert api_module._ENGINE is None


def test_valid_request_returns_audit_with_fake_engine(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "vit_policy_head.joblib"
    artifact.write_bytes(b"test sentinel")
    monkeypatch.setattr(api_module, "CLASSIFIER_ARTIFACT", artifact)
    _mark_model_snapshots_ready(monkeypatch)

    class FakeResult:
        def audit_record(self) -> dict[str, Any]:
            return {
                "decision": {"verdict": "APPROVE", "primary_label": "safe"},
                "classifier_scores": {"safe": 0.9},
                "detections": [],
                "ocr": {"text": "", "engine": "disabled", "tokens": []},
                "latency_ms": {"total": 1.0},
                "input_size": [64, 48],
                "applied_thresholds": {
                    "firearms": 0.991356,
                    "explosives": 0.172763,
                    "financial_promotion": 0.994603,
                    "review": 0.35,
                    "uncertainty_margin": 0.12,
                },
            }

    class FakeEngine:
        detector = None

        def analyze(self, payload: bytes, **options: Any) -> FakeResult:
            assert payload.startswith(b"\x89PNG")
            assert options == {"run_detector": False, "run_ocr": False, "explain": False}
            return FakeResult()

    monkeypatch.setattr(api_module, "_ENGINE", FakeEngine())

    response = client.post(
        "/analyze?run_detector=false&run_ocr=false&explain=false",
        files={"file": ("creative.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["decision"]["verdict"] == "APPROVE"
    assert payload["input"]["filename"] == "creative.png"
    assert payload["input"]["width"] == 64
    assert len(payload["input"]["sha256"]) == 64
    assert payload["audit_schema_version"] == "1.1"
    assert payload["analysis"]["applied_thresholds"]["firearms"] == pytest.approx(0.991356)


def test_runtime_error_does_not_expose_internal_detail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "vit_policy_head.joblib"
    artifact.write_bytes(b"test sentinel")
    monkeypatch.setattr(api_module, "CLASSIFIER_ARTIFACT", artifact)
    _mark_model_snapshots_ready(monkeypatch)
    monkeypatch.setattr(
        api_module,
        "_perform_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("/private/tmp/secret.bin")),
    )

    response = client.post(
        "/analyze?run_detector=false&run_ocr=false&explain=false",
        files={"file": ("creative.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "secret.bin" not in response.json()["detail"]


def test_unexpected_error_returns_opaque_reference(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "vit_policy_head.joblib"
    artifact.write_bytes(b"test sentinel")
    monkeypatch.setattr(api_module, "CLASSIFIER_ARTIFACT", artifact)
    _mark_model_snapshots_ready(monkeypatch)
    monkeypatch.setattr(
        api_module,
        "_perform_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("private implementation detail")),
    )

    response = client.post(
        "/analyze?run_detector=false&run_ocr=false&explain=false",
        files={"file": ("creative.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert re.fullmatch(r"Analysis failed safely\. Reference [0-9A-F]{8}\.", detail)
    assert "ValueError" not in detail
    assert "private implementation detail" not in detail
