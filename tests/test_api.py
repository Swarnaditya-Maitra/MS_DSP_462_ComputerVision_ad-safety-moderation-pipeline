from __future__ import annotations

import io
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


def test_oversized_api_input_is_rejected_before_decode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "MAX_IMAGE_BYTES", 8)
    monkeypatch.setattr(api_module, "_build_engine", lambda: pytest.fail("engine should not load"))

    response = client.post("/analyze", files={"file": ("large.png", b"123456789", "image/png")})

    assert response.status_code == 413


def test_valid_request_returns_audit_with_fake_engine(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "vit_policy_head.joblib"
    artifact.write_bytes(b"test sentinel")
    monkeypatch.setattr(api_module, "CLASSIFIER_ARTIFACT", artifact)

    class FakeResult:
        def audit_record(self) -> dict[str, Any]:
            return {
                "decision": {"verdict": "APPROVE", "primary_label": "safe"},
                "classifier_scores": {"safe": 0.9},
                "detections": [],
                "ocr": {"text": "", "engine": "disabled", "tokens": []},
                "latency_ms": {"total": 1.0},
                "input_size": [64, 48],
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
