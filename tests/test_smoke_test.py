from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import smoke_test  # noqa: E402


def test_synthetic_smoke_image_is_deterministic_and_valid() -> None:
    first = smoke_test._synthetic_png()
    second = smoke_test._synthetic_png()

    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
    image = Image.open(io.BytesIO(first))
    assert image.size == (256, 256)
    assert image.mode == "RGB"


def test_validate_health_reports_missing_components() -> None:
    with pytest.raises(RuntimeError, match="bootstrap.py --profile core"):
        smoke_test.validate_health(
            {
                "status": "degraded",
                "ready_for_analysis": False,
                "components": {"visual_backbone_snapshot": False},
            }
        )


def test_validate_analysis_returns_compact_summary() -> None:
    payload = {
        "audit_schema_version": "1.1",
        "input": {"sha256": "a" * 64},
        "analysis": {
            "decision": {"verdict": "REVIEW", "primary_label": "financial_promotion"},
            "latency_ms": {"total": 42.0},
        },
    }

    summary = smoke_test.validate_analysis(payload)

    assert summary["smoke_test"] == "passed"
    assert summary["verdict"] == "REVIEW"
    assert summary["latency_ms"]["total"] == 42.0
