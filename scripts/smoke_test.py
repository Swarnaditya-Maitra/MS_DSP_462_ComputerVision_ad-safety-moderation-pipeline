#!/usr/bin/env python3
"""Run one real, detector-free API analysis against the local model stack."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from fastapi.testclient import TestClient
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api as api_module  # noqa: E402


VALID_VERDICTS = {"APPROVE", "REVIEW", "BLOCK"}


def _synthetic_png() -> bytes:
    """Return a deterministic 256 by 256 RGB image without external assets."""

    image = Image.new("RGB", (256, 256))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (x, y, (x + y) // 2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def validate_health(payload: Mapping[str, Any]) -> None:
    """Raise a useful error when the core model stack is not ready."""

    if payload.get("status") != "ok" or payload.get("ready_for_analysis") is not True:
        components = payload.get("components", {})
        raise RuntimeError(
            "Core analysis is not ready. Run "
            "`python scripts/bootstrap.py --profile core`, then retry. "
            f"Components: {json.dumps(components, sort_keys=True)}"
        )


def validate_analysis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the stable success contract and return a compact summary."""

    analysis = payload.get("analysis")
    input_record = payload.get("input")
    if not isinstance(analysis, Mapping) or not isinstance(input_record, Mapping):
        raise RuntimeError("Analysis response is missing the input or analysis object.")
    decision = analysis.get("decision")
    if not isinstance(decision, Mapping) or decision.get("verdict") not in VALID_VERDICTS:
        raise RuntimeError("Analysis response has no valid policy verdict.")
    sha256 = input_record.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise RuntimeError("Analysis response has no 64-character input SHA-256.")
    if payload.get("audit_schema_version") != "1.1":
        raise RuntimeError("Analysis response has an unexpected audit schema version.")
    return {
        "smoke_test": "passed",
        "audit_schema_version": payload["audit_schema_version"],
        "verdict": decision["verdict"],
        "primary_label": decision.get("primary_label"),
        "input_sha256": sha256,
        "latency_ms": analysis.get("latency_ms", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local JPEG, PNG, or WebP. A generated PNG is used by default.",
    )
    args = parser.parse_args()

    if args.image is None:
        filename = "synthetic_smoke_test.png"
        content_type = "image/png"
        image_bytes = _synthetic_png()
    else:
        if not args.image.is_file():
            parser.error(f"Image does not exist: {args.image}")
        suffix = args.image.suffix.casefold()
        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        if suffix not in content_types:
            parser.error("--image must be a JPEG, PNG, or WebP file")
        filename = args.image.name
        content_type = content_types[suffix]
        image_bytes = args.image.read_bytes()

    with TestClient(api_module.app) as client:
        health_response = client.get("/health")
        health_response.raise_for_status()
        validate_health(health_response.json())
        response = client.post(
            "/analyze?run_detector=false&run_ocr=false&explain=false",
            files={"file": (filename, image_bytes, content_type)},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Analysis returned HTTP {response.status_code}: {response.text}"
            )
        summary = validate_analysis(response.json())

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
