from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ad_safety.preprocessing import (  # noqa: E402
    MAX_IMAGE_BYTES,
    ImageValidationError,
    load_image,
)


SERVICE_VERSION = "1.0.0"
CLASSIFIER_ARTIFACT = PROJECT_ROOT / "models" / "vit_policy_head.joblib"
MODEL_MANIFEST = PROJECT_ROOT / "models" / "pretrained_model_manifest.json"
ALLOWED_CONTENT_TYPES = {
    "application/octet-stream",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

app = FastAPI(
    title="Ad Safety Moderation API",
    version=SERVICE_VERSION,
    description=(
        "Local policy-triage API for ad creatives. Uploaded pixels are analyzed inside the host process "
        "and are not forwarded to a third-party inference service."
    ),
)

_ENGINE: Any | None = None
_ENGINE_LOCK = Lock()
_ANALYSIS_LOCK = Lock()


def _build_engine() -> Any:
    # The import and all model construction stay behind the first valid analysis
    # request. Health checks therefore remain fast and offline-safe.
    from ad_safety.inference import ModerationEngine

    return ModerationEngine(enable_detector=True)


def get_engine() -> Any:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = _build_engine()
    return _ENGINE


def _model_snapshot_ready(repo_id: str) -> bool:
    if not MODEL_MANIFEST.is_file():
        return False
    try:
        manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    row = next(
        (
            item
            for item in manifest.get("models", [])
            if item.get("repo_id") == repo_id
        ),
        None,
    )
    if row is None:
        return False
    snapshot = Path(str(row.get("local_snapshot", "")))
    if not snapshot.is_absolute():
        snapshot = PROJECT_ROOT / snapshot
    return (snapshot / "model.safetensors").is_file() and (snapshot / "config.json").is_file()


def _perform_analysis(payload: bytes, run_detector: bool, run_ocr: bool, explain: bool) -> Any:
    # A single engine owns shared torch modules. Serialize access to avoid racing
    # optional lazy model initialization and to cap memory spikes on local hosts.
    with _ANALYSIS_LOCK:
        return get_engine().analyze(
            payload,
            run_detector=run_detector,
            run_ocr=run_ocr,
            explain=explain,
        )


@app.get("/health", tags=["operations"])
def health() -> dict[str, Any]:
    classifier_ready = CLASSIFIER_ARTIFACT.is_file()
    backbone_ready = _model_snapshot_ready("timm/vit_base_patch16_224.augreg2_in21k_ft_in1k")
    analysis_ready = classifier_ready and backbone_ready
    engine_loaded = _ENGINE is not None
    return {
        "status": "ok" if analysis_ready else "degraded",
        "service": "ad-safety-moderation",
        "version": SERVICE_VERSION,
        "ready_for_analysis": analysis_ready,
        "engine_loaded": engine_loaded,
        "components": {
            "classifier_artifact": classifier_ready,
            "visual_backbone_snapshot": backbone_ready,
            "detector_snapshot": _model_snapshot_ready("IDEA-Research/grounding-dino-tiny"),
            "detector_loaded": bool(engine_loaded and getattr(_ENGINE, "detector", None) is not None),
            "tesseract": shutil.which("tesseract") is not None,
        },
    }


@app.post("/analyze", tags=["moderation"])
async def analyze(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP ad creative"),
    run_detector: bool = Query(False, description="Fuse local Grounding DINO object evidence"),
    run_ocr: bool = Query(True, description="Fuse local Tesseract OCR evidence"),
    explain: bool = Query(False, description="Generate a slower occlusion sensitivity explanation"),
) -> dict[str, Any]:
    content_type = (file.content_type or "application/octet-stream").split(";", 1)[0].casefold()
    if content_type not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG, PNG, and WebP uploads are supported.",
        )

    try:
        payload = await file.read(MAX_IMAGE_BYTES + 1)
    finally:
        await file.close()
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 20 MB upload limit.",
        )

    # Validate before constructing any model so corrupt and unsafe uploads are
    # rejected cheaply and cannot force a heavy model load.
    try:
        validated = await run_in_threadpool(load_image, payload)
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not CLASSIFIER_ARTIFACT.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The trained classifier artifact models/vit_policy_head.joblib is missing.",
        )

    try:
        result = await run_in_threadpool(_perform_analysis, payload, run_detector, run_ocr, explain)
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"A required local model component could not start: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed safely ({type(exc).__name__}).",
        ) from exc

    audit = result.audit_record()
    return {
        "audit_schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "filename": Path(file.filename or "upload").name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "width": validated.width,
            "height": validated.height,
        },
        "options": {
            "detector": run_detector,
            "ocr": run_ocr,
            "explanation": explain,
        },
        "analysis": audit,
    }
