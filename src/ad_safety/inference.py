from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

from .classifier import PolicyClassifier, overlay_heatmap
from .detection import Detection, GroundingDinoDetector
from .ocr import OCRResult, run_tesseract
from .paths import CONFIG_DIR, MODEL_DIR
from .policy import PolicyDecision, PolicyEngine
from .preprocessing import load_image


@dataclass
class AnalysisResult:
    decision: PolicyDecision
    classifier_scores: dict[str, float]
    detections: list[Detection]
    ocr: OCRResult
    annotated_image: Image.Image
    heatmap_image: Image.Image | None
    latency_ms: dict[str, float]
    input_size: tuple[int, int]
    applied_thresholds: dict[str, float]

    def audit_record(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "classifier_scores": self.classifier_scores,
            "detections": [item.to_dict() for item in self.detections],
            "ocr": {
                "text": self.ocr.text,
                "engine": self.ocr.engine,
                "tokens": [token.to_dict() for token in self.ocr.tokens],
            },
            "latency_ms": self.latency_ms,
            "input_size": list(self.input_size),
            "applied_thresholds": self.applied_thresholds,
        }


def _annotate(image: Image.Image, detections: list[Detection], ocr: OCRResult) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for detection in detections:
        x0, y0, x1, y1 = detection.box
        color = (226, 63, 87)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=max(2, min(image.size) // 180))
        label = f"{detection.label} {detection.score:.2f}"
        bbox = draw.textbbox((x0, y0), label, font=font)
        draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=color)
        draw.text((x0, y0), label, fill="white", font=font)
    for token in ocr.tokens:
        x0, y0, x1, y1 = token.box
        draw.rectangle((x0, y0, x1, y1), outline=(25, 154, 130), width=2)
    return canvas


class ModerationEngine:
    def __init__(
        self,
        classifier_path: str | Path | None = None,
        device: str = "auto",
        enable_detector: bool = True,
    ) -> None:
        classifier_path = Path(classifier_path or MODEL_DIR / "vit_policy_head.joblib")
        self.classifier = PolicyClassifier(classifier_path, device=device)
        config = yaml.safe_load((CONFIG_DIR / "policy.yaml").read_text(encoding="utf-8"))
        self.policy = PolicyEngine(config, thresholds=self.classifier.artifact.get("thresholds", {}))
        self._detector_enabled = enable_detector
        self._detector_lock = Lock()
        self.detector: GroundingDinoDetector | None = None

    def _get_detector(self) -> GroundingDinoDetector | None:
        if not self._detector_enabled:
            return None
        if self.detector is None:
            with self._detector_lock:
                if self.detector is None:
                    self.detector = GroundingDinoDetector(device="cpu")
        return self.detector

    def analyze(
        self,
        source: Any,
        *,
        run_detector: bool = True,
        run_ocr: bool = True,
        explain: bool = False,
    ) -> AnalysisResult:
        total_started = perf_counter()
        image = load_image(source)
        classification = self.classifier.predict(image)
        detections: list[Detection] = []
        detector_ms = 0.0
        detector = self._get_detector() if run_detector else None
        if detector is not None:
            detections, detector_ms = detector.detect(image)
        ocr = run_tesseract(image) if run_ocr else OCRResult("", (), 0.0, "disabled")
        decision = self.policy.decide(
            classification.scores,
            ocr_text=ocr.text,
            detections=[item.to_dict() for item in detections],
            model_version=classification.model_version,
        )
        heatmap_image = None
        explain_ms = 0.0
        if explain:
            started = perf_counter()
            heatmap, _ = self.classifier.occlusion_heatmap(image, decision.primary_label, grid_size=4)
            heatmap_image = overlay_heatmap(image, heatmap)
            explain_ms = (perf_counter() - started) * 1000.0
        latency = {
            "classification": classification.latency_ms,
            "detection": detector_ms,
            "ocr": ocr.latency_ms,
            "explanation": explain_ms,
            "total": (perf_counter() - total_started) * 1000.0,
        }
        return AnalysisResult(
            decision=decision,
            classifier_scores=classification.scores,
            detections=detections,
            ocr=ocr,
            annotated_image=_annotate(image, detections, ocr),
            heatmap_image=heatmap_image,
            latency_ms=latency,
            input_size=image.size,
            applied_thresholds={
                str(label): float(value) for label, value in self.policy.thresholds.items()
            },
        )
