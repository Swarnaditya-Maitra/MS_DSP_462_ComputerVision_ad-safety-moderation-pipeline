from __future__ import annotations

from PIL import Image

from ad_safety.inference import AnalysisResult
from ad_safety.ocr import OCRResult
from ad_safety.policy import PolicyDecision


def test_analysis_audit_preserves_exact_applied_threshold_snapshot() -> None:
    thresholds = {
        "firearms": 0.9913563345650912,
        "explosives": 0.1727632618375518,
        "financial_promotion": 0.9946029323485182,
        "review": 0.35,
        "uncertainty_margin": 0.12,
    }
    decision = PolicyDecision(
        verdict="APPROVE",
        primary_label="safe",
        primary_score=0.99,
        review_reasons=(),
        fused_scores={
            "safe": 0.99,
            "firearms": 0.004,
            "explosives": 0.003,
            "financial_promotion": 0.003,
        },
        classifier_scores={
            "safe": 0.99,
            "firearms": 0.004,
            "explosives": 0.003,
            "financial_promotion": 0.003,
        },
        model_version="vit-test",
        policy_version="1.0.0-pilot",
    )
    image = Image.new("RGB", (64, 48), "white")
    result = AnalysisResult(
        decision=decision,
        classifier_scores=decision.classifier_scores,
        detections=[],
        ocr=OCRResult("", (), 0.0, "disabled"),
        annotated_image=image,
        heatmap_image=None,
        latency_ms={
            "classification": 5.0,
            "detection": 0.0,
            "ocr": 0.0,
            "explanation": 0.0,
            "total": 5.5,
        },
        input_size=image.size,
        applied_thresholds=thresholds,
    )

    audit = result.audit_record()

    assert audit["applied_thresholds"] == thresholds
    assert audit["decision"]["verdict"] == "APPROVE"
    assert audit["input_size"] == [64, 48]
