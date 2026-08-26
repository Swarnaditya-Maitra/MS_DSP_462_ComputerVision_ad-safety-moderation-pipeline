from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ad_safety.policy import PolicyEngine


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "policy.yaml"


@pytest.fixture()
def engine() -> PolicyEngine:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return PolicyEngine(config)


def test_clear_safe_case_is_approved(engine: PolicyEngine) -> None:
    decision = engine.decide(
        {"safe": 0.88, "firearms": 0.03, "explosives": 0.04, "financial_promotion": 0.05},
        model_version="test-model",
    )

    assert decision.verdict == "APPROVE"
    assert decision.primary_label == "safe"
    assert decision.review_reasons == ()


def test_financial_classifier_signal_routes_to_review(engine: PolicyEngine) -> None:
    decision = engine.decide(
        {"safe": 0.61, "firearms": 0.02, "explosives": 0.03, "financial_promotion": 0.45}
    )

    assert decision.verdict == "REVIEW"
    assert decision.primary_label == "financial_promotion"
    assert "jurisdiction review" in " ".join(decision.review_reasons)


@pytest.mark.parametrize("label", ["firearms", "explosives"])
def test_validated_restricted_classifier_signal_is_blocked(engine: PolicyEngine, label: str) -> None:
    scores = {"safe": 0.20, "firearms": 0.06, "explosives": 0.06, "financial_promotion": 0.06}
    scores[label] = 0.68

    decision = engine.decide(scores)

    assert decision.verdict == "BLOCK"
    assert decision.primary_label == label
    assert "top classifier class" in " ".join(decision.review_reasons)


def test_secondary_restricted_score_never_auto_blocks(engine: PolicyEngine) -> None:
    decision = engine.decide(
        {"safe": 0.55, "firearms": 0.02, "explosives": 0.40, "financial_promotion": 0.03}
    )

    assert decision.verdict == "REVIEW"
    assert decision.verdict != "BLOCK"


def test_overlapping_detector_phrases_count_once(engine: PolicyEngine) -> None:
    decision = engine.decide(
        {"safe": 0.82, "firearms": 0.05, "explosives": 0.03, "financial_promotion": 0.10},
        detections=[
            {"label": "handgun", "score": 0.40, "box": [1, 2, 30, 40]},
            {"label": "handgun pistol", "score": 0.35, "box": [1, 2, 30, 40]},
        ],
    )

    expected = 1.0 - ((1.0 - 0.05) * (1.0 - 0.45 * 0.40))
    assert decision.fused_scores["firearms"] == pytest.approx(expected)
    assert sum("Object context" in reason for reason in decision.review_reasons) == 1


def test_ocr_keywords_fuse_with_classifier_signal(engine: PolicyEngine) -> None:
    decision = engine.decide(
        {"safe": 0.80, "firearms": 0.03, "explosives": 0.03, "financial_promotion": 0.02},
        ocr_text="Guaranteed BITCOIN returns",
    )

    expected = 1.0 - ((1.0 - 0.02) * (1.0 - 0.42) * (1.0 - 0.25))
    assert decision.verdict == "REVIEW"
    assert decision.primary_label == "financial_promotion"
    assert decision.fused_scores["financial_promotion"] == pytest.approx(expected)
    assert "bitcoin" in " ".join(decision.review_reasons).casefold()


def test_detector_context_fuses_without_automatic_block(engine: PolicyEngine) -> None:
    decision = engine.decide(
        {"safe": 0.82, "firearms": 0.05, "explosives": 0.03, "financial_promotion": 0.02},
        detections=[{"label": "handgun", "score": 0.90, "box": [1, 2, 3, 4]}],
    )

    expected = 1.0 - ((1.0 - 0.05) * (1.0 - 0.45 * 0.90))
    assert decision.verdict == "REVIEW"
    assert decision.primary_label == "firearms"
    assert decision.fused_scores["firearms"] == pytest.approx(expected)
    assert any(reason.startswith("Object context: handgun") for reason in decision.review_reasons)


def test_fused_ocr_and_detector_evidence_cannot_bypass_block_threshold(engine: PolicyEngine) -> None:
    decision = engine.decide(
        {"safe": 0.75, "firearms": 0.10, "explosives": 0.03, "financial_promotion": 0.02},
        ocr_text="Pistol and ammunition",
        detections=[{"label": "pistol", "score": 0.95}],
    )

    assert decision.fused_scores["firearms"] > 0.65
    assert decision.verdict == "REVIEW"
    assert decision.verdict != "BLOCK"


def test_ocr_keyword_does_not_match_inside_another_word(engine: PolicyEngine) -> None:
    decision = engine.decide(
        {"safe": 0.70, "firearms": 0.10, "explosives": 0.04, "financial_promotion": 0.03},
        ocr_text="A process has begun",
    )

    assert decision.verdict == "APPROVE"
    assert decision.fused_scores["firearms"] == pytest.approx(0.10)
    assert not decision.review_reasons
