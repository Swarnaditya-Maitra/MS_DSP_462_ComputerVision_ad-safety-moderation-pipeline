from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


RESTRICTED_LABELS = ("firearms", "explosives", "financial_promotion")


@dataclass(frozen=True)
class PolicyDecision:
    verdict: str
    primary_label: str
    primary_score: float
    review_reasons: tuple[str, ...]
    fused_scores: dict[str, float]
    classifier_scores: dict[str, float]
    model_version: str
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["review_reasons"] = list(self.review_reasons)
        return payload


def _noisy_or(values: Sequence[float]) -> float:
    remaining = 1.0
    for value in values:
        remaining *= 1.0 - max(0.0, min(1.0, float(value)))
    return 1.0 - remaining


def _contains_term(text: str, term: str) -> bool:
    """Match a cue as a complete token or phrase, not inside another word."""

    pattern = rf"(?<!\w){re.escape(term.casefold())}(?!\w)"
    return re.search(pattern, text.casefold()) is not None


class PolicyEngine:
    """Fuse visual, object, and OCR evidence into an auditable triage action."""

    def __init__(self, config: Mapping[str, Any], thresholds: Mapping[str, float] | None = None) -> None:
        self.config = dict(config)
        defaults = dict(self.config.get("default_thresholds", {}))
        defaults.update(dict(thresholds or {}))
        self.thresholds = defaults
        self.policy_version = str(self.config.get("policy_version", "unknown"))

    def decide(
        self,
        classifier_scores: Mapping[str, float],
        ocr_text: str = "",
        detections: Sequence[Mapping[str, Any]] = (),
        model_version: str = "unknown",
    ) -> PolicyDecision:
        scores = {label: float(classifier_scores.get(label, 0.0)) for label in self.config["model_labels"]}
        signals: dict[str, list[float]] = {label: [scores.get(label, 0.0)] for label in RESTRICTED_LABELS}
        reasons: list[str] = []
        lowered = ocr_text.casefold()

        for label, keywords in self.config.get("ocr_keywords", {}).items():
            matches = [(word, float(weight)) for word, weight in keywords.items() if _contains_term(lowered, word)]
            if matches:
                signals.setdefault(label, [scores.get(label, 0.0)]).extend(weight for _, weight in matches)
                reasons.append(f"OCR cue for {label}: " + ", ".join(word for word, _ in matches[:4]))

        detector_context = self.config.get("detector_context", {})
        detector_max: dict[str, tuple[float, str, float]] = {}
        for detection in detections:
            name = str(detection.get("label", "")).casefold()
            matched = [
                (term, context)
                for term, context in detector_context.items()
                if _contains_term(name, str(term)) and context.get("policy_label")
            ]
            if not matched:
                continue
            term, context = max(matched, key=lambda item: len(str(item[0])))
            label = str(context["policy_label"])
            strength = float(context.get("evidence_strength", 0.0)) * float(detection.get("score", 0.0))
            score = float(detection.get("score", 0.0))
            previous = detector_max.get(label)
            if previous is None or strength > previous[0]:
                detector_max[label] = (strength, str(term), score)

        # Grounding DINO can return several overlapping phrases for one object.
        # Count only the strongest cue per policy label so duplicate boxes do
        # not look like independent evidence.
        for label, (strength, term, score) in detector_max.items():
            signals.setdefault(label, [scores.get(label, 0.0)]).append(strength)
            reasons.append(f"Object context: {term} ({score:.2f})")

        fused = {"safe": scores.get("safe", 0.0)}
        for label in RESTRICTED_LABELS:
            fused[label] = _noisy_or(signals.get(label, [scores.get(label, 0.0)]))

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        primary_label, primary_score = ranked[0]
        margin = primary_score - ranked[1][1] if len(ranked) > 1 else primary_score

        block_labels = ("firearms", "explosives")
        classifier_primary = max(scores, key=scores.get)
        block_candidates = [
            label
            for label in block_labels
            if label == classifier_primary
            and scores.get(label, 0.0) >= float(self.thresholds.get(label, 0.65))
        ]
        if block_candidates:
            selected = max(block_candidates, key=lambda label: scores[label])
            verdict = "BLOCK"
            primary_label = selected
            primary_score = fused[selected]
            reasons.append(
                f"{selected} was the top classifier class and crossed its validated block threshold"
            )
        elif fused.get("financial_promotion", 0.0) >= float(self.thresholds.get("financial_promotion", 0.40)):
            verdict = "REVIEW"
            primary_label = "financial_promotion"
            primary_score = fused[primary_label]
            reasons.append("Financial imagery requires human policy and jurisdiction review")
        elif max(fused[label] for label in RESTRICTED_LABELS) >= float(self.thresholds.get("review", 0.35)):
            verdict = "REVIEW"
            primary_label = max(RESTRICTED_LABELS, key=lambda label: fused[label])
            primary_score = fused[primary_label]
            reasons.append("Restricted-class evidence crossed the review threshold")
        elif margin < float(self.thresholds.get("uncertainty_margin", 0.12)):
            verdict = "REVIEW"
            reasons.append("Top scores are too close for an automatic decision")
        else:
            verdict = "APPROVE"

        return PolicyDecision(
            verdict=verdict,
            primary_label=primary_label,
            primary_score=float(primary_score),
            review_reasons=tuple(dict.fromkeys(reasons)),
            fused_scores={key: float(value) for key, value in fused.items()},
            classifier_scores=scores,
            model_version=model_version,
            policy_version=self.policy_version,
        )
