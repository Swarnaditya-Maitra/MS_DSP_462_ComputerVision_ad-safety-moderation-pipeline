#!/usr/bin/env python3
"""Reapply the current policy to saved external classifier/OCR/detector evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ad_safety.policy import PolicyEngine  # noqa: E402


CSV_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "external_spot_check.csv"
JSON_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "external_spot_check.json"


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Run evaluate_external_spot_check.py first: {CSV_PATH}")
    frame = pd.read_csv(CSV_PATH)
    artifact = joblib.load(PROJECT_ROOT / "models" / "vit_policy_head.joblib")
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "policy.yaml").read_text(encoding="utf-8"))
    policy = PolicyEngine(config, thresholds=artifact.get("thresholds", {}))

    for index, row in frame.iterrows():
        scores = json.loads(str(row["classifier_scores_json"]))
        detections = json.loads(str(row["detections_json"]))
        decision = policy.decide(
            scores,
            ocr_text=str(row.get("ocr_text", "") if pd.notna(row.get("ocr_text", "")) else ""),
            detections=detections,
            model_version=str(artifact.get("model_version", "unknown")),
        )
        frame.at[index, "policy_verdict"] = decision.verdict
        frame.at[index, "policy_primary_label"] = decision.primary_label
        frame.at[index, "fused_scores_json"] = json.dumps(decision.fused_scores, sort_keys=True)
        frame.at[index, "review_reasons_json"] = json.dumps(list(decision.review_reasons))

    frame.to_csv(CSV_PATH, index=False)
    expected = frame["expected_label"].astype(str).tolist()
    predicted = frame["classifier_prediction"].astype(str).tolist()
    report = classification_report(expected, predicted, output_dict=True, zero_division=0)
    safe = frame[frame["expected_label"].eq("safe")]
    weapon = frame[frame["expected_label"].isin(["firearms", "explosives"])]
    non_weapon = frame[~frame["expected_label"].isin(["firearms", "explosives"])]
    financial = frame[frame["expected_label"].eq("financial_promotion")]
    summary = {
        "scope": "manually reviewed external Wikimedia diagnostic spot check; not a formal test set",
        "sample_count": int(len(frame)),
        "class_counts": frame["expected_label"].value_counts().sort_index().astype(int).to_dict(),
        "classifier_accuracy": float(frame["classifier_correct"].mean()),
        "classifier_macro_f1": float(report["macro avg"]["f1-score"]),
        "safe_policy_non_approve_rate": ratio(int((safe["policy_verdict"] != "APPROVE").sum()), len(safe)),
        "weapon_policy_block_rate": ratio(int((weapon["policy_verdict"] == "BLOCK").sum()), len(weapon)),
        "financial_correct_review_rate": ratio(
            int(
                (
                    financial["policy_verdict"].eq("REVIEW")
                    & financial["policy_primary_label"].eq("financial_promotion")
                ).sum()
            ),
            len(financial),
        ),
        "weapon_detector_image_recall": ratio(int(weapon["weapon_detector_signal"].sum()), len(weapon)),
        "weapon_detector_nonweapon_false_positive_rate": ratio(
            int(non_weapon["weapon_detector_signal"].sum()), len(non_weapon)
        ),
        "classification_report": report,
        "detector_enabled": True,
        "policy_recomputed_from_saved_evidence": True,
        "limitations": [
            "The sample is small and selected through Wikimedia keyword search.",
            "Historical advertisements dominate the safe and financial examples.",
            "Image-level detector recall is not box-level mAP.",
            "The source search produced one false explosive label, which manual review excluded.",
            "Detector signals use an uncalibrated external score threshold and show a high false-positive rate.",
        ],
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
