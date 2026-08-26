#!/usr/bin/env python3
"""Run a manually reviewed external spot check on Wikimedia examples.

This is diagnostic evidence only. It is kept separate from the untouched model
test split because the examples are small, historically skewed, and were found
through keyword searches rather than a probability sample of modern ads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ad_safety.inference import ModerationEngine  # noqa: E402


MANIFEST = PROJECT_ROOT / "data" / "wikimedia_external_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def safe_div(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-detector", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = pd.read_csv(MANIFEST)
    rows = rows[rows["include_in_spot_check"].astype(str).str.casefold().eq("true")].copy()
    if args.limit:
        rows = rows.head(args.limit)
    if rows.empty:
        raise RuntimeError("No reviewed external samples were selected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated_dir = OUTPUT_DIR / "external_annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    engine = ModerationEngine(device=args.device, enable_detector=not args.disable_detector)
    records: list[dict[str, object]] = []

    for index, row in rows.reset_index(drop=True).iterrows():
        path = PROJECT_ROOT / str(row["local_path"])
        result = engine.analyze(
            path,
            run_detector=not args.disable_detector,
            run_ocr=True,
            explain=False,
        )
        predicted = max(result.classifier_scores, key=result.classifier_scores.get)
        detection_labels = [item.label for item in result.detections]
        has_weapon_detection = any(
            term in label
            for label in detection_labels
            for term in ("handgun", "pistol", "rifle", "shotgun", "ammunition", "grenade", "bomb", "explosive", "fireworks")
        )
        output_name = f"{index:02d}_{path.stem}.jpg"
        result.annotated_image.save(annotated_dir / output_name, "JPEG", quality=91)
        records.append(
            {
                "local_path": str(row["local_path"]),
                "expected_label": str(row["reviewed_label"]),
                "classifier_prediction": predicted,
                "policy_verdict": result.decision.verdict,
                "policy_primary_label": result.decision.primary_label,
                "classifier_correct": predicted == str(row["reviewed_label"]),
                "weapon_detector_signal": has_weapon_detection,
                "detections_json": json.dumps([item.to_dict() for item in result.detections]),
                "ocr_text": result.ocr.text,
                "classifier_scores_json": json.dumps(result.classifier_scores, sort_keys=True),
                "fused_scores_json": json.dumps(result.decision.fused_scores, sort_keys=True),
                "review_reasons_json": json.dumps(list(result.decision.review_reasons)),
                "classification_ms": result.latency_ms["classification"],
                "detection_ms": result.latency_ms["detection"],
                "ocr_ms": result.latency_ms["ocr"],
                "total_ms": result.latency_ms["total"],
                "annotated_path": str((annotated_dir / output_name).relative_to(PROJECT_ROOT)),
            }
        )
        print(
            f"[{index + 1:02d}/{len(rows):02d}] {row['reviewed_label']:<21s} "
            f"classifier={predicted:<21s} policy={result.decision.verdict:<7s} "
            f"total={result.latency_ms['total']:.0f} ms"
        )

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(OUTPUT_DIR / "external_spot_check.csv", index=False)
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
        "safe_policy_non_approve_rate": safe_div(int((safe["policy_verdict"] != "APPROVE").sum()), len(safe)),
        "weapon_policy_block_rate": safe_div(int((weapon["policy_verdict"] == "BLOCK").sum()), len(weapon)),
        "financial_correct_review_rate": safe_div(
            int(
                (
                    financial["policy_verdict"].eq("REVIEW")
                    & financial["policy_primary_label"].eq("financial_promotion")
                ).sum()
            ),
            len(financial),
        ),
        "weapon_detector_image_recall": safe_div(int(weapon["weapon_detector_signal"].sum()), len(weapon)),
        "weapon_detector_nonweapon_false_positive_rate": safe_div(
            int(non_weapon["weapon_detector_signal"].sum()), len(non_weapon)
        ),
        "classification_report": report,
        "detector_enabled": not args.disable_detector,
        "limitations": [
            "The sample is small and selected through Wikimedia keyword search.",
            "Historical advertisements dominate the safe and financial examples.",
            "Image-level detector recall is not box-level mAP.",
            "The source search produced one false explosive label, which manual review excluded.",
        ],
    }
    (OUTPUT_DIR / "external_spot_check.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {OUTPUT_DIR / 'external_spot_check.csv'}")
    print(f"Saved {OUTPUT_DIR / 'external_spot_check.json'}")


if __name__ == "__main__":
    main()
