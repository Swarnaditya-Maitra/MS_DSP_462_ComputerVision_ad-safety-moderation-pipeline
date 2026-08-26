#!/usr/bin/env python3
"""Run representative and edge-case inferences for the app, deck, and video."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ad_safety.inference import ModerationEngine  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo_cases"


def select_formal_cases() -> list[dict[str, object]]:
    predictions = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "predictions.csv")
    failures = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "failure_cases.csv")
    cases: list[dict[str, object]] = []
    options = {
        "safe": {"detector": False, "ocr": True, "explain": False},
        "firearms": {"detector": True, "ocr": False, "explain": True},
        "explosives": {"detector": True, "ocr": False, "explain": False},
        "financial_promotion": {"detector": False, "ocr": True, "explain": True},
    }
    for label in ("safe", "firearms", "explosives", "financial_promotion"):
        subset = predictions[predictions["true_label"].eq(label) & predictions["correct"].astype(bool)].copy()
        probability_column = f"probability_{label}"
        row = subset.sort_values(probability_column, ascending=False).iloc[0]
        cases.append(
            {
                "case_id": f"formal_{label}",
                "scope": "formal grouped test",
                "expected_label": label,
                "path": str(row["local_path"]),
                **options[label],
            }
        )
    if not failures.empty:
        row = failures.iloc[0]
        cases.append(
            {
                "case_id": "formal_known_failure",
                "scope": "formal grouped test failure",
                "expected_label": str(row["true_label"]),
                "path": str(row["local_path"]),
                "detector": True,
                "ocr": False,
                "explain": True,
            }
        )
    cases.append(
        {
            "case_id": "external_safe_domain_shift",
            "scope": "external Wikimedia diagnostic",
            "expected_label": "safe",
            "path": "data/wikimedia_pilot/train/safe/safe_train_c456321d7b18.jpg",
            "detector": True,
            "ocr": True,
            "explain": True,
        }
    )
    return cases


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = ModerationEngine(device="cpu", enable_detector=True)
    rows: list[dict[str, object]] = []
    cases = select_formal_cases()
    for index, case in enumerate(cases, start=1):
        source = PROJECT_ROOT / str(case["path"])
        result = engine.analyze(
            source,
            run_detector=bool(case["detector"]),
            run_ocr=bool(case["ocr"]),
            explain=bool(case["explain"]),
        )
        case_dir = OUTPUT_DIR / str(case["case_id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        source_copy = case_dir / f"source{source.suffix.casefold()}"
        shutil.copy2(source, source_copy)
        result.annotated_image.save(case_dir / "evidence_overlay.jpg", "JPEG", quality=92)
        if result.heatmap_image is not None:
            result.heatmap_image.save(case_dir / "occlusion_heatmap.jpg", "JPEG", quality=92)
        audit = {
            "case": case,
            "analysis": result.audit_record(),
            "expected_label": case["expected_label"],
            "classifier_prediction": max(result.classifier_scores, key=result.classifier_scores.get),
        }
        (case_dir / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "scope": case["scope"],
                "expected_label": case["expected_label"],
                "classifier_prediction": audit["classifier_prediction"],
                "policy_verdict": result.decision.verdict,
                "policy_primary_label": result.decision.primary_label,
                "policy_score": result.decision.primary_score,
                "detector_enabled": case["detector"],
                "ocr_enabled": case["ocr"],
                "explanation_enabled": case["explain"],
                "detections": len(result.detections),
                "ocr_tokens": len(result.ocr.tokens),
                "classification_ms": result.latency_ms["classification"],
                "detection_ms": result.latency_ms["detection"],
                "ocr_ms": result.latency_ms["ocr"],
                "explanation_ms": result.latency_ms["explanation"],
                "total_ms": result.latency_ms["total"],
                "source_path": str(source.relative_to(PROJECT_ROOT)),
                "output_dir": str(case_dir.relative_to(PROJECT_ROOT)),
            }
        )
        print(
            f"[{index}/{len(cases)}] {case['case_id']}: "
            f"{audit['classifier_prediction']} -> {result.decision.verdict} "
            f"({result.latency_ms['total']:.0f} ms)"
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "case_summary.csv", index=False)
    (OUTPUT_DIR / "case_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {OUTPUT_DIR / 'case_summary.csv'}")


if __name__ == "__main__":
    main()
