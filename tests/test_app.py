from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import app as app_module


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_shell_renders_without_loading_models() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

    assert not app.exception
    assert len(app.get("file_uploader")) == 1
    assert len(app.sidebar.toggle) == 3
    assert [toggle.value for toggle in app.sidebar.toggle] == [False, True, False]
    assert len(app.get("button_group")) == 1


def test_analysis_identity_changes_only_for_inference_inputs() -> None:
    options = {"detector": False, "ocr": True, "explanation": False}
    source, fingerprint = app_module._analysis_identity("creative.png", b"abc", options)

    assert source.endswith(":" + hashlib.sha256(b"abc").hexdigest())
    assert app_module._analysis_identity("creative.png", b"abc", options) == (source, fingerprint)
    assert app_module._analysis_identity("renamed.png", b"abc", options)[1] != fingerprint
    assert app_module._analysis_identity("creative.png", b"abd", options)[1] != fingerprint
    assert app_module._analysis_identity(
        "creative.png", b"abc", {**options, "detector": True}
    )[1] != fingerprint


def test_benchmark_loader_supports_results_layout_and_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "results" / "evaluation"
    demos = tmp_path / "results" / "demo_cases"
    evaluation.mkdir(parents=True)
    demos.mkdir(parents=True)
    (evaluation / "metrics.json").write_text(json.dumps({"samples": 48}), encoding="utf-8")
    (evaluation / "evaluation_metrics.json").write_text(
        json.dumps({"dataset": {"rows": 288}}), encoding="utf-8"
    )
    (evaluation / "external_spot_check.json").write_text("{broken", encoding="utf-8")
    (demos / "case_summary.json").write_text(json.dumps([{"case_id": "safe"}]), encoding="utf-8")

    bundle = app_module._load_benchmark_from_directories(
        app_module._evaluation_directories(tmp_path), app_module._demo_directories(tmp_path)
    )

    assert bundle["available"] is True
    assert bundle["formal"]["samples"] == 48
    assert bundle["evaluation"]["dataset"]["rows"] == 288
    assert bundle["external"] == {}
    assert bundle["demo_cases"][0]["case_id"] == "safe"


def test_saved_benchmark_scope_and_chart_values_match_artifacts() -> None:
    bundle = app_module._load_benchmark_from_directories(
        app_module._evaluation_directories(), app_module._demo_directories()
    )

    assert bundle["formal"]["samples"] == 48
    assert bundle["formal"]["macro_f1"] == pytest.approx(0.9791304347826086)
    assert bundle["external"]["sample_count"] == 26
    assert bundle["external"]["classifier_macro_f1"] == pytest.approx(0.5548871063576946)
    assert bundle["external"]["weapon_detector_nonweapon_false_positive_rate"] == pytest.approx(0.9)

    quality = app_module.build_model_quality_figure(bundle)
    latency = app_module.build_model_latency_figure(bundle)
    dataset = app_module.build_dataset_figure(bundle)
    confusion = app_module.build_confusion_figure(bundle)
    recalls = app_module.build_threshold_recall_figure(bundle)

    assert len(quality.data) == 2
    assert quality.data[1].y[0] == pytest.approx(0.9791304347826086)
    assert latency.data[0].x[0] == pytest.approx(71.52729103690945)
    assert sum(int(value) for trace in dataset.data for value in trace.x) == 288
    assert sum(sum(int(value) for value in row) for row in confusion.data[0].z) == 48
    assert list(recalls.data[0].x) == pytest.approx([11 / 12, 11 / 12, 1.0])


def test_score_chart_does_not_assume_policy_focus_is_numeric_maximum() -> None:
    result = SimpleNamespace(
        classifier_scores={
            "safe": 0.05,
            "firearms": 0.60,
            "explosives": 0.05,
            "financial_promotion": 0.30,
        },
        decision=SimpleNamespace(
            primary_label="financial_promotion",
            fused_scores={
                "safe": 0.05,
                "firearms": 0.60,
                "explosives": 0.05,
                "financial_promotion": 0.45,
            },
        ),
    )

    figure = app_module.build_score_figure(result, {"model_labels": list(app_module.LABEL_ORDER)})

    assert len(figure.data) == 2
    assert list(figure.data[0].x) == pytest.approx([0.05, 0.60, 0.05, 0.30])
    assert list(figure.data[1].x) == pytest.approx([0.05, 0.60, 0.05, 0.45])
    assert figure.layout.shapes in (None, ())


def test_error_message_hides_internal_exception_text_and_css_reduces_motion() -> None:
    message = app_module._analysis_error_message(RuntimeError("/private/tmp/secret-model.bin"))

    assert "secret-model" not in message
    re_match = re.search(r"Reference [0-9A-F]{8}", message)
    assert re_match is not None
    assert "prefers-reduced-motion:reduce" in APP_PATH.read_text(encoding="utf-8")
