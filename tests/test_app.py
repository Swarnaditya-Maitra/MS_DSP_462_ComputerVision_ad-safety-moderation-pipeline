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


def test_new_upload_identity_clears_previous_analysis_before_decode() -> None:
    state_store = {
        "analysis_state": {
            "source_id": "old.png:old-sha",
            "result": object(),
        }
    }

    state = app_module._analysis_state_for_source(state_store, "corrupt.png:new-sha")

    assert state is None
    assert "analysis_state" not in state_store


def test_empty_uploader_preserves_saved_analysis_state_for_workspace_navigation() -> None:
    saved_state = {
        "source_id": "creative.png:sha",
        "result": object(),
    }
    state_store = {"analysis_state": saved_state, "unrelated": "preserved"}

    state = app_module._analysis_state_for_upload(state_store, None)

    assert state is saved_state
    assert state_store["analysis_state"] is saved_state
    assert state_store["unrelated"] == "preserved"


def test_benchmark_loader_supports_canonical_outputs_and_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "outputs" / "evaluation"
    demos = tmp_path / "outputs" / "demo_cases"
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


def test_accessible_confusion_table_has_all_cells_and_matches_test_sample() -> None:
    bundle = app_module._load_benchmark_from_directories(
        app_module._evaluation_directories(), app_module._demo_directories()
    )

    rows = app_module._confusion_rows(bundle)

    labels = {app_module._compact_label(label) for label in app_module.LABEL_ORDER}
    assert len(rows) == 16
    assert {(row["actual"], row["predicted"]) for row in rows} == {
        (actual, predicted) for actual in labels for predicted in labels
    }
    assert sum(row["count"] for row in rows) == 48


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


def test_formal_kpi_scope_is_caption_not_metric_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []
    captions: list[str] = []

    class FakeColumn:
        def metric(self, *args: object, **kwargs: object) -> None:
            metrics.append((args, kwargs))

        def caption(self, text: str) -> None:
            captions.append(text)

    monkeypatch.setattr(app_module.st, "columns", lambda count: [FakeColumn() for _ in range(count)])
    bundle = {
        "formal": {"macro_f1": 0.979, "restricted_recall": 0.944},
        "evaluation": {
            "models": [
                {
                    "key": "vit",
                    "test_metrics": {"per_class": {"safe": {"precision": 1.0}}},
                    "cpu_batch1_benchmark": {"p95_ms": 71.5},
                }
            ]
        },
    }

    app_module._render_formal_kpis(bundle)

    assert len(metrics) == 4
    assert all(len(args) == 2 and "delta" not in kwargs for args, kwargs in metrics)
    assert captions == [
        "Scope: n=48 untouched test images",
        "Scope: 12 safe test images",
        "Scope: argmax across 36 restricted images",
        "Scope: CPU batch 1 only",
    ]


def test_result_kpi_context_is_caption_not_metric_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []
    captions: list[str] = []

    class StopAfterKpis(Exception):
        pass

    class FakeColumn:
        def metric(self, *args: object, **kwargs: object) -> None:
            metrics.append((args, kwargs))

        def caption(self, text: str) -> None:
            captions.append(text)

    def stop_before_tabs(*args: object, **kwargs: object) -> None:
        raise StopAfterKpis

    monkeypatch.setattr(app_module.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module.st, "columns", lambda count: [FakeColumn() for _ in range(count)])
    monkeypatch.setattr(app_module.st, "tabs", stop_before_tabs)
    result = SimpleNamespace(
        decision=SimpleNamespace(
            verdict="REVIEW",
            primary_label="firearms",
            primary_score=0.72,
            fused_scores={
                "safe": 0.10,
                "firearms": 0.72,
                "explosives": 0.08,
                "financial_promotion": 0.10,
            },
        ),
        latency_ms={"total": 84.4},
        detections=[object(), object()],
        ocr=SimpleNamespace(tokens=["offer"], engine="tesseract"),
    )

    with pytest.raises(StopAfterKpis):
        app_module._render_results(
            result,
            {"label_display_names": {}},
            {},
            None,
            {},
            stale=False,
        )

    assert len(metrics) == 5
    assert all(len(args) == 2 and "delta" not in kwargs for args, kwargs in metrics)
    assert captions == [
        "Scope: one local run",
        "Value: 72.0% fused evidence",
        "Meaning: fused-score gap",
        "Scope: raw cues, not unique objects",
        "Engine: tesseract",
    ]
