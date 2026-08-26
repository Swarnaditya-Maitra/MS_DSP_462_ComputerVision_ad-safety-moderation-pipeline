from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import plotly.graph_objects as go
import streamlit as st
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ad_safety.paths import CONFIG_DIR, MODEL_DIR  # noqa: E402
from ad_safety.model_assets import (  # noqa: E402
    DETECTOR_REPO_ID,
    VISUAL_BACKBONE_REPO_ID,
    model_snapshot_ready,
    trained_head_ready,
)
from ad_safety.preprocessing import ImageValidationError, load_image  # noqa: E402


st.set_page_config(
    page_title="Ad Safety Studio",
    page_icon="AS",
    layout="wide",
    initial_sidebar_state="expanded",
)


LABEL_ORDER = ("safe", "firearms", "explosives", "financial_promotion")
SPLIT_ORDER = ("train", "val", "test")
CHART_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


@st.cache_data(show_spinner=False)
def load_policy_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "policy.yaml").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_engine() -> Any:
    # Delayed import keeps the navigation shell responsive and avoids model
    # construction until a user explicitly runs an analysis.
    from ad_safety.inference import ModerationEngine

    return ModerationEngine(enable_detector=True)


def _evaluation_directories(root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    return (root / "outputs" / "evaluation", root / "results" / "evaluation")


def _demo_directories(root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    return (root / "outputs" / "demo_cases", root / "results" / "demo_cases")


def _artifact_signature(paths: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        if path.is_file():
            stat = path.stat()
            signature.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def _read_first_json(filename: str, directories: Sequence[Path]) -> tuple[Any, str | None]:
    for directory in directories:
        path = directory / filename
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8")), str(path)
            except (OSError, json.JSONDecodeError):
                continue
    return None, None


def _load_benchmark_from_directories(
    evaluation_directories: Sequence[Path],
    demo_directories: Sequence[Path],
) -> dict[str, Any]:
    formal, formal_path = _read_first_json("metrics.json", evaluation_directories)
    evaluation, evaluation_path = _read_first_json("evaluation_metrics.json", evaluation_directories)
    external, external_path = _read_first_json("external_spot_check.json", evaluation_directories)
    demo_cases, demo_path = _read_first_json("case_summary.json", demo_directories)
    return {
        "formal": formal or {},
        "evaluation": evaluation or {},
        "external": external or {},
        "demo_cases": demo_cases or [],
        "paths": {
            "formal": formal_path,
            "evaluation": evaluation_path,
            "external": external_path,
            "demo_cases": demo_path,
        },
        "available": bool(formal and evaluation),
    }


@st.cache_data(show_spinner=False)
def _load_benchmark_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _load_benchmark_from_directories(_evaluation_directories(), _demo_directories())


def load_benchmark_data() -> dict[str, Any]:
    watched = [
        *(directory / name for directory in _evaluation_directories() for name in (
            "metrics.json",
            "evaluation_metrics.json",
            "external_spot_check.json",
        )),
        *(directory / "case_summary.json" for directory in _demo_directories()),
    ]
    return _load_benchmark_cached(_artifact_signature(watched))


def _safe_download_stem(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_")
    return stem[:60] or "image"


def _analysis_identity(
    filename: str,
    payload: bytes,
    options: Mapping[str, bool],
) -> tuple[str, str]:
    payload_sha = hashlib.sha256(payload).hexdigest()
    source_id = f"{Path(filename).name}:{payload_sha}"
    fingerprint = json.dumps({"source": source_id, **dict(options)}, sort_keys=True)
    return source_id, fingerprint


def _analysis_state_for_source(
    state_store: MutableMapping[str, Any], source_id: str
) -> Mapping[str, Any] | None:
    """Drop a saved decision immediately when the uploaded bytes or name change."""
    state = state_store.get("analysis_state")
    if state and state.get("source_id") != source_id:
        state_store.pop("analysis_state", None)
        return None
    return state


def _analysis_state_for_upload(
    state_store: MutableMapping[str, Any], uploaded: Any | None
) -> Mapping[str, Any] | None:
    """Clear a saved decision when the uploader no longer contains a file."""
    if uploaded is None:
        state_store.pop("analysis_state", None)
        return None
    return state_store.get("analysis_state")


def _display_label(config: Mapping[str, Any], label: str) -> str:
    return str(config.get("label_display_names", {}).get(label, label.replace("_", " ").title()))


def _compact_label(label: str) -> str:
    return {
        "safe": "Safe",
        "firearms": "Firearms",
        "explosives": "Explosives",
        "financial_promotion": "Financial",
    }.get(label, label.replace("_", " ").title())


def _reason_text(result: Any, config: Mapping[str, Any]) -> list[str]:
    reasons = list(result.decision.review_reasons)
    if reasons:
        return reasons
    thresholds = getattr(result, "applied_thresholds", config.get("default_thresholds", {}))
    return [
        "No restricted signal crossed an applied policy rule, and the top-two score margin "
        f"cleared {float(thresholds.get('uncertainty_margin', 0.12)):.2f}."
    ]


def _analysis_error_message(exc: Exception) -> str:
    reference = hashlib.sha256(
        f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
    ).hexdigest()[:8].upper()
    return (
        "A local analysis component could not complete. Check model readiness or disable optional "
        f"evidence sources, then retry. Reference {reference}."
    )


def _audit_payload(
    result: Any,
    *,
    filename: str,
    payload: bytes,
    run_detector: bool,
    run_ocr: bool,
    explain: bool,
) -> dict[str, Any]:
    audit = result.audit_record()
    return {
        "audit_schema_version": "1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "filename": Path(filename).name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "width": result.input_size[0],
            "height": result.input_size[1],
        },
        "options": {"detector": run_detector, "ocr": run_ocr, "explanation": explain},
        "analysis": audit,
    }


def _display_thumbnail(image: Image.Image, max_side: int = 1400) -> Image.Image:
    thumbnail = image.copy()
    thumbnail.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return thumbnail


def _model_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in bundle.get("evaluation", {}).get("models", []):
        test = model.get("test_metrics", {})
        benchmark = model.get("cpu_batch1_benchmark", {})
        key = str(model.get("key", "model"))
        rows.append(
            {
                "key": key,
                "model": "ViT" if key == "vit" else "ResNet50" if key.startswith("resnet") else key,
                "accuracy": float(test.get("accuracy", 0.0)),
                "macro_f1": float(test.get("macro_f1", 0.0)),
                "p95_ms": float(benchmark.get("p95_ms", 0.0)),
            }
        )
    return rows


def _chart_style(fig: go.Figure, *, height: int, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 17}},
        height=height,
        margin={"l": 28, "r": 24, "t": 64, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Arial, sans-serif", "color": "#26354f", "size": 12},
        hoverlabel={"bgcolor": "#14213d", "font": {"color": "white"}},
        legend={"orientation": "h", "y": 1.08, "x": 1.0, "xanchor": "right"},
    )
    fig.update_xaxes(gridcolor="rgba(95,107,122,.14)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(95,107,122,.10)", zeroline=False)
    return fig


def build_model_quality_figure(bundle: Mapping[str, Any]) -> go.Figure:
    rows = _model_rows(bundle)
    if not rows:
        raise ValueError("Model comparison data is unavailable.")
    fig = go.Figure()
    for key, label, color in (
        ("accuracy", "Accuracy", "#2f6fed"),
        ("macro_f1", "Macro F1", "#31b7a5"),
    ):
        fig.add_trace(
            go.Bar(
                name=label,
                x=[row["model"] for row in rows],
                y=[row[key] for row in rows],
                text=[f"{row[key]:.1%}" for row in rows],
                textposition="outside",
                marker_color=color,
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    return _chart_style(fig, height=330, title="Model quality on the untouched test set")


def build_model_latency_figure(bundle: Mapping[str, Any]) -> go.Figure:
    rows = _model_rows(bundle)
    if not rows:
        raise ValueError("Model latency data is unavailable.")
    fig = go.Figure(
        go.Bar(
            x=[row["p95_ms"] for row in rows],
            y=[row["model"] for row in rows],
            orientation="h",
            text=[f"{row['p95_ms']:.1f} ms" for row in rows],
            textposition="outside",
            marker_color=["#2f6fed", "#91a8cf"][: len(rows)],
            hovertemplate="%{y}<br>Classifier p95: %{x:.1f} ms<extra></extra>",
        )
    )
    maximum = max(row["p95_ms"] for row in rows)
    fig.update_xaxes(title="Milliseconds", range=[0, maximum * 1.18 if maximum else 1])
    return _chart_style(fig, height=330, title="Classifier-only CPU latency")


def build_dataset_figure(bundle: Mapping[str, Any]) -> go.Figure:
    counts = bundle.get("evaluation", {}).get("dataset", {}).get("counts", {})
    if not counts:
        raise ValueError("Dataset composition data is unavailable.")
    fig = go.Figure()
    colors = {"train": "#2f6fed", "val": "#31b7a5", "test": "#ffb44a"}
    for split in SPLIT_ORDER:
        values = [int(counts.get(label, {}).get(split, 0)) for label in LABEL_ORDER]
        fig.add_trace(
            go.Bar(
                name={"train": "Train", "val": "Validation", "test": "Test"}[split],
                x=values,
                y=[_compact_label(label) for label in LABEL_ORDER],
                orientation="h",
                text=values,
                textposition="inside",
                marker_color=colors[split],
                hovertemplate="%{y}<br>%{fullData.name}: %{x}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Images")
    fig.update_yaxes(autorange="reversed")
    return _chart_style(fig, height=350, title="Grouped split by policy label")


def build_confusion_figure(bundle: Mapping[str, Any]) -> go.Figure:
    models = bundle.get("evaluation", {}).get("models", [])
    selected = next((model for model in models if model.get("key") == "vit"), None)
    matrix = (selected or {}).get("test_metrics", {}).get("confusion_matrix")
    if not matrix:
        raise ValueError("Confusion matrix data is unavailable.")
    labels = [_compact_label(label) for label in LABEL_ORDER]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=matrix,
            texttemplate="%{text}",
            colorscale=[[0, "#f3f6fb"], [0.2, "#c8d8f5"], [1, "#2f6fed"]],
            showscale=False,
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicted label", side="bottom")
    fig.update_yaxes(title="Actual label", autorange="reversed")
    return _chart_style(fig, height=350, title="ViT confusion matrix, n=48")


def _confusion_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = bundle.get("evaluation", {}).get("models", [])
    selected = next((model for model in models if model.get("key") == "vit"), None)
    matrix = (selected or {}).get("test_metrics", {}).get("confusion_matrix")
    if not matrix:
        return []
    labels = [_compact_label(label) for label in LABEL_ORDER]
    return [
        {"actual": actual, "predicted": predicted, "count": int(matrix[row][column])}
        for row, actual in enumerate(labels)
        for column, predicted in enumerate(labels)
    ]


def build_threshold_recall_figure(bundle: Mapping[str, Any]) -> go.Figure:
    points = bundle.get("formal", {}).get("threshold_operating_points", {})
    labels = ("firearms", "explosives", "financial_promotion")
    if not points:
        raise ValueError("Threshold recall data is unavailable.")
    recalls = [float(points.get(label, {}).get("recall", 0.0)) for label in labels]
    fig = go.Figure(
        go.Bar(
            x=recalls,
            y=[_compact_label(label) for label in labels],
            orientation="h",
            text=[f"{value:.1%}" for value in recalls],
            textposition="inside",
            marker_color=["#31b7a5" if value >= 0.9 else "#e35d6a" for value in recalls],
            hovertemplate="%{y}<br>Untouched-test recall: %{x:.1%}<extra></extra>",
        )
    )
    fig.add_vline(
        x=0.90,
        line_width=2,
        line_dash="dash",
        line_color="#7a4b00",
        annotation_text="Calibration target 90%",
        annotation_position="top left",
    )
    fig.update_xaxes(range=[0, 1.03], tickformat=".0%")
    return _chart_style(fig, height=290, title="Restricted-class recall at validation-selected thresholds")


def build_score_figure(result: Any, config: Mapping[str, Any]) -> go.Figure:
    labels = list(config.get("model_labels", LABEL_ORDER))
    display = [_compact_label(label) for label in labels]
    classifier = [float(result.classifier_scores.get(label, 0.0)) for label in labels]
    fused = [float(result.decision.fused_scores.get(label, 0.0)) for label in labels]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Raw classifier",
            x=classifier,
            y=display,
            orientation="h",
            marker_color="#9db3d7",
            text=[f"{value:.1%}" for value in classifier],
            textposition="inside",
            hovertemplate="%{y}<br>Raw classifier: %{x:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Fused evidence",
            x=fused,
            y=display,
            orientation="h",
            marker_color="#2f6fed",
            text=[f"{value:.1%}" for value in fused],
            textposition="inside",
            hovertemplate="%{y}<br>Fused evidence: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return _chart_style(fig, height=390, title="Raw model scores and fused policy evidence")


def build_latency_figure(result: Any) -> go.Figure:
    stages = ("classification", "detection", "ocr", "explanation")
    labels = [stage.title() for stage in stages]
    values = [max(0.0, float(result.latency_ms.get(stage, 0.0))) for stage in stages]
    total = max(0.0, float(result.latency_ms.get("total", 0.0)))
    overhead = max(0.0, total - sum(values))
    labels.append("Other overhead")
    values.append(overhead)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#2f6fed", "#8f6bea", "#31b7a5", "#ffb44a", "#91a0b5"],
            text=[f"{value:.1f} ms" for value in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f} ms<extra></extra>",
        )
    )
    fig.update_xaxes(title="Milliseconds", rangemode="tozero")
    return _chart_style(fig, height=340, title="Observed latency by stage")


def _render_header(*, model_ready: bool, evidence_ready: bool) -> None:
    model_state = "MODEL READY" if model_ready else "MODEL SETUP NEEDED"
    evidence_state = "EVIDENCE LOADED" if evidence_ready else "EVIDENCE UNAVAILABLE"
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-copy">
            <div class="eyebrow">LOCAL POLICY TRIAGE</div>
            <h1>Ad Safety Studio</h1>
            <p>Move one creative from upload to an auditable policy decision. Inspect raw model
            scores, fused evidence, stage timing, overlays, and the exact thresholds applied.</p>
          </div>
          <div class="hero-pills" aria-label="System status">
            <span class="pill pill-neutral">LOCAL PROCESSING</span>
            <span class="pill {'pill-good' if model_ready else 'pill-warn'}">{model_state}</span>
            <span class="pill {'pill-good' if evidence_ready else 'pill-warn'}">{evidence_state}</span>
            <span class="pill pill-risk">NOT PRODUCTION-READY</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline_cards() -> None:
    st.markdown("### One clear path from image to action")
    columns = st.columns(4)
    cards = [
        ("01", "Upload", "Validate one static JPEG, PNG, or WebP without saving it."),
        ("02", "Score", "Run the frozen ViT feature extractor and trained policy head."),
        ("03", "Enrich", "Optionally add OCR, object context, and an occlusion view."),
        ("04", "Decide", "Apply versioned rules and export a case-level audit record."),
    ]
    for column, (number, title, body) in zip(columns, cards):
        column.markdown(
            f"""
            <div class="pipeline-card">
              <span>{number}</span><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_sidebar(config: Mapping[str, Any], bundle: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    st.sidebar.markdown("## Evidence settings")
    st.sidebar.caption("The core classifier always runs. Optional evidence can add seconds.")
    run_detector = st.sidebar.toggle(
        "Object context",
        value=False,
        help="Adds Grounding DINO box and phrase cues. This is diagnostic and substantially slower.",
    )
    run_ocr = st.sidebar.toggle(
        "Text extraction",
        value=True,
        help="Runs local Tesseract OCR and fuses configured policy terms.",
    )
    explain = st.sidebar.toggle(
        "Occlusion view",
        value=False,
        help="Masks image regions and reruns the classifier. This is qualitative and slower.",
    )

    enabled = ["classifier"]
    if run_detector:
        enabled.append("object context")
    if run_ocr:
        enabled.append("OCR")
    if explain:
        enabled.append("occlusion")
    st.sidebar.caption("Active: " + " + ".join(enabled))

    st.sidebar.markdown("---")
    st.sidebar.markdown("## System readiness")
    artifact = MODEL_DIR / "vit_policy_head.joblib"
    classifier_ready = trained_head_ready(artifact)
    backbone_ready = model_snapshot_ready(VISUAL_BACKBONE_REPO_ID)
    detector_ready = model_snapshot_ready(DETECTOR_REPO_ID)
    if classifier_ready:
        st.sidebar.success("Classifier head ready")
    else:
        st.sidebar.warning("Classifier head missing or invalid")
    if backbone_ready:
        st.sidebar.success("ViT backbone snapshot ready")
    else:
        st.sidebar.warning("ViT backbone snapshot missing")
    if detector_ready:
        st.sidebar.success("Detector snapshot ready")
    elif run_detector:
        st.sidebar.warning("Detector snapshot missing")
    else:
        st.sidebar.info("Detector snapshot unavailable; object context is off")
    if shutil.which("tesseract"):
        st.sidebar.success("Tesseract available")
    else:
        st.sidebar.info("Tesseract unavailable; OCR returns no text")
    if bundle.get("available"):
        st.sidebar.success("Saved evaluation evidence loaded")
    else:
        st.sidebar.warning("Saved evaluation evidence unavailable")

    st.sidebar.markdown("### Privacy")
    st.sidebar.info(
        "Uploads stay inside this host process. The app does not send images to a third-party "
        "inference API or save them unless you download the audit file."
    )
    with st.sidebar.expander("Pilot limitations"):
        for limitation in config.get("limitations", []):
            st.markdown(f"- {limitation}")
    return run_detector, run_ocr, explain


def _render_formal_kpis(bundle: Mapping[str, Any]) -> None:
    formal = bundle.get("formal", {})
    models = bundle.get("evaluation", {}).get("models", [])
    vit = next((model for model in models if model.get("key") == "vit"), {})
    safe_precision = vit.get("test_metrics", {}).get("per_class", {}).get("safe", {}).get("precision")
    p95 = vit.get("cpu_batch1_benchmark", {}).get("p95_ms")
    columns = st.columns(4)
    columns[0].metric("Formal macro F1", f"{float(formal.get('macro_f1', 0)):.1%}")
    columns[0].caption("Scope: n=48 untouched test images")
    columns[1].metric("Safe precision", f"{float(safe_precision or 0):.1%}")
    columns[1].caption("Scope: 12 safe test images")
    columns[2].metric("Restricted recall", f"{float(formal.get('restricted_recall', 0)):.1%}")
    columns[2].caption("Scope: argmax across 36 restricted images")
    columns[3].metric("Classifier p95", f"{float(p95 or 0):.1f} ms")
    columns[3].caption("Scope: CPU batch 1 only")


def _render_pilot_evidence(bundle: Mapping[str, Any]) -> None:
    st.markdown("## Pilot evidence")
    if not bundle.get("available"):
        st.warning("Evaluation artifacts are unavailable. I will not substitute hard-coded metrics.")
        return
    st.warning(
        "Limited benchmark: the 288-image dataset is source-confounded and the untouched test set has "
        "only 12 images per class. These figures do not establish production readiness."
    )
    _render_formal_kpis(bundle)

    quality_column, latency_column = st.columns(2)
    with quality_column:
        st.plotly_chart(build_model_quality_figure(bundle), width="stretch", config=CHART_CONFIG, key="quality_chart")
    with latency_column:
        st.plotly_chart(build_model_latency_figure(bundle), width="stretch", config=CHART_CONFIG, key="model_latency_chart")
    st.caption(
        "Quality and latency use separate charts because they have different units. Latency covers decoding-free "
        "classifier execution only, not OCR, detection, transfer, or interface rendering."
    )

    dataset_column, confusion_column = st.columns(2)
    with dataset_column:
        st.plotly_chart(build_dataset_figure(bundle), width="stretch", config=CHART_CONFIG, key="dataset_chart")
    with confusion_column:
        st.plotly_chart(build_confusion_figure(bundle), width="stretch", config=CHART_CONFIG, key="confusion_chart")
    st.caption(
        "The split has no source-group or exact-hash overlap, but class labels remain tied to source or generator style."
    )

    st.plotly_chart(build_threshold_recall_figure(bundle), width="stretch", config=CHART_CONFIG, key="recall_chart")
    st.caption(
        "The dashed line is the validation selection rule of at least 90% recall. A future MVP gate should be stricter "
        "and tested on a larger independent set."
    )

    external = bundle.get("external", {})
    if external:
        st.markdown("### External domain-shift diagnostic")
        st.error(
            "The 26-image Wikimedia spot check breaks the formal-test confidence. It is a small, selected diagnostic, "
            "not a second test set and not directly comparable to the formal benchmark."
        )
        columns = st.columns(5)
        columns[0].metric("Accuracy", f"{float(external.get('classifier_accuracy', 0)):.1%}")
        columns[1].metric("Macro F1", f"{float(external.get('classifier_macro_f1', 0)):.1%}")
        columns[2].metric("Safe non-approve", f"{float(external.get('safe_policy_non_approve_rate', 0)):.1%}")
        columns[3].metric("Detector nonweapon FPR", f"{float(external.get('weapon_detector_nonweapon_false_positive_rate', 0)):.1%}")
        columns[4].metric("Sample", str(int(external.get("sample_count", 0))))

    with st.expander("Accessible chart data"):
        st.markdown("**Model comparison**")
        st.dataframe(_model_rows(bundle), hide_index=True, width="stretch")
        st.markdown("**Threshold operating points**")
        st.dataframe(
            [
                {"label": _compact_label(label), **values}
                for label, values in bundle.get("formal", {}).get("threshold_operating_points", {}).items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Dataset counts**")
        counts = bundle.get("evaluation", {}).get("dataset", {}).get("counts", {})
        st.dataframe(
            [{"label": _compact_label(label), **counts.get(label, {})} for label in LABEL_ORDER],
            hide_index=True,
            width="stretch",
        )
        st.markdown("**ViT confusion matrix**")
        st.dataframe(_confusion_rows(bundle), hide_index=True, width="stretch")


def _threshold_rows(thresholds: Mapping[str, float]) -> list[dict[str, Any]]:
    return [
        {"rule": "Firearms block", "signal": "Raw classifier, top class", "threshold": float(thresholds.get("firearms", 0.0))},
        {"rule": "Explosives block", "signal": "Raw classifier, top class", "threshold": float(thresholds.get("explosives", 0.0))},
        {"rule": "Financial review", "signal": "Fused evidence", "threshold": float(thresholds.get("financial_promotion", 0.0))},
        {"rule": "Restricted review", "signal": "Maximum restricted fused evidence", "threshold": float(thresholds.get("review", 0.0))},
        {"rule": "Uncertainty review", "signal": "Top-two fused-score margin below", "threshold": float(thresholds.get("uncertainty_margin", 0.0))},
    ]


def _render_how_it_works(config: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
    st.markdown("## How the pilot works")
    _render_pipeline_cards()
    st.markdown("### Decision rules")
    saved_thresholds = bundle.get("formal", {}).get("thresholds_selected_on_validation_only", {})
    thresholds = dict(config.get("default_thresholds", {}))
    thresholds.update(saved_thresholds)
    st.dataframe(_threshold_rows(thresholds), hide_index=True, width="stretch")
    st.caption(
        "The table combines the saved ViT validation-selected classifier thresholds with policy defaults for review "
        "and uncertainty. Each completed audit records the exact thresholds applied by the loaded artifact."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("### What the charts mean")
        st.markdown(
            "- Raw classifier scores come from the four-way model head.\n"
            "- Fused evidence can add OCR and the strongest detector cue per policy label.\n"
            "- A fused score is a triage signal, not a calibrated probability of illegality.\n"
            "- The policy-selected focus can differ from the numerically highest fused score."
        )
    with right:
        st.markdown("### Scope boundary")
        st.markdown(
            "- One static creative at a time. Animated images are rejected.\n"
            "- No landing page, audience, jurisdiction, or campaign history.\n"
            "- Review cases require a human decision maker.\n"
            "- Session charts are not production monitoring or population evidence."
        )


def _action_copy(verdict: str) -> str:
    return {
        "APPROVE": "Passes the current pilot rules. This is not a legal or platform-compliance clearance.",
        "REVIEW": "Hold publication and route the creative to a human policy reviewer.",
        "BLOCK": "Do not auto-publish in this pilot. Preserve the audit record for review and appeal.",
    }.get(verdict, "Route the case to a human reviewer.")


def _render_decision_trace(result: Any, config: Mapping[str, Any]) -> None:
    ranked = sorted(result.decision.fused_scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    margin = top_score - ranked[1][1] if len(ranked) > 1 else top_score
    policy_label = _display_label(config, result.decision.primary_label)
    st.markdown(
        f"""
        <div class="trace-grid">
          <div class="trace-card"><span>1 · TOP NUMERIC SIGNAL</span><strong>{html.escape(_display_label(config, top_label))}</strong><p>{top_score:.1%} fused evidence</p></div>
          <div class="trace-card"><span>2 · SEPARATION</span><strong>{margin:.1%}</strong><p>top-two fused-score margin</p></div>
          <div class="trace-card"><span>3 · POLICY FOCUS</span><strong>{html.escape(policy_label)}</strong><p>selected by the rule path</p></div>
          <div class="trace-card trace-accent"><span>4 · ACTION</span><strong>{html.escape(result.decision.verdict)}</strong><p>{html.escape(_action_copy(result.decision.verdict))}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_results(
    result: Any,
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    preview: Image.Image,
    audit: Mapping[str, Any],
    *,
    stale: bool,
) -> None:
    if stale:
        st.warning("Evidence settings changed after this run. The saved result remains visible, but rerun analysis before acting on it.")

    verdict = result.decision.verdict
    label = _display_label(config, result.decision.primary_label)
    st.markdown(
        f"""
        <div class="verdict verdict-{verdict.casefold()}">
          <div><span>POLICY DECISION</span><strong>{html.escape(verdict)}</strong></div>
          <div><span>POLICY FOCUS</span><strong>{html.escape(label)}</strong></div>
          <div><span>POLICY EVIDENCE SCORE</span><strong>{result.decision.primary_score:.1%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ranked = sorted(result.decision.fused_scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    margin = top_score - ranked[1][1] if len(ranked) > 1 else top_score
    primary_metrics = st.columns(3)
    primary_metrics[0].metric("Total latency", f"{result.latency_ms['total']:.0f} ms")
    primary_metrics[0].caption("Scope: one local run")
    primary_metrics[1].metric("Top numeric signal", _compact_label(top_label))
    primary_metrics[1].caption(f"Value: {top_score:.1%} fused evidence")
    primary_metrics[2].metric("Top-two margin", f"{margin:.1%}")
    primary_metrics[2].caption("Meaning: fused-score gap")

    evidence_metrics = st.columns(2)
    evidence_metrics[0].metric("Detector boxes", len(result.detections))
    evidence_metrics[0].caption("Scope: raw cues, not unique objects")
    evidence_metrics[1].metric("Text tokens", len(result.ocr.tokens))
    evidence_metrics[1].caption(f"Engine: {result.ocr.engine}")

    overview_tab, evidence_tab, timing_tab, benchmark_tab, audit_tab = st.tabs(
        ["Decision", "Evidence", "Timing", "Benchmark", "Audit"]
    )
    with overview_tab:
        _render_decision_trace(result, config)
        st.plotly_chart(build_score_figure(result, config), width="stretch", config=CHART_CONFIG, key="result_score_chart")
        st.caption(
            "Fused evidence combines model output with optional cues. It is not a legal risk score or a calibrated "
            "probability. No single universal threshold applies to every bar."
        )
        st.markdown("#### Decision rationale")
        for reason in _reason_text(result, config):
            st.markdown(f"<div class='reason-row'>{html.escape(reason)}</div>", unsafe_allow_html=True)
        st.caption(
            f"Policy {result.decision.policy_version} | Model {result.decision.model_version}. "
            "Review and block remain bounded pilot actions."
        )
        with st.expander("Score table"):
            st.dataframe(
                [
                    {
                        "label": _display_label(config, raw_label),
                        "raw_classifier": round(float(result.classifier_scores.get(raw_label, 0.0)), 6),
                        "fused_evidence": round(float(result.decision.fused_scores.get(raw_label, 0.0)), 6),
                    }
                    for raw_label in config.get("model_labels", result.classifier_scores)
                ],
                hide_index=True,
                width="stretch",
            )

    with evidence_tab:
        image_columns = st.columns(2)
        with image_columns[0]:
            st.markdown("#### Source creative")
            st.image(_display_thumbnail(preview), width="stretch")
        with image_columns[1]:
            st.markdown("#### Evidence overlay")
            if result.detections or result.ocr.tokens:
                st.image(_display_thumbnail(result.annotated_image), width="stretch")
            else:
                st.info("No spatial evidence was returned. The overlay would be identical to the source image.")

        if result.heatmap_image is not None:
            st.markdown("#### Occlusion sensitivity")
            st.image(
                _display_thumbnail(result.heatmap_image),
                caption="Warm regions changed the selected class score most when hidden. This is qualitative, not causal evidence.",
                width="stretch",
            )

        objects_tab, ocr_tab = st.tabs(["Detector cues", "Extracted text"])
        with objects_tab:
            if result.detections:
                st.dataframe([item.to_dict() for item in result.detections], hide_index=True, width="stretch")
                st.caption("Overlapping boxes or phrases can refer to the same object. The policy keeps only the strongest cue per label.")
            else:
                st.caption("No detector cues were returned, or object context was disabled.")
        with ocr_tab:
            st.caption(f"OCR engine: {result.ocr.engine}")
            st.code(result.ocr.text or "No readable text returned.", language=None)
            if result.ocr.tokens:
                st.dataframe([token.to_dict() for token in result.ocr.tokens], hide_index=True, width="stretch")

    with timing_tab:
        st.plotly_chart(build_latency_figure(result), width="stretch", config=CHART_CONFIG, key="result_latency_chart")
        st.caption(
            "These values describe this single local run. They are not concurrent-load telemetry or a production service-level measurement."
        )
        st.dataframe(
            [
                {"stage": stage.replace("_", " ").title(), "milliseconds": round(value, 2)}
                for stage, value in result.latency_ms.items()
            ],
            hide_index=True,
            width="stretch",
        )

    with benchmark_tab:
        _render_pilot_evidence(bundle)

    with audit_tab:
        thresholds = getattr(result, "applied_thresholds", {})
        st.markdown("#### Exact thresholds applied")
        if thresholds:
            st.dataframe(_threshold_rows(thresholds), hide_index=True, width="stretch")
        else:
            st.warning("The loaded result does not expose an applied-threshold snapshot.")
        st.markdown("#### Case metadata")
        st.json({"input": audit["input"], "options": audit["options"]}, expanded=False)
        encoded = json.dumps(audit, indent=2, sort_keys=True).encode("utf-8")
        st.download_button(
            "Download audit JSON",
            data=encoded,
            file_name=f"ad_safety_audit_{_safe_download_stem(audit['input']['filename'])}.json",
            mime="application/json",
            type="primary",
            width="stretch",
            on_click="ignore",
        )


def _render_analyze(
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    run_detector: bool,
    run_ocr: bool,
    explain: bool,
) -> None:
    st.markdown("## Analyze one creative")
    st.markdown(
        "<div class='step-strip'><span><b>1</b> Upload</span><span><b>2</b> Configure</span><span><b>3</b> Review evidence</span><span><b>4</b> Export audit</span></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload one static JPEG, PNG, or WebP image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Maximum encoded size: 20 MB. Maximum decoded size: 40 megapixels. Animated images are rejected.",
    )
    _analysis_state_for_upload(st.session_state, uploaded)
    if uploaded is None:
        _render_pipeline_cards()
        st.info("Start with one creative. No upload leaves this device or persists after the session.")
        return

    payload = uploaded.getvalue()
    option_snapshot = {"detector": run_detector, "ocr": run_ocr, "explanation": explain}
    source_id, fingerprint = _analysis_identity(uploaded.name, payload, option_snapshot)
    state = _analysis_state_for_source(st.session_state, source_id)
    try:
        preview = load_image(payload)
    except ImageValidationError as exc:
        st.error(str(exc))
        return

    payload_sha = source_id.rsplit(":", 1)[1]
    st.caption(
        f"{Path(uploaded.name).name} | {len(payload) / 1024:.1f} KB | "
        f"{preview.width} x {preview.height} | SHA-256 {payload_sha[:12]}..."
    )

    run_clicked = st.button("Run policy analysis", type="primary", width="stretch")
    if run_clicked:
        artifact = MODEL_DIR / "vit_policy_head.joblib"
        if not trained_head_ready(artifact):
            st.error(
                "The trained classifier artifact models/vit_policy_head.joblib is missing or "
                "failed its integrity check. Restore the tracked file and run preflight before "
                "running moderation."
            )
            return
        if not model_snapshot_ready(VISUAL_BACKBONE_REPO_ID):
            st.error(
                "The required ViT backbone snapshot is incomplete. Complete the documented "
                "model setup before running moderation."
            )
            return
        if run_detector and not model_snapshot_ready(DETECTOR_REPO_ID):
            st.error(
                "Object context is enabled, but the local detector snapshot is incomplete. "
                "Complete the detector setup or turn off object context."
            )
            return
        try:
            with st.status("Running local policy analysis...", expanded=True) as status:
                st.write("Validated the static image and prepared the core classifier.")
                if run_detector:
                    st.write("Object context is enabled and may add several seconds.")
                if run_ocr:
                    st.write("Local text extraction is enabled.")
                if explain:
                    st.write("Occlusion sensitivity is enabled and reruns the classifier.")
                result = load_engine().analyze(
                    payload,
                    run_detector=run_detector,
                    run_ocr=run_ocr,
                    explain=explain,
                )
                status.update(label="Analysis complete", state="complete", expanded=False)
        except ImageValidationError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(_analysis_error_message(exc))
            return
        audit = _audit_payload(
            result,
            filename=uploaded.name,
            payload=payload,
            run_detector=run_detector,
            run_ocr=run_ocr,
            explain=explain,
        )
        state = {
            "source_id": source_id,
            "fingerprint": fingerprint,
            "result": result,
            "audit": audit,
        }
        st.session_state["analysis_state"] = state

    if state and state.get("source_id") == source_id:
        _render_results(
            state["result"],
            config,
            bundle,
            preview,
            state["audit"],
            stale=state.get("fingerprint") != fingerprint,
        )
    else:
        st.image(_display_thumbnail(preview), caption="Validated local preview", width="stretch")
        st.caption("Choose optional evidence in the sidebar, then run the analysis.")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink:#14213d; --muted:#5f6b7a; --paper:#f7f9fc; --card:#ffffff;
          --line:#dfe5ef; --blue:#2f6fed; --mint:#31b7a5; --amber:#c27a00; --red:#b3334d;
        }
        .stApp { background:radial-gradient(circle at 90% 0%,rgba(74,125,238,.14) 0,transparent 30%),var(--paper); }
        .block-container { max-width:1420px; padding-top:1.5rem; padding-bottom:4rem; }
        .hero { position:relative; overflow:hidden; padding:2rem 2.2rem; border:1px solid var(--line); border-radius:26px;
          background:linear-gradient(125deg,rgba(255,255,255,.98),rgba(232,240,255,.94));
          box-shadow:0 22px 60px rgba(30,55,95,.10); margin-bottom:1.1rem; animation:riseIn .42s ease-out both; }
        .hero::after { content:""; position:absolute; width:300px; height:300px; right:-115px; top:-155px;
          border-radius:50%; background:radial-gradient(circle,rgba(47,111,237,.24),rgba(47,111,237,0)); }
        .hero-copy { position:relative; z-index:1; }
        .hero .eyebrow { color:#3156b3; font-size:.78rem; font-weight:800; letter-spacing:.16em; }
        .hero h1 { color:var(--ink); font-size:clamp(2.35rem,5vw,4.7rem); line-height:1; margin:.55rem 0 .8rem; letter-spacing:-.05em; }
        .hero p { color:var(--muted); font-size:1.06rem; max-width:850px; margin:0; line-height:1.62; }
        .hero-pills { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.25rem; position:relative; z-index:1; }
        .pill { border-radius:999px; padding:.38rem .68rem; font-size:.72rem; font-weight:800; letter-spacing:.05em; border:1px solid; }
        .pill-neutral { color:#3156b3; background:#eef4ff; border-color:#bed0f4; }
        .pill-good { color:#17634f; background:#eaf8f3; border-color:#bce6d6; }
        .pill-warn { color:#7a4b00; background:#fff7df; border-color:#f0d68a; }
        .pill-risk { color:#8b2539; background:#fff0f2; border-color:#efbdc6; }
        .pipeline-card { min-height:178px; background:var(--card); border:1px solid var(--line); border-radius:18px;
          padding:1.3rem; box-shadow:0 10px 28px rgba(30,55,95,.055); animation:riseIn .42s ease-out both;
          transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease; }
        .pipeline-card:hover { transform:translateY(-3px); box-shadow:0 16px 35px rgba(30,55,95,.10); border-color:#b8c9e8; }
        .pipeline-card span { color:#3156b3; font-weight:800; font-size:.8rem; letter-spacing:.1em; }
        .pipeline-card h3 { color:var(--ink); margin:.65rem 0 .45rem; font-size:1.08rem; }
        .pipeline-card p { color:var(--muted); line-height:1.55; margin:0; font-size:.93rem; }
        .step-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin:.5rem 0 1rem; }
        .step-strip span { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.7rem .8rem; color:var(--muted); }
        .step-strip b { display:inline-grid; place-items:center; width:1.55rem; height:1.55rem; margin-right:.35rem;
          border-radius:50%; color:white; background:var(--blue); }
        .verdict { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; padding:1.25rem 1.4rem;
          border-radius:18px; margin:.7rem 0 1.3rem; border:1px solid; animation:riseIn .36s ease-out both; }
        .verdict > div { display:flex; flex-direction:column; gap:.3rem; min-width:0; }
        .verdict span { font-size:.75rem; font-weight:800; letter-spacing:.09em; opacity:.74; }
        .verdict strong { font-size:1.26rem; overflow-wrap:anywhere; }
        .verdict-approve { color:#17634f; background:#eaf8f3; border-color:#bce6d6; }
        .verdict-review { color:#7a4b00; background:#fff7df; border-color:#f0d68a; }
        .verdict-block { color:#92263a; background:#fff0f2; border-color:#efbdc6; }
        .trace-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem; margin:.7rem 0 1rem; }
        .trace-card { min-height:142px; padding:1rem; background:var(--card); border:1px solid var(--line); border-radius:15px; }
        .trace-card span { color:#55709f; font-size:.73rem; font-weight:800; letter-spacing:.06em; }
        .trace-card strong { display:block; color:var(--ink); font-size:1.06rem; margin:.55rem 0 .35rem; overflow-wrap:anywhere; }
        .trace-card p { color:var(--muted); line-height:1.45; margin:0; font-size:.88rem; }
        .trace-accent { border-color:#9cb9ee; background:linear-gradient(150deg,#f7faff,#edf3ff); }
        .reason-row { padding:.82rem .95rem; margin:.5rem 0; border-left:4px solid var(--blue);
          background:var(--card); border-radius:0 10px 10px 0; color:#26354f; }
        div[data-testid="stFileUploader"] { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:.65rem; }
        div[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:.82rem 1rem; min-height:112px; }
        div[data-testid="stPlotlyChart"] { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:.35rem; }
        button[kind="primary"] { box-shadow:0 9px 22px rgba(47,111,237,.20); }
        @keyframes riseIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        @media (max-width:900px) { .trace-grid { grid-template-columns:repeat(2,1fr); } .step-strip { grid-template-columns:repeat(2,1fr); } }
        @media (max-width:760px) { .verdict,.trace-grid,.step-strip { grid-template-columns:1fr; } .hero { padding:1.45rem; } }
        @media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none !important; transition:none !important; scroll-behavior:auto !important; } }
        @media (prefers-color-scheme:dark) {
          :root { --ink:#e8eefb; --muted:#bac6d8; --paper:#0e1524; --card:#151f31; --line:#2e3b51; }
          .hero { background:linear-gradient(125deg,rgba(21,31,49,.98),rgba(29,48,83,.95)); }
          .reason-row,.trace-card p,.pipeline-card p { color:var(--muted); }
          .trace-accent { background:#192944; }
          div[data-testid="stPlotlyChart"] { background:#f7f9fc; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    config = load_policy_config()
    bundle = load_benchmark_data()
    model_ready = trained_head_ready(
        MODEL_DIR / "vit_policy_head.joblib"
    ) and model_snapshot_ready(VISUAL_BACKBONE_REPO_ID)
    _render_header(model_ready=model_ready, evidence_ready=bool(bundle.get("available")))
    run_detector, run_ocr, explain = _render_sidebar(config, bundle)

    navigation = st.segmented_control(
        "Workspace",
        ["Analyze", "Pilot evidence", "How it works"],
        default="Analyze",
        label_visibility="collapsed",
        width="stretch",
        key="workspace_navigation",
    )
    if navigation == "Pilot evidence":
        _render_pilot_evidence(bundle)
    elif navigation == "How it works":
        _render_how_it_works(config, bundle)
    else:
        _render_analyze(
            config,
            bundle,
            run_detector=run_detector,
            run_ocr=run_ocr,
            explain=explain,
        )


if __name__ == "__main__":
    main()
