from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ad_safety.paths import CONFIG_DIR, MODEL_DIR  # noqa: E402
from ad_safety.preprocessing import ImageValidationError, load_image  # noqa: E402


st.set_page_config(
    page_title="Ad Safety Studio",
    page_icon="AS",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_policy_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "policy.yaml").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_engine() -> Any:
    # Importing here keeps the page shell fast and prevents model construction
    # until the user explicitly asks for an analysis.
    from ad_safety.inference import ModerationEngine

    return ModerationEngine(enable_detector=True)


def _safe_download_stem(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_")
    return stem[:60] or "image"


def _display_label(config: dict[str, Any], label: str) -> str:
    return str(config.get("label_display_names", {}).get(label, label.replace("_", " ").title()))


def _reason_text(result: Any, config: dict[str, Any]) -> list[str]:
    reasons = list(result.decision.review_reasons)
    if reasons:
        return reasons
    thresholds = config.get("default_thresholds", {})
    return [
        "No restricted signal crossed the review or block thresholds, and the leading score "
        f"cleared the {float(thresholds.get('uncertainty_margin', 0.12)):.2f} uncertainty margin."
    ]


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
        "audit_schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "filename": Path(filename).name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "width": result.input_size[0],
            "height": result.input_size[1],
        },
        "options": {
            "detector": run_detector,
            "ocr": run_ocr,
            "explanation": explain,
        },
        "analysis": audit,
    }


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">LOCAL POLICY TRIAGE</div>
          <h1>Ad Safety Studio</h1>
          <p>Screen ad creatives for firearms, explosives, and financial-promotion cues.
          Every decision keeps its evidence, thresholds, and timing visible.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline_empty_state() -> None:
    st.markdown("### What this pilot evaluates")
    columns = st.columns(3)
    cards = [
        (
            "01",
            "Visual policy model",
            "A frozen vision backbone and trained policy head produce four class scores.",
        ),
        (
            "02",
            "Optional evidence",
            "Grounding DINO object cues and local Tesseract text can raise a case for review.",
        ),
        (
            "03",
            "Auditable decision",
            "A versioned policy converts signals into approve, review, or block.",
        ),
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


def _render_score_rows(result: Any, config: dict[str, Any]) -> None:
    for label in config.get("model_labels", result.decision.fused_scores):
        score = float(result.decision.fused_scores.get(label, 0.0))
        st.progress(
            min(1.0, max(0.0, score)),
            text=f"{_display_label(config, label)}  {score:.1%}",
        )


def _render_results(result: Any, config: dict[str, Any], preview: Any, audit: dict[str, Any]) -> None:
    verdict = result.decision.verdict
    label = _display_label(config, result.decision.primary_label)
    st.markdown(
        f"""
        <div class="verdict verdict-{verdict.casefold()}">
          <div><span>POLICY DECISION</span><strong>{html.escape(verdict)}</strong></div>
          <div><span>LEADING SIGNAL</span><strong>{html.escape(label)}</strong></div>
          <div><span>EVIDENCE SCORE</span><strong>{result.decision.primary_score:.1%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total latency", f"{result.latency_ms['total']:.0f} ms")
    metric_columns[1].metric("Detected objects", len(result.detections))
    metric_columns[2].metric("OCR words", len(result.ocr.tokens))
    metric_columns[3].metric("Input size", f"{result.input_size[0]} x {result.input_size[1]}")

    image_columns = st.columns(2)
    with image_columns[0]:
        st.markdown("#### Source creative")
        st.image(preview, width="stretch")
    with image_columns[1]:
        st.markdown("#### Evidence overlay")
        st.image(result.annotated_image, width="stretch")

    if result.heatmap_image is not None:
        st.markdown("#### Occlusion sensitivity")
        st.image(
            result.heatmap_image,
            caption="Warm regions changed the selected class score most when hidden. This is qualitative evidence, not a causal explanation.",
            width="stretch",
        )

    detail_columns = st.columns([1.05, 0.95])
    with detail_columns[0]:
        st.markdown("#### Decision rationale")
        for reason in _reason_text(result, config):
            st.markdown(f"<div class='reason-row'>{html.escape(reason)}</div>", unsafe_allow_html=True)
        st.caption(
            f"Policy {result.decision.policy_version} | Model {result.decision.model_version}. "
            "Scores are triage signals and do not establish legality or platform compliance."
        )

    with detail_columns[1]:
        st.markdown("#### Fused evidence scores")
        _render_score_rows(result, config)

    with st.expander("Inspect evidence and timing"):
        evidence_tabs = st.tabs(["Objects", "OCR", "Latency", "Raw scores"])
        with evidence_tabs[0]:
            if result.detections:
                st.dataframe(
                    [item.to_dict() for item in result.detections],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption("No object evidence was returned, or the detector was disabled.")
        with evidence_tabs[1]:
            st.caption(f"OCR engine: {result.ocr.engine}")
            st.code(result.ocr.text or "No readable text returned.", language=None)
            if result.ocr.tokens:
                st.dataframe(
                    [token.to_dict() for token in result.ocr.tokens],
                    hide_index=True,
                    width="stretch",
                )
        with evidence_tabs[2]:
            st.dataframe(
                [
                    {"stage": stage.replace("_", " ").title(), "milliseconds": round(value, 2)}
                    for stage, value in result.latency_ms.items()
                ],
                hide_index=True,
                width="stretch",
            )
        with evidence_tabs[3]:
            rows = []
            for raw_label in config.get("model_labels", result.classifier_scores):
                rows.append(
                    {
                        "label": _display_label(config, raw_label),
                        "classifier_score": round(float(result.classifier_scores.get(raw_label, 0.0)), 6),
                        "fused_score": round(float(result.decision.fused_scores.get(raw_label, 0.0)), 6),
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")

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


def _render_sidebar(config: dict[str, Any]) -> tuple[bool, bool, bool]:
    st.sidebar.markdown("## Analysis controls")
    run_detector = st.sidebar.toggle(
        "Object detector",
        value=False,
        help="Adds local Grounding DINO object context. It increases latency and memory use.",
    )
    run_ocr = st.sidebar.toggle(
        "Text extraction",
        value=True,
        help="Runs local Tesseract OCR and fuses configured policy keywords.",
    )
    explain = st.sidebar.toggle(
        "Occlusion explanation",
        value=False,
        help="Masks image regions and reruns the classifier. This is substantially slower.",
    )

    st.sidebar.markdown("---")
    artifact = MODEL_DIR / "vit_policy_head.joblib"
    if artifact.is_file():
        st.sidebar.success("Classifier artifact ready")
    else:
        st.sidebar.warning("Classifier artifact missing")
    if shutil.which("tesseract"):
        st.sidebar.caption("Tesseract OCR is available locally.")
    else:
        st.sidebar.caption("Tesseract is unavailable. OCR will return no text.")

    st.sidebar.markdown("### Privacy")
    st.sidebar.info(
        "Uploaded images stay inside this host process and are not sent to a third-party inference API. "
        "OCR may create a short-lived local temporary PNG that is deleted automatically. "
        "The app does not save uploads unless you download the audit file."
    )

    with st.sidebar.expander("Pilot limitations"):
        for limitation in config.get("limitations", []):
            st.markdown(f"- {limitation}")
    return run_detector, run_ocr, explain


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#14213d; --muted:#5f6b7a; --paper:#f6f8fc; --line:#dfe5ef; }
        .stApp { background: radial-gradient(circle at 85% 0%, #e8efff 0, transparent 28%), #f7f9fc; }
        .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }
        .hero { padding: 2.1rem 2.3rem; border: 1px solid #dce3f0; border-radius: 24px;
          background: linear-gradient(130deg, rgba(255,255,255,.98), rgba(237,243,255,.92));
          box-shadow: 0 18px 55px rgba(30, 55, 95, .09); margin-bottom: 1.5rem; }
        .hero .eyebrow { color:#3156b3; font-size:.73rem; font-weight:800; letter-spacing:.16em; }
        .hero h1 { color:var(--ink); font-size:clamp(2.25rem,5vw,4.4rem); line-height:1; margin:.55rem 0 .8rem; letter-spacing:-.045em; }
        .hero p { color:var(--muted); font-size:1.08rem; max-width:760px; margin:0; line-height:1.65; }
        .pipeline-card { min-height:175px; background:white; border:1px solid var(--line); border-radius:18px;
          padding:1.35rem; box-shadow:0 9px 25px rgba(30,55,95,.05); }
        .pipeline-card span { color:#3156b3; font-weight:800; font-size:.78rem; letter-spacing:.1em; }
        .pipeline-card h3 { color:var(--ink); margin:.65rem 0 .45rem; font-size:1.03rem; }
        .pipeline-card p { color:var(--muted); line-height:1.55; margin:0; }
        .verdict { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; padding:1.25rem 1.4rem;
          border-radius:18px; margin:.5rem 0 1.4rem; border:1px solid; }
        .verdict > div { display:flex; flex-direction:column; gap:.25rem; }
        .verdict span { font-size:.69rem; font-weight:800; letter-spacing:.11em; opacity:.72; }
        .verdict strong { font-size:1.24rem; }
        .verdict-approve { color:#17634f; background:#eaf8f3; border-color:#bce6d6; }
        .verdict-review { color:#7a4b00; background:#fff7df; border-color:#f0d68a; }
        .verdict-block { color:#92263a; background:#fff0f2; border-color:#efbdc6; }
        .reason-row { padding:.8rem .9rem; margin:.5rem 0; border-left:4px solid #3156b3;
          background:white; border-radius:0 10px 10px 0; color:#26354f; }
        div[data-testid="stFileUploader"] { background:white; border:1px solid var(--line); border-radius:16px; padding:.65rem; }
        div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem; }
        @media (max-width: 760px) { .verdict { grid-template-columns:1fr; } .hero { padding:1.5rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    config = load_policy_config()
    _render_header()
    run_detector, run_ocr, explain = _render_sidebar(config)

    st.markdown("### Analyze an ad creative")
    uploaded = st.file_uploader(
        "Upload one JPEG, PNG, or WebP image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Maximum encoded size: 20 MB. Maximum decoded size: 40 megapixels.",
    )
    if uploaded is None:
        _render_pipeline_empty_state()
        return

    payload = uploaded.getvalue()
    try:
        preview = load_image(payload)
    except ImageValidationError as exc:
        st.error(str(exc))
        return

    st.caption(f"{Path(uploaded.name).name} | {len(payload) / 1024:.1f} KB | {preview.width} x {preview.height}")
    if not st.button("Run policy analysis", type="primary", width="stretch"):
        st.image(preview, caption="Validated local preview", width="stretch")
        return

    artifact = MODEL_DIR / "vit_policy_head.joblib"
    if not artifact.is_file():
        st.error(
            "The trained classifier artifact models/vit_policy_head.joblib is missing. "
            "Complete the training pipeline before running moderation."
        )
        return

    try:
        with st.spinner("Running local policy analysis..."):
            result = load_engine().analyze(
                payload,
                run_detector=run_detector,
                run_ocr=run_ocr,
                explain=explain,
            )
    except ImageValidationError as exc:
        st.error(str(exc))
        return
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError) as exc:
        st.error(f"A required local model component could not start: {exc}")
        return
    except Exception as exc:
        st.error(f"Analysis failed safely: {type(exc).__name__}: {exc}")
        return

    audit = _audit_payload(
        result,
        filename=uploaded.name,
        payload=payload,
        run_detector=run_detector,
        run_ocr=run_ocr,
        explain=explain,
    )
    _render_results(result, config, preview, audit)


if __name__ == "__main__":
    main()
