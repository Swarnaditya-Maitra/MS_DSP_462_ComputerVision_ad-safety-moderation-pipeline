#!/usr/bin/env python3
"""Execute the evidence notebook and build a local static technical book."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = PROJECT_ROOT / "book"
SOURCE_NOTEBOOK = PROJECT_ROOT / "ad_safety_moderation_pipeline.ipynb"
BUILD_DIR = BOOK_DIR / "_build"
HTML_DIR = BUILD_DIR / "html"

REQUIRED_EVALUATION_ARTIFACTS = (
    "outputs/evaluation/evaluation_metrics.json",
    "outputs/evaluation/independent_validation.json",
    "outputs/evaluation/per_class_metrics.csv",
    "outputs/evaluation/test_predictions.csv",
    "outputs/evaluation/thresholds.csv",
    "outputs/evaluation/metrics_summary.csv",
    "outputs/evaluation/benchmark_cpu_batch1.csv",
    "outputs/evaluation/split_summary.csv",
    "outputs/evaluation/output_checksums.csv",
    "outputs/evaluation/dataset_distribution.png",
    "outputs/evaluation/dataset_contact_sheet.jpg",
    "outputs/evaluation/confusion_matrix_vit.png",
    "outputs/evaluation/confusion_matrix_resnet50.png",
    "outputs/evaluation/precision_recall_curves.png",
    "outputs/evaluation/threshold_calibration.png",
    "outputs/evaluation/model_comparison.png",
)
REQUIRED_RUNTIME_ARTIFACTS = (
    "models/vit_policy_head.joblib",
    "app.py",
    "api.py",
)


def validate_inputs() -> None:
    required = REQUIRED_EVALUATION_ARTIFACTS + REQUIRED_RUNTIME_ARTIFACTS
    missing = [relative for relative in required if not (PROJECT_ROOT / relative).is_file()]
    if not SOURCE_NOTEBOOK.is_file():
        missing.insert(0, SOURCE_NOTEBOOK.name)
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "Static-book evidence gate failed. Missing required files:\n"
            f"{formatted}\n"
            "The builder will not create a report with placeholder results. Run the "
            "project training/evaluation workflow and create the app entrypoints first."
        )

    empty = [relative for relative in required if (PROJECT_ROOT / relative).stat().st_size == 0]
    if empty:
        formatted = "\n".join(f"  - {item}" for item in empty)
        raise ValueError(f"Static-book evidence gate found empty required files:\n{formatted}")

    for relative in (
        "outputs/evaluation/evaluation_metrics.json",
        "outputs/evaluation/independent_validation.json",
    ):
        path = PROJECT_ROOT / relative
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {relative}: {exc}") from exc


def execute_notebook(timeout: int, execute: bool) -> nbformat.NotebookNode:
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    if execute:
        processor = ExecutePreprocessor(timeout=timeout, kernel_name="python3", allow_errors=False)
        processor.preprocess(notebook, {"metadata": {"path": str(PROJECT_ROOT)}})
    else:
        print("WARNING: --no-execute creates a source-only site without refreshed outputs.")
    return notebook


def save_executed_notebook(notebook: nbformat.NotebookNode) -> None:
    nbformat.write(notebook, SOURCE_NOTEBOOK)
    print(f"Saved canonical executed notebook: {SOURCE_NOTEBOOK}")


def build_with_jupyter_book(executable: str, notebook: nbformat.NotebookNode) -> Path:
    with tempfile.TemporaryDirectory(prefix="ad-safety-book-") as temp_root:
        staging_book = Path(temp_root) / "book"
        shutil.copytree(
            BOOK_DIR,
            staging_book,
            ignore=shutil.ignore_patterns(
                "_build",
                SOURCE_NOTEBOOK.name,
                "__pycache__",
                "*.pyc",
            ),
        )
        nbformat.write(notebook, staging_book / SOURCE_NOTEBOOK.name)
        subprocess.run([executable, "build", str(staging_book)], cwd=PROJECT_ROOT, check=True)
        staged_html = staging_book / "_build" / "html"
        if not (staged_html / "index.html").is_file():
            raise FileNotFoundError(
                f"jupyter-book completed without creating {staged_html / 'index.html'}"
            )
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged_html, HTML_DIR, dirs_exist_ok=True)

    output = HTML_DIR / "index.html"
    return output


def fallback_landing_page(generated_at: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Ad Safety Moderation Pilot</title>
  <link rel=\"icon\" href=\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%232f6fed'/%3E%3Cpath d='M18 46 29 17h7l11 29h-8l-2-7H27l-2 7Zm11-14h6l-3-10Z' fill='white'/%3E%3C/svg%3E\">
  <link rel=\"stylesheet\" href=\"custom.css\">
</head>
<body class=\"fallback-home\">
  <main>
    <p class=\"eyebrow\">MS_DSP_462 Computer Vision</p>
    <h1>Ad Safety Moderation Pilot</h1>
    <p class=\"lede\">A reproducible static-ad triage system combining visual
    classification, optional open-vocabulary detection, OCR cues, and an
    auditable policy engine.</p>
    <section class=\"evidence-boundary\">
      <h2>Evidence boundary</h2>
      <p>All results come from the saved evaluation artifacts used during this
      build. Financial-promotion imagery does not establish fraud, legality, or
      regulatory status. Pilot metrics do not establish production readiness.</p>
    </section>
    <a class=\"primary-link\" href=\"ad_safety_moderation_pipeline.html\">Open the executable technical chapter</a>
    <p class=\"generated\">Generated {html.escape(generated_at)}</p>
  </main>
</body>
</html>
"""


def build_with_nbconvert(notebook: nbformat.NotebookNode) -> Path:
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(notebook)
    favicon = (
        "<link rel=\"icon\" href=\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
        "viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%232f6fed'/%3E"
        "%3Cpath d='M18 46 29 17h7l11 29h-8l-2-7H27l-2 7Zm11-14h6l-3-10Z' "
        "fill='white'/%3E%3C/svg%3E\">"
    )
    body = body.replace("</head>", f"{favicon}\n</head>", 1)
    (HTML_DIR / "ad_safety_moderation_pipeline.html").write_text(body, encoding="utf-8")

    custom_css = (BOOK_DIR / "_static" / "custom.css").read_text(encoding="utf-8")
    fallback_css = """
body.fallback-home { background: #f7f9fc; margin: 0; font-family: Inter, system-ui, sans-serif; }
.fallback-home main { max-width: 920px; margin: 0 auto; padding: 7rem 2rem; }
.eyebrow { color: #2f6fed; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; }
.lede { font-size: 1.35rem; line-height: 1.55; max-width: 760px; }
.evidence-boundary { background: white; border-left: 5px solid #e75b65; border-radius: 8px; margin: 2rem 0; padding: 1rem 1.35rem; box-shadow: 0 8px 28px rgb(17 36 58 / 9%); }
.primary-link { display: inline-block; background: #2f6fed; border-radius: 8px; color: white; font-weight: 700; margin-top: .5rem; padding: .85rem 1.2rem; text-decoration: none; }
.generated { color: #5d6b7d; font-size: .85rem; margin-top: 2rem; }
"""
    (HTML_DIR / "custom.css").write_text(custom_css + fallback_css, encoding="utf-8")
    generated_at = datetime.now(timezone.utc).isoformat()
    (HTML_DIR / "index.html").write_text(fallback_landing_page(generated_at), encoding="utf-8")
    return HTML_DIR / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--builder",
        choices=("auto", "jupyter-book", "nbconvert"),
        default="auto",
        help="Static-site backend. auto prefers jupyter-book when installed.",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Notebook execution timeout in seconds.")
    parser.add_argument("--no-execute", action="store_true", help="Build without refreshing notebook outputs.")
    args = parser.parse_args()

    validate_inputs()
    notebook = execute_notebook(timeout=args.timeout, execute=not args.no_execute)
    if not args.no_execute:
        save_executed_notebook(notebook)

    jupyter_book = shutil.which("jupyter-book")
    if args.builder == "jupyter-book" and not jupyter_book:
        raise FileNotFoundError("--builder jupyter-book was requested, but jupyter-book is not installed.")

    if args.builder == "jupyter-book" or (args.builder == "auto" and jupyter_book):
        output = build_with_jupyter_book(str(jupyter_book), notebook)
        backend = "jupyter-book"
    else:
        output = build_with_nbconvert(notebook)
        backend = "nbconvert"

    print(f"Static book built with {backend}: {output}")


if __name__ == "__main__":
    main()
