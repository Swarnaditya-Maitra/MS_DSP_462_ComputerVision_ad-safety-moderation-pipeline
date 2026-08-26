#!/usr/bin/env python3
"""Build the strict two-page Ad Safety technical synopsis.

The generator is evidence-gated. It reads only saved project artifacts, rejects
missing or placeholder evidence, and writes no PDF unless every required input
passes validation. The formal dataset is data/capstone_registry.csv. The older
data/dataset_registry.csv file is intentionally excluded because it is a
rejected sparse Wikimedia search artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Flowable,
    Frame,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
REPORT_DIR = PROJECT_ROOT / "outputs" / "report"
DEFAULT_OUTPUT = REPORT_DIR / "ad_safety_technical_synopsis.pdf"

FORMAL_REGISTRY = PROJECT_ROOT / "data" / "capstone_registry.csv"
REJECTED_WIKIMEDIA_REGISTRY = PROJECT_ROOT / "data" / "dataset_registry.csv"
PROPOSAL_PATH = PROJECT_ROOT / "Capstone Project Idea - Ad Safety.pdf"
POLICY_PATH = PROJECT_ROOT / "configs" / "policy.yaml"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "pretrained_model_manifest.json"
SOURCES_PATH = PROJECT_ROOT / "SOURCES.md"

REQUIRED_EVALUATION_FILES = (
    "metrics.json",
    "evaluation_metrics.json",
    "independent_validation.json",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "confusion_matrix_vit.png",
    "predictions.csv",
    "test_predictions.csv",
    "latency.json",
    "benchmark_cpu_batch1.csv",
    "failure_cases.csv",
    "thresholds.csv",
    "metrics_summary.csv",
    "model_comparison.csv",
    "dataset_distribution.png",
    "evaluation_manifest.json",
    "output_checksums.csv",
)
OPTIONAL_EXTERNAL_DIAGNOSTIC = "external_spot_check.json"

EXPECTED_LABELS = ("safe", "firearms", "explosives", "financial_promotion")
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_MODEL_REPOS = (
    "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "timm/resnet50.a1_in1k",
    "IDEA-Research/grounding-dino-tiny",
)
TEAM_NAMES = (
    "Swarnaditya Maitra",
    "Vijay Agnihotri",
    "Myetchae Thu",
    "Bickramjit Basu",
)

PLACEHOLDER_VALUES = {
    "placeholder",
    "tbd",
    "todo",
    "not measured",
    "not available",
    "pending",
    "n/a",
    "na",
}


class EvidenceError(RuntimeError):
    """Raised when measured evidence cannot support the synopsis."""


class LayoutError(RuntimeError):
    """Raised when content does not fit the strict two-page layout."""


@dataclass(frozen=True)
class Reference:
    title: str
    url: str


@dataclass(frozen=True)
class DatasetEvidence:
    row_count: int
    group_count: int
    split_counts: Mapping[str, int]
    label_split_counts: Mapping[str, Mapping[str, int]]
    source_counts: Mapping[str, int]
    synthetic_count: int
    width_range: tuple[int, int]
    height_range: tuple[int, int]
    mime_counts: Mapping[str, int]
    license_counts: Mapping[str, int]
    label_audit_notes: tuple[str, ...]
    group_overlap_across_splits: int
    exact_hash_overlap_across_splits: int


@dataclass(frozen=True)
class ProposalTargets:
    safe_class_precision_floor: float
    restricted_class_recall_floor: float
    latency_ceiling_ms: float


@dataclass(frozen=True)
class StaticEvidence:
    dataset: DatasetEvidence
    proposal_targets: ProposalTargets
    policy: Mapping[str, Any]
    model_manifest: Mapping[str, Any]
    references: tuple[Reference, ...]


@dataclass(frozen=True)
class RawEvaluation:
    metrics: Mapping[str, Any]
    evaluation_metrics: Mapping[str, Any]
    independent_validation: Mapping[str, Any]
    latency: Mapping[str, Any]
    evaluation_manifest: Mapping[str, Any]
    per_class_metrics: tuple[Mapping[str, str], ...]
    confusion_matrix: tuple[Mapping[str, str], ...]
    predictions: tuple[Mapping[str, str], ...]
    test_predictions: tuple[Mapping[str, str], ...]
    failure_cases: tuple[Mapping[str, str], ...]
    thresholds: tuple[Mapping[str, str], ...]
    metrics_summary: tuple[Mapping[str, str], ...]
    model_comparison: tuple[Mapping[str, str], ...]
    benchmark_cpu_batch1: tuple[Mapping[str, str], ...]
    output_checksums: tuple[Mapping[str, str], ...]
    csv_headers: Mapping[str, tuple[str, ...]]
    confusion_matrix_image: Path
    dataset_distribution_image: Path
    external_spot_check: Mapping[str, Any] | None


@dataclass(frozen=True)
class MetricValue:
    label: str
    value: str
    note: str = ""


@dataclass(frozen=True)
class MeasuredEvidence:
    """Normalized values accepted by the renderer after exact schema mapping."""

    selected_model: str
    model_version: str
    evaluation_scope: str
    architecture_paragraphs: tuple[str, ...]
    math_lines: tuple[str, ...]
    preprocessing_steps: tuple[str, ...]
    training_items: tuple[tuple[str, str], ...]
    overall_metrics: tuple[MetricValue, ...]
    performance_definition_note: str
    per_class_headers: tuple[str, ...]
    per_class_rows: tuple[tuple[str, ...], ...]
    confusion_headers: tuple[str, ...]
    confusion_rows: tuple[tuple[str, ...], ...]
    comparison_headers: tuple[str, ...]
    comparison_rows: tuple[tuple[str, ...], ...]
    latency_metrics: tuple[MetricValue, ...]
    latency_note: str
    challenges_and_solutions: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]
    reproducibility_items: tuple[tuple[str, str], ...]
    external_diagnostic: tuple[MetricValue, ...] | None
    external_diagnostic_scope: str | None


@dataclass(frozen=True)
class SynopsisEvidence:
    static: StaticEvidence
    measured: MeasuredEvidence


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _required_paths() -> tuple[Path, ...]:
    return (
        FORMAL_REGISTRY,
        PROPOSAL_PATH,
        POLICY_PATH,
        MODEL_MANIFEST_PATH,
        SOURCES_PATH,
        *(EVALUATION_DIR / name for name in REQUIRED_EVALUATION_FILES),
    )


def validate_required_files() -> None:
    missing = [path for path in _required_paths() if not path.is_file()]
    empty = [path for path in _required_paths() if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        lines = ["Measured-evidence gate failed. The synopsis was not generated."]
        if missing:
            lines.append("Missing required artifacts:")
            lines.extend(f"  - {_relative(path)}" for path in missing)
        if empty:
            lines.append("Empty required artifacts:")
            lines.extend(f"  - {_relative(path)}" for path in empty)
        lines.append(
            "Run scripts/build_capstone_dataset.py and scripts/train_and_evaluate.py, "
            "then rerun this generator."
        )
        raise EvidenceError("\n".join(lines))


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"Invalid JSON in {_relative(path)}: {exc}") from exc
    except OSError as exc:
        raise EvidenceError(f"Cannot read {_relative(path)}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise EvidenceError(f"{_relative(path)} must contain a non-empty JSON object.")
    _reject_placeholders(value, _relative(path))
    return value


def read_csv_rows(path: Path, *, allow_empty_rows: bool = False) -> tuple[tuple[str, ...], tuple[Mapping[str, str], ...]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = tuple(reader.fieldnames or ())
            rows = tuple(dict(row) for row in reader)
    except (OSError, csv.Error) as exc:
        raise EvidenceError(f"Cannot parse {_relative(path)}: {exc}") from exc
    if not headers:
        raise EvidenceError(f"{_relative(path)} has no CSV header.")
    if not rows and not allow_empty_rows:
        raise EvidenceError(f"{_relative(path)} contains no measured rows.")
    for row_index, row in enumerate(rows, start=2):
        for column, value in row.items():
            if _is_placeholder(value):
                raise EvidenceError(
                    f"Placeholder value in {_relative(path)} at row {row_index}, column {column!r}: {value!r}"
                )
    return headers, rows


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() in PLACEHOLDER_VALUES


def _reject_placeholders(value: Any, source: str, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _reject_placeholders(child, source, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, source, f"{path}[{index}]")
    elif _is_placeholder(value):
        raise EvidenceError(f"Placeholder value in {source} at {path}: {value!r}")


def _require_columns(headers: Sequence[str], required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(headers))
    if missing:
        raise EvidenceError(
            f"{_relative(source)} is missing required columns: {', '.join(missing)}"
        )


def _parse_int(value: str, source: Path, row_number: int, column: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(
            f"Invalid integer in {_relative(source)} at row {row_number}, column {column!r}: {value!r}"
        ) from exc
    return parsed


def load_dataset_evidence() -> DatasetEvidence:
    required_columns = {
        "sample_id",
        "label",
        "split",
        "local_path",
        "source_dataset",
        "source_revision",
        "source_group",
        "declared_license",
        "is_synthetic",
        "local_sha256",
        "mime",
        "width",
        "height",
        "label_audit_status",
    }
    headers, rows = read_csv_rows(FORMAL_REGISTRY)
    _require_columns(headers, required_columns, FORMAL_REGISTRY)

    labels = {str(row["label"]).strip() for row in rows}
    splits = {str(row["split"]).strip() for row in rows}
    if labels != set(EXPECTED_LABELS):
        raise EvidenceError(
            f"{_relative(FORMAL_REGISTRY)} labels are {sorted(labels)}; expected exactly {list(EXPECTED_LABELS)}."
        )
    if splits != set(EXPECTED_SPLITS):
        raise EvidenceError(
            f"{_relative(FORMAL_REGISTRY)} splits are {sorted(splits)}; expected exactly {list(EXPECTED_SPLITS)}."
        )

    sample_ids: set[str] = set()
    local_paths: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    label_split_counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[str] = Counter()
    mime_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    audit_notes: set[str] = set()
    widths: list[int] = []
    heights: list[int] = []
    synthetic_count = 0
    missing_local_files: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row["sample_id"]).strip()
        local_path = str(row["local_path"]).strip()
        label = str(row["label"]).strip()
        split = str(row["split"]).strip()
        group = str(row["source_group"]).strip()
        digest = str(row["local_sha256"]).strip().casefold()
        if not sample_id or not local_path or not group:
            raise EvidenceError(
                f"Blank sample_id, local_path, or source_group in {_relative(FORMAL_REGISTRY)} row {row_number}."
            )
        if sample_id in sample_ids:
            raise EvidenceError(f"Duplicate sample_id {sample_id!r} in {_relative(FORMAL_REGISTRY)}.")
        if local_path in local_paths:
            raise EvidenceError(f"Duplicate local_path {local_path!r} in {_relative(FORMAL_REGISTRY)}.")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EvidenceError(
                f"Invalid local_sha256 in {_relative(FORMAL_REGISTRY)} row {row_number}: {digest!r}"
            )
        source_file = PROJECT_ROOT / local_path
        if not source_file.is_file():
            missing_local_files.append(local_path)
        elif hashlib.sha256(source_file.read_bytes()).hexdigest() != digest:
            raise EvidenceError(
                f"Local image hash does not match {_relative(FORMAL_REGISTRY)} at row {row_number}: {local_path}"
            )

        synthetic_text = str(row["is_synthetic"]).strip().casefold()
        if synthetic_text not in {"true", "false"}:
            raise EvidenceError(
                f"Invalid is_synthetic in {_relative(FORMAL_REGISTRY)} row {row_number}: {synthetic_text!r}"
            )
        synthetic_count += int(synthetic_text == "true")
        sample_ids.add(sample_id)
        local_paths.add(local_path)
        group_splits[group].add(split)
        hash_splits[digest].add(split)
        split_counts[split] += 1
        label_split_counts[(label, split)] += 1
        source_counts[str(row["source_dataset"]).strip()] += 1
        mime_counts[str(row["mime"]).strip()] += 1
        license_counts[str(row["declared_license"]).strip()] += 1
        audit_notes.add(str(row["label_audit_status"]).strip())
        widths.append(_parse_int(str(row["width"]), FORMAL_REGISTRY, row_number, "width"))
        heights.append(_parse_int(str(row["height"]), FORMAL_REGISTRY, row_number, "height"))

    if missing_local_files:
        preview = "\n".join(f"  - {item}" for item in missing_local_files[:12])
        suffix = f"\n  ... and {len(missing_local_files) - 12} more" if len(missing_local_files) > 12 else ""
        raise EvidenceError(
            f"{_relative(FORMAL_REGISTRY)} references missing local images:\n{preview}{suffix}"
        )

    leaking_groups = {key: value for key, value in group_splits.items() if len(value) > 1}
    leaking_hashes = {key: value for key, value in hash_splits.items() if len(value) > 1}
    if leaking_groups or leaking_hashes:
        raise EvidenceError(
            "Formal dataset split leakage detected: "
            f"{len(leaking_groups)} source groups and {len(leaking_hashes)} exact hashes cross splits."
        )

    return DatasetEvidence(
        row_count=len(rows),
        group_count=len(group_splits),
        split_counts={split: split_counts[split] for split in EXPECTED_SPLITS},
        label_split_counts={
            label: {split: label_split_counts[(label, split)] for split in EXPECTED_SPLITS}
            for label in EXPECTED_LABELS
        },
        source_counts=dict(source_counts.most_common()),
        synthetic_count=synthetic_count,
        width_range=(min(widths), max(widths)),
        height_range=(min(heights), max(heights)),
        mime_counts=dict(mime_counts.most_common()),
        license_counts=dict(license_counts.most_common()),
        label_audit_notes=tuple(sorted(audit_notes)),
        group_overlap_across_splits=0,
        exact_hash_overlap_across_splits=0,
    )


def load_proposal_targets() -> ProposalTargets:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise EvidenceError(
            "pypdf is required to verify the proposal success criteria before rendering the synopsis."
        ) from exc
    try:
        text = " ".join(
            (page.extract_text() or "") for page in PdfReader(str(PROPOSAL_PATH)).pages
        )
    except Exception as exc:
        raise EvidenceError(f"Cannot extract proposal criteria from {_relative(PROPOSAL_PATH)}: {exc}") from exc
    normalized = re.sub(r"\s+", " ", text)
    precision_match = re.search(
        r"Precision\s*>\s*(\d+(?:\.\d+)?)%\s*\(Safe\s+class\)",
        normalized,
        flags=re.IGNORECASE,
    )
    recall_match = re.search(
        r"Recall\s*>\s*(\d+(?:\.\d+)?)%\s*\(restricted\s+classes\)",
        normalized,
        flags=re.IGNORECASE,
    )
    latency_match = re.search(
        r"Latency\s+below\s+(\d+(?:\.\d+)?)\s*ms\s+per\s+asset",
        normalized,
        flags=re.IGNORECASE,
    )
    if precision_match is None or recall_match is None or latency_match is None:
        missing = []
        if precision_match is None:
            missing.append("Safe-class precision > percentage")
        if recall_match is None:
            missing.append("restricted-class recall > percentage")
        if latency_match is None:
            missing.append("latency below milliseconds per asset")
        raise EvidenceError(
            f"{_relative(PROPOSAL_PATH)} lacks parseable success criteria: {', '.join(missing)}."
        )
    precision_floor = float(precision_match.group(1)) / 100.0
    recall_floor = float(recall_match.group(1)) / 100.0
    latency_ceiling_ms = float(latency_match.group(1))
    if not 0.0 < precision_floor < 1.0 or not 0.0 < recall_floor < 1.0 or latency_ceiling_ms <= 0.0:
        raise EvidenceError(
            f"Invalid proposal criteria in {_relative(PROPOSAL_PATH)}: "
            f"precision>{precision_floor}, recall>{recall_floor}, latency<{latency_ceiling_ms} ms."
        )
    return ProposalTargets(
        safe_class_precision_floor=precision_floor,
        restricted_class_recall_floor=recall_floor,
        latency_ceiling_ms=latency_ceiling_ms,
    )


def load_policy() -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceError(f"Cannot parse {_relative(POLICY_PATH)}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{_relative(POLICY_PATH)} must contain a YAML mapping.")
    labels = tuple(value.get("model_labels", ()))
    if labels != EXPECTED_LABELS:
        raise EvidenceError(
            f"{_relative(POLICY_PATH)} model_labels are {labels}; expected {EXPECTED_LABELS}."
        )
    if not value.get("policy_version"):
        raise EvidenceError(f"{_relative(POLICY_PATH)} has no policy_version.")
    thresholds = value.get("default_thresholds")
    if not isinstance(thresholds, dict):
        raise EvidenceError(f"{_relative(POLICY_PATH)} has no default_thresholds mapping.")
    for key in ("firearms", "explosives", "financial_promotion", "review", "uncertainty_margin"):
        raw = thresholds.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 0.0 <= float(raw) <= 1.0:
            raise EvidenceError(f"Invalid policy threshold {key!r} in {_relative(POLICY_PATH)}: {raw!r}")
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise EvidenceError(f"{_relative(POLICY_PATH)} has no documented limitations.")
    return value


def load_model_manifest() -> Mapping[str, Any]:
    manifest = read_json(MODEL_MANIFEST_PATH)
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise EvidenceError(f"{_relative(MODEL_MANIFEST_PATH)} has no models list.")
    by_repo = {str(row.get("repo_id")): row for row in models if isinstance(row, dict)}
    missing = [repo for repo in EXPECTED_MODEL_REPOS if repo not in by_repo]
    if missing:
        raise EvidenceError(
            f"{_relative(MODEL_MANIFEST_PATH)} lacks required pinned models: {', '.join(missing)}"
        )
    for repo in EXPECTED_MODEL_REPOS:
        row = by_repo[repo]
        revision = str(row.get("revision", ""))
        digest = str(row.get("snapshot_sha256", ""))
        license_name = str(row.get("license", "")).strip()
        local_snapshot = Path(str(row.get("local_snapshot", "")))
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise EvidenceError(f"Model {repo} has no immutable 40-character revision.")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EvidenceError(f"Model {repo} has no valid snapshot_sha256.")
        if not license_name:
            raise EvidenceError(f"Model {repo} has no declared license.")
        if not local_snapshot.is_absolute():
            local_snapshot = PROJECT_ROOT / local_snapshot
        if not (local_snapshot / "model.safetensors").is_file():
            raise EvidenceError(
                f"Pinned weights for {repo} are absent from {row.get('local_snapshot')}/model.safetensors."
            )
    return manifest


def load_references() -> tuple[Reference, ...]:
    text = SOURCES_PATH.read_text(encoding="utf-8")
    references: list[Reference] = []
    pattern = re.compile(r"^\s*-\s+(.+?):\s+(https?://\S+)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        title = re.sub(r"[*_`]", "", match.group(1)).strip()
        url = match.group(2).rstrip(".,);]")
        references.append(Reference(title=title, url=url))
    if not references:
        raise EvidenceError(f"{_relative(SOURCES_PATH)} contains no parseable title and URL references.")
    required_fragments = (
        "arxiv.org/abs/2010.11929",
        "arxiv.org/abs/2303.05499",
        "huggingface.co/timm/vit_base_patch16_224",
        "huggingface.co/datasets/EthanGabis/ADautoGen-DS",
        "huggingface.co/datasets/rajshivanshuu/weapons_set1",
    )
    urls = "\n".join(reference.url for reference in references)
    missing = [fragment for fragment in required_fragments if fragment not in urls]
    if missing:
        raise EvidenceError(
            f"{_relative(SOURCES_PATH)} lacks required source references: {', '.join(missing)}"
        )
    return tuple(references)


def load_raw_evaluation() -> RawEvaluation:
    csv_names = (
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "predictions.csv",
        "test_predictions.csv",
        "failure_cases.csv",
        "thresholds.csv",
        "metrics_summary.csv",
        "model_comparison.csv",
        "benchmark_cpu_batch1.csv",
        "output_checksums.csv",
    )
    headers: dict[str, tuple[str, ...]] = {}
    rows: dict[str, tuple[Mapping[str, str], ...]] = {}
    for name in csv_names:
        allow_empty = name == "failure_cases.csv"
        header, records = read_csv_rows(EVALUATION_DIR / name, allow_empty_rows=allow_empty)
        headers[name] = header
        rows[name] = records

    external_path = EVALUATION_DIR / OPTIONAL_EXTERNAL_DIAGNOSTIC
    external = read_json(external_path) if external_path.is_file() else None
    if external is not None:
        scope = str(external.get("scope", "")).casefold()
        if "external" not in scope or "not a formal test set" not in scope:
            raise EvidenceError(
                f"{_relative(external_path)} must explicitly identify itself as external and not a formal test set."
            )

    return RawEvaluation(
        metrics=read_json(EVALUATION_DIR / "metrics.json"),
        evaluation_metrics=read_json(EVALUATION_DIR / "evaluation_metrics.json"),
        independent_validation=read_json(EVALUATION_DIR / "independent_validation.json"),
        latency=read_json(EVALUATION_DIR / "latency.json"),
        evaluation_manifest=read_json(EVALUATION_DIR / "evaluation_manifest.json"),
        per_class_metrics=rows["per_class_metrics.csv"],
        confusion_matrix=rows["confusion_matrix.csv"],
        predictions=rows["predictions.csv"],
        test_predictions=rows["test_predictions.csv"],
        failure_cases=rows["failure_cases.csv"],
        thresholds=rows["thresholds.csv"],
        metrics_summary=rows["metrics_summary.csv"],
        model_comparison=rows["model_comparison.csv"],
        benchmark_cpu_batch1=rows["benchmark_cpu_batch1.csv"],
        output_checksums=rows["output_checksums.csv"],
        csv_headers=headers,
        confusion_matrix_image=EVALUATION_DIR / "confusion_matrix_vit.png",
        dataset_distribution_image=EVALUATION_DIR / "dataset_distribution.png",
        external_spot_check=external,
    )


def _expect_exact_headers(raw: RawEvaluation, name: str, expected: Sequence[str]) -> None:
    actual = raw.csv_headers.get(name)
    if actual != tuple(expected):
        raise EvidenceError(
            f"Unexpected schema in outputs/evaluation/{name}. Expected {list(expected)}, found {list(actual or ())}."
        )


def _require_mapping(value: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"Expected a mapping at {source}, found {type(value).__name__}.")
    return value


def _require_list(value: Any, source: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceError(f"Expected a list at {source}, found {type(value).__name__}.")
    return value


def _number(value: Any, source: str) -> float:
    if isinstance(value, bool):
        raise EvidenceError(f"Expected a measured number at {source}, found boolean {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"Expected a measured number at {source}, found {value!r}.") from exc
    if not math.isfinite(parsed):
        raise EvidenceError(f"Non-finite measured number at {source}: {value!r}")
    return parsed


def _integer(value: Any, source: str) -> int:
    parsed = _number(value, source)
    if not parsed.is_integer():
        raise EvidenceError(f"Expected an integer at {source}, found {value!r}.")
    return int(parsed)


def _row_by(
    rows: Sequence[Mapping[str, str]],
    source: str,
    **conditions: str,
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(column, "")) == str(expected) for column, expected in conditions.items())
    ]
    if len(matches) != 1:
        detail = ", ".join(f"{key}={value!r}" for key, value in conditions.items())
        raise EvidenceError(f"Expected one row in {source} for {detail}; found {len(matches)}.")
    return matches[0]


def _pct(value: Any, source: str, digits: int = 1) -> str:
    return f"{100.0 * _number(value, source):.{digits}f}%"


def _decimal(value: Any, source: str, digits: int = 3) -> str:
    return f"{_number(value, source):.{digits}f}"


def _ms(value: Any, source: str) -> str:
    return f"{_number(value, source):.1f} ms"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_output_checksums(raw: RawEvaluation) -> None:
    _expect_exact_headers(raw, "output_checksums.csv", ("path", "bytes", "sha256"))
    records: dict[str, Mapping[str, str]] = {}
    for index, row in enumerate(raw.output_checksums, start=2):
        relative = str(row.get("path", ""))
        if not relative or relative in records:
            raise EvidenceError(f"Duplicate or blank path in outputs/evaluation/output_checksums.csv row {index}.")
        candidate = (PROJECT_ROOT / relative).resolve()
        if PROJECT_ROOT.resolve() not in candidate.parents:
            raise EvidenceError(f"Checksum path escapes the project at row {index}: {relative!r}")
        if not candidate.is_file():
            raise EvidenceError(f"Checksum manifest references a missing file: {relative}")
        expected_bytes = _integer(row.get("bytes"), f"output_checksums.csv row {index}.bytes")
        expected_sha = str(row.get("sha256", "")).casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise EvidenceError(f"Invalid SHA-256 in output_checksums.csv row {index}: {expected_sha!r}")
        if candidate.stat().st_size != expected_bytes:
            raise EvidenceError(f"Byte-count mismatch for {relative} against output_checksums.csv.")
        if _sha256_file(candidate) != expected_sha:
            raise EvidenceError(f"SHA-256 mismatch for {relative} against output_checksums.csv.")
        records[relative] = row

    required = {
        str((EVALUATION_DIR / name).relative_to(PROJECT_ROOT))
        for name in REQUIRED_EVALUATION_FILES
        if name != "output_checksums.csv"
    }
    required.add("models/vit_policy_head.joblib")
    missing = sorted(required - set(records))
    if missing:
        raise EvidenceError(
            "outputs/evaluation/output_checksums.csv lacks required evidence entries: " + ", ".join(missing)
        )


def map_measured_evidence(raw: RawEvaluation, static: StaticEvidence) -> MeasuredEvidence:
    """Map the executed schema to renderer fields with exact cross-checks."""

    expected_headers = {
        "per_class_metrics.csv": (
            "model", "split", "label", "precision", "recall", "f1", "support",
            "average_precision_ovr", "roc_auc_ovr",
        ),
        "confusion_matrix.csv": ("true_label", *EXPECTED_LABELS),
        "predictions.csv": (
            "model", "model_version", "sample_id", "local_path", "source_group", "source_dataset",
            "true_label", "predicted_label", "correct", *[f"probability_{label}" for label in EXPECTED_LABELS],
            "threshold_flag_firearms", "threshold_flag_explosives", "threshold_flag_financial_promotion",
        ),
        "test_predictions.csv": (
            "model", "sample_id", "true_label", "predicted_label", "source_group", "source_dataset",
            "threshold_flags", *[f"probability_{label}" for label in EXPECTED_LABELS],
        ),
        "failure_cases.csv": (
            "model", "model_version", "sample_id", "local_path", "source_group", "source_dataset",
            "true_label", "predicted_label", "correct", *[f"probability_{label}" for label in EXPECTED_LABELS],
            "threshold_flag_firearms", "threshold_flag_explosives", "threshold_flag_financial_promotion",
        ),
        "thresholds.csv": (
            "model", "label", "threshold", "selection_split", "target_validation_recall",
            "validation_precision", "validation_recall", "validation_f1", "validation_fp", "validation_fn",
            "test_precision", "test_recall", "test_f1", "test_fp", "test_fn",
        ),
        "metrics_summary.csv": (
            "model", "role", "validation_accuracy", "validation_macro_f1", "test_accuracy", "test_macro_f1",
            "test_macro_average_precision_ovr", "test_macro_roc_auc_ovr", "test_log_loss",
            "cpu_batch1_median_ms", "cpu_batch1_p95_ms", "feature_extraction_seconds", "head_fit_seconds",
        ),
        "model_comparison.csv": (
            "model", "role", "backbone", "backbone_revision", "feature_dim", "total_backbone_parameters",
            "trainable_backbone_parameters", "validation_accuracy", "validation_macro_f1", "test_accuracy",
            "test_macro_f1", "test_macro_average_precision_ovr", "cpu_batch1_median_ms", "cpu_batch1_p95_ms",
            "head_fit_seconds", "artifact_path", "artifact_sha256",
        ),
        "benchmark_cpu_batch1.csv": (
            "model", "scope", "device", "batch_size", "warmup_runs", "measured_runs", "torch_cpu_threads",
            "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "throughput_images_per_second_from_mean",
        ),
    }
    for name, headers in expected_headers.items():
        _expect_exact_headers(raw, name, headers)
    _validate_output_checksums(raw)

    metrics = raw.metrics
    evaluation = raw.evaluation_metrics
    independent = raw.independent_validation
    manifest = raw.evaluation_manifest
    if str(metrics.get("schema_version")) != "1.0.0" or str(manifest.get("schema_version")) != "1.0.0":
        raise EvidenceError("Synopsis supports only metrics and evaluation manifest schema_version 1.0.0.")
    if manifest.get("canonical_metrics") != metrics:
        raise EvidenceError("evaluation_manifest.json canonical_metrics does not exactly match metrics.json.")
    if manifest.get("validation_result") != independent:
        raise EvidenceError("evaluation_manifest.json validation_result does not exactly match independent_validation.json.")
    if independent.get("all_checks_passed") is not True:
        raise EvidenceError("independent_validation.json does not report all_checks_passed=true.")

    dataset_result = _require_mapping(evaluation.get("dataset"), "evaluation_metrics.json.dataset")
    if _integer(dataset_result.get("rows"), "evaluation_metrics.json.dataset.rows") != static.dataset.row_count:
        raise EvidenceError("Evaluation dataset row count does not match data/capstone_registry.csv.")
    if _integer(dataset_result.get("source_groups"), "evaluation_metrics.json.dataset.source_groups") != static.dataset.group_count:
        raise EvidenceError("Evaluation source-group count does not match data/capstone_registry.csv.")
    registry_sha = _sha256_file(FORMAL_REGISTRY)
    if str(dataset_result.get("registry_sha256")) != registry_sha:
        raise EvidenceError("Evaluation registry SHA-256 does not match data/capstone_registry.csv.")
    if _integer(dataset_result.get("group_overlap_across_splits"), "evaluation dataset group overlap") != 0:
        raise EvidenceError("Evaluation reports source-group leakage across splits.")
    if _integer(dataset_result.get("exact_hash_overlap_across_splits"), "evaluation dataset hash overlap") != 0:
        raise EvidenceError("Evaluation reports exact-hash leakage across splits.")
    if _integer(independent.get("registry_hashes_recomputed"), "independent_validation.registry_hashes_recomputed") != static.dataset.row_count:
        raise EvidenceError("Independent validation did not recompute every formal registry image hash.")

    if str(metrics.get("model")) != "vit" or str(metrics.get("split")) != "untouched_test":
        raise EvidenceError("metrics.json must identify the predeclared ViT on split='untouched_test'.")
    if _integer(metrics.get("samples"), "metrics.json.samples") != static.dataset.split_counts["test"]:
        raise EvidenceError("metrics.json sample count does not match the formal test split.")
    model_version = str(metrics.get("model_version", ""))
    if not model_version:
        raise EvidenceError("metrics.json has no model_version.")

    run = _require_mapping(evaluation.get("run"), "evaluation_metrics.json.run")
    protocol = _require_mapping(evaluation.get("protocol"), "evaluation_metrics.json.protocol")
    if protocol.get("primary_model_predeclared") != "vit" or protocol.get("baseline_predeclared") != "resnet50":
        raise EvidenceError("Evaluation protocol does not preserve the predeclared ViT primary and ResNet50 baseline.")
    if protocol.get("frozen_backbones") is not True:
        raise EvidenceError("Evaluation protocol does not report frozen backbones.")
    if protocol.get("fit_split") != "train only" or protocol.get("threshold_selection_split") != "validation only":
        raise EvidenceError("Evaluation protocol does not isolate training and threshold selection splits.")

    model_records = _require_list(evaluation.get("models"), "evaluation_metrics.json.models")
    by_model = {
        str(item.get("key")): _require_mapping(item, "evaluation_metrics.json.models[]")
        for item in model_records
        if isinstance(item, Mapping)
    }
    if set(by_model) != {"vit", "resnet50"}:
        raise EvidenceError(f"Expected executed models vit and resnet50, found {sorted(by_model)}.")
    primary = by_model["vit"]
    if primary.get("model_version") != model_version:
        raise EvidenceError("Primary model version differs between evaluation_metrics.json and metrics.json.")
    backbone = _require_mapping(primary.get("backbone"), "evaluation_metrics.json.models[vit].backbone")
    head = _require_mapping(primary.get("head"), "evaluation_metrics.json.models[vit].head")
    extraction = _require_mapping(primary.get("feature_extraction"), "evaluation_metrics.json.models[vit].feature_extraction")
    pinned = {
        str(item.get("repo_id")): item
        for item in _require_list(static.model_manifest.get("models"), "pretrained_model_manifest.json.models")
        if isinstance(item, Mapping)
    }
    repo_id = str(backbone.get("repo_id"))
    if repo_id not in pinned or pinned[repo_id].get("revision") != backbone.get("revision"):
        raise EvidenceError("Executed ViT revision does not match the pinned pretrained model manifest.")
    if _integer(backbone.get("trainable_backbone_parameters"), "ViT trainable_backbone_parameters") != 0:
        raise EvidenceError("Executed ViT backbone was not fully frozen.")
    if str(primary.get("artifact_path")) != "models/vit_policy_head.joblib":
        raise EvidenceError("Primary model artifact path is not models/vit_policy_head.joblib.")
    artifact_path = PROJECT_ROOT / str(primary["artifact_path"])
    if _sha256_file(artifact_path) != str(primary.get("artifact_sha256")):
        raise EvidenceError("Primary model artifact SHA-256 does not match evaluation_metrics.json.")
    independent_vit = _require_mapping(
        _require_mapping(independent.get("artifacts"), "independent_validation.artifacts").get("vit"),
        "independent_validation.artifacts.vit",
    )
    if _number(independent_vit.get("reload_max_abs_probability_difference"), "artifact reload difference") != 0.0:
        raise EvidenceError("Reloaded ViT artifact does not reproduce saved probabilities exactly.")

    if len(raw.predictions) != static.dataset.split_counts["test"]:
        raise EvidenceError(
            f"predictions.csv has {len(raw.predictions)} rows; expected {static.dataset.split_counts['test']}."
        )
    sample_ids = [str(row.get("sample_id")) for row in raw.predictions]
    if len(set(sample_ids)) != len(sample_ids):
        raise EvidenceError("predictions.csv contains duplicate sample_id values.")
    if {str(row.get("true_label")) for row in raw.predictions} != set(EXPECTED_LABELS):
        raise EvidenceError("predictions.csv does not contain all four expected true labels.")
    incorrect = [row for row in raw.predictions if str(row.get("correct")).casefold() != "true"]
    if {row.get("sample_id") for row in incorrect} != {row.get("sample_id") for row in raw.failure_cases}:
        raise EvidenceError("failure_cases.csv does not exactly match incorrect rows in predictions.csv.")
    if len(raw.test_predictions) != 2 * static.dataset.split_counts["test"]:
        raise EvidenceError("test_predictions.csv must contain one test prediction per image for both models.")
    for model_name in ("vit", "resnet50"):
        rows = [row for row in raw.test_predictions if row.get("model") == model_name]
        if len(rows) != static.dataset.split_counts["test"]:
            raise EvidenceError(f"test_predictions.csv has {len(rows)} rows for {model_name}; expected 48.")

    per_class_test: list[Mapping[str, str]] = []
    for label in EXPECTED_LABELS:
        row = _row_by(raw.per_class_metrics, "per_class_metrics.csv", model="vit", split="test", label=label)
        if _integer(row.get("support"), f"per_class_metrics vit/test/{label}.support") != static.dataset.label_split_counts[label]["test"]:
            raise EvidenceError(f"Per-class support mismatch for {label}.")
        per_class_test.append(row)

    matrix_rows = [
        _row_by(raw.confusion_matrix, "confusion_matrix.csv", true_label=label)
        for label in EXPECTED_LABELS
    ]
    matrix_total = sum(
        _integer(row.get(predicted), f"confusion_matrix {row.get('true_label')}->{predicted}")
        for row in matrix_rows
        for predicted in EXPECTED_LABELS
    )
    if matrix_total != static.dataset.split_counts["test"]:
        raise EvidenceError(f"confusion_matrix.csv sums to {matrix_total}; expected 48 test samples.")
    diagonal = sum(
        _integer(row.get(str(row.get("true_label"))), "confusion matrix diagonal") for row in matrix_rows
    )
    if diagonal != len(raw.predictions) - len(incorrect):
        raise EvidenceError("confusion_matrix.csv accuracy count disagrees with predictions.csv.")

    threshold_rows = {
        label: _row_by(raw.thresholds, "thresholds.csv", model="vit", label=label)
        for label in ("firearms", "explosives", "financial_promotion")
    }
    selected_thresholds = _require_mapping(
        metrics.get("thresholds_selected_on_validation_only"),
        "metrics.json.thresholds_selected_on_validation_only",
    )
    for label, row in threshold_rows.items():
        if row.get("selection_split") != "validation":
            raise EvidenceError(f"Threshold for {label} was not selected on validation.")
        if not math.isclose(
            _number(row.get("threshold"), f"thresholds.csv vit/{label}.threshold"),
            _number(selected_thresholds.get(label), f"metrics.json selected threshold {label}"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise EvidenceError(f"Selected threshold mismatch for {label}.")
    threshold_tuning_floors = {
        label: _number(
            row.get("target_validation_recall"),
            f"thresholds.csv vit/{label}.target_validation_recall",
        )
        for label, row in threshold_rows.items()
    }
    threshold_tuning_floor = next(iter(threshold_tuning_floors.values()))
    if not 0.0 < threshold_tuning_floor <= 1.0 or any(
        not math.isclose(value, threshold_tuning_floor, rel_tol=0.0, abs_tol=1e-12)
        for value in threshold_tuning_floors.values()
    ):
        raise EvidenceError(
            "Restricted thresholds do not share one valid target_validation_recall tuning floor."
        )
    expected_threshold_rule = f"restricted-class recall >= {threshold_tuning_floor:.2f};"
    if not str(protocol.get("threshold_rule", "")).startswith(expected_threshold_rule):
        raise EvidenceError(
            "evaluation_metrics.json threshold_rule does not match the tuning floor in thresholds.csv."
        )

    per_class_by_label = dict(zip(EXPECTED_LABELS, per_class_test))
    restricted_labels = ("firearms", "explosives", "financial_promotion")
    safe_precision = _number(
        per_class_by_label["safe"].get("precision"),
        "per_class_metrics.csv vit/test/safe.precision",
    )
    if not 0.0 <= safe_precision <= 1.0:
        raise EvidenceError(f"Safe-class precision is outside [0, 1]: {safe_precision}.")
    proposal_safe_precision_floor = static.proposal_targets.safe_class_precision_floor
    safe_precision_assessment = "passes" if safe_precision > proposal_safe_precision_floor else "fails"
    safe_false_positive_rate = _number(
        metrics.get("safe_false_positive_rate"), "metrics.json.safe_false_positive_rate"
    )
    if not 0.0 <= safe_false_positive_rate <= 1.0:
        raise EvidenceError(
            f"Safe-ad false-positive rate is outside [0, 1]: {safe_false_positive_rate}."
        )
    argmax_recall_by_label = {
        label: _number(
            per_class_by_label[label].get("recall"),
            f"per_class_metrics.csv vit/test/{label}.recall",
        )
        for label in restricted_labels
    }
    argmax_restricted_mean = sum(argmax_recall_by_label.values()) / len(restricted_labels)
    reported_restricted_recall = _number(
        metrics.get("restricted_recall"), "metrics.json.restricted_recall"
    )
    if not math.isclose(
        reported_restricted_recall,
        argmax_restricted_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise EvidenceError(
            "metrics.json restricted_recall is not the multiclass-argmax class-average recall "
            "across firearms, explosives, and financial promotion."
        )

    threshold_operating_recall = {
        label: _number(
            threshold_rows[label].get("test_recall"),
            f"thresholds.csv vit/{label}.test_recall",
        )
        for label in restricted_labels
    }
    for label, value in threshold_operating_recall.items():
        if not 0.0 <= value <= 1.0:
            raise EvidenceError(f"Threshold operating recall for {label} is outside [0, 1]: {value}.")
    threshold_operating_mean = sum(threshold_operating_recall.values()) / len(restricted_labels)
    threshold_operating_worst = min(threshold_operating_recall.values())
    proposal_recall_floor = static.proposal_targets.restricted_class_recall_floor
    below_proposal_recall = tuple(
        label for label, value in threshold_operating_recall.items() if not value > proposal_recall_floor
    )
    recall_assessment = "fails" if below_proposal_recall else "passes"
    below_target_names = tuple(label.replace("_", " ") for label in below_proposal_recall)
    if len(below_target_names) == 1:
        recall_failure_detail = f" for {below_target_names[0]}"
    elif len(below_target_names) == 2:
        recall_failure_detail = f" for {below_target_names[0]} and {below_target_names[1]}"
    elif below_target_names:
        recall_failure_detail = " for " + ", ".join(below_target_names[:-1]) + f", and {below_target_names[-1]}"
    else:
        recall_failure_detail = ""

    comparison = {
        model: _row_by(raw.model_comparison, "model_comparison.csv", model=model)
        for model in ("vit", "resnet50")
    }
    summaries = {
        model: _row_by(raw.metrics_summary, "metrics_summary.csv", model=model)
        for model in ("vit", "resnet50")
    }
    benchmarks = {
        model: _row_by(raw.benchmark_cpu_batch1, "benchmark_cpu_batch1.csv", model=model)
        for model in ("vit", "resnet50")
    }
    for model in ("vit", "resnet50"):
        for column in ("validation_macro_f1", "test_accuracy", "test_macro_f1", "cpu_batch1_median_ms", "cpu_batch1_p95_ms"):
            if not math.isclose(
                _number(comparison[model].get(column), f"model_comparison {model}.{column}"),
                _number(summaries[model].get(column), f"metrics_summary {model}.{column}"),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise EvidenceError(f"Model comparison and summary disagree for {model}.{column}.")

    latency = raw.latency
    vit_benchmark = benchmarks["vit"]
    latency_pairs = (
        ("classifier_mean_ms", "mean_ms"),
        ("classifier_median_ms", "median_ms"),
        ("classifier_p95_ms", "p95_ms"),
        ("classifier_min_ms", "min_ms"),
        ("classifier_max_ms", "max_ms"),
    )
    for json_key, csv_key in latency_pairs:
        if not math.isclose(
            _number(latency.get(json_key), f"latency.json.{json_key}"),
            _number(vit_benchmark.get(csv_key), f"benchmark_cpu_batch1 vit.{csv_key}"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise EvidenceError(f"Latency artifacts disagree for {json_key}.")
    if str(latency.get("model")) != "vit" or str(latency.get("model_version")) != model_version:
        raise EvidenceError("latency.json does not describe the primary ViT artifact.")
    if str(latency.get("protocol")) != str(vit_benchmark.get("scope")):
        raise EvidenceError("Latency protocol differs between latency.json and benchmark_cpu_batch1.csv.")
    classifier_p95_ms = _number(latency.get("classifier_p95_ms"), "latency.json.classifier_p95_ms")
    proposal_latency_ceiling_ms = static.proposal_targets.latency_ceiling_ms
    latency_assessment = "exceeds" if not classifier_p95_ms < proposal_latency_ceiling_ms else "meets"

    macro_f1_ci = _require_mapping(metrics.get("group_bootstrap_95pct"), "metrics.json.group_bootstrap_95pct")
    ci_values = _require_list(macro_f1_ci.get("macro_f1_95pct"), "metrics macro_f1_95pct")
    if len(ci_values) != 2:
        raise EvidenceError("metrics.json macro_f1_95pct must have exactly two bounds.")
    ci_low = _number(ci_values[0], "metrics macro F1 CI lower")
    ci_high = _number(ci_values[1], "metrics macro F1 CI upper")
    replicates = _integer(macro_f1_ci.get("replicates"), "metrics bootstrap replicates")

    label_names = {
        "safe": "Safe",
        "firearms": "Firearms",
        "explosives": "Explosives",
        "financial_promotion": "Financial promo",
    }
    per_class_rows = tuple(
        (
            label_names[label],
            _decimal(row.get("precision"), f"per-class {label}.precision"),
            _decimal(row.get("recall"), f"per-class {label}.recall"),
            _decimal(row.get("f1"), f"per-class {label}.f1"),
            str(_integer(row.get("support"), f"per-class {label}.support")),
        )
        for label, row in zip(EXPECTED_LABELS, per_class_test)
    )
    confusion_rows = tuple(
        (
            label_names[label],
            *(str(_integer(row.get(predicted), f"confusion {label}->{predicted}")) for predicted in EXPECTED_LABELS),
        )
        for label, row in zip(EXPECTED_LABELS, matrix_rows)
    )
    comparison_rows = tuple(
        (
            "ViT primary" if model == "vit" else "ResNet50 baseline",
            _decimal(comparison[model].get("validation_macro_f1"), f"comparison {model} val F1"),
            _decimal(comparison[model].get("test_macro_f1"), f"comparison {model} test F1"),
            _decimal(comparison[model].get("test_accuracy"), f"comparison {model} test accuracy"),
            _ms(comparison[model].get("cpu_batch1_p95_ms"), f"comparison {model} p95"),
        )
        for model in ("vit", "resnet50")
    )

    failure_description = "No primary-model errors were saved."
    if incorrect:
        first_failure = incorrect[0]
        failure_description = (
            f"The saved failure was {first_failure.get('true_label')} classified as "
            f"{first_failure.get('predicted_label')} ({len(incorrect)}/{len(raw.predictions)} test images)."
        )

    evaluation_limitations = [str(item) for item in _require_list(evaluation.get("limitations"), "evaluation limitations")]
    selected_limitations = (
        evaluation_limitations[0],
        evaluation_limitations[1],
        evaluation_limitations[3],
        evaluation_limitations[4],
        evaluation_limitations[5],
        evaluation_limitations[7],
        evaluation_limitations[8],
        evaluation_limitations[9],
        str(static.policy["limitations"][2]),
    )

    external_metrics: tuple[MetricValue, ...] | None = None
    external_scope: str | None = None
    if raw.external_spot_check is not None:
        external = raw.external_spot_check
        external_scope = str(external.get("scope"))
        detector_recall = _number(
            external.get("weapon_detector_image_recall"), "external weapon_detector_image_recall"
        )
        detector_false_positive_rate = _number(
            external.get("weapon_detector_nonweapon_false_positive_rate"),
            "external weapon_detector_nonweapon_false_positive_rate",
        )
        external_scope += (
            f" Detector image recall was {detector_recall:.1%}, but the non-weapon detector false-positive rate "
            f"was {detector_false_positive_rate:.1%}; this is a generalization warning, not formal test evidence."
        )
        external_metrics = (
            MetricValue("Samples", str(_integer(external.get("sample_count"), "external sample_count"))),
            MetricValue("Classifier accuracy", _pct(external.get("classifier_accuracy"), "external accuracy")),
            MetricValue("Macro F1", _pct(external.get("classifier_macro_f1"), "external macro F1")),
            MetricValue("Detector nonweapon FP", _pct(detector_false_positive_rate, "external detector FPR")),
        )

    threshold_text = "; ".join(
        f"{label.replace('_', ' ')} {_number(row['threshold'], f'threshold {label}'):.4f}"
        for label, row in threshold_rows.items()
    )
    total_parameters = _integer(backbone.get("total_parameters"), "ViT total parameters")
    feature_dim = _integer(backbone.get("feature_dim"), "ViT feature dimension")
    training_items = (
        ("Split isolation", "192 train fit; 48 validation calibration; 48 test once"),
        ("Feature run", f"{feature_dim}-D; batch {extraction.get('batch_size')}; {extraction.get('device')}"),
        ("Head", str(head.get("type"))),
        ("Head settings", f"C={head.get('C')}; {head.get('solver')}; {head.get('class_weight')}; max_iter={head.get('max_iter')}"),
        ("Frozen backbone", f"{total_parameters / 1_000_000:.2f}M parameters; 0 trainable"),
        ("Artifact thresholds", threshold_text),
        (
            "Threshold tuning",
            f"validation recall >={threshold_tuning_floor:.2f}; distinct from proposal acceptance",
        ),
        (
            "Recorded timing",
            f"features {_number(extraction.get('elapsed_seconds'), 'ViT extraction seconds'):.3f}s; "
            f"head fit {_number(head.get('fit_seconds'), 'ViT head fit seconds'):.4f}s",
        ),
    )

    exclusion_items = tuple(
        str(item) for item in _require_list(latency.get("exclusions"), "latency exclusions")
    )
    required_end_to_end_exclusions = {
        "file decoding",
        "OCR",
        "object detection",
        "application rendering",
    }
    missing_exclusions = required_end_to_end_exclusions - set(exclusion_items)
    if missing_exclusions:
        raise EvidenceError(
            "latency.json cannot support the full end-to-end boundary; missing exclusions: "
            + ", ".join(sorted(missing_exclusions))
        )
    exclusions = ", ".join(exclusion_items)
    benchmark_runs = _integer(latency.get("measured_runs"), "latency measured_runs")
    warmups = _integer(latency.get("warmup_runs"), "latency warmup_runs")
    throughput = _number(vit_benchmark.get("throughput_images_per_second_from_mean"), "ViT throughput")
    created_at = str(run.get("created_at_utc", ""))
    if not created_at:
        raise EvidenceError("evaluation_metrics.json.run has no created_at_utc.")

    return MeasuredEvidence(
        selected_model=str(primary.get("display_name")),
        model_version=model_version,
        evaluation_scope=(
            f"Untouched grouped test: n={len(raw.predictions)} ({static.dataset.label_split_counts['safe']['test']} per class). "
            f"Macro AP={_number(metrics.get('macro_average_precision_ovr'), 'metrics macro AP'):.3f}; "
            f"group-bootstrap macro F1 95% interval={ci_low:.3f} to {ci_high:.3f} "
            f"({replicates:,} source-group replicates)."
        ),
        architecture_paragraphs=(
            f"A pinned ViT-B/16 backbone ({total_parameters:,} parameters, all frozen) converts each RGB creative into a "
            f"{feature_dim}-dimensional embedding. A StandardScaler and multinomial logistic head produce four class probabilities.",
            "Grounding DINO and Tesseract are optional evidence channels. The policy layer applies noisy-OR fusion to restricted cues; "
            "only the calibrated classifier can trigger an automatic firearms or explosives block.",
        ),
        math_lines=(
            f"z = f_theta(x) in R^{feature_dim};  u = (z - mu) / sigma",
            "p(y=k|x) = exp(w_k^T u + b_k) / sum_j exp(w_j^T u + b_j)",
            "fused_score_k = 1 - product_i (1 - evidence_k,i)",
        ),
        preprocessing_steps=(
            f"The formal registry audit confirms {static.dataset.row_count} hash-verified JPEGs with a maximum saved side of "
            f"{max(static.dataset.width_range[1], static.dataset.height_range[1])} pixels.",
            "Source or generated campaign groups stay within one split; the exact-file and group overlap counts are both zero.",
            "Decoded RGB images use the pinned timm backbone transform before frozen feature extraction; scaling is learned on train embeddings only.",
        ),
        training_items=training_items,
        overall_metrics=(
            MetricValue("Test accuracy", _pct(metrics.get("accuracy"), "metrics accuracy"), "47 / 48 correct"),
            MetricValue("Macro F1", _pct(metrics.get("macro_f1"), "metrics macro_f1"), "four classes"),
            MetricValue(
                "Argmax restricted mean",
                f"{argmax_restricted_mean:.6f}",
                "multiclass class-average",
            ),
            MetricValue(
                "Threshold-op mean",
                f"{threshold_operating_mean:.6f}",
                "validation-selected thresholds",
            ),
            MetricValue(
                "Safe-class precision",
                f"{safe_precision:.3f}",
                f"proposal >{proposal_safe_precision_floor:.2f}: {safe_precision_assessment}",
            ),
            MetricValue(
                "Safe-ad false flag",
                f"{100.0 * safe_false_positive_rate:.1f}%",
                "1 - Safe recall",
            ),
        ),
        performance_definition_note=(
            f"The {argmax_restricted_mean:.6f} figure is the multiclass-argmax class-average recall across "
            "firearms, explosives, and financial promotion; it is not the final threshold operating point. "
            f"Safe-class precision is {safe_precision:.3f} and {safe_precision_assessment} the proposal "
            f">{proposal_safe_precision_floor:.2f} precision target; it is distinct from the "
            f"{100.0 * safe_false_positive_rate:.1f}% Safe-ad false-flag rate (1 - Safe recall). "
            f"Threshold selection used a >={threshold_tuning_floor:.2f} validation-recall tuning floor, "
            f"distinct from the proposal per-class >{proposal_recall_floor:.2f} acceptance target. "
            "At thresholds selected on validation, untouched-test recalls are "
            f"firearms {threshold_operating_recall['firearms']:.6f}, "
            f"explosives {threshold_operating_recall['explosives']:.6f}, and financial promotion "
            f"{threshold_operating_recall['financial_promotion']:.6f}; mean {threshold_operating_mean:.6f}, "
            f"worst class {threshold_operating_worst:.6f}. The proposal criterion of per-class recall "
            f">{proposal_recall_floor:.2f} {recall_assessment}{recall_failure_detail}."
        ),
        per_class_headers=("Class", "P", "R", "F1", "n"),
        per_class_rows=per_class_rows,
        confusion_headers=("True / pred", "Safe", "Fire", "Expl", "Fin"),
        confusion_rows=confusion_rows,
        comparison_headers=("Model", "Val F1", "Test F1", "Test acc", "CPU p95"),
        comparison_rows=comparison_rows,
        latency_metrics=(
            MetricValue("Median", _ms(latency.get("classifier_median_ms"), "latency median"), "batch 1"),
            MetricValue("Classifier-path P95", f"{classifier_p95_ms:.3f} ms", "batch 1; excludes auxiliaries"),
            MetricValue("Mean", _ms(latency.get("classifier_mean_ms"), "latency mean"), "single thread"),
            MetricValue("Serial throughput", f"{throughput:.1f} img/s", "derived; not concurrent"),
        ),
        latency_note=(
            f"The measured ViT classifier-path p95 is {classifier_p95_ms:.3f} ms and {latency_assessment} the "
            f"proposal's <{proposal_latency_ceiling_ms:.0f} ms per-asset benchmark. Measured on "
            f"{latency.get('hardware')} / {latency.get('device')}: {warmups} warmups, {benchmark_runs} runs. "
            f"Full end-to-end p95 was not measured because {exclusions} are excluded. "
            "Concurrent throughput and MPS resource use were not measured. "
            f"Artifact scope: {latency.get('protocol')}."
        ),
        challenges_and_solutions=(
            (
                "Source-style confounding remains.",
                "The implementation grouped campaigns before splitting and recomputed all 288 hashes; this controls direct leakage but cannot remove source cues.",
            ),
            (
                "Threshold leakage could inflate safety claims.",
                "The head fit on train only, restricted thresholds were selected on validation only, and the 48-image test was scored once afterward.",
            ),
            (
                "One grenade crossed the visual taxonomy boundary.",
                f"{failure_description} The failure remains in the audit table; ambiguous restricted evidence routes to review rather than being hidden.",
            ),
            (
                "The measured classifier path misses the proposal latency benchmark.",
                f"ViT classifier-path CPU p95 was {classifier_p95_ms:.3f} ms, above "
                f"{proposal_latency_ceiling_ms:.0f} ms, versus "
                f"{_number(comparison['resnet50']['cpu_batch1_p95_ms'], 'ResNet p95'):.1f} ms for the CNN baseline. "
                "Neither value is a full end-to-end p95.",
            ),
        ),
        limitations=selected_limitations,
        reproducibility_items=(
            ("Run", f"{created_at} | seed {run.get('seed')}"),
            ("Host", f"{run.get('platform')} | feature device {run.get('actual_feature_device')}"),
            ("Registry SHA-256", registry_sha[:16] + "..."),
            ("Backbone revision", str(backbone.get("revision"))[:16] + "..."),
            ("Artifact SHA-256", str(primary.get("artifact_sha256"))[:16] + "..."),
            ("Independent check", f"{independent.get('registry_hashes_recomputed')} image hashes; reload max diff 0"),
        ),
        external_diagnostic=external_metrics,
        external_diagnostic_scope=external_scope,
    )


class Palette:
    NAVY = colors.HexColor("#14213D")
    BLUE = colors.HexColor("#3156B3")
    TEAL = colors.HexColor("#18856F")
    RED = colors.HexColor("#A43449")
    GOLD = colors.HexColor("#B67812")
    INK = colors.HexColor("#1E2B43")
    MUTED = colors.HexColor("#5F6B7A")
    LINE = colors.HexColor("#DDE4EF")
    PAPER = colors.HexColor("#F6F8FC")
    PALE_BLUE = colors.HexColor("#EAF0FF")
    PALE_TEAL = colors.HexColor("#EAF8F3")
    PALE_GOLD = colors.HexColor("#FFF7DF")
    WHITE = colors.white


def build_styles() -> Mapping[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "SynopsisBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.0,
            leading=9.8,
            textColor=Palette.INK,
            spaceAfter=3.5,
            allowWidows=0,
            allowOrphans=0,
        ),
        "small": ParagraphStyle(
            "SynopsisSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=8.35,
            textColor=Palette.MUTED,
            spaceAfter=2.0,
        ),
        "bullet": ParagraphStyle(
            "SynopsisBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9.65,
            leftIndent=9,
            firstLineIndent=-6,
            bulletIndent=0,
            textColor=Palette.INK,
            spaceAfter=2.0,
        ),
        "section": ParagraphStyle(
            "SynopsisSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=8.8,
            leading=10.3,
            textColor=Palette.NAVY,
            spaceAfter=0,
            uppercase=True,
            letterSpacing=0.4,
        ),
        "card": ParagraphStyle(
            "SynopsisCard",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.75,
            leading=7.6,
            textColor=Palette.MUTED,
            alignment=TA_LEFT,
        ),
        "metric_label": ParagraphStyle(
            "SynopsisMetricLabel",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.75,
            leading=7.5,
            textColor=Palette.MUTED,
            alignment=TA_LEFT,
            spaceAfter=1.5,
        ),
        "metric_value": ParagraphStyle(
            "SynopsisMetricValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=12.0,
            leading=12.8,
            textColor=Palette.NAVY,
            alignment=TA_LEFT,
            spaceAfter=1.0,
        ),
        "metric_note": ParagraphStyle(
            "SynopsisMetricNote",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.5,
            leading=7.25,
            textColor=Palette.MUTED,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "SynopsisTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.5,
            leading=7.4,
            textColor=Palette.WHITE,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "SynopsisTableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.5,
            leading=7.4,
            textColor=Palette.INK,
            alignment=TA_CENTER,
        ),
        "math": ParagraphStyle(
            "SynopsisMath",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=6.8,
            leading=8.55,
            textColor=Palette.NAVY,
            backColor=Palette.PALE_BLUE,
            borderPadding=(4, 5, 4, 5),
            borderColor=Palette.LINE,
            borderWidth=0.5,
            borderRadius=4,
            spaceAfter=2,
        ),
        "reference": ParagraphStyle(
            "SynopsisReference",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.5,
            leading=7.6,
            textColor=Palette.MUTED,
            leftIndent=8,
            firstLineIndent=-8,
            spaceAfter=1.2,
        ),
    }


class SectionRule(Flowable):
    def __init__(self, title: str, styles: Mapping[str, ParagraphStyle]) -> None:
        super().__init__()
        self.title = title.upper()
        self.styles = styles
        self.height = 17
        self.width = 0

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = available_width
        return available_width, self.height

    def draw(self) -> None:
        self.canv.setFillColor(Palette.BLUE)
        self.canv.roundRect(0, 4.5, 3.2, 10, 1.6, fill=1, stroke=0)
        paragraph = Paragraph(html.escape(self.title), self.styles["section"])
        paragraph.wrapOn(self.canv, self.width - 10, 12)
        paragraph.drawOn(self.canv, 9, 5)
        self.canv.setStrokeColor(Palette.LINE)
        self.canv.setLineWidth(0.45)
        self.canv.line(0, 1, self.width, 1)


def paragraph(text: str, styles: Mapping[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str, styles: Mapping[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, styles["bullet"], bulletText="-")


def section(title: str, styles: Mapping[str, ParagraphStyle], *flowables: Flowable) -> list[Flowable]:
    if not flowables:
        raise ValueError(f"Section {title!r} has no content.")
    return [SectionRule(title, styles), *flowables, Spacer(1, 2.5)]


def key_value_table(
    rows: Sequence[tuple[str, str]],
    styles: Mapping[str, ParagraphStyle],
    width: float,
) -> Table:
    data = [
        [
            Paragraph(f"<b>{html.escape(str(key))}</b>", styles["small"]),
            Paragraph(html.escape(str(value)), styles["small"]),
        ]
        for key, value in rows
    ]
    table = Table(data, colWidths=[width * 0.34, width * 0.66], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), Palette.PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.45, Palette.LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, Palette.LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def metric_grid(
    metrics: Sequence[MetricValue],
    styles: Mapping[str, ParagraphStyle],
    width: float,
) -> Table:
    if not 1 <= len(metrics) <= 6:
        raise LayoutError(f"Metric grid requires 1 to 6 values, received {len(metrics)}.")
    cards: list[list[Flowable]] = []
    for metric in metrics:
        card: list[Flowable] = [
            Paragraph(html.escape(metric.label.upper()), styles["metric_label"]),
            Paragraph(html.escape(metric.value), styles["metric_value"]),
        ]
        if metric.note:
            card.append(Paragraph(html.escape(metric.note), styles["metric_note"]))
        cards.append(card)
    columns = 2
    while len(cards) % columns:
        cards.append([Paragraph("", styles["card"])])
    data = [cards[index : index + columns] for index in range(0, len(cards), columns)]
    table = Table(data, colWidths=[(width - 6) / 2] * 2, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, Palette.LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, Palette.LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def evidence_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    styles: Mapping[str, ParagraphStyle],
    width: float,
    column_weights: Sequence[float] | None = None,
) -> Table:
    if not headers or not rows:
        raise LayoutError("Evidence table requires non-empty headers and rows.")
    if any(len(row) != len(headers) for row in rows):
        raise LayoutError("Evidence table row width does not match its header width.")
    weights = tuple(column_weights or [1.0] * len(headers))
    if len(weights) != len(headers) or any(weight <= 0 for weight in weights):
        raise LayoutError("Evidence table has invalid column weights.")
    total = sum(weights)
    widths = [width * weight / total for weight in weights]
    data: list[list[Paragraph]] = [
        [Paragraph(html.escape(str(value)), styles["table_header"]) for value in headers]
    ]
    data.extend(
        [Paragraph(html.escape(str(value)), styles["table_cell"]) for value in row]
        for row in rows
    )
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), Palette.NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, Palette.PAPER]),
                ("BOX", (0, 0), (-1, -1), 0.45, Palette.LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, Palette.LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
            ]
        )
    )
    return table


def _dataset_distribution_table(
    dataset: DatasetEvidence,
    policy: Mapping[str, Any],
    styles: Mapping[str, ParagraphStyle],
    width: float,
) -> Table:
    display_names = policy.get("label_display_names", {})
    headers = ("Label", "Train", "Val", "Test", "Total")
    rows: list[tuple[str, ...]] = []
    for label in EXPECTED_LABELS:
        counts = dataset.label_split_counts[label]
        rows.append(
            (
                str(display_names.get(label, label.replace("_", " "))),
                str(counts["train"]),
                str(counts["val"]),
                str(counts["test"]),
                str(sum(counts.values())),
            )
        )
    return evidence_table(headers, rows, styles, width, column_weights=(2.5, 1, 1, 1, 1))


def _reference_flowables(
    references: Sequence[Reference],
    styles: Mapping[str, ParagraphStyle],
    limit: int = 9,
) -> list[Flowable]:
    priorities = (
        "Ad Product Taxonomy",
        "An Image is Worth",
        "Grounding DINO",
        "ViT-B/16",
        "ResNet-50",
        "ADautoGen-DS",
        "Weapons Set 1",
        "Tesseract",
        "MPS backend",
    )
    selected: list[Reference] = []
    for phrase in priorities:
        match = next((item for item in references if phrase.casefold() in item.title.casefold()), None)
        if match and match not in selected:
            selected.append(match)
    for item in references:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    flowables: list[Flowable] = []
    for index, reference in enumerate(selected[:limit], start=1):
        escaped_title = html.escape(reference.title)
        escaped_url = html.escape(reference.url, quote=True)
        flowables.append(
            Paragraph(
                f"[{index}] <link href='{escaped_url}' color='#3156B3'>{escaped_title}</link><br/>"
                f"<font color='#5F6B7A'>{escaped_url}</font>",
                styles["reference"],
            )
        )
    return flowables


def build_page_columns(
    evidence: SynopsisEvidence,
    styles: Mapping[str, ParagraphStyle],
    column_width: float,
) -> tuple[tuple[list[Flowable], list[Flowable]], tuple[list[Flowable], list[Flowable]]]:
    dataset = evidence.static.dataset
    policy = evidence.static.policy
    measured = evidence.measured
    thresholds = policy["default_thresholds"]

    objective_text = (
        "Static ad creatives can contain restricted visual objects, visible promotion language, or ambiguous context. "
        "This pilot converts one local image into an auditable <b>APPROVE</b>, <b>REVIEW</b>, or <b>BLOCK</b> triage result. "
        "It does not infer legality, audience eligibility, landing-page claims, or jurisdictional compliance."
    )
    objective_bullets = (
        "Classify four bounded visual labels on an untouched grouped test split.",
        "Fuse optional object and OCR evidence without allowing auxiliary cues to trigger an automatic block.",
        "Expose the verdict, raw and fused score distributions, evidence provenance, stage timing, overlays, and a downloadable audit record.",
    )

    source_summary = ", ".join(
        f"{name} ({count})" for name, count in list(dataset.source_counts.items())[:4]
    )
    mime_summary = ", ".join(f"{name}: {count}" for name, count in dataset.mime_counts.items())
    dataset_text = (
        f"The formal registry contains <b>{dataset.row_count}</b> images in <b>{dataset.group_count}</b> source or campaign groups. "
        f"Grouped splitting produced train/validation/test counts of {dataset.split_counts['train']}/"
        f"{dataset.split_counts['val']}/{dataset.split_counts['test']}. Verified overlap across splits: "
        f"{dataset.group_overlap_across_splits} source groups and {dataset.exact_hash_overlap_across_splits} exact hashes."
    )
    observed_text = (
        f"Registry-observed dimensions span {dataset.width_range[0]}-{dataset.width_range[1]} px wide and "
        f"{dataset.height_range[0]}-{dataset.height_range[1]} px high. Encodings: {html.escape(mime_summary)}. "
        f"Synthetic-declared rows: {dataset.synthetic_count}/{dataset.row_count}. Sources: {html.escape(source_summary)}."
    )

    page1_left: list[Flowable] = []
    page1_left.extend(
        section(
            "1. Problem and objectives",
            styles,
            paragraph(objective_text, styles),
            *(bullet(html.escape(item), styles) for item in objective_bullets),
        )
    )
    page1_left.extend(
        section(
            "2. Dataset and preprocessing",
            styles,
            paragraph(dataset_text, styles),
            _dataset_distribution_table(dataset, policy, styles, column_width),
            Spacer(1, 2),
            paragraph(observed_text, styles, "small"),
            *(bullet(html.escape(item), styles) for item in measured.preprocessing_steps),
        )
    )
    architecture_flowables: list[Flowable] = [
        paragraph(html.escape(item), styles) for item in measured.architecture_paragraphs
    ]
    architecture_flowables.extend(
        Paragraph(html.escape(line), styles["math"]) for line in measured.math_lines
    )
    page1_left.extend(section("3. Model architecture and math", styles, *architecture_flowables))

    training_rows = list(measured.training_items)
    training_rows.extend(
        [
            ("Policy", str(policy["policy_version"])),
            ("Auxiliary review", f"fused restricted evidence {thresholds['review']:.2f}; margin {thresholds['uncertainty_margin']:.2f}"),
        ]
    )
    page1_right: list[Flowable] = []
    page1_right.extend(
        section(
            "4. Training and hyperparameters",
            styles,
            paragraph(
                f"Selected runtime model: <b>{html.escape(measured.selected_model)}</b> "
                f"({html.escape(measured.model_version)}). Test data remained outside fitting and threshold selection.",
                styles,
            ),
            key_value_table(training_rows, styles, column_width),
        )
    )
    page1_right.extend(
        section(
            "5. Detailed measured performance",
            styles,
            paragraph(html.escape(measured.evaluation_scope), styles, "small"),
            metric_grid(measured.overall_metrics, styles, column_width),
            paragraph(html.escape(measured.performance_definition_note), styles, "small"),
            Spacer(1, 3),
            evidence_table(
                measured.per_class_headers,
                measured.per_class_rows,
                styles,
                column_width,
                column_weights=(2.0, *([1.0] * (len(measured.per_class_headers) - 1))),
            ),
            Spacer(1, 3),
            paragraph("Untouched test confusion matrix (counts)", styles, "small"),
            evidence_table(
                measured.confusion_headers,
                measured.confusion_rows,
                styles,
                column_width,
                column_weights=(1.7, *([1.0] * (len(measured.confusion_headers) - 1))),
            ),
        )
    )

    page2_left: list[Flowable] = []
    page2_left.extend(
        section(
            "6. Baseline and backbone comparison",
            styles,
            paragraph(
                "The comparison below is reproduced from the saved model-selection artifact. "
                "Its column labels preserve whether each score is validation or test evidence.",
                styles,
            ),
            evidence_table(
                measured.comparison_headers,
                measured.comparison_rows,
                styles,
                column_width,
                column_weights=(2.0, *([1.0] * (len(measured.comparison_headers) - 1))),
            ),
        )
    )
    page2_left.extend(
        section(
            "7. Measured latency",
            styles,
            metric_grid(measured.latency_metrics, styles, column_width),
            paragraph(html.escape(measured.latency_note), styles, "small"),
        )
    )
    challenge_flowables: list[Flowable] = []
    for challenge, solution in measured.challenges_and_solutions:
        challenge_flowables.append(
            paragraph(
                f"<b>{html.escape(challenge)}</b><br/><font color='#5F6B7A'>{html.escape(solution)}</font>",
                styles,
                "body",
            )
        )
    page2_left.extend(section("8. Challenges and implemented solutions", styles, *challenge_flowables))
    if measured.external_diagnostic is not None:
        page2_left.extend(
            section(
                "External diagnostic only",
                styles,
                paragraph(html.escape(measured.external_diagnostic_scope or ""), styles, "small"),
                metric_grid(measured.external_diagnostic, styles, column_width),
            )
        )

    limitations = list(dict.fromkeys(measured.limitations))
    page2_right: list[Flowable] = []
    page2_right.extend(
        section(
            "9. Limitations and deployment boundary",
            styles,
            *(bullet(html.escape(item), styles) for item in limitations),
        )
    )
    page2_right.extend(
        section(
            "10. Reproducibility and evidence",
            styles,
            key_value_table(measured.reproducibility_items, styles, column_width),
            paragraph(
                "Every reported metric comes from outputs/evaluation. The formal dataset claim comes only from "
                "data/capstone_registry.csv. The rejected Wikimedia registry is excluded from training and test evidence.",
                styles,
                "small",
            ),
            paragraph(
                "The Streamlit workspace visualizes the current case and saved benchmark context; its session charts are "
                "descriptive interface evidence, not production throughput, monitoring, or population-level performance evidence.",
                styles,
                "small",
            ),
        )
    )
    page2_right.extend(
        section(
            "11. References",
            styles,
            *_reference_flowables(evidence.static.references, styles),
        )
    )
    return ((page1_left, page1_right), (page2_left, page2_right))


def draw_page_chrome(pdf: canvas.Canvas, page_number: int) -> None:
    width, height = LETTER
    pdf.setFillColor(Palette.PAPER)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.setFillColor(Palette.NAVY)
    pdf.rect(0, height - 94, width, 94, fill=1, stroke=0)
    pdf.setFillColor(Palette.BLUE)
    pdf.roundRect(34, height - 30, 70, 13, 6.5, fill=1, stroke=0)
    pdf.setFillColor(Palette.WHITE)
    pdf.setFont("Helvetica-Bold", 6.6)
    pdf.drawCentredString(69, height - 25.5, "EVIDENCE-LED PILOT")
    pdf.setFont("Helvetica-Bold", 17.5)
    pdf.drawString(34, height - 51, "Ad Safety Moderation Pipeline")
    pdf.setFont("Helvetica", 7.2)
    pdf.setFillColor(colors.HexColor("#D7E1F5"))
    pdf.drawString(34, height - 66, "Strict two-page technical synopsis | MS_DSP_462 Computer Vision")
    team = " | ".join(TEAM_NAMES)
    pdf.setFont("Helvetica", 6.5)
    pdf.drawString(34, height - 82, team)
    pdf.setFillColor(Palette.WHITE)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawRightString(width - 34, height - 26, f"PAGE {page_number} / 2")
    pdf.setFont("Helvetica", 6.5)
    pdf.setFillColor(colors.HexColor("#D7E1F5"))
    pdf.drawRightString(width - 34, height - 67, "Measured results | proposal benchmarks labeled")

    footer_y = 19
    pdf.setStrokeColor(Palette.LINE)
    pdf.setLineWidth(0.5)
    pdf.line(34, footer_y + 9, width - 34, footer_y + 9)
    pdf.setFillColor(Palette.MUTED)
    pdf.setFont("Helvetica", 6.0)
    pdf.drawString(34, footer_y, "Generated from proposal, saved registry, policy, model manifest, sources, and evaluation artifacts")
    pdf.drawRightString(width - 34, footer_y, "Course pilot | Not production-ready")


def _add_column(
    pdf: canvas.Canvas,
    flowables: Sequence[Flowable],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    page_number: int,
    column_name: str,
) -> None:
    remaining = list(flowables)
    frame = Frame(
        x,
        y,
        width,
        height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        showBoundary=0,
        id=f"page-{page_number}-{column_name}",
    )
    frame.addFromList(remaining, pdf)
    if remaining:
        first = remaining[0]
        raise LayoutError(
            f"Page {page_number} {column_name} column overflowed at {type(first).__name__}. "
            "Reduce content or typography; the generator will not create a third page."
        )


def count_pdf_pages(path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except ModuleNotFoundError:
        payload = path.read_bytes()
        count = len(re.findall(rb"/Type\s*/Page(?!s)\b", payload))
        if count <= 0:
            raise LayoutError("Could not verify PDF page count. Install pypdf for strict validation.")
        return count


def render_pdf(evidence: SynopsisEvidence, output: Path) -> None:
    output = output.resolve()
    report_root = REPORT_DIR.resolve()
    if output != report_root and report_root not in output.parents:
        raise EvidenceError(f"Output must stay under {_relative(REPORT_DIR)}: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.building.pdf")
    if temporary.exists():
        temporary.unlink()
    styles = build_styles()
    page_width, page_height = LETTER
    left_margin = 34.0
    right_margin = 34.0
    gutter = 16.0
    column_width = (page_width - left_margin - right_margin - gutter) / 2.0
    frame_y = 34.0
    frame_top = page_height - 98.0
    frame_height = frame_top - frame_y
    pages = build_page_columns(evidence, styles, column_width)
    if len(pages) != 2 or any(len(columns) != 2 for columns in pages):
        raise LayoutError("Renderer must receive exactly two pages with two columns each.")

    try:
        pdf = canvas.Canvas(str(temporary), pagesize=LETTER, pageCompression=1)
        pdf.setTitle("Ad Safety Moderation Pipeline - Technical Synopsis")
        pdf.setAuthor(", ".join(TEAM_NAMES))
        pdf.setSubject("Evidence-led two-page computer vision capstone synopsis")
        for page_number, (left_story, right_story) in enumerate(pages, start=1):
            draw_page_chrome(pdf, page_number)
            _add_column(
                pdf,
                left_story,
                x=left_margin,
                y=frame_y,
                width=column_width,
                height=frame_height,
                page_number=page_number,
                column_name="left",
            )
            _add_column(
                pdf,
                right_story,
                x=left_margin + column_width + gutter,
                y=frame_y,
                width=column_width,
                height=frame_height,
                page_number=page_number,
                column_name="right",
            )
            pdf.showPage()
        pdf.save()
        page_count = count_pdf_pages(temporary)
        if page_count != 2:
            raise LayoutError(f"Pagination invariant failed: generated {page_count} pages, expected exactly 2.")
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def load_evidence() -> SynopsisEvidence:
    validate_required_files()
    static = StaticEvidence(
        dataset=load_dataset_evidence(),
        proposal_targets=load_proposal_targets(),
        policy=load_policy(),
        model_manifest=load_model_manifest(),
        references=load_references(),
    )
    raw = load_raw_evaluation()
    measured = map_measured_evidence(raw, static)
    return SynopsisEvidence(static=static, measured=measured)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PDF path under {_relative(REPORT_DIR)} (default: {_relative(DEFAULT_OUTPUT)})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate every evidence input and schema mapping without writing the PDF.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evidence = load_evidence()
        if args.validate_only:
            print("PASS: synopsis evidence gate and exact schema mapping are valid.")
            return 0
        render_pdf(evidence, args.output)
        print(f"Saved {_relative(args.output.resolve())}")
        print("PASS: generated exactly two US Letter portrait pages.")
        return 0
    except (EvidenceError, LayoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
