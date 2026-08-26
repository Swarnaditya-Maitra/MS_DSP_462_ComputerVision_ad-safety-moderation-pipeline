#!/usr/bin/env python3
"""Independently validate every required Ad Safety capstone deliverable."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import timm
import torch
from PIL import Image, ImageStat
from pypdf import PdfReader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "validation"
RESULT_PATH = OUTPUT_DIR / "final_validation.json"
MARKER_PATH = OUTPUT_DIR / "FINAL_VALIDATION_PASSED.txt"

EXPECTED_PINNED_MODELS: dict[str, dict[str, Any]] = {
    "vit": {
        "repo_id": "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "revision": "063c6c38a5d8510b2e57df480445e94b231dad2c",
        "model_name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "feature_dim": 768,
        "weight_sha256": "32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b",
        "snapshot_sha256": "e19b7516e8c2b523f452598f6add160b8f80450f1412feb68924e6a957a1ea9f",
        "snapshot_bytes": 692616164,
        "snapshot_files": 5,
    },
    "resnet50": {
        "repo_id": "timm/resnet50.a1_in1k",
        "revision": "767268603ca0cb0bfe326fa87277f19c419566ef",
        "model_name": "resnet50.a1_in1k",
        "feature_dim": 2048,
        "weight_sha256": "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb",
        "snapshot_sha256": "4421e496a8e63ea21d10d905320a5d8d1442996f91bed429e9f2f0f000390488",
        "snapshot_bytes": 205055737,
        "snapshot_files": 5,
    },
    "grounding_dino": {
        "repo_id": "IDEA-Research/grounding-dino-tiny",
        "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "model_name": "grounding-dino-tiny",
        "feature_dim": None,
        "weight_sha256": "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3",
        "snapshot_sha256": "1d9b6d4538c3dfbc30cc8c1a9defeedfde8e57e540c4901e4ee3e6bf16b804d8",
        "snapshot_bytes": 1382224246,
        "snapshot_files": 11,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_fingerprint(snapshot_path: Path) -> tuple[str, int, int]:
    """Hash ordered relative paths and bytes for one pinned local snapshot."""
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(item for item in snapshot_path.rglob("*") if item.is_file()):
        digest.update(path.relative_to(snapshot_path).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        file_count += 1
    return digest.hexdigest(), total_bytes, file_count


def decoded_dhash(path: Path) -> str:
    """Compute a 64-bit dHash from the current decoded saved image."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.get_flattened_data())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return f"{bits:016x}"


def image_pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        payload = f"{rgb.width}x{rgb.height}:RGB:".encode("ascii") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def rendered_image_evidence(path: Path) -> dict[str, Any]:
    """Return portable evidence that a freshly rendered image is decodable and nonblank."""
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        sample = rgb.copy()
        sample.thumbnail((256, 256), Image.Resampling.LANCZOS)
        gray = sample.convert("L")
        gray_extrema = list(gray.getextrema())
        gray_stddev = float(ImageStat.Stat(gray).stddev[0])
        return {
            "width": rgb.width,
            "height": rgb.height,
            "format": image.format,
            "gray_extrema": gray_extrema,
            "gray_stddev": gray_stddev,
            "nonblank": gray_extrema[0] < 245 and gray_stddev > 3.0,
            "pixel_sha256": image_pixel_sha256(path),
        }


def portable_text(value: str) -> str:
    """Remove machine-specific absolute paths while preserving useful diagnostics."""
    replacements = (
        (str(ROOT), "."),
        (str(Path.home()), "<HOME>"),
        (str(Path(tempfile.gettempdir())), "<TMP>"),
        (str(Path(sys.prefix)), "<PYTHON_PREFIX>"),
    )
    for source, replacement in replacements:
        if source:
            value = value.replace(source, replacement)
    value = re.sub(
        r"(^|[\s\"'(=])/(?!/)(?:[^\s\"'<>|,;]+/)*([^/\s\"'<>|,;]+)",
        r"\1<ABSOLUTE_PATH>/\2",
        value,
        flags=re.MULTILINE,
    )
    value = re.sub(
        r"(^|[\s\"'(=])[A-Za-z]:\\(?:[^\s\"'<>|,;]+\\)*([^\\\s\"'<>|,;]+)",
        r"\1<ABSOLUTE_PATH>\\\2",
        value,
        flags=re.MULTILINE,
    )
    return value


def portable_payload(value: Any) -> Any:
    """Recursively make a validation payload safe to commit and share."""
    if isinstance(value, dict):
        return {portable_text(str(key)): portable_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [portable_payload(item) for item in value]
    if isinstance(value, Path):
        return portable_text(str(value))
    if isinstance(value, str):
        return portable_text(value)
    return value


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.write_text(
        json.dumps(portable_payload(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_book_in_temporary_checkout() -> dict[str, Any]:
    """Exercise the tracked book builder without creating ignored files in the checkout."""
    relative_files = (
        "scripts/build_book.py",
        "ad_safety_moderation_pipeline.ipynb",
        "app.py",
        "api.py",
        "models/vit_policy_head.joblib",
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
    with tempfile.TemporaryDirectory(prefix="ad-safety-book-validation-") as temp_dir:
        staged_root = Path(temp_dir) / "project"
        shutil.copytree(
            ROOT / "book",
            staged_root / "book",
            ignore=shutil.ignore_patterns("_build", ".DS_Store", "__pycache__", "*.pyc"),
        )
        for relative in relative_files:
            source = ROOT / relative
            destination = staged_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        build = subprocess.run(
            [
                sys.executable,
                str(staged_root / "scripts/build_book.py"),
                "--builder",
                "nbconvert",
                "--no-execute",
            ],
            cwd=staged_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        index_path = staged_root / "book/_build/html/index.html"
        chapter_path = staged_root / "book/_build/html/ad_safety_moderation_pipeline.html"
        return {
            "returncode": build.returncode,
            "output": f"{build.stdout}\n{build.stderr}".strip()[-4000:],
            "index_present": index_path.is_file() and index_path.stat().st_size > 0,
            "chapter_present": chapter_path.is_file() and chapter_path.stat().st_size > 0,
            "index_html": index_path.read_text(encoding="utf-8") if index_path.is_file() else "",
            "chapter_html": chapter_path.read_text(encoding="utf-8") if chapter_path.is_file() else "",
        }


def add_check(checks: dict[str, Any], name: str, passed: bool, **details: Any) -> None:
    checks[name] = {"passed": bool(passed), **details}


def close_enough(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def classification_facts(rows: list[dict[str, str]], labels: list[str]) -> dict[str, Any]:
    """Recompute accuracy, per-class metrics, and macro F1 from saved labels."""
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(row["true_label"] == label and row["predicted_label"] == label for row in rows)
        fp = sum(row["true_label"] != label and row["predicted_label"] == label for row in rows)
        fn = sum(row["true_label"] == label and row["predicted_label"] != label for row in rows)
        support = sum(row["true_label"] == label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    accuracy = sum(row["true_label"] == row["predicted_label"] for row in rows) / len(rows)
    macro_f1 = sum(float(per_class[label]["f1"]) for label in labels) / len(labels)
    return {"rows": len(rows), "accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


def binary_threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Recompute one binary operating point without importing training code."""
    predicted = scores >= threshold
    actual = y_true.astype(bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positives": int(np.sum(actual)),
        "negatives": int(np.sum(~actual)),
    }


def select_validation_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    recall_floor: float = 0.90,
) -> dict[str, Any]:
    """Independently reproduce the declared validation-only selection rule."""
    candidates = np.unique(np.concatenate((np.asarray([0.0, 1.0]), scores.astype(float))))
    evaluated = [
        binary_threshold_metrics(y_true, scores, float(threshold))
        for threshold in candidates
    ]
    eligible = [row for row in evaluated if row["recall"] >= recall_floor]
    if not eligible:
        raise RuntimeError("No validation threshold meets the declared recall floor")
    selected = dict(max(eligible, key=lambda row: (row["precision"], row["f1"], row["threshold"])))
    selected["selection_rule"] = (
        f"validation recall >= {recall_floor:.2f}; then maximize precision, F1, and threshold"
    )
    selected["candidate_count"] = len(evaluated)
    return selected


def canonical_policy_head() -> Pipeline:
    """Return the validator-owned, predeclared head specification."""
    return Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=462,
                ),
            ),
        ]
    )


def regenerate_backbone_embeddings(
    model_name: str,
    weight_path: Path,
    image_paths: list[Path],
    batch_size: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Regenerate features from current pixels, weights, and timm preprocessing."""
    from time import perf_counter

    torch.set_num_threads(1)
    started = perf_counter()
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        pretrained_cfg_overlay={"source": "", "file": str(weight_path)},
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().cpu()
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            tensors: list[torch.Tensor] = []
            for path in image_paths[start : start + batch_size]:
                with Image.open(path) as raw:
                    tensors.append(transform(raw.convert("RGB")))
            outputs.append(model(torch.stack(tensors)).detach().float().cpu().numpy())
    features = np.concatenate(outputs).astype(np.float32, copy=False)
    serializable_config = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in data_config.items()
        if isinstance(value, (str, int, float, bool, tuple, list)) or value is None
    }
    details = {
        "elapsed_seconds": perf_counter() - started,
        "device": "cpu",
        "batch_size": batch_size,
        "feature_shape": list(features.shape),
        "weight_path": str(weight_path.relative_to(ROOT)),
        "weight_sha256": sha256_file(weight_path),
        "timm_version": getattr(timm, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "resolved_data_config": serializable_config,
        "transform": repr(transform),
    }
    del model
    gc.collect()
    return features, details


def extract_pptx_text(xml_blob: bytes) -> str:
    root = ET.fromstring(xml_blob)
    return "\n".join((node.text or "") for node in root.iter() if node.tag.endswith("}t"))


def parse_ffmpeg_probe(video_path: Path) -> dict[str, Any]:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    text = f"{probe.stdout}\n{probe.stderr}"
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    resolution_match = re.search(r"Video:\s*([^,]+).*?\b(\d{2,5})x(\d{2,5})\b", text)
    video_codec_match = re.search(r"Video:\s*([^,\s]+)", text)
    audio_codec_match = re.search(r"Audio:\s*([^,\s]+)", text)
    if not duration_match:
        raise RuntimeError(f"Could not parse video duration from ffmpeg output:\n{text[-2500:]}")
    hours, minutes, seconds = duration_match.groups()
    duration_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return {
        "ffmpeg": ffmpeg,
        "duration_seconds": duration_seconds,
        "width": int(resolution_match.group(2)) if resolution_match else None,
        "height": int(resolution_match.group(3)) if resolution_match else None,
        "video_codec": video_codec_match.group(1) if video_codec_match else None,
        "audio_codec": audio_codec_match.group(1) if audio_codec_match else None,
        "has_video_stream": "Video:" in text,
        "has_audio_stream": "Audio:" in text,
        "has_subtitle_stream": "Subtitle:" in text,
    }


def main() -> int:
    checks: dict[str, Any] = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MARKER_PATH.exists():
        MARKER_PATH.unlink()

    required = [
        "README.md",
        "USER_MANUAL.md",
        "SOURCES.md",
        "requirements-lock.txt",
        "configs/policy.yaml",
        "app.py",
        "api.py",
        "models/pretrained_model_manifest.json",
        "models/vit_policy_head.joblib",
        "models/resnet50_policy_head.joblib",
        "outputs/evaluation/embeddings_vit.npz",
        "outputs/evaluation/embeddings_resnet50.npz",
        "data/capstone_registry.csv",
        "data/wikimedia_external_manifest.csv",
        "outputs/evaluation/evaluation_metrics.json",
        "outputs/evaluation/metrics.json",
        "outputs/evaluation/independent_validation.json",
        "outputs/evaluation/external_spot_check.json",
        "outputs/evaluation/external_spot_check.csv",
        "outputs/evaluation/test_predictions.csv",
        "outputs/evaluation/thresholds.csv",
        "outputs/evaluation/output_checksums.csv",
        "outputs/evaluation/confusion_matrix_vit.png",
        "outputs/demo_cases/case_summary.json",
        "ad_safety_moderation_pipeline.ipynb",
        "scripts/build_book.py",
        "book/README.md",
        "book/_config.yml",
        "book/_toc.yml",
        "book/index.md",
        "book/_static/custom.css",
        "outputs/report/ad_safety_technical_synopsis.pdf",
        "outputs/presentation/ad_safety_management_presentation.pptx",
        "outputs/presentation/presentation_validation.json",
        "outputs/presentation/qa_report.json",
        "outputs/presentation/artifact_tool_montage.png",
        "outputs/presentation/pptx_montage.png",
        "outputs/video/ad_safety_demo.mp4",
        "outputs/video/narration_script.md",
        "outputs/video/storyboard.json",
        "outputs/video/storyboard_actual.json",
        "outputs/video/ad_safety_demo.srt",
        "outputs/video/video_manifest.json",
        "outputs/video/narration_provenance.json",
        "outputs/video/qa_report.json",
        "outputs/app/app_shell.png",
        "outputs/app/app_safe_result.png",
        "outputs/app/app_financial_result.png",
        "outputs/app/app_firearm_result.png",
        "outputs/app/browser_validation.json",
    ]
    missing = [item for item in required if not (ROOT / item).is_file() or (ROOT / item).stat().st_size == 0]
    add_check(checks, "required_files", not missing, expected=len(required), missing=missing)
    if missing:
        result = {
            "schema_version": "1.0.0",
            "project_root": ".",
            "all_checks_passed": False,
            "check_count": len(checks),
            "checks": checks,
        }
        write_result(result)
        print("FINAL_VALIDATION_FAILED: required_files")
        print(RESULT_PATH.relative_to(ROOT).as_posix())
        return 1

    with (ROOT / "data/capstone_registry.csv").open(newline="", encoding="utf-8") as handle:
        availability_registry = list(csv.DictReader(handle))
    missing_formal_rows = [
        row for row in availability_registry if not (ROOT / row.get("local_path", "")).is_file()
    ]
    if missing_formal_rows:
        missing_by_label = Counter(row.get("label", "unknown") for row in missing_formal_rows)
        add_check(
            checks,
            "full_local_dataset_available",
            False,
            expected_images=len(availability_registry),
            present_images=len(availability_registry) - len(missing_formal_rows),
            missing_images=len(missing_formal_rows),
            missing_by_label=dict(sorted(missing_by_label.items())),
            action=(
                "Run python scripts/build_capstone_dataset.py in this full-local audit checkout, "
                "then rerun the validator. Do not commit the 216 local-only third-party images."
            ),
        )
        result = {
            "schema_version": "1.0.0",
            "project_root": ".",
            "all_checks_passed": False,
            "check_count": len(checks),
            "checks": checks,
        }
        write_result(result)
        print("FINAL_VALIDATION_FAILED: full_local_dataset_available")
        print(RESULT_PATH.relative_to(ROOT).as_posix())
        return 1
    add_check(
        checks,
        "full_local_dataset_available",
        len(availability_registry) == 288,
        expected_images=288,
        present_images=len(availability_registry),
        missing_images=0,
    )

    pretrained_manifest = json.loads(
        (ROOT / "models/pretrained_model_manifest.json").read_text(encoding="utf-8")
    )
    manifest_records = {
        str(row.get("repo_id")): row for row in pretrained_manifest.get("models", [])
    }
    pinned_model_issues: list[str] = []
    pinned_model_evidence: dict[str, Any] = {}
    for model_key, expected in EXPECTED_PINNED_MODELS.items():
        repo_id = str(expected["repo_id"])
        record = manifest_records.get(repo_id)
        if record is None:
            pinned_model_issues.append(f"{model_key}:manifest_record")
            continue
        expected_relative_snapshot = Path(
            ".cache/huggingface/hub"
        ) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / str(expected["revision"])
        if Path(str(record.get("local_snapshot", ""))) != expected_relative_snapshot:
            pinned_model_issues.append(f"{model_key}:snapshot_path")
        snapshot_path = ROOT / expected_relative_snapshot
        if not snapshot_path.is_dir():
            pinned_model_issues.append(f"{model_key}:snapshot_missing")
            continue
        actual_snapshot_sha256, actual_snapshot_bytes, actual_snapshot_files = snapshot_fingerprint(
            snapshot_path
        )
        weight_path = snapshot_path / "model.safetensors"
        actual_weight_sha256 = sha256_file(weight_path) if weight_path.is_file() else None
        expected_fields = {
            "revision": expected["revision"],
            "snapshot_sha256": expected["snapshot_sha256"],
            "snapshot_bytes": expected["snapshot_bytes"],
            "snapshot_files": expected["snapshot_files"],
        }
        for field, expected_value in expected_fields.items():
            if record.get(field) != expected_value:
                pinned_model_issues.append(f"{model_key}:manifest:{field}")
        if actual_snapshot_sha256 != expected["snapshot_sha256"]:
            pinned_model_issues.append(f"{model_key}:current_snapshot_sha256")
        if actual_snapshot_bytes != expected["snapshot_bytes"]:
            pinned_model_issues.append(f"{model_key}:current_snapshot_bytes")
        if actual_snapshot_files != expected["snapshot_files"]:
            pinned_model_issues.append(f"{model_key}:current_snapshot_files")
        if actual_weight_sha256 != expected["weight_sha256"]:
            pinned_model_issues.append(f"{model_key}:current_weight_sha256")
        pinned_model_evidence[model_key] = {
            "repo_id": repo_id,
            "revision": expected["revision"],
            "snapshot_path": str(expected_relative_snapshot),
            "snapshot_sha256": actual_snapshot_sha256,
            "snapshot_bytes": actual_snapshot_bytes,
            "snapshot_files": actual_snapshot_files,
            "model_safetensors_sha256": actual_weight_sha256,
            "expected_feature_dim": expected["feature_dim"],
        }
    if set(manifest_records) != {
        str(expected["repo_id"]) for expected in EXPECTED_PINNED_MODELS.values()
    }:
        pinned_model_issues.append("manifest_repo_id_set")
    add_check(
        checks,
        "pinned_model_identities_and_snapshots",
        not pinned_model_issues,
        models=pinned_model_evidence,
        issues=pinned_model_issues,
    )

    registry_path = ROOT / "data" / "capstone_registry.csv"
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    counts = Counter((row["label"], row["split"]) for row in registry)
    expected_counts = {
        (label, split): count
        for label in ("safe", "firearms", "explosives", "financial_promotion")
        for split, count in (("train", 48), ("val", 12), ("test", 12))
    }
    groups_to_splits: dict[str, set[str]] = defaultdict(set)
    hashes_to_splits: dict[str, set[str]] = defaultdict(set)
    stored_dhashes_to_splits: dict[str, set[str]] = defaultdict(set)
    decoded_dhashes_to_splits: dict[str, set[str]] = defaultdict(set)
    decoded_dhash_by_id: dict[str, str] = {}
    stored_vs_decoded_dhash_mismatches: list[str] = []
    bad_hashes: list[str] = []
    decode_failures: list[str] = []
    for row in registry:
        groups_to_splits[row["source_group"]].add(row["split"])
        hashes_to_splits[row["local_sha256"]].add(row["split"])
        stored_dhashes_to_splits[row["perceptual_dhash"]].add(row["split"])
        image_path = ROOT / row["local_path"]
        if not image_path.is_file() or sha256_file(image_path) != row["local_sha256"]:
            bad_hashes.append(row["sample_id"])
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            current_dhash = decoded_dhash(image_path)
            decoded_dhash_by_id[row["sample_id"]] = current_dhash
            decoded_dhashes_to_splits[current_dhash].add(row["split"])
            if current_dhash != row["perceptual_dhash"]:
                stored_vs_decoded_dhash_mismatches.append(row["sample_id"])
        except Exception:
            decode_failures.append(row["sample_id"])
    group_overlap = sorted(group for group, splits in groups_to_splits.items() if len(splits) > 1)
    hash_overlap = sorted(value for value, splits in hashes_to_splits.items() if len(splits) > 1)
    stored_dhash_overlap = sorted(
        value for value, splits in stored_dhashes_to_splits.items() if len(splits) > 1
    )
    decoded_dhash_overlap = sorted(
        value for value, splits in decoded_dhashes_to_splits.items() if len(splits) > 1
    )
    cross_split_near_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(registry):
        left_hash = decoded_dhash_by_id.get(left["sample_id"])
        if left_hash is None:
            continue
        for right in registry[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            right_hash = decoded_dhash_by_id.get(right["sample_id"])
            if right_hash is None:
                continue
            distance = (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()
            if distance <= 4:
                cross_split_near_pairs.append(
                    {
                        "hamming_distance": distance,
                        "left_sample_id": left["sample_id"],
                        "left_split": left["split"],
                        "left_label": left["label"],
                        "left_source_group": left["source_group"],
                        "right_sample_id": right["sample_id"],
                        "right_split": right["split"],
                        "right_label": right["label"],
                        "right_source_group": right["source_group"],
                        "both_declared_synthetic": (
                            left["is_synthetic"].casefold() == "true"
                            and right["is_synthetic"].casefold() == "true"
                        ),
                    }
                )
    declared_near_duplicate_threshold = 2
    rejected_near_duplicates = [
        pair
        for pair in cross_split_near_pairs
        if pair["hamming_distance"] <= declared_near_duplicate_threshold
    ]
    diagnostic_similarity_band = [
        pair for pair in cross_split_near_pairs if 3 <= pair["hamming_distance"] <= 4
    ]
    diagnostic_band_is_expected_synthetic_financial_style = all(
        pair["left_label"] == "financial_promotion"
        and pair["right_label"] == "financial_promotion"
        and pair["both_declared_synthetic"] is True
        for pair in diagnostic_similarity_band
    )
    dataset_ok = (
        len(registry) == 288
        and counts == Counter(expected_counts)
        and not group_overlap
        and not hash_overlap
        and not decoded_dhash_overlap
        and not rejected_near_duplicates
        and diagnostic_band_is_expected_synthetic_financial_style
        and not bad_hashes
        and not decode_failures
    )
    add_check(
        checks,
        "dataset_registry",
        dataset_ok,
        rows=len(registry),
        source_groups=len(groups_to_splits),
        counts={f"{label}:{split}": count for (label, split), count in sorted(counts.items())},
        group_overlap_count=len(group_overlap),
        exact_hash_overlap_count=len(hash_overlap),
        stored_pre_encoding_dhash_overlap_count=len(stored_dhash_overlap),
        decoded_pixel_dhash_overlap_count=len(decoded_dhash_overlap),
        stored_pre_encoding_vs_decoded_dhash_mismatch_count=len(
            stored_vs_decoded_dhash_mismatches
        ),
        stored_pre_encoding_vs_decoded_dhash_mismatch_sample_ids=(
            stored_vs_decoded_dhash_mismatches
        ),
        declared_near_duplicate_hamming_threshold=declared_near_duplicate_threshold,
        rejected_cross_split_near_duplicate_count=len(rejected_near_duplicates),
        diagnostic_cross_split_hamming_3_to_4_count=len(diagnostic_similarity_band),
        diagnostic_hamming_3_to_4_pairs=diagnostic_similarity_band,
        diagnostic_band_interpretation=(
            "synthetic financial creatives reuse a shared visual grammar; this is reported as "
            "source-style confounding, not treated as proof of pixel duplication"
        ),
        hash_mismatches=bad_hashes,
        decode_failures=decode_failures,
    )

    evaluation = json.loads((ROOT / "outputs/evaluation/evaluation_metrics.json").read_text())
    primary_metrics = json.loads((ROOT / "outputs/evaluation/metrics.json").read_text())
    independent = json.loads((ROOT / "outputs/evaluation/independent_validation.json").read_text())
    external = json.loads((ROOT / "outputs/evaluation/external_spot_check.json").read_text())
    vit = next(model for model in evaluation["models"] if model["key"] == "vit")
    resnet = next(model for model in evaluation["models"] if model["key"] == "resnet50")
    metric_facts = {
        "vit_macro_f1": vit["test_metrics"]["macro_f1"],
        "vit_accuracy": vit["test_metrics"]["accuracy"],
        "vit_cpu_p95_ms": vit["cpu_batch1_benchmark"]["p95_ms"],
        "resnet_macro_f1": resnet["test_metrics"]["macro_f1"],
        "resnet_cpu_p95_ms": resnet["cpu_batch1_benchmark"]["p95_ms"],
        "external_macro_f1": external["classifier_macro_f1"],
        "external_detector_nonweapon_fpr": external["weapon_detector_nonweapon_false_positive_rate"],
    }
    with (ROOT / "outputs/evaluation/test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    labels = ["safe", "firearms", "explosives", "financial_promotion"]
    restricted_labels = ["firearms", "explosives", "financial_promotion"]
    rows_by_model = {
        model_key: [row for row in prediction_rows if row["model"] == model_key]
        for model_key in ("vit", "resnet50")
    }
    test_registry = {row["sample_id"]: row for row in registry if row["split"] == "test"}
    prediction_registry_issues: list[str] = []
    for model_key, rows in rows_by_model.items():
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(test_registry):
            prediction_registry_issues.append(f"{model_key}:sample_id_set")
        for row in rows:
            registry_row = test_registry.get(row["sample_id"])
            if registry_row is None or any(
                row[field] != registry_row[field]
                for field in ("label", "source_group", "source_dataset")
                if field in row
            ):
                prediction_registry_issues.append(f"{model_key}:{row['sample_id']}:registry_join")
            elif row["true_label"] != registry_row["label"]:
                prediction_registry_issues.append(f"{model_key}:{row['sample_id']}:true_label")
    recomputed = {key: classification_facts(rows, labels) for key, rows in rows_by_model.items()}
    probability_issues: list[str] = []
    argmax_issues: list[str] = []
    for row in prediction_rows:
        probabilities = {label: float(row[f"probability_{label}"]) for label in labels}
        if any(not 0.0 <= value <= 1.0 for value in probabilities.values()) or not close_enough(
            sum(probabilities.values()), 1.0, 1e-6
        ):
            probability_issues.append(f"{row['model']}:{row['sample_id']}")
        recomputed_argmax = max(labels, key=lambda label: probabilities[label])
        if recomputed_argmax != row["predicted_label"]:
            argmax_issues.append(f"{row['model']}:{row['sample_id']}")

    threshold_rows: list[dict[str, str]]
    with (ROOT / "outputs/evaluation/thresholds.csv").open(newline="", encoding="utf-8") as handle:
        threshold_rows = list(csv.DictReader(handle))
    vit_threshold_rows = {row["label"]: row for row in threshold_rows if row["model"] == "vit"}
    threshold_row_keys = {(row["model"], row["label"]) for row in threshold_rows}
    expected_threshold_row_keys = {
        (model_key, label)
        for model_key in ("vit", "resnet50")
        for label in restricted_labels
    }
    threshold_provenance_ok = (
        len(threshold_rows) == 6
        and threshold_row_keys == expected_threshold_row_keys
        and all(row["selection_split"] == "validation" for row in threshold_rows)
        and all(close_enough(row["target_validation_recall"], 0.90) for row in threshold_rows)
    )

    # Refit each saved head from the training embeddings only. Exact agreement
    # with the persisted fitted head proves that its learned parameters are
    # reproducible without validation or test rows. Then derive every operating
    # threshold from the validation rows only and compare all persisted copies.
    registry_sha256 = sha256_file(registry_path)
    registry_by_id = {row["sample_id"]: row for row in registry}
    threshold_rows_by_key = {(row["model"], row["label"]): row for row in threshold_rows}
    calibration_issues: list[str] = []
    calibration_reproduction: dict[str, Any] = {}
    for model_key, model_record in (("vit", vit), ("resnet50", resnet)):
        expected_identity = EXPECTED_PINNED_MODELS[model_key]
        expected_embedding_relative = Path(f"outputs/evaluation/embeddings_{model_key}.npz")
        expected_artifact_relative = Path(f"models/{model_key}_policy_head.joblib")
        embedding_path = ROOT / expected_embedding_relative
        artifact_path = ROOT / expected_artifact_relative
        artifact = joblib.load(artifact_path)
        with np.load(embedding_path, allow_pickle=False) as cache:
            features = np.asarray(cache["features"])
            sample_ids = [str(value) for value in cache["sample_ids"].tolist()]
            cached_registry_sha256 = str(cache["registry_sha256"].item())
            cached_model_name = str(cache["model_name"].item())
            cached_revision = str(cache["revision"].item())

        model_issues: list[str] = []
        if model_record.get("key") != model_key:
            model_issues.append("evaluation_model_key")
        if Path(str(model_record["feature_extraction"].get("embedding_path", ""))) != expected_embedding_relative:
            model_issues.append("evaluation_embedding_path")
        if Path(str(model_record.get("artifact_path", ""))) != expected_artifact_relative:
            model_issues.append("evaluation_artifact_path")
        for field, expected_value in (
            ("name", expected_identity["model_name"]),
            ("repo_id", expected_identity["repo_id"]),
            ("revision", expected_identity["revision"]),
            ("feature_dim", expected_identity["feature_dim"]),
            ("weight_sha256", expected_identity["weight_sha256"]),
        ):
            if model_record["backbone"].get(field) != expected_value:
                model_issues.append(f"evaluation_backbone_identity:{field}")
        for field, expected_value in (
            ("backbone_name", expected_identity["model_name"]),
            ("backbone_repo_id", expected_identity["repo_id"]),
            ("backbone_revision", expected_identity["revision"]),
            ("backbone_weight_sha256", expected_identity["weight_sha256"]),
        ):
            if artifact.get(field) != expected_value:
                model_issues.append(f"artifact_backbone_identity:{field}")
        class_names = [str(value) for value in artifact.get("class_names", [])]
        if class_names != labels:
            model_issues.append("artifact_class_order")
        if len(sample_ids) != 288 or len(sample_ids) != len(set(sample_ids)):
            model_issues.append("embedding_sample_id_count_or_uniqueness")
        if set(sample_ids) != set(registry_by_id):
            model_issues.append("embedding_registry_sample_id_set")
        if features.ndim != 2 or features.shape[0] != len(sample_ids):
            model_issues.append("embedding_shape")
        if not np.isfinite(features).all():
            model_issues.append("embedding_nonfinite_values")
        if cached_registry_sha256 != registry_sha256:
            model_issues.append("embedding_registry_sha256")
        if sha256_file(embedding_path) != model_record["feature_extraction"]["embedding_sha256"]:
            model_issues.append("embedding_file_sha256")
        if sha256_file(artifact_path) != model_record["artifact_sha256"]:
            model_issues.append("artifact_file_sha256")
        if cached_model_name != expected_identity["model_name"]:
            model_issues.append("embedding_model_name")
        if cached_revision != expected_identity["revision"]:
            model_issues.append("embedding_revision")

        manifest_record = manifest_records.get(str(expected_identity["repo_id"]))
        weight_path: Path | None = None
        if manifest_record is None:
            model_issues.append("pinned_backbone_manifest_record")
        else:
            weight_path = ROOT / str(manifest_record.get("local_snapshot", "")) / "model.safetensors"
            if manifest_record.get("revision") != expected_identity["revision"]:
                model_issues.append("pinned_backbone_manifest_revision")
            if not weight_path.is_file():
                model_issues.append("pinned_backbone_weight_file")
            else:
                current_weight_sha256 = sha256_file(weight_path)
                if current_weight_sha256 != expected_identity["weight_sha256"]:
                    model_issues.append("current_backbone_weight_sha256")

        training_metadata = artifact.get("training", {})
        expected_metadata = {
            "registry_sha256": registry_sha256,
            "training_rows": 192,
            "validation_rows": 48,
            "test_rows_excluded_from_fit_and_calibration": 48,
            "feature_dim": expected_identity["feature_dim"],
            "trainable_backbone_parameters": 0,
        }
        for field, expected in expected_metadata.items():
            if training_metadata.get(field) != expected:
                model_issues.append(f"artifact_training_metadata:{field}")
        if not close_enough(training_metadata.get("threshold_recall_target"), 0.90):
            model_issues.append("artifact_training_metadata:threshold_recall_target")
        if model_record["backbone"].get("trainable_backbone_parameters") != 0:
            model_issues.append("evaluation_trainable_backbone_parameters")

        if (
            model_issues
            or class_names != labels
            or set(sample_ids) != set(registry_by_id)
            or weight_path is None
        ):
            calibration_issues.extend(f"{model_key}:{issue}" for issue in model_issues)
            calibration_reproduction[model_key] = {"issues": model_issues}
            continue

        label_to_index = {label: index for index, label in enumerate(class_names)}
        y = np.asarray([label_to_index[registry_by_id[sample_id]["label"]] for sample_id in sample_ids])
        train_mask = np.asarray([registry_by_id[sample_id]["split"] == "train" for sample_id in sample_ids])
        val_mask = np.asarray([registry_by_id[sample_id]["split"] == "val" for sample_id in sample_ids])
        test_mask = np.asarray([registry_by_id[sample_id]["split"] == "test" for sample_id in sample_ids])
        if (int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())) != (192, 48, 48):
            model_issues.append("reconstructed_split_counts")

        image_paths = [ROOT / registry_by_id[sample_id]["local_path"] for sample_id in sample_ids]
        regenerated_features, regeneration_details = regenerate_backbone_embeddings(
            str(expected_identity["model_name"]),
            weight_path,
            image_paths,
            batch_size=8,
        )
        cache_max_abs_feature_difference = float(
            np.max(np.abs(features - regenerated_features))
        )
        if not np.array_equal(features, regenerated_features):
            model_issues.append("raw_pixel_regeneration_vs_embedding_cache")

        reproduced_pipeline = canonical_policy_head()
        canonical_step_names = [name for name, _ in reproduced_pipeline.steps]
        artifact_step_names = [name for name, _ in artifact["pipeline"].steps]
        if artifact_step_names != canonical_step_names:
            model_issues.append("artifact_pipeline_steps")
        canonical_logistic_params = reproduced_pipeline.named_steps["logistic"].get_params()
        artifact_logistic_params = artifact["pipeline"].named_steps["logistic"].get_params()
        for field in (
            "C",
            "class_weight",
            "dual",
            "fit_intercept",
            "intercept_scaling",
            "l1_ratio",
            "max_iter",
            "n_jobs",
            "penalty",
            "random_state",
            "solver",
            "tol",
            "verbose",
            "warm_start",
        ):
            if artifact_logistic_params.get(field) != canonical_logistic_params.get(field):
                model_issues.append(f"artifact_pipeline_hyperparameter:{field}")
            if training_metadata.get("head_hyperparameters", {}).get(field) != canonical_logistic_params.get(field):
                model_issues.append(f"artifact_training_hyperparameter:{field}")
        canonical_scaler_params = reproduced_pipeline.named_steps["standardize"].get_params()
        artifact_scaler_params = artifact["pipeline"].named_steps["standardize"].get_params()
        if artifact_scaler_params != canonical_scaler_params:
            model_issues.append("artifact_standard_scaler_hyperparameters")

        reproduced_pipeline.fit(regenerated_features[train_mask], y[train_mask])
        saved_probabilities = artifact["pipeline"].predict_proba(regenerated_features)
        reproduced_probabilities = reproduced_pipeline.predict_proba(regenerated_features)
        max_abs_probability_difference = float(
            np.max(np.abs(saved_probabilities - reproduced_probabilities))
        )
        if max_abs_probability_difference > 1e-12:
            model_issues.append("train_only_refit_probability_difference")
        expected_classes = np.arange(len(labels))
        if not np.array_equal(artifact["pipeline"].classes_, expected_classes):
            model_issues.append("saved_pipeline_classes")
        if not np.array_equal(reproduced_pipeline.classes_, expected_classes):
            model_issues.append("reproduced_pipeline_classes")

        reproduced_thresholds: dict[str, Any] = {}
        for label in restricted_labels:
            label_index = label_to_index[label]
            selected = select_validation_threshold(
                y[val_mask] == label_index,
                reproduced_probabilities[val_mask, label_index],
                recall_floor=0.90,
            )
            reproduced_thresholds[label] = selected
            persisted_sources = {
                "artifact_calibration": artifact["threshold_calibration"][label],
                "evaluation_calibration": model_record["threshold_calibration_validation_only"][label],
            }
            float_fields = (
                "threshold",
                "precision",
                "recall",
                "f1",
                "specificity",
                "false_positive_rate",
            )
            integer_fields = ("tp", "fp", "tn", "fn", "positives", "negatives", "candidate_count")
            for source_name, persisted in persisted_sources.items():
                for field in float_fields:
                    if not close_enough(selected[field], persisted.get(field)):
                        model_issues.append(f"{label}:{source_name}:{field}")
                for field in integer_fields:
                    if selected[field] != persisted.get(field):
                        model_issues.append(f"{label}:{source_name}:{field}")
                if selected["selection_rule"] != persisted.get("selection_rule"):
                    model_issues.append(f"{label}:{source_name}:selection_rule")

            if not close_enough(selected["threshold"], artifact["thresholds"].get(label)):
                model_issues.append(f"{label}:artifact_threshold")
            if not close_enough(selected["threshold"], model_record["thresholds"].get(label)):
                model_issues.append(f"{label}:evaluation_threshold")
            if model_key == "vit" and not close_enough(
                selected["threshold"],
                primary_metrics["thresholds_selected_on_validation_only"].get(label),
            ):
                model_issues.append(f"{label}:primary_metrics_threshold")

            csv_row = threshold_rows_by_key.get((model_key, label), {})
            csv_float_fields = {
                "threshold": "threshold",
                "precision": "validation_precision",
                "recall": "validation_recall",
                "f1": "validation_f1",
            }
            for selected_field, csv_field in csv_float_fields.items():
                if not close_enough(selected[selected_field], csv_row.get(csv_field)):
                    model_issues.append(f"{label}:threshold_csv:{csv_field}")
            for selected_field, csv_field in (("fp", "validation_fp"), ("fn", "validation_fn")):
                try:
                    csv_value = int(csv_row[csv_field])
                except (KeyError, TypeError, ValueError):
                    csv_value = None
                if selected[selected_field] != csv_value:
                    model_issues.append(f"{label}:threshold_csv:{csv_field}")
            if selected["candidate_count"] != 50:
                model_issues.append(f"{label}:candidate_count_expected_50")

        # Bind the independently reproduced train-only model to every saved
        # formal test row. This prevents a self-consistent CSV, metrics JSON,
        # and checksum manifest from substituting results not produced by the
        # fitted artifact.
        csv_test_rows_by_id = {row["sample_id"]: row for row in rows_by_model[model_key]}
        test_indices = np.flatnonzero(test_mask)
        expected_test_ids = {sample_ids[int(index)] for index in test_indices}
        if set(csv_test_rows_by_id) != expected_test_ids:
            model_issues.append("formal_test_csv_sample_id_set")
        csv_probability_max_abs_difference = 0.0
        csv_probability_mismatches: list[str] = []
        csv_argmax_mismatches: list[str] = []
        csv_threshold_flag_mismatches: list[str] = []
        direct_test_rows: list[dict[str, str]] = []
        for index in test_indices:
            row_index = int(index)
            sample_id = sample_ids[row_index]
            probability_vector = reproduced_probabilities[row_index]
            predicted_label = class_names[int(np.argmax(probability_vector))]
            true_label = registry_by_id[sample_id]["label"]
            direct_test_rows.append({"true_label": true_label, "predicted_label": predicted_label})
            csv_row = csv_test_rows_by_id.get(sample_id)
            if csv_row is None:
                continue
            if csv_row.get("true_label") != true_label:
                model_issues.append(f"{sample_id}:formal_test_csv_true_label")
            if csv_row.get("predicted_label") != predicted_label:
                csv_argmax_mismatches.append(sample_id)
            for label_index, label in enumerate(class_names):
                try:
                    csv_probability = float(csv_row[f"probability_{label}"])
                except (KeyError, TypeError, ValueError):
                    csv_probability_mismatches.append(f"{sample_id}:{label}:missing_or_invalid")
                    continue
                difference = abs(float(probability_vector[label_index]) - csv_probability)
                csv_probability_max_abs_difference = max(
                    csv_probability_max_abs_difference,
                    difference,
                )
                if difference > 1e-12:
                    csv_probability_mismatches.append(f"{sample_id}:{label}")
            try:
                stored_flags = json.loads(csv_row["threshold_flags"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                stored_flags = {}
            for label in restricted_labels:
                label_index = label_to_index[label]
                reproduced_flag = bool(
                    probability_vector[label_index] >= reproduced_thresholds[label]["threshold"]
                )
                if stored_flags.get(label) is not reproduced_flag:
                    csv_threshold_flag_mismatches.append(f"{sample_id}:{label}")

        if csv_probability_mismatches:
            model_issues.append("formal_test_csv_probabilities")
        if csv_argmax_mismatches:
            model_issues.append("formal_test_csv_argmax_labels")
        if csv_threshold_flag_mismatches:
            model_issues.append("formal_test_csv_threshold_flags")

        direct_test_facts = classification_facts(direct_test_rows, labels)
        persisted_test_metrics = model_record["test_metrics"]
        if not close_enough(direct_test_facts["accuracy"], persisted_test_metrics.get("accuracy")):
            model_issues.append("evaluation_test_accuracy")
        if not close_enough(direct_test_facts["macro_f1"], persisted_test_metrics.get("macro_f1")):
            model_issues.append("evaluation_test_macro_f1")
        for label in labels:
            direct_class = direct_test_facts["per_class"][label]
            persisted_class = persisted_test_metrics["per_class"][label]
            for field in ("precision", "recall", "f1"):
                if not close_enough(direct_class[field], persisted_class.get(field)):
                    model_issues.append(f"{label}:evaluation_test_per_class:{field}")
            if direct_class["support"] != persisted_class.get("support"):
                model_issues.append(f"{label}:evaluation_test_per_class:support")

        direct_test_threshold_metrics: dict[str, Any] = {}
        for label in restricted_labels:
            label_index = label_to_index[label]
            direct_operating_point = binary_threshold_metrics(
                y[test_mask] == label_index,
                reproduced_probabilities[test_mask, label_index],
                reproduced_thresholds[label]["threshold"],
            )
            direct_test_threshold_metrics[label] = direct_operating_point
            persisted_operating_point = model_record["threshold_test_metrics"][label]
            for field in (
                "threshold",
                "precision",
                "recall",
                "f1",
                "specificity",
                "false_positive_rate",
            ):
                if not close_enough(direct_operating_point[field], persisted_operating_point.get(field)):
                    model_issues.append(f"{label}:evaluation_test_threshold:{field}")
            for field in ("tp", "fp", "tn", "fn", "positives", "negatives"):
                if direct_operating_point[field] != persisted_operating_point.get(field):
                    model_issues.append(f"{label}:evaluation_test_threshold:{field}")

            csv_row = threshold_rows_by_key.get((model_key, label), {})
            for direct_field, csv_field in (
                ("precision", "test_precision"),
                ("recall", "test_recall"),
                ("f1", "test_f1"),
            ):
                if not close_enough(direct_operating_point[direct_field], csv_row.get(csv_field)):
                    model_issues.append(f"{label}:threshold_csv:{csv_field}")
            for direct_field, csv_field in (("fp", "test_fp"), ("fn", "test_fn")):
                try:
                    csv_value = int(csv_row[csv_field])
                except (KeyError, TypeError, ValueError):
                    csv_value = None
                if direct_operating_point[direct_field] != csv_value:
                    model_issues.append(f"{label}:threshold_csv:{csv_field}")

        if model_key == "vit":
            direct_restricted_argmax_recall = sum(
                float(direct_test_facts["per_class"][label]["recall"])
                for label in restricted_labels
            ) / len(restricted_labels)
            direct_safe_false_flag_rate = 1.0 - float(
                direct_test_facts["per_class"]["safe"]["recall"]
            )
            for direct_value, primary_field in (
                (direct_test_facts["accuracy"], "accuracy"),
                (direct_test_facts["macro_f1"], "macro_f1"),
                (direct_restricted_argmax_recall, "restricted_recall"),
                (direct_safe_false_flag_rate, "safe_false_positive_rate"),
            ):
                if not close_enough(direct_value, primary_metrics.get(primary_field)):
                    model_issues.append(f"primary_metrics:{primary_field}")

        calibration_issues.extend(f"{model_key}:{issue}" for issue in model_issues)
        calibration_reproduction[model_key] = {
            "embedding_path": str(embedding_path.relative_to(ROOT)),
            "embedding_sha256": sha256_file(embedding_path),
            "artifact_path": str(artifact_path.relative_to(ROOT)),
            "artifact_sha256": sha256_file(artifact_path),
            "raw_pixel_backbone_regeneration": {
                **regeneration_details,
                "embedding_cache_max_abs_difference": cache_max_abs_feature_difference,
                "embedding_cache_exact_match": bool(np.array_equal(features, regenerated_features)),
                "ordered_image_count": len(image_paths),
                "ordered_image_sha256": [
                    {
                        "sample_id": sample_id,
                        "sha256": registry_by_id[sample_id]["local_sha256"],
                    }
                    for sample_id in sample_ids
                ],
            },
            "split_rows": {
                "train_fit_only": int(train_mask.sum()),
                "validation_calibration_only": int(val_mask.sum()),
                "test_excluded_from_fit_and_calibration": int(test_mask.sum()),
            },
            "max_abs_probability_difference_after_train_only_refit": max_abs_probability_difference,
            "reproduced_thresholds": reproduced_thresholds,
            "formal_test_output_binding": {
                "rows": len(direct_test_rows),
                "csv_probability_max_abs_difference": csv_probability_max_abs_difference,
                "csv_probability_mismatches": csv_probability_mismatches,
                "csv_argmax_mismatches": csv_argmax_mismatches,
                "csv_threshold_flag_mismatches": csv_threshold_flag_mismatches,
                "direct_classification_metrics": direct_test_facts,
                "direct_threshold_metrics": direct_test_threshold_metrics,
            },
            "issues": model_issues,
        }

    add_check(
        checks,
        "train_only_fit_and_validation_calibration_reproduced",
        not calibration_issues,
        registry_sha256=registry_sha256,
        recall_floor=0.90,
        models=calibration_reproduction,
        issues=calibration_issues,
    )
    vit_thresholds = {label: float(vit_threshold_rows[label]["threshold"]) for label in restricted_labels}
    threshold_flag_issues: list[str] = []
    threshold_recalls: dict[str, float] = {}
    for label in restricted_labels:
        positives = [row for row in rows_by_model["vit"] if row["true_label"] == label]
        hits = 0
        for row in positives:
            stored_flags = json.loads(row["threshold_flags"])
            recomputed_flag = float(row[f"probability_{label}"]) >= vit_thresholds[label]
            if bool(stored_flags[label]) != recomputed_flag:
                threshold_flag_issues.append(f"{row['sample_id']}:{label}")
            hits += int(recomputed_flag)
        threshold_recalls[label] = hits / len(positives)
    for row in rows_by_model["vit"]:
        stored_flags = json.loads(row["threshold_flags"])
        for label in restricted_labels:
            recomputed_flag = float(row[f"probability_{label}"]) >= vit_thresholds[label]
            if bool(stored_flags[label]) != recomputed_flag:
                issue = f"{row['sample_id']}:{label}"
                if issue not in threshold_flag_issues:
                    threshold_flag_issues.append(issue)

    argmax_restricted_recalls = {
        label: float(recomputed["vit"]["per_class"][label]["recall"])
        for label in restricted_labels
    }
    argmax_restricted_mean = sum(argmax_restricted_recalls.values()) / len(restricted_labels)
    threshold_recall_mean = sum(threshold_recalls.values()) / len(restricted_labels)
    threshold_recall_min = min(threshold_recalls.values())
    safe_precision = float(recomputed["vit"]["per_class"]["safe"]["precision"])
    safe_ad_false_flag_rate = (
        sum(row["true_label"] == "safe" and row["predicted_label"] != "safe" for row in rows_by_model["vit"])
        / sum(row["true_label"] == "safe" for row in rows_by_model["vit"])
    )
    model_metrics_match = all(
        len(rows_by_model[key]) == 48
        and close_enough(recomputed[key]["accuracy"], model["test_metrics"]["accuracy"])
        and close_enough(recomputed[key]["macro_f1"], model["test_metrics"]["macro_f1"])
        for key, model in (("vit", vit), ("resnet50", resnet))
    )
    threshold_csv_matches = all(
        close_enough(threshold_recalls[label], float(vit_threshold_rows[label]["test_recall"]))
        and close_enough(threshold_recalls[label], vit["threshold_test_metrics"][label]["recall"])
        and close_enough(vit_thresholds[label], primary_metrics["thresholds_selected_on_validation_only"][label])
        for label in restricted_labels
    )
    metrics_ok = (
        independent["all_checks_passed"] is True
        and len(prediction_rows) == 96
        and model_metrics_match
        and not prediction_registry_issues
        and threshold_provenance_ok
        and not probability_issues
        and not argmax_issues
        and not threshold_flag_issues
        and close_enough(recomputed["vit"]["accuracy"], primary_metrics["accuracy"])
        and close_enough(recomputed["vit"]["macro_f1"], primary_metrics["macro_f1"])
        and close_enough(argmax_restricted_mean, primary_metrics["restricted_recall"])
        and close_enough(safe_ad_false_flag_rate, primary_metrics["safe_false_positive_rate"])
        and close_enough(safe_precision, 1.0)
        and safe_precision > 0.98
        and close_enough(argmax_restricted_mean, 0.9722222222222222)
        and close_enough(threshold_recall_mean, 0.9444444444444444)
        and close_enough(threshold_recall_min, 0.9166666666666666)
        and not all(value > 0.95 for value in threshold_recalls.values())
        and threshold_csv_matches
        and close_enough(metric_facts["vit_macro_f1"], 0.9791304347826086)
        and close_enough(metric_facts["resnet_macro_f1"], 0.8731521739130435)
        and external["sample_count"] == 26
        and external["scope"].startswith("manually reviewed external Wikimedia diagnostic")
    )
    add_check(
        checks,
        "measured_evidence_recomputed",
        metrics_ok,
        prediction_rows=len(prediction_rows),
        recomputed=recomputed,
        argmax_restricted_recalls=argmax_restricted_recalls,
        argmax_restricted_mean=argmax_restricted_mean,
        threshold_operating_recalls=threshold_recalls,
        threshold_operating_mean=threshold_recall_mean,
        threshold_operating_min=threshold_recall_min,
        safe_precision=safe_precision,
        safe_precision_target_gt_0_98_met=safe_precision > 0.98,
        safe_ad_false_flag_rate_one_minus_safe_recall=safe_ad_false_flag_rate,
        per_class_recall_target_gt_0_95_met=all(value > 0.95 for value in threshold_recalls.values()),
        threshold_selection_validation_recall_floor=0.90,
        probability_issues=probability_issues,
        argmax_issues=argmax_issues,
        threshold_flag_issues=threshold_flag_issues,
        prediction_registry_issues=prediction_registry_issues,
        threshold_provenance_ok=threshold_provenance_ok,
        **metric_facts,
    )

    with (ROOT / "outputs/evaluation/external_spot_check.csv").open(newline="", encoding="utf-8") as handle:
        external_rows = list(csv.DictReader(handle))
    external_recomputed = classification_facts(
        [
            {"true_label": row["expected_label"], "predicted_label": row["classifier_prediction"]}
            for row in external_rows
        ],
        labels,
    )
    weapon_labels = {"firearms", "explosives"}
    weapon_rows = [row for row in external_rows if row["expected_label"] in weapon_labels]
    nonweapon_rows = [row for row in external_rows if row["expected_label"] not in weapon_labels]
    safe_rows = [row for row in external_rows if row["expected_label"] == "safe"]
    financial_rows = [row for row in external_rows if row["expected_label"] == "financial_promotion"]

    def detector_signal(row: dict[str, str]) -> bool:
        return row["weapon_detector_signal"].strip().casefold() == "true"

    external_detector_recall = sum(detector_signal(row) for row in weapon_rows) / len(weapon_rows)
    external_detector_nonweapon_fpr = sum(detector_signal(row) for row in nonweapon_rows) / len(nonweapon_rows)
    external_financial_review_rate = sum(
        row["policy_verdict"] == "REVIEW" and row["policy_primary_label"] == "financial_promotion"
        for row in financial_rows
    ) / len(financial_rows)
    external_safe_nonapprove_rate = sum(row["policy_verdict"] != "APPROVE" for row in safe_rows) / len(safe_rows)
    external_weapon_block_rate = sum(row["policy_verdict"] == "BLOCK" for row in weapon_rows) / len(weapon_rows)
    external_row_issues: list[str] = []
    for row in external_rows:
        probabilities = json.loads(row["classifier_scores_json"])
        if (
            not close_enough(sum(float(value) for value in probabilities.values()), 1.0, 1e-6)
            or max(labels, key=lambda label: float(probabilities[label])) != row["classifier_prediction"]
            or (row["classifier_correct"].strip().casefold() == "true")
            != (row["expected_label"] == row["classifier_prediction"])
        ):
            external_row_issues.append(row["local_path"])
        image_path = ROOT / row["local_path"]
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            external_row_issues.append(f"{row['local_path']}:decode")
    with (ROOT / "data/wikimedia_external_manifest.csv").open(newline="", encoding="utf-8") as handle:
        external_manifest = list(csv.DictReader(handle))
    included_manifest_rows = [
        row
        for row in external_manifest
        if row["include_in_spot_check"].strip().casefold() == "true"
    ]
    included_manifest = {
        (row["local_path"], row["reviewed_label"])
        for row in included_manifest_rows
    }
    saved_external_rows = {(row["local_path"], row["expected_label"]) for row in external_rows}

    # Rerun the current classifier, detector, OCR, and policy from the reviewed
    # external pixels. Saved CSV fields are comparisons, not trusted inputs.
    src_root = ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from ad_safety.inference import ModerationEngine

    external_rows_by_path = {row["local_path"]: row for row in external_rows}
    external_live_issues: list[str] = []
    external_live_records: list[dict[str, Any]] = []
    external_image_hashes: list[dict[str, str]] = []
    external_classifier_max_abs_difference = 0.0
    moderation_engine = ModerationEngine(device="cpu", enable_detector=True)
    detector_terms = (
        "handgun",
        "pistol",
        "rifle",
        "shotgun",
        "ammunition",
        "grenade",
        "bomb",
        "explosive",
        "fireworks",
    )
    for manifest_row in included_manifest_rows:
        local_path = manifest_row["local_path"]
        image_path = ROOT / local_path
        external_image_hashes.append(
            {"local_path": local_path, "sha256": sha256_file(image_path)}
        )
        result = moderation_engine.analyze(
            image_path,
            run_detector=True,
            run_ocr=True,
            explain=False,
        )
        live_prediction = max(result.classifier_scores, key=result.classifier_scores.get)
        live_detector_signal = any(
            term in detection.label
            for detection in result.detections
            for term in detector_terms
        )
        saved_row = external_rows_by_path.get(local_path)
        if saved_row is None:
            external_live_issues.append(f"{local_path}:missing_saved_row")
            continue
        try:
            saved_scores = json.loads(saved_row["classifier_scores_json"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            saved_scores = {}
        for label in labels:
            try:
                difference = abs(float(result.classifier_scores[label]) - float(saved_scores[label]))
            except (KeyError, TypeError, ValueError):
                external_live_issues.append(f"{local_path}:classifier_score:{label}:missing_or_invalid")
                continue
            external_classifier_max_abs_difference = max(
                external_classifier_max_abs_difference,
                difference,
            )
            if difference > 1e-12:
                external_live_issues.append(f"{local_path}:classifier_score:{label}")
        if live_prediction != saved_row.get("classifier_prediction"):
            external_live_issues.append(f"{local_path}:classifier_prediction")
        if live_detector_signal != detector_signal(saved_row):
            external_live_issues.append(f"{local_path}:weapon_detector_signal")
        if result.decision.verdict != saved_row.get("policy_verdict"):
            external_live_issues.append(f"{local_path}:policy_verdict")
        if result.decision.primary_label != saved_row.get("policy_primary_label"):
            external_live_issues.append(f"{local_path}:policy_primary_label")
        external_live_records.append(
            {
                "local_path": local_path,
                "expected_label": manifest_row["reviewed_label"],
                "predicted_label": live_prediction,
                "weapon_detector_signal": live_detector_signal,
                "policy_verdict": result.decision.verdict,
                "policy_primary_label": result.decision.primary_label,
                "detection_count": len(result.detections),
            }
        )
    if moderation_engine.detector is None:
        external_live_issues.append("grounding_dino:not_loaded")
    elif moderation_engine.detector.revision != EXPECTED_PINNED_MODELS["grounding_dino"]["revision"]:
        external_live_issues.append("grounding_dino:revision")
    del moderation_engine
    gc.collect()

    external_live_classification = classification_facts(
        [
            {
                "true_label": row["expected_label"],
                "predicted_label": row["predicted_label"],
            }
            for row in external_live_records
        ],
        labels,
    )
    live_weapon_rows = [
        row for row in external_live_records if row["expected_label"] in weapon_labels
    ]
    live_nonweapon_rows = [
        row for row in external_live_records if row["expected_label"] not in weapon_labels
    ]
    live_safe_rows = [
        row for row in external_live_records if row["expected_label"] == "safe"
    ]
    live_financial_rows = [
        row for row in external_live_records if row["expected_label"] == "financial_promotion"
    ]
    external_live_detector_recall = (
        sum(bool(row["weapon_detector_signal"]) for row in live_weapon_rows)
        / len(live_weapon_rows)
    )
    external_live_detector_nonweapon_fpr = (
        sum(bool(row["weapon_detector_signal"]) for row in live_nonweapon_rows)
        / len(live_nonweapon_rows)
    )
    external_live_financial_review_rate = (
        sum(
            row["policy_verdict"] == "REVIEW"
            and row["policy_primary_label"] == "financial_promotion"
            for row in live_financial_rows
        )
        / len(live_financial_rows)
    )
    external_live_safe_nonapprove_rate = (
        sum(row["policy_verdict"] != "APPROVE" for row in live_safe_rows)
        / len(live_safe_rows)
    )
    external_live_weapon_block_rate = (
        sum(row["policy_verdict"] == "BLOCK" for row in live_weapon_rows)
        / len(live_weapon_rows)
    )
    external_ok = (
        len(external_rows) == external["sample_count"] == 26
        and len(external_live_records) == 26
        and included_manifest == saved_external_rows
        and not external_row_issues
        and not external_live_issues
        and external_classifier_max_abs_difference <= 1e-12
        and close_enough(external_recomputed["accuracy"], external["classifier_accuracy"])
        and close_enough(external_recomputed["macro_f1"], external["classifier_macro_f1"])
        and close_enough(external_live_classification["accuracy"], external["classifier_accuracy"])
        and close_enough(external_live_classification["macro_f1"], external["classifier_macro_f1"])
        and close_enough(external_detector_recall, external["weapon_detector_image_recall"])
        and close_enough(external_live_detector_recall, external["weapon_detector_image_recall"])
        and close_enough(
            external_detector_nonweapon_fpr,
            external["weapon_detector_nonweapon_false_positive_rate"],
        )
        and close_enough(
            external_live_detector_nonweapon_fpr,
            external["weapon_detector_nonweapon_false_positive_rate"],
        )
        and close_enough(external_financial_review_rate, external["financial_correct_review_rate"])
        and close_enough(
            external_live_financial_review_rate,
            external["financial_correct_review_rate"],
        )
        and close_enough(external_safe_nonapprove_rate, external["safe_policy_non_approve_rate"])
        and close_enough(
            external_live_safe_nonapprove_rate,
            external["safe_policy_non_approve_rate"],
        )
        and close_enough(external_weapon_block_rate, external["weapon_policy_block_rate"])
        and close_enough(
            external_live_weapon_block_rate,
            external["weapon_policy_block_rate"],
        )
        and external["scope"].startswith("manually reviewed external Wikimedia diagnostic")
    )
    add_check(
        checks,
        "external_diagnostic_recomputed",
        external_ok,
        rows=len(external_rows),
        manifest_included_rows=len(included_manifest),
        classifier_accuracy=external_recomputed["accuracy"],
        classifier_macro_f1=external_recomputed["macro_f1"],
        detector_weapon_image_recall=external_detector_recall,
        detector_nonweapon_false_positive_rate=external_detector_nonweapon_fpr,
        financial_review_rate=external_financial_review_rate,
        safe_nonapprove_rate=external_safe_nonapprove_rate,
        weapon_block_rate=external_weapon_block_rate,
        row_issues=external_row_issues,
        fresh_current_pipeline_inference={
            "rows": len(external_live_records),
            "classifier_max_abs_saved_score_difference": external_classifier_max_abs_difference,
            "classifier_accuracy": external_live_classification["accuracy"],
            "classifier_macro_f1": external_live_classification["macro_f1"],
            "detector_weapon_image_recall": external_live_detector_recall,
            "detector_nonweapon_false_positive_rate": external_live_detector_nonweapon_fpr,
            "financial_review_rate": external_live_financial_review_rate,
            "safe_nonapprove_rate": external_live_safe_nonapprove_rate,
            "weapon_block_rate": external_live_weapon_block_rate,
            "image_sha256": external_image_hashes,
            "records": external_live_records,
            "issues": external_live_issues,
        },
        csv_sha256=sha256_file(ROOT / "outputs/evaluation/external_spot_check.csv"),
        json_sha256=sha256_file(ROOT / "outputs/evaluation/external_spot_check.json"),
        reviewed_manifest_sha256=sha256_file(ROOT / "data/wikimedia_external_manifest.csv"),
    )

    checksum_rows: list[dict[str, str]]
    with (ROOT / "outputs/evaluation/output_checksums.csv").open(newline="", encoding="utf-8") as handle:
        checksum_rows = list(csv.DictReader(handle))
    expected_checksum_paths = {
        "outputs/evaluation/benchmark_cpu_batch1.csv",
        "outputs/evaluation/confusion_matrix.csv",
        "outputs/evaluation/confusion_matrix_resnet50.png",
        "outputs/evaluation/confusion_matrix_vit.png",
        "outputs/evaluation/dataset_contact_sheet.jpg",
        "outputs/evaluation/dataset_distribution.png",
        "outputs/evaluation/embeddings_resnet50.npz",
        "outputs/evaluation/embeddings_vit.npz",
        "outputs/evaluation/evaluation_metrics.json",
        "outputs/evaluation/failure_cases.csv",
        "outputs/evaluation/independent_validation.json",
        "outputs/evaluation/latency.json",
        "outputs/evaluation/metrics.json",
        "outputs/evaluation/metrics_summary.csv",
        "outputs/evaluation/model_comparison.csv",
        "outputs/evaluation/model_comparison.png",
        "outputs/evaluation/per_class_metrics.csv",
        "outputs/evaluation/precision_recall_curves.png",
        "outputs/evaluation/predictions.csv",
        "outputs/evaluation/split_summary.csv",
        "outputs/evaluation/test_predictions.csv",
        "outputs/evaluation/threshold_calibration.png",
        "outputs/evaluation/thresholds.csv",
        "models/vit_policy_head.joblib",
        "models/resnet50_policy_head.joblib",
        "outputs/evaluation/evaluation_manifest.json",
    }
    checksum_failures: list[dict[str, Any]] = []
    checksum_paths: list[str] = []
    for row in checksum_rows:
        relative = row["path"]
        path = (ROOT / relative).resolve()
        checksum_paths.append(relative)
        inside_root = path == ROOT or ROOT in path.parents
        actual_bytes = path.stat().st_size if inside_root and path.is_file() else None
        actual_sha256 = sha256_file(path) if actual_bytes is not None else None
        if (
            not inside_root
            or actual_bytes != int(row["bytes"])
            or actual_sha256 != row["sha256"]
        ):
            checksum_failures.append(
                {
                    "path": relative,
                    "expected_bytes": int(row["bytes"]),
                    "actual_bytes": actual_bytes,
                    "expected_sha256": row["sha256"],
                    "actual_sha256": actual_sha256,
                }
            )
    add_check(
        checks,
        "evaluation_output_checksums_recomputed",
        set(checksum_paths) == expected_checksum_paths
        and len(checksum_paths) == len(set(checksum_paths))
        and not checksum_failures,
        entries=len(checksum_rows),
        unique_paths=len(set(checksum_paths)),
        path_set_matches=set(checksum_paths) == expected_checksum_paths,
        failures=checksum_failures,
    )

    notebook_path = ROOT / "ad_safety_moderation_pipeline.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    output_count = sum(len(cell.get("outputs", [])) for cell in code_cells)
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    notebook_hash = sha256_file(notebook_path)
    notebook_text = json.dumps(notebook, ensure_ascii=False)
    notebook_outputs_text = json.dumps(
        [output for cell in code_cells for output in cell.get("outputs", [])],
        ensure_ascii=False,
    )
    book_build = build_book_in_temporary_checkout()
    book_html = str(book_build.pop("chapter_html"))
    book_index_html = str(book_build.pop("index_html"))
    notebook_facts = [
        "Safe-class precision",
        "Safe-ad argmax false-flag rate",
        "Safe-ad restricted-threshold crossing rate",
        "validation-threshold operating recall",
        "0.944444",
        "0.916667",
        "concurrent-load throughput",
        "MPS resource use",
        "NOT ASSESSED",
    ]
    executed_output_facts = [
        "Safe-class precision",
        "Safe-ad argmax false-flag rate",
        "Safe-ad restricted-threshold crossing rate",
        "validation-threshold operating recall",
        "0.944444",
        "0.916667",
        "NOT ASSESSED",
    ]
    notebook_ok = (
        len(code_cells) == 22
        and execution_counts == list(range(1, 23))
        and output_count == 53
        and not error_outputs
        and all(fact.casefold() in notebook_text.casefold() for fact in notebook_facts)
        and all(fact.casefold() in notebook_outputs_text.casefold() for fact in executed_output_facts)
        and book_build["returncode"] == 0
        and book_build["index_present"] is True
        and book_build["chapter_present"] is True
        and "Ad Safety Moderation Pilot".casefold() in book_index_html.casefold()
        and all(fact.casefold() in book_html.casefold() for fact in notebook_facts)
        and "generic coco" not in notebook_text.casefold()
        and "generic coco" not in book_html.casefold()
    )
    add_check(
        checks,
        "executed_notebook_and_book",
        notebook_ok,
        code_cells=len(code_cells),
        execution_counts=execution_counts,
        output_count=output_count,
        error_outputs=len(error_outputs),
        temporary_book_build=book_build,
        missing_required_facts=[
            fact
            for fact in notebook_facts
            if fact.casefold() not in notebook_text.casefold() or fact.casefold() not in book_html.casefold()
        ],
        missing_executed_output_facts=[
            fact for fact in executed_output_facts if fact.casefold() not in notebook_outputs_text.casefold()
        ],
        sha256=notebook_hash,
    )

    pdf_path = ROOT / "outputs/report/ad_safety_technical_synopsis.pdf"
    pdf = PdfReader(str(pdf_path))
    page_sizes = [
        [float(page.mediabox.width), float(page.mediabox.height)]
        for page in pdf.pages
    ]
    report_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    normalized_report_text = re.sub(r"\s+", " ", report_text)
    report_render_evidence: list[dict[str, Any]] = []
    report_render_returncode: int | None = None
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        with tempfile.TemporaryDirectory(prefix="ad-safety-pdf-") as temp_dir:
            prefix = Path(temp_dir) / "page"
            render_env = dict(os.environ)
            render_env["XDG_CACHE_HOME"] = str(Path(temp_dir) / "cache")
            render = subprocess.run(
                [pdftoppm, "-png", "-r", "180", str(pdf_path), str(prefix)],
                check=False,
                capture_output=True,
                text=True,
                env=render_env,
                timeout=120,
            )
            report_render_returncode = render.returncode
            generated = [Path(f"{prefix}-1.png"), Path(f"{prefix}-2.png")]
            if render.returncode == 0 and all(path.is_file() for path in generated):
                report_render_evidence = [rendered_image_evidence(path) for path in generated]
    report_facts = [
        "0.972222",
        "0.944444",
        "0.916667",
        "per-class recall >0.95 fails",
        "71.527 ms",
        "Full end-to-end p95 was not measured",
        "Safe-class precision",
        "Safe-ad false-flag rate (1 - Safe recall)",
        ">0.98",
        "0.90",
        "55.5%",
        "90.0%",
        "concurrent throughput",
        "MPS",
        "session charts are descriptive interface evidence",
    ]
    report_render_ok = (
        report_render_returncode == 0
        and len(report_render_evidence) == 2
        and all(
            row["width"] == 1530
            and row["height"] == 1980
            and row["format"] == "PNG"
            and row["nonblank"] is True
            for row in report_render_evidence
        )
        and len({row["pixel_sha256"] for row in report_render_evidence}) == 2
    )
    report_ok = (
        len(pdf.pages) == 2
        and all(abs(width - 612.0) < 1 and abs(height - 792.0) < 1 for width, height in page_sizes)
        and report_render_ok
        and all(fact.casefold() in normalized_report_text.casefold() for fact in report_facts)
    )
    add_check(
        checks,
        "strict_two_page_report",
        report_ok,
        pages=len(pdf.pages),
        page_sizes=page_sizes,
        current_pdf_render_valid=report_render_ok,
        render_returncode=report_render_returncode,
        render_evidence=report_render_evidence,
        pdftoppm_binary=Path(pdftoppm).name if pdftoppm else None,
        missing_required_facts=[
            fact for fact in report_facts if fact.casefold() not in normalized_report_text.casefold()
        ],
    )

    pptx_path = ROOT / "outputs/presentation/ad_safety_management_presentation.pptx"
    with zipfile.ZipFile(pptx_path) as archive:
        names = archive.namelist()
        slide_xml = sorted(
            (name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
            key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1)),
        )
        notes_xml = sorted(
            (name for name in names if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)),
            key=lambda name: int(re.search(r"notesSlide(\d+)\.xml$", name).group(1)),
        )
        slide_texts = [extract_pptx_text(archive.read(name)) for name in slide_xml]
        notes_texts = [extract_pptx_text(archive.read(name)) for name in notes_xml]
        slide_text = "\n".join(slide_texts)
        notes_text = "\n".join(notes_texts)
        presentation_root = ET.fromstring(archive.read("ppt/presentation.xml"))
        slide_size_node = next(
            (node for node in presentation_root.iter() if node.tag.endswith("}sldSz")),
            None,
        )
        slide_size_emu = (
            [int(slide_size_node.attrib["cx"]), int(slide_size_node.attrib["cy"])]
            if slide_size_node is not None
            else [0, 0]
        )
        notes_relationship_count = 0
        for slide_name in slide_xml:
            slide_file = Path(slide_name).name
            rel_name = f"ppt/slides/_rels/{slide_file}.rels"
            if rel_name not in names:
                continue
            rel_root = ET.fromstring(archive.read(rel_name))
            notes_relationship_count += sum(
                node.attrib.get("Type", "").endswith("/notesSlide")
                for node in rel_root.iter()
                if node.tag.endswith("}Relationship")
            )
        archive_test = archive.testzip()
    presentation_validation = json.loads(
        (ROOT / "outputs/presentation/presentation_validation.json").read_text()
    )
    presentation_qa = json.loads((ROOT / "outputs/presentation/qa_report.json").read_text())
    pptx_sha256 = sha256_file(pptx_path)
    presenter_names = ["Swarnaditya Maitra", "Vijay Agnihotri", "Myetchae Thu", "Bickramjit Basu"]
    presentation_facts = [
        "0.97913",
        "0.87315",
        "71.527",
        "0.97222",
        "0.94444",
        "0.91667",
        "0.55489",
        "0.90 detector nonweapon FPR",
        "end-to-end p95 unassessed",
        ">0.95 per-class target: FAIL",
        "Safe precision",
        ">0.98",
        "calibration floor ≥0.90",
        "concurrency",
        "MPS",
        "Session analytics describe review",
        "not production or population evidence",
    ]
    mvp_slide_text = slide_texts[14]
    mvp_roadmap_facts = [
        "recommended 10-week",
        "WEEKS 1–3",
        "WEEKS 4–6",
        "WEEKS 7–10",
        "2 annotators",
        "1 ML engineer",
        "app/MLOps engineer",
        "10 concurrent requests",
        "CPU/GPU/RAM",
    ]
    validation_facts = presentation_validation.get("measured_facts_used", {})
    roadmap_plan = presentation_validation.get("mvp_roadmap_plan", {})
    presentation_validation_path = ROOT / "outputs/presentation/presentation_validation.json"
    presentation_builder_path = ROOT / "scripts/build_presentation.mjs"
    artifact_montage_path = ROOT / "outputs/presentation/artifact_tool_montage.png"
    pptx_montage_path = ROOT / "outputs/presentation/pptx_montage.png"
    presentation_support_hashes_match = (
        presentation_qa.get("validation", {}).get("sha256") == sha256_file(presentation_validation_path)
        and presentation_qa.get("builder", {}).get("sha256") == sha256_file(presentation_builder_path)
        and presentation_qa.get("rendering", {}).get("artifact_tool_montage", {}).get("sha256")
        == sha256_file(artifact_montage_path)
        and presentation_qa.get("rendering", {}).get("independent_pptx_montage", {}).get("sha256")
        == sha256_file(pptx_montage_path)
    )
    notes_portable = "/Users/" not in notes_text and "src/ad_safety/pipeline.py" not in notes_text
    each_note_has_sources = all("[Sources]" in text for text in notes_texts)
    each_note_has_reference = all("https://" in text or "Project-relative:" in text for text in notes_texts)
    presentation_manual_status = str(
        presentation_qa.get("rendering", {}).get("manual_visual_inspection", "")
    )
    presentation_layout = presentation_qa.get("layout_tests", {})
    montage_sizes: list[list[int]] = []
    for path in (artifact_montage_path, pptx_montage_path):
        with Image.open(path) as image:
            montage_sizes.append([image.width, image.height])
    soffice_binary = shutil.which("soffice")
    presentation_pdftoppm_binary = shutil.which("pdftoppm")
    pptx_rerender_failures: list[str] = []
    pptx_rerender_returncode: int | None = None
    pptx_rerender_page_count = 0
    pptx_rerender_evidence: list[dict[str, Any]] = []
    if soffice_binary is None or presentation_pdftoppm_binary is None:
        pptx_rerender_failures.append("missing_soffice_or_pdftoppm")
    else:
        with tempfile.TemporaryDirectory(prefix="ad-safety-pptx-rerender-") as temp_dir:
            temp_root = Path(temp_dir)
            conversion = subprocess.run(
                [
                    soffice_binary,
                    f"-env:UserInstallation={(temp_root / 'profile').as_uri()}",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(temp_root),
                    str(pptx_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            pptx_rerender_returncode = conversion.returncode
            rendered_pdf = temp_root / "ad_safety_management_presentation.pdf"
            if conversion.returncode != 0 or not rendered_pdf.is_file():
                pptx_rerender_failures.append("pptx_to_pdf_conversion")
            else:
                try:
                    pptx_rerender_page_count = len(PdfReader(str(rendered_pdf)).pages)
                except Exception:
                    pptx_rerender_failures.append("rendered_pdf_parse")
                if pptx_rerender_page_count != 15:
                    pptx_rerender_failures.append("rendered_pdf_page_count")
                for page_number in range(1, 16):
                    prefix = temp_root / f"slide-{page_number}"
                    raster = subprocess.run(
                        [
                            presentation_pdftoppm_binary,
                            "-f",
                            str(page_number),
                            "-l",
                            str(page_number),
                            "-singlefile",
                            "-png",
                            "-r",
                            "120",
                            str(rendered_pdf),
                            str(prefix),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    fresh_slide = prefix.with_suffix(".png")
                    if raster.returncode != 0 or not fresh_slide.is_file():
                        pptx_rerender_failures.append(f"slide-{page_number}")
                    else:
                        evidence = rendered_image_evidence(fresh_slide)
                        evidence["slide"] = page_number
                        pptx_rerender_evidence.append(evidence)
                        if not (
                            evidence["width"] in {1600, 1601}
                            and evidence["height"] == 900
                            and evidence["format"] == "PNG"
                            and evidence["nonblank"] is True
                        ):
                            pptx_rerender_failures.append(f"slide-{page_number}:render_evidence")
                if len({row["pixel_sha256"] for row in pptx_rerender_evidence}) != 15:
                    pptx_rerender_failures.append("rendered_slides_not_unique")
    presentation_ok = (
        len(slide_xml) == 15
        and len(notes_xml) == 15
        and each_note_has_sources
        and each_note_has_reference
        and archive_test is None
        and notes_relationship_count == 15
        and slide_size_emu[0] > 0
        and close_enough(slide_size_emu[0] / slide_size_emu[1], 16 / 9, 1e-9)
        and notes_portable
        and all(name in f"{slide_text}\n{notes_text}" for name in presenter_names)
        and all(fact.casefold() in slide_text.casefold() for fact in presentation_facts)
        and all(fact.casefold() in mvp_slide_text.casefold() for fact in mvp_roadmap_facts)
        and pptx_sha256 == presentation_validation.get("pptx_sha256")
        and pptx_sha256 == presentation_qa.get("artifact", {}).get("sha256")
        and pptx_path.stat().st_size == presentation_qa.get("artifact", {}).get("bytes")
        and presentation_support_hashes_match
        and presentation_validation["semantic_overlap_and_bounds_test"]["passed"] is True
        and presentation_validation.get("authored_slide_count") == 15
        and presentation_validation.get("reopened_slide_count") == 15
        and presentation_validation.get("rendered_png_count") == 15
        and presentation_validation.get("slide_size") == {"width": 1280, "height": 720}
        and presentation_validation["sources_notes"]["all_have_sources_block"] is True
        and presentation_validation["sources_notes"].get("all_local_sources_portable") is True
        and roadmap_plan.get("status") == "recommended plan, not a measured commitment"
        and roadmap_plan.get("timeline_weeks") == 10
        and roadmap_plan.get("phase_windows") == ["weeks 1-3", "weeks 4-6", "weeks 7-10"]
        and len(roadmap_plan.get("resource_requirements", [])) == 3
        and roadmap_plan.get("scale_gate", {}).get("concurrent_requests") == 10
        and roadmap_plan.get("scale_gate", {}).get("measurements")
        == ["end-to-end p95", "CPU", "GPU", "RAM"]
        and close_enough(validation_facts.get("formal_multiclass_argmax_restricted_class_average_recall"), 0.97222, 1e-5)
        and close_enough(validation_facts.get("formal_validation_threshold_operating_recall_mean"), 0.94444, 1e-5)
        and close_enough(validation_facts.get("formal_validation_threshold_operating_recall_worst_class"), 0.91667, 1e-5)
        and validation_facts.get("formal_per_class_recall_target_met") is False
        and close_enough(validation_facts.get("formal_safe_precision"), 1.0)
        and validation_facts.get("formal_safe_precision_target_met") is True
        and validation_facts.get("full_end_to_end_p95_assessed") is False
        and validation_facts.get("concurrent_throughput_assessed") is False
        and validation_facts.get("mps_resource_use_assessed") is False
        and presentation_manual_status.upper().startswith("PASS:")
        and presentation_qa.get("rendering", {}).get("artifact_tool_png_count") == 15
        and presentation_qa.get("rendering", {}).get("reopened_pptx_png_count") == 15
        and presentation_qa.get("speaker_notes", {}).get("notes_slide_count") == 15
        and presentation_qa.get("speaker_notes", {}).get("sources_block_count") == 15
        and presentation_qa.get("speaker_notes", {}).get("absolute_project_path_count") == 0
        and presentation_qa.get("speaker_notes", {}).get("pipeline_py_mention_count") == 0
        and presentation_layout.get("semantic_overlap_issues") == 0
        and presentation_layout.get("semantic_out_of_bounds_issues") == 0
        and presentation_layout.get("artifact_tool_reopen_slide_count") == 15
        and not pptx_rerender_failures
        and len(montage_sizes) == 2
        and all(width >= 1900 and height >= 700 for width, height in montage_sizes)
    )
    add_check(
        checks,
        "management_presentation",
        presentation_ok,
        slides=len(slide_xml),
        notes_slides=len(notes_xml),
        sources_blocks=sum("[Sources]" in text for text in notes_texts),
        notes_portable=notes_portable,
        notes_relationship_count=notes_relationship_count,
        slide_size_emu=slide_size_emu,
        montage_sizes=montage_sizes,
        manual_visual_inspection=presentation_manual_status,
        presenters_found={name: name in f"{slide_text}\n{notes_text}" for name in presenter_names},
        missing_required_facts=[fact for fact in presentation_facts if fact.casefold() not in slide_text.casefold()],
        missing_mvp_roadmap_facts=[
            fact for fact in mvp_roadmap_facts if fact.casefold() not in mvp_slide_text.casefold()
        ],
        mvp_roadmap_plan=roadmap_plan,
        archive_error=archive_test,
        sha256=pptx_sha256,
        validation_sha256=presentation_validation.get("pptx_sha256"),
        qa_sha256=presentation_qa.get("artifact", {}).get("sha256"),
        support_hashes_match=presentation_support_hashes_match,
        current_pptx_rerender={
            "soffice_binary": Path(soffice_binary).name if soffice_binary else None,
            "pdftoppm_binary": (
                Path(presentation_pdftoppm_binary).name
                if presentation_pdftoppm_binary
                else None
            ),
            "conversion_returncode": pptx_rerender_returncode,
            "page_count": pptx_rerender_page_count,
            "render_failures": pptx_rerender_failures,
            "slides_validated": len(pptx_rerender_evidence),
            "render_evidence": pptx_rerender_evidence,
        },
    )

    app_source = (ROOT / "app.py").read_text()
    app_shots = [
        ROOT / "outputs/app/app_shell.png",
        ROOT / "outputs/app/app_safe_result.png",
        ROOT / "outputs/app/app_financial_result.png",
        ROOT / "outputs/app/app_firearm_result.png",
    ]
    shot_sizes = []
    for path in app_shots:
        with Image.open(path) as image:
            shot_sizes.append([image.width, image.height])
    browser_validation = json.loads((ROOT / "outputs/app/browser_validation.json").read_text())
    browser_verdicts = {row["id"]: row["observed_verdict"] for row in browser_validation["cases"]}
    interface_validation = browser_validation.get("interface_validation", {})
    live_api_validation = browser_validation.get("live_api_validation", {})
    audit_download = interface_validation.get("audit_download", {})
    expected_threshold_keys = {
        "explosives",
        "financial_promotion",
        "firearms",
        "review",
        "safe",
        "uncertainty_margin",
    }
    app_hash_failures: list[str] = []
    provenance_paths = {
        "app_py_sha256": ROOT / "app.py",
        "api_py_sha256": ROOT / "api.py",
        "user_manual_sha256": ROOT / "USER_MANUAL.md",
        "policy_yaml_sha256": ROOT / "configs/policy.yaml",
        "vit_policy_head_sha256": ROOT / "models/vit_policy_head.joblib",
        "shell_screenshot_sha256": ROOT / "outputs/app/app_shell.png",
        "pilot_evidence_screenshot_sha256": ROOT / "outputs/app/pilot_evidence_dashboard.png",
        "pilot_evidence_charts_sha256": ROOT / "outputs/app/pilot_evidence_charts.png",
        "decision_chart_screenshot_sha256": ROOT / "outputs/app/safe_decision_chart.png",
    }
    for key, path in provenance_paths.items():
        if browser_validation.get("provenance", {}).get(key) != sha256_file(path):
            app_hash_failures.append(key)
    runtime_hashes = browser_validation.get("provenance", {}).get("runtime_files_sha256", {})
    expected_runtime_files = {
        "models/pretrained_model_manifest.json",
        "requirements-lock.txt",
        *{
            str(path.relative_to(ROOT))
            for path in (ROOT / "src/ad_safety").glob("*.py")
            if path.is_file()
        },
    }
    if set(runtime_hashes) != expected_runtime_files:
        app_hash_failures.append("runtime_files:path_set")
    for relative, expected_sha256 in runtime_hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_sha256:
            app_hash_failures.append(f"runtime_files:{relative}")
    runtime_model_hashes = browser_validation.get("provenance", {}).get("runtime_model_files_sha256", {})
    expected_runtime_model_files = {
        ".cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/config.json",
        ".cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors",
        ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/config.json",
        ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/model.safetensors",
        ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/preprocessor_config.json",
        ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/tokenizer.json",
        ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/tokenizer_config.json",
        ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/special_tokens_map.json",
        ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/added_tokens.json",
        ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/vocab.txt",
    }
    if set(runtime_model_hashes) != expected_runtime_model_files:
        app_hash_failures.append("runtime_model_files:path_set")
    for relative, expected_sha256 in runtime_model_hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_sha256:
            app_hash_failures.append(f"runtime_model_files:{relative}")
    for row in browser_validation["cases"]:
        input_path = ROOT / row["file"]
        screenshot_path = ROOT / row["screenshot"]
        if not input_path.is_file() or row.get("input_sha256") != sha256_file(input_path):
            app_hash_failures.append(f"{row['id']}:input")
        if not screenshot_path.is_file() or row.get("screenshot_sha256") != sha256_file(screenshot_path):
            app_hash_failures.append(f"{row['id']}:screenshot")
    app_ok = (
        browser_validation.get("schema_version") == "1.3.0"
        and "use_container_width" not in app_source
        and all(width >= 1200 and height >= 700 for width, height in shot_sizes)
        and browser_validation["layout"]["horizontal_overflow"] is False
        and not browser_validation["console"]["errors"]
        and not browser_validation["console"]["warnings"]
        and not app_hash_failures
        and browser_validation["layout"]["document_scroll_width"] == browser_validation["layout"]["document_client_width"]
        and len(browser_validation["cases"]) == 3
        and len({row["id"] for row in browser_validation["cases"]}) == 3
        and all(row["expected_label"] == row["observed_label"] for row in browser_validation["cases"])
        and browser_verdicts == {
            "formal_safe": "APPROVE",
            "formal_financial_promotion": "REVIEW",
            "formal_firearms": "BLOCK",
        }
        and interface_validation.get("navigation_sections") == ["Analyze", "Pilot evidence", "How it works"]
        and interface_validation.get("pilot_evidence_plotly_charts") == 5
        and interface_validation.get("completed_result_plotly_charts", 0) >= 2
        and all(
            interface_validation.get(key) is True
            for key in (
                "formal_and_external_scopes_separated",
                "accessible_chart_data_table",
                "reduced_motion_css_present",
                "result_persists_after_option_rerun",
                "changed_options_mark_result_stale",
                "option_change_did_not_reanalyze",
                "new_upload_clears_previous_result",
                "result_persists_across_workspace_navigation",
                "same_file_reupload_does_not_resurrect_result",
                "corrupt_upload_rejected_before_analysis",
                "invalid_replacement_clears_previous_result",
                "core_readiness_checks_head_and_vit_snapshot",
                "detector_snapshot_status_visible",
            )
        )
        and interface_validation.get("accessible_chart_data_tables") == 4
        and interface_validation.get("accessible_confusion_matrix_cells") == 16
        and interface_validation.get("formal_metric_delta_elements") == 0
        and interface_validation.get("result_metric_delta_elements") == 0
        and audit_download.get("schema_version") == "1.1"
        and audit_download.get("verdict") == "BLOCK"
        and audit_download.get("options") == {"detector": True, "ocr": False, "explanation": True}
        and set(audit_download.get("applied_threshold_keys", [])) == expected_threshold_keys
        and all(
            live_api_validation.get(key) is True
            for key in (
                "cold_health_ready",
                "post_safe_engine_loaded",
                "post_firearm_detector_loaded",
                "corrupt_input_rejected",
                "unsupported_media_rejected",
                "unexpected_error_opaque_reference_tested",
                "missing_visual_backbone_returns_503_tested",
                "missing_requested_detector_returns_503_tested",
            )
        )
        and live_api_validation.get("cold_engine_loaded") is False
        and live_api_validation.get("safe_verdict") == "APPROVE"
        and live_api_validation.get("firearm_verdict") == "BLOCK"
        and live_api_validation.get("firearm_detector_boxes", 0) > 0
        and live_api_validation.get("schema_version") == "1.1"
        and "prefers-reduced-motion:reduce" in app_source.replace(" ", "")
    )
    add_check(
        checks,
        "app_browser_evidence",
        app_ok,
        screenshot_sizes=shot_sizes,
        deprecated_width_api=False,
        horizontal_overflow=browser_validation["layout"]["horizontal_overflow"],
        console_errors=browser_validation["console"]["errors"],
        console_warnings=browser_validation["console"]["warnings"],
        provenance_hash_failures=app_hash_failures,
        verdicts=browser_verdicts,
        interface_validation=interface_validation,
        live_api_validation=live_api_validation,
    )

    demo = json.loads((ROOT / "outputs/demo_cases/case_summary.json").read_text())
    demo_rows = demo if isinstance(demo, list) else demo.get("cases", [])
    demo_decisions = Counter(
        row.get("decision") or row.get("verdict") or row.get("policy_verdict") for row in demo_rows
    )
    demo_ok = len(demo_rows) >= 6 and all(demo_decisions.get(value, 0) for value in ("APPROVE", "REVIEW", "BLOCK"))
    add_check(checks, "representative_demo_cases", demo_ok, cases=len(demo_rows), decisions=dict(demo_decisions))

    video_path = ROOT / "outputs/video/ad_safety_demo.mp4"
    video = parse_ffmpeg_probe(video_path)
    narration_text = (ROOT / "outputs/video/narration_script.md").read_text().casefold()
    video_manifest = json.loads((ROOT / "outputs/video/video_manifest.json").read_text())
    video_qa = json.loads((ROOT / "outputs/video/qa_report.json").read_text())
    narration_provenance_path = ROOT / "outputs/video/narration_provenance.json"
    narration_provenance = json.loads(narration_provenance_path.read_text())
    workflow_narration_provenance = next(
        (row for row in narration_provenance if row.get("scene_id") == "13_demo_workflow"),
        {},
    )
    workflow_narration_text = str(workflow_narration_provenance.get("narration_text", ""))
    workflow_narration_provenance_ok = (
        len(narration_provenance) == 19
        and len({row.get("scene_id") for row in narration_provenance}) == 19
        and workflow_narration_provenance.get("forced_regeneration") is True
        and workflow_narration_text.startswith("The workspace has four steps.")
        and "three steps" not in workflow_narration_text.casefold()
        and workflow_narration_provenance.get("narration_text_sha256")
        == hashlib.sha256(workflow_narration_text.encode("utf-8")).hexdigest()
        and video_qa.get("workflow_narration_provenance") == workflow_narration_provenance
    )
    manifest_text = json.dumps(video_manifest).casefold()
    narration_stale_text_absent = (
        "three steps" not in narration_text
        and "from pixels to a review queue" not in narration_text
        and "three steps" not in manifest_text
        and "from pixels to a review queue" not in manifest_text
    )
    offline_disclosure_ok = (
        "offline macos" in narration_text
        and "cedar" not in narration_text
        and "gpt-4o-mini-tts" not in manifest_text
        and "cedar" not in manifest_text
    )
    narration_facts = [
        "multiclass argmax",
        "0.97913",
        "0.87315",
        "0.97222",
        "threshold-operating mean",
        "0.94444",
        "minimum 0.91667",
        "per-class greater-than-0.95 target failed",
        "safe precision",
        "greater-than-0.98",
        "0.90 validation-recall tuning floor",
        "71.527",
        "40.064",
        "0.55489",
        "detector's nonweapon false-positive rate reached 0.90",
        "concurrent throughput",
        "M-P-S resource use",
    ]
    expected_video_artifacts = {
        "outputs/video/ad_safety_demo.mp4",
        "outputs/video/ad_safety_demo.srt",
        "outputs/video/narration_script.md",
        "outputs/video/storyboard.csv",
        "outputs/video/storyboard.json",
        "outputs/video/storyboard_actual.csv",
        "outputs/video/storyboard_actual.json",
        "outputs/video/video_manifest.json",
        "outputs/video/narration_provenance.json",
    }
    video_checksum_failures: list[str] = []
    video_checksum_rows = video_qa.get("artifact_checksums", [])
    if {item.get("path") for item in video_checksum_rows} != expected_video_artifacts:
        video_checksum_failures.append("artifact_checksums:path_set")
    for item in video_checksum_rows:
        path = ROOT / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            video_checksum_failures.append(item["path"])
    representative_frames = video_qa.get("representative_frames", [])
    representative_frame_failures: list[str] = []
    representative_frame_times: list[float] = []
    if len(representative_frames) != 6 or len(set(representative_frames)) != 6:
        representative_frame_failures.append("representative_frames:path_set")
    for relative in representative_frames:
        match = re.search(r"_(\d+(?:\.\d+)?)s\.jpg$", str(relative))
        if match:
            representative_frame_times.append(float(match.group(1)))
        else:
            representative_frame_failures.append(f"{relative}:timestamp")
    caption_cues = (ROOT / "outputs/video/ad_safety_demo.srt").read_text().count(" --> ")
    decode = subprocess.run(
        [video["ffmpeg"], "-v", "error", "-i", str(video_path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    subtitle_extract = subprocess.run(
        [video["ffmpeg"], "-v", "error", "-i", str(video_path), "-map", "0:s:0", "-f", "srt", "-"],
        check=False,
        capture_output=True,
        timeout=60,
    )
    saved_srt = (ROOT / "outputs/video/ad_safety_demo.srt").read_bytes()
    subtitle_roundtrip_ok = subtitle_extract.returncode == 0 and subtitle_extract.stdout == saved_srt
    audio_signal = subprocess.run(
        [
            video["ffmpeg"],
            "-hide_banner",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    mean_volume_match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", audio_signal.stderr)
    max_volume_match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", audio_signal.stderr)
    mean_volume_db = float(mean_volume_match.group(1)) if mean_volume_match else None
    max_volume_db = float(max_volume_match.group(1)) if max_volume_match else None
    audio_signal_ok = (
        audio_signal.returncode == 0
        and mean_volume_db is not None
        and max_volume_db is not None
        and mean_volume_db > -60.0
        and max_volume_db > -30.0
    )
    representative_frame_evidence: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ad-safety-video-frames-") as temp_dir:
        for index, timestamp in enumerate(representative_frame_times, start=1):
            output = Path(temp_dir) / f"frame-{index:02d}.jpg"
            frame_result = subprocess.run(
                [
                    video["ffmpeg"],
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if frame_result.returncode != 0 or not output.is_file():
                representative_frame_failures.append(f"frame-{index}:extract")
                continue
            evidence = rendered_image_evidence(output)
            evidence["timestamp_seconds"] = timestamp
            representative_frame_evidence.append(evidence)
            if not (
                evidence["width"] == 1280
                and evidence["height"] == 720
                and evidence["format"] == "JPEG"
                and evidence["nonblank"] is True
            ):
                representative_frame_failures.append(f"frame-{index}:render_evidence")
    if len({row["pixel_sha256"] for row in representative_frame_evidence}) != 6:
        representative_frame_failures.append("representative_frames_not_unique")
    representative_frames_regenerated = (
        len(representative_frame_times) == 6
        and all(0.0 <= value < video["duration_seconds"] for value in representative_frame_times)
        and len(representative_frame_evidence) == 6
        and not representative_frame_failures
    )
    expected_video_qa_checks = {
        "duration_300_to_480_seconds",
        "resolution_1280x720",
        "video_codec_h264",
        "audio_codec_aac",
        "caption_stream_mov_text",
        "frame_count_nontrivial",
        "decode_completed_without_errors",
        "narration_disclosure_present",
        "external_speech_api_not_used",
        "all_15_slide_renders_used",
        "captions_present",
        "workflow_narration_declares_four_steps",
        "workflow_caption_declares_four_steps",
        "workflow_caption_has_no_three_step_text",
        "workflow_narration_provenance_current",
    }
    video_qa_checks = video_qa.get("checks", {})
    manual_visual_status = str(video_qa.get("manual_visual_inspection", ""))
    video_ok = (
        300 <= video["duration_seconds"] <= 480
        and video["width"] == 1280
        and video["height"] == 720
        and video["has_video_stream"]
        and video["has_audio_stream"]
        and video["has_subtitle_stream"]
        and str(video["video_codec"]).startswith("h264")
        and str(video["audio_codec"]).startswith("aac")
        and offline_disclosure_ok
        and all(fact.casefold() in narration_text for fact in narration_facts)
        and video_manifest.get("scene_count") == 19
        and video_manifest.get("caption_cue_count") == caption_cues
        and len(video_manifest.get("storyboard_actual", [])) == 19
        and not video_checksum_failures
        and representative_frames_regenerated
        and subtitle_roundtrip_ok
        and audio_signal_ok
        and workflow_narration_provenance_ok
        and narration_stale_text_absent
        and video_qa.get("media", {}).get("sha256") == sha256_file(video_path)
        and video_qa.get("media", {}).get("file_bytes") == video_path.stat().st_size
        and decode.returncode == 0
        and video_qa.get("status") == "PASS"
        and set(video_qa_checks) == expected_video_qa_checks
        and all(value is True for value in video_qa_checks.values())
        and manual_visual_status.upper().startswith("PASS:")
    )
    add_check(
        checks,
        "narrated_demo_video",
        video_ok,
        **video,
        sha256=sha256_file(video_path),
        offline_disclosure_ok=offline_disclosure_ok,
        missing_required_facts=[fact for fact in narration_facts if fact.casefold() not in narration_text],
        caption_cues=caption_cues,
        video_checksum_failures=video_checksum_failures,
        representative_frame_failures=representative_frame_failures,
        representative_frames_regenerated=representative_frames_regenerated,
        representative_frame_evidence=representative_frame_evidence,
        subtitle_roundtrip_ok=subtitle_roundtrip_ok,
        audio_signal_ok=audio_signal_ok,
        workflow_narration_provenance_ok=workflow_narration_provenance_ok,
        narration_stale_text_absent=narration_stale_text_absent,
        workflow_narration_provenance=workflow_narration_provenance,
        audio_mean_volume_db=mean_volume_db,
        audio_max_volume_db=max_volume_db,
        independent_decode_returncode=decode.returncode,
        independent_decode_stderr=decode.stderr[-4000:],
        video_qa_status=video_qa.get("status"),
        manual_visual_inspection=manual_visual_status,
    )

    MARKER_PATH.write_text("FINAL_VALIDATION_IN_PROGRESS\n", encoding="utf-8")
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(ROOT / "src")
    test_env.setdefault("OMP_NUM_THREADS", "1")
    test_env.setdefault("MKL_NUM_THREADS", "1")
    test_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        env=test_env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    tests_text = f"{tests.stdout}\n{tests.stderr}".strip()
    add_check(checks, "automated_tests", tests.returncode == 0, returncode=tests.returncode, output=tests_text[-4000:])

    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "app.py", "api.py", "src", "scripts", "tests"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    add_check(
        checks,
        "python_compilation",
        compile_result.returncode == 0,
        returncode=compile_result.returncode,
        output=f"{compile_result.stdout}\n{compile_result.stderr}".strip()[-4000:],
    )

    all_passed = all(item["passed"] for item in checks.values())
    result = {
        "schema_version": "1.0.0",
        "project_root": ".",
        "all_checks_passed": all_passed,
        "check_count": len(checks),
        "checks": checks,
    }
    write_result(result)
    if all_passed:
        MARKER_PATH.write_text("FINAL_VALIDATION_PASSED\n", encoding="utf-8")
        print(f"FINAL_VALIDATION_PASSED ({len(checks)} checks)")
        print(RESULT_PATH.relative_to(ROOT).as_posix())
        return 0
    failed = [name for name, item in checks.items() if not item["passed"]]
    if MARKER_PATH.exists():
        MARKER_PATH.unlink()
    print(f"FINAL_VALIDATION_FAILED: {', '.join(failed)}")
    print(RESULT_PATH.relative_to(ROOT).as_posix())
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if MARKER_PATH.exists():
            MARKER_PATH.unlink()
        failure = {
            "schema_version": "1.0.0",
            "project_root": ".",
            "all_checks_passed": False,
            "check_count": 1,
            "checks": {
                "validator_completed_without_exception": {
                    "passed": False,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            },
        }
        write_result(failure)
        print(f"FINAL_VALIDATION_FAILED: {type(exc).__name__}: {portable_text(str(exc))}")
        print(RESULT_PATH.relative_to(ROOT).as_posix())
        raise SystemExit(1)
