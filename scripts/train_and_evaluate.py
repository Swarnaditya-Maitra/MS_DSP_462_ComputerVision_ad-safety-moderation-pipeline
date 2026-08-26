#!/usr/bin/env python3
"""Train frozen visual policy heads and evaluate an untouched grouped test split.

The ViT is the predeclared primary model. ResNet50 is the frozen CNN baseline.
Only logistic heads are fit. Restricted-class operating thresholds are selected
from validation predictions under a recall-first rule, then applied once to the
held-out test split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
MODEL_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = PROJECT_ROOT / "data" / "capstone_registry.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "capstone_dataset" / "dataset_manifest.json"
PRETRAINED_MANIFEST_PATH = MODEL_DIR / "pretrained_model_manifest.json"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _head_worker_main() -> None:
    """Fit or verify sklearn artifacts without importing torch or timm."""

    import argparse as worker_argparse

    import joblib as worker_joblib
    import numpy as worker_np
    from sklearn.linear_model import LogisticRegression as WorkerLogisticRegression
    from sklearn.pipeline import Pipeline as WorkerPipeline
    from sklearn.preprocessing import StandardScaler as WorkerStandardScaler

    parser = worker_argparse.ArgumentParser()
    parser.add_argument("--head-worker", choices=("fit", "predict", "verify"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    with worker_np.load(input_path, allow_pickle=False) as payload:
        features = payload["features"]
        if args.head_worker == "fit":
            y = payload["y"]
            train_mask = payload["train_mask"].astype(bool)
            val_mask = payload["val_mask"].astype(bool)
            pipeline = WorkerPipeline(
                [
                    ("standardize", WorkerStandardScaler()),
                    (
                        "logistic",
                        WorkerLogisticRegression(
                            C=0.1,
                            class_weight="balanced",
                            max_iter=5000,
                            solver="lbfgs",
                            random_state=462,
                        ),
                    ),
                ]
            )
            started = perf_counter()
            pipeline.fit(features[train_mask], y[train_mask])
            fit_seconds = perf_counter() - started
            worker_joblib.dump(pipeline, output_path, compress=3)
            worker_np.savez_compressed(
                output_path.with_suffix(".npz"),
                val_probabilities=pipeline.predict_proba(features[val_mask]),
                fit_seconds=worker_np.asarray(fit_seconds),
                n_iter=pipeline.named_steps["logistic"].n_iter_,
                classes=pipeline.classes_,
            )
        elif args.head_worker == "predict":
            if not args.artifact:
                raise ValueError("--artifact is required for predict mode")
            loaded = worker_joblib.load(args.artifact)
            pipeline = loaded["pipeline"] if isinstance(loaded, dict) else loaded
            worker_np.savez_compressed(output_path, probabilities=pipeline.predict_proba(features))
        else:
            if not args.artifact:
                raise ValueError("--artifact is required for verify mode")
            expected = payload["expected_probabilities"]
            artifact = worker_joblib.load(args.artifact)
            observed = artifact["pipeline"].predict_proba(features)
            difference = float(worker_np.max(worker_np.abs(observed - expected)))
            output_path.write_text(
                json.dumps({"max_abs_probability_difference": difference}, sort_keys=True) + "\n",
                encoding="utf-8",
            )


if "--head-worker" in sys.argv:
    _head_worker_main()
    raise SystemExit(0)

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402

from ad_safety.features import CNN_MODEL_NAME, VIT_MODEL_NAME, select_device  # noqa: E402


SEED = 462
LABELS = ("safe", "firearms", "explosives", "financial_promotion")
RESTRICTED_LABELS = ("firearms", "explosives", "financial_promotion")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
SPLITS = ("train", "val", "test")
RECALL_TARGET = 0.90
BOOTSTRAP_REPLICATES = 2000

MODEL_SPECS = (
    {"key": "vit", "display_name": "Frozen ViT + logistic head", "backbone": VIT_MODEL_NAME},
    {"key": "resnet50", "display_name": "Frozen ResNet50 + logistic head", "backbone": CNN_MODEL_NAME},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hardware_name() -> str:
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chip:"):
                return stripped.split(":", 1)[1].strip()
        return platform.processor() or platform.machine()
    except (OSError, subprocess.SubprocessError):
        return platform.processor() or platform.machine()


def package_versions() -> dict[str, str]:
    names = ("torch", "torchvision", "timm", "scikit-learn", "pandas", "numpy", "Pillow", "joblib")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def validate_registry(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "sample_id",
        "label",
        "split",
        "local_path",
        "local_sha256",
        "source_group",
        "source_dataset",
        "is_synthetic",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Registry missing columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 288:
        raise RuntimeError(f"Expected 288 registry rows, found {len(frame)}")
    if set(frame["label"]) != set(LABELS) or set(frame["split"]) != set(SPLITS):
        raise RuntimeError("Registry labels or splits differ from the declared experiment")
    expected = {"train": 48, "val": 12, "test": 12}
    counts = frame.groupby(["label", "split"]).size().to_dict()
    for label in LABELS:
        for split, count in expected.items():
            if int(counts.get((label, split), 0)) != count:
                raise RuntimeError(f"Expected {count} rows for {label}/{split}")

    group_split_count = frame.groupby("source_group")["split"].nunique()
    if int((group_split_count > 1).sum()) != 0:
        raise RuntimeError("A source group crosses split boundaries")
    hash_split_count = frame.groupby("local_sha256")["split"].nunique()
    if int((hash_split_count > 1).sum()) != 0:
        raise RuntimeError("An exact image hash crosses split boundaries")
    if frame["sample_id"].duplicated().any() or frame["local_path"].duplicated().any():
        raise RuntimeError("Duplicate sample ID or local path")

    bad_hashes: list[str] = []
    for record in frame.itertuples(index=False):
        path = PROJECT_ROOT / str(record.local_path)
        if not path.is_file() or sha256_file(path) != str(record.local_sha256):
            bad_hashes.append(str(record.sample_id))
    if bad_hashes:
        raise RuntimeError(f"Missing or hash-mismatched local images: {bad_hashes[:5]}")
    return {
        "rows": len(frame),
        "source_groups": int(frame["source_group"].nunique()),
        "group_overlap_across_splits": 0,
        "exact_hash_overlap_across_splits": 0,
        "counts": {
            label: {split: int(counts[(label, split)]) for split in SPLITS}
            for label in LABELS
        },
        "registry_sha256": sha256_file(REGISTRY_PATH),
    }


class PinnedBackboneExtractor:
    """Frozen timm extractor that resolves a manifest-pinned local snapshot."""

    def __init__(self, model_name: str, device: str) -> None:
        pretrained_manifest = json.loads(PRETRAINED_MANIFEST_PATH.read_text(encoding="utf-8"))
        repo_id = f"timm/{model_name}"
        row = next(
            (item for item in pretrained_manifest.get("models", []) if item.get("repo_id") == repo_id),
            None,
        )
        if row is None:
            raise RuntimeError(f"Pinned model missing from manifest: {repo_id}")
        self.model_name = model_name
        self.repo_id = repo_id
        self.revision = str(row["revision"])
        self.license = str(row.get("license", "unknown"))
        self.weight_path = PROJECT_ROOT / str(row["local_snapshot"]) / "model.safetensors"
        if not self.weight_path.is_file():
            raise RuntimeError(f"Pinned local weights are missing: {self.weight_path}")
        self.weight_sha256 = sha256_file(self.weight_path)
        self.device = select_device(device)
        # Keep the .safetensors suffix instead of resolving the symlink to its
        # extensionless cache blob. timm uses the suffix to choose safe loading.
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            pretrained_cfg_overlay={"source": "", "file": str(self.weight_path)},
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval().to(self.device)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.feature_dim = int(self.model.num_features)
        self.total_parameters = int(sum(parameter.numel() for parameter in self.model.parameters()))
        self.trainable_backbone_parameters = int(
            sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        )

    def synchronize(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def embed_paths(self, paths: Sequence[Path], batch_size: int) -> tuple[np.ndarray, float]:
        outputs: list[np.ndarray] = []
        self.synchronize()
        started = perf_counter()
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as raw:
                    tensors.append(self.transform(raw.convert("RGB")))
            batch = torch.stack(tensors).to(self.device)
            features = self.model(batch)
            outputs.append(features.detach().float().cpu().numpy())
        self.synchronize()
        return np.concatenate(outputs).astype(np.float32, copy=False), perf_counter() - started

    @torch.inference_mode()
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        tensors = [self.transform(image.convert("RGB")) for image in images]
        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch)
        self.synchronize()
        return features.detach().float().cpu().numpy().astype(np.float32, copy=False)


def embedding_cache_valid(
    cache_path: Path,
    sample_ids: np.ndarray,
    registry_sha256: str,
    model_name: str,
    revision: str,
) -> bool:
    if not cache_path.is_file():
        return False
    with np.load(cache_path, allow_pickle=False) as cache:
        return bool(
            np.array_equal(cache["sample_ids"].astype(str), sample_ids.astype(str))
            and str(cache["registry_sha256"].item()) == registry_sha256
            and str(cache["model_name"].item()) == model_name
            and str(cache["revision"].item()) == revision
        )


def get_embeddings(
    extractor: PinnedBackboneExtractor,
    frame: pd.DataFrame,
    key: str,
    registry_sha256: str,
    batch_size: int,
    reuse: bool,
) -> tuple[np.ndarray, float, bool, Path]:
    cache_path = OUTPUT_DIR / f"embeddings_{key}.npz"
    sample_ids = frame["sample_id"].astype(str).to_numpy()
    if reuse and embedding_cache_valid(
        cache_path,
        sample_ids,
        registry_sha256,
        extractor.model_name,
        extractor.revision,
    ):
        with np.load(cache_path, allow_pickle=False) as cache:
            return cache["features"].astype(np.float32), float(cache["elapsed_seconds"].item()), True, cache_path
    paths = [PROJECT_ROOT / path for path in frame["local_path"].astype(str)]
    features, elapsed = extractor.embed_paths(paths, batch_size=batch_size)
    np.savez_compressed(
        cache_path,
        features=features,
        sample_ids=sample_ids.astype("U"),
        registry_sha256=np.asarray(registry_sha256),
        model_name=np.asarray(extractor.model_name),
        revision=np.asarray(extractor.revision),
        elapsed_seconds=np.asarray(elapsed),
    )
    return features, elapsed, False, cache_path


def isolated_worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
        }
    )
    return environment


def fit_head_isolated(
    features: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[Pipeline, np.ndarray, float, list[int]]:
    """Fit sklearn in a clean process that never imports torch or timm."""

    with tempfile.TemporaryDirectory(prefix="ad-safety-head-fit-") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "head_input.npz"
        pipeline_path = temp_root / "pipeline.joblib"
        np.savez_compressed(
            input_path,
            features=features,
            y=y,
            train_mask=masks["train"],
            val_mask=masks["val"],
        )
        command = [
            sys.executable,
            str(Path(__file__)),
            "--head-worker",
            "fit",
            "--input",
            str(input_path),
            "--output",
            str(pipeline_path),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            env=isolated_worker_environment(),
        )
        if completed.stderr.strip():
            print(f"head worker stderr: {completed.stderr.strip()}", flush=True)
        pipeline = joblib.load(pipeline_path)
        with np.load(pipeline_path.with_suffix(".npz"), allow_pickle=False) as worker_output:
            classes = worker_output["classes"].astype(int).tolist()
            if classes != list(range(len(LABELS))):
                raise RuntimeError(f"Probability class order mismatch: {classes}")
            val_probabilities = worker_output["val_probabilities"].astype(np.float64)
            fit_seconds = float(worker_output["fit_seconds"].item())
            n_iter = worker_output["n_iter"].astype(int).tolist()
    return pipeline, val_probabilities, fit_seconds, n_iter


def predict_head_isolated(pipeline: Pipeline, features: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="ad-safety-head-predict-") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "predict_input.npz"
        pipeline_path = temp_root / "pipeline.joblib"
        output_path = temp_root / "predict_output.npz"
        np.savez_compressed(input_path, features=features)
        joblib.dump(pipeline, pipeline_path, compress=3)
        command = [
            sys.executable,
            str(Path(__file__)),
            "--head-worker",
            "predict",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--artifact",
            str(pipeline_path),
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env=isolated_worker_environment(),
        )
        with np.load(output_path, allow_pickle=False) as payload:
            probabilities = payload["probabilities"].astype(np.float64)
    return probabilities


def verify_artifact_isolated(
    artifact_path: Path,
    test_features: np.ndarray,
    expected_probabilities: np.ndarray,
) -> float:
    with tempfile.TemporaryDirectory(prefix="ad-safety-head-verify-") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "verify_input.npz"
        output_path = temp_root / "verify_result.json"
        np.savez_compressed(
            input_path,
            features=test_features,
            expected_probabilities=expected_probabilities,
        )
        command = [
            sys.executable,
            str(Path(__file__)),
            "--head-worker",
            "verify",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--artifact",
            str(artifact_path),
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env=isolated_worker_environment(),
        )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    return float(payload["max_abs_probability_difference"])


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
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


def calibrate_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Meet validation recall first, then maximize precision and threshold."""

    candidates = np.unique(np.concatenate((np.asarray([0.0, 1.0]), scores.astype(float))))
    evaluated = [binary_metrics(y_true, scores, float(threshold)) for threshold in candidates]
    eligible = [row for row in evaluated if row["recall"] >= RECALL_TARGET]
    if not eligible:
        raise RuntimeError("No validation threshold can meet the declared recall target")
    selected = max(eligible, key=lambda row: (row["precision"], row["f1"], row["threshold"]))
    selected = dict(selected)
    selected["selection_rule"] = (
        f"validation recall >= {RECALL_TARGET:.2f}; then maximize precision, F1, and threshold"
    )
    selected["candidate_count"] = len(evaluated)
    return selected


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_true,
        predictions,
        labels=np.arange(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    binary = label_binarize(y_true, classes=np.arange(len(LABELS)))
    per_class: dict[str, Any] = {}
    for index, label in enumerate(LABELS):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true == index,
            predictions == index,
            average="binary",
            zero_division=0,
        )
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(y_true == index)),
            "average_precision_ovr": float(average_precision_score(binary[:, index], probabilities[:, index])),
            "roc_auc_ovr": float(roc_auc_score(binary[:, index], probabilities[:, index])),
        }
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "macro_average_precision_ovr": float(average_precision_score(binary, probabilities, average="macro")),
        "macro_roc_auc_ovr": float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(len(LABELS)))),
        "multiclass_brier": float(np.mean(np.sum((probabilities - binary) ** 2, axis=1))),
        "expected_calibration_error_10_bin": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=np.arange(len(LABELS))).tolist(),
        "per_class": per_class,
    }


def group_bootstrap_intervals(
    y_true: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    label_groups: dict[int, list[str]] = {}
    for label_index in range(len(LABELS)):
        label_groups[label_index] = sorted(set(groups[y_true == label_index].astype(str)))
    accuracy_values: list[float] = []
    macro_f1_values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices: list[int] = []
        for label_index, available_groups in label_groups.items():
            sampled = rng.choice(available_groups, size=len(available_groups), replace=True)
            for group in sampled:
                indices.extend(np.flatnonzero((y_true == label_index) & (groups == group)).tolist())
        sample = np.asarray(indices, dtype=int)
        accuracy_values.append(float(accuracy_score(y_true[sample], predictions[sample])))
        macro_f1_values.append(float(f1_score(y_true[sample], predictions[sample], average="macro", zero_division=0)))
    return {
        "method": "stratified nonparametric bootstrap over source groups",
        "replicates": BOOTSTRAP_REPLICATES,
        "accuracy_95pct": [float(value) for value in np.percentile(accuracy_values, [2.5, 97.5])],
        "macro_f1_95pct": [float(value) for value in np.percentile(macro_f1_values, [2.5, 97.5])],
    }


def benchmark_cpu_batch1(
    extractor: PinnedBackboneExtractor,
    pipeline: Pipeline,
    images: Sequence[Image.Image],
    warmups: int = 3,
    runs: int = 20,
) -> dict[str, Any]:
    if extractor.device.type != "cpu":
        raise RuntimeError("Batch-1 benchmark extractor must be on CPU")
    for index in range(warmups):
        features = extractor.embed_images([images[index % len(images)]])
        pipeline.predict_proba(features)
    latencies: list[float] = []
    for index in range(runs):
        started = perf_counter()
        features = extractor.embed_images([images[index % len(images)]])
        pipeline.predict_proba(features)
        latencies.append((perf_counter() - started) * 1000.0)
    return {
        "scope": "decoded in-memory PIL image; preprocessing + frozen backbone + logistic head",
        "device": "cpu",
        "batch_size": 1,
        "warmup_runs": warmups,
        "measured_runs": runs,
        "torch_cpu_threads": int(torch.get_num_threads()),
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "throughput_images_per_second_from_mean": float(1000.0 / np.mean(latencies)),
        "latencies_ms": latencies,
    }


def train_one_model(
    spec: dict[str, str],
    frame: pd.DataFrame,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    registry_validation: dict[str, Any],
    device: str,
    batch_size: int,
    reuse_embeddings: bool,
    benchmark_images: Sequence[Image.Image],
) -> dict[str, Any]:
    print(f"Loading pinned backbone: {spec['display_name']}", flush=True)
    extractor = PinnedBackboneExtractor(spec["backbone"], device=device)
    features, extraction_seconds, reused, embedding_path = get_embeddings(
        extractor,
        frame,
        spec["key"],
        registry_validation["registry_sha256"],
        batch_size,
        reuse_embeddings,
    )
    if features.shape != (len(frame), extractor.feature_dim):
        raise RuntimeError(f"Unexpected feature shape for {spec['key']}: {features.shape}")
    print(
        f"{spec['key']} features: {features.shape}, {extraction_seconds:.3f}s, "
        f"device={extractor.device}, cache_reused={reused}",
        flush=True,
    )

    pipeline, val_probabilities, fit_seconds, head_iterations = fit_head_isolated(features, y, masks)

    # Validation is the only source used for threshold selection. The isolated
    # fit worker does not even score test features.
    val_y = y[masks["val"]]
    threshold_calibration: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for label in RESTRICTED_LABELS:
        label_index = LABEL_TO_INDEX[label]
        calibration = calibrate_threshold(val_y == label_index, val_probabilities[:, label_index])
        threshold_calibration[label] = calibration
        thresholds[label] = float(calibration["threshold"])

    # The test split is first scored after the fitted head and thresholds exist.
    test_probabilities = predict_head_isolated(pipeline, features[masks["test"]])
    test_y = y[masks["test"]]
    val_metrics = multiclass_metrics(val_y, val_probabilities)
    test_metrics = multiclass_metrics(test_y, test_probabilities)
    test_predictions = test_probabilities.argmax(axis=1)
    test_groups = frame.loc[masks["test"], "source_group"].astype(str).to_numpy()
    test_metrics["group_bootstrap_95pct"] = group_bootstrap_intervals(test_y, test_predictions, test_groups)

    threshold_test_metrics: dict[str, Any] = {}
    for label in RESTRICTED_LABELS:
        label_index = LABEL_TO_INDEX[label]
        threshold_test_metrics[label] = binary_metrics(
            test_y == label_index,
            test_probabilities[:, label_index],
            thresholds[label],
        )

    model_version = f"{spec['key']}-frozen-{extractor.revision[:12]}-logreg-v1"
    artifact_path = MODEL_DIR / f"{spec['key']}_policy_head.joblib"
    artifact = {
        "artifact_version": "1.0.0",
        "model_version": model_version,
        "backbone_name": extractor.model_name,
        "backbone_repo_id": extractor.repo_id,
        "backbone_revision": extractor.revision,
        "backbone_license": extractor.license,
        "backbone_weight_sha256": extractor.weight_sha256,
        "class_names": list(LABELS),
        "pipeline": pipeline,
        "thresholds": thresholds,
        "threshold_calibration": threshold_calibration,
        "training": {
            "seed": SEED,
            "registry_sha256": registry_validation["registry_sha256"],
            "training_rows": int(masks["train"].sum()),
            "validation_rows": int(masks["val"].sum()),
            "test_rows_excluded_from_fit_and_calibration": int(masks["test"].sum()),
            "feature_dim": extractor.feature_dim,
            "backbone_parameters": extractor.total_parameters,
            "trainable_backbone_parameters": extractor.trainable_backbone_parameters,
            "head_hyperparameters": pipeline.named_steps["logistic"].get_params(),
            "threshold_recall_target": RECALL_TARGET,
        },
    }
    joblib.dump(artifact, artifact_path, compress=3)

    if extractor.device.type == "cpu":
        benchmark = benchmark_cpu_batch1(extractor, pipeline, benchmark_images)
    else:
        cpu_extractor = PinnedBackboneExtractor(spec["backbone"], device="cpu")
        benchmark = benchmark_cpu_batch1(cpu_extractor, pipeline, benchmark_images)
        del cpu_extractor

    reload_max_abs_difference = verify_artifact_isolated(
        artifact_path,
        features[masks["test"]],
        test_probabilities,
    )
    if reload_max_abs_difference > 1e-12:
        raise RuntimeError(f"Reloaded artifact predictions changed: {reload_max_abs_difference}")

    result = {
        "key": spec["key"],
        "display_name": spec["display_name"],
        "model_version": model_version,
        "role": "predeclared primary" if spec["key"] == "vit" else "predeclared CNN baseline",
        "backbone": {
            "name": extractor.model_name,
            "repo_id": extractor.repo_id,
            "revision": extractor.revision,
            "license": extractor.license,
            "weight_sha256": extractor.weight_sha256,
            "feature_dim": extractor.feature_dim,
            "total_parameters": extractor.total_parameters,
            "trainable_backbone_parameters": extractor.trainable_backbone_parameters,
        },
        "head": {
            "type": "StandardScaler + multinomial logistic regression",
            "C": 0.1,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 5000,
            "fit_seconds": fit_seconds,
            "iterations_by_class": head_iterations,
        },
        "feature_extraction": {
            "device": str(extractor.device),
            "batch_size": batch_size,
            "elapsed_seconds": extraction_seconds,
            "cache_reused": reused,
            "embedding_path": str(embedding_path.relative_to(PROJECT_ROOT)),
            "embedding_sha256": sha256_file(embedding_path),
        },
        "thresholds": thresholds,
        "threshold_calibration_validation_only": threshold_calibration,
        "threshold_test_metrics": threshold_test_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cpu_batch1_benchmark": benchmark,
        "artifact_path": str(artifact_path.relative_to(PROJECT_ROOT)),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_reload_max_abs_probability_difference": reload_max_abs_difference,
        "val_probabilities": val_probabilities,
        "test_probabilities": test_probabilities,
        "val_y": val_y,
        "test_y": test_y,
    }
    del extractor
    return result


def save_split_summary(frame: pd.DataFrame) -> None:
    summary = (
        frame.assign(is_synthetic=frame["is_synthetic"].astype(str).str.casefold().eq("true"))
        .groupby(["label", "split", "source_dataset"], as_index=False)
        .agg(samples=("sample_id", "size"), source_groups=("source_group", "nunique"), synthetic_samples=("is_synthetic", "sum"))
    )
    summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)


def save_prediction_tables(results: Sequence[dict[str, Any]], frame: pd.DataFrame, masks: dict[str, np.ndarray]) -> None:
    test_frame = frame.loc[masks["test"]].reset_index(drop=True)
    prediction_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    for result in results:
        probabilities = result["test_probabilities"]
        predicted = probabilities.argmax(axis=1)
        for row_index, record in test_frame.iterrows():
            flags = {
                label: bool(probabilities[row_index, LABEL_TO_INDEX[label]] >= result["thresholds"][label])
                for label in RESTRICTED_LABELS
            }
            row = {
                "model": result["key"],
                "sample_id": record["sample_id"],
                "true_label": record["label"],
                "predicted_label": LABELS[int(predicted[row_index])],
                "source_group": record["source_group"],
                "source_dataset": record["source_dataset"],
                "threshold_flags": json.dumps(flags, sort_keys=True),
            }
            row.update({f"probability_{label}": float(probabilities[row_index, index]) for index, label in enumerate(LABELS)})
            prediction_rows.append(row)

        for split_key, metrics_key in (("validation", "validation_metrics"), ("test", "test_metrics")):
            for label, metrics in result[metrics_key]["per_class"].items():
                per_class_rows.append({"model": result["key"], "split": split_key, "label": label, **metrics})
        for label in RESTRICTED_LABELS:
            calibration = result["threshold_calibration_validation_only"][label]
            test_metrics = result["threshold_test_metrics"][label]
            threshold_rows.append(
                {
                    "model": result["key"],
                    "label": label,
                    "threshold": result["thresholds"][label],
                    "selection_split": "validation",
                    "target_validation_recall": RECALL_TARGET,
                    "validation_precision": calibration["precision"],
                    "validation_recall": calibration["recall"],
                    "validation_f1": calibration["f1"],
                    "validation_fp": calibration["fp"],
                    "validation_fn": calibration["fn"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                    "test_fp": test_metrics["fp"],
                    "test_fn": test_metrics["fn"],
                }
            )
        summary_rows.append(
            {
                "model": result["key"],
                "role": result["role"],
                "validation_accuracy": result["validation_metrics"]["accuracy"],
                "validation_macro_f1": result["validation_metrics"]["macro_f1"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "test_macro_f1": result["test_metrics"]["macro_f1"],
                "test_macro_average_precision_ovr": result["test_metrics"]["macro_average_precision_ovr"],
                "test_macro_roc_auc_ovr": result["test_metrics"]["macro_roc_auc_ovr"],
                "test_log_loss": result["test_metrics"]["log_loss"],
                "cpu_batch1_median_ms": result["cpu_batch1_benchmark"]["median_ms"],
                "cpu_batch1_p95_ms": result["cpu_batch1_benchmark"]["p95_ms"],
                "feature_extraction_seconds": result["feature_extraction"]["elapsed_seconds"],
                "head_fit_seconds": result["head"]["fit_seconds"],
            }
        )
        benchmark_rows.append({"model": result["key"], **result["cpu_batch1_benchmark"]})

    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUTPUT_DIR / "thresholds.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(benchmark_rows).drop(columns=["latencies_ms"]).to_csv(
        OUTPUT_DIR / "benchmark_cpu_batch1.csv", index=False
    )


def plot_dataset_distribution(frame: pd.DataFrame) -> None:
    counts = frame.groupby(["label", "split"]).size().unstack(fill_value=0).loc[list(LABELS), list(SPLITS)]
    figure, axis = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(LABELS))
    width = 0.24
    colors = ("#2457A6", "#E49B24", "#2A9D73")
    for index, split in enumerate(SPLITS):
        bars = axis.bar(x + (index - 1) * width, counts[split], width, label=split, color=colors[index])
        axis.bar_label(bars, padding=3, fontsize=9)
    axis.set_xticks(x, [label.replace("_", " ").title() for label in LABELS])
    axis.set_ylabel("Images")
    axis.set_title("Source-grouped capstone dataset")
    axis.set_ylim(0, max(counts.max()) * 1.18)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "dataset_distribution.png", dpi=180)
    plt.close(figure)


def make_contact_sheet(frame: pd.DataFrame) -> None:
    tile_width, tile_height = 224, 252
    canvas = Image.new("RGB", (tile_width * 3, tile_height * 4), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    for row_index, label in enumerate(LABELS):
        for column_index, split in enumerate(SPLITS):
            subset = frame[(frame["label"] == label) & (frame["split"] == split)].sort_values("sample_id")
            record = subset.iloc[0]
            with Image.open(PROJECT_ROOT / record["local_path"]) as raw:
                image = raw.convert("RGB")
                image.thumbnail((tile_width - 10, tile_height - 38), Image.Resampling.LANCZOS)
            x = column_index * tile_width
            y = row_index * tile_height
            canvas.paste(image, (x + (tile_width - image.width) // 2, y + 4))
            draw.rectangle((x, y + tile_height - 32, x + tile_width, y + tile_height), fill=(242, 245, 248))
            caption = f"{split} | {label.replace('_', ' ')}"
            draw.text((x + 7, y + tile_height - 26), caption, font=font, fill=(22, 32, 47))
    canvas.save(OUTPUT_DIR / "dataset_contact_sheet.jpg", "JPEG", quality=92, optimize=True)


def plot_confusion(result: dict[str, Any]) -> None:
    matrix = np.asarray(result["test_metrics"]["confusion_matrix"], dtype=int)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black", fontsize=12)
    names = [label.replace("_", "\n") for label in LABELS]
    axis.set_xticks(range(len(LABELS)), names)
    axis.set_yticks(range(len(LABELS)), names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"Untouched test confusion matrix\n{result['display_name']}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"confusion_matrix_{result['key']}.png", dpi=180)
    if result["key"] == "vit":
        figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def plot_precision_recall(results: Sequence[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    colors = ("#4F6D7A", "#B33F62", "#F09D51", "#2A9D73")
    for axis, result in zip(axes, results):
        binary = label_binarize(result["test_y"], classes=np.arange(len(LABELS)))
        for index, label in enumerate(LABELS):
            precision, recall, _ = precision_recall_curve(binary[:, index], result["test_probabilities"][:, index])
            ap = result["test_metrics"]["per_class"][label]["average_precision_ovr"]
            axis.plot(recall, precision, color=colors[index], linewidth=2, label=f"{label} (AP={ap:.3f})")
        axis.set_title(result["display_name"])
        axis.set_xlabel("Recall")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Precision")
    figure.suptitle("One-vs-rest precision-recall curves on untouched test")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "precision_recall_curves.png", dpi=180)
    plt.close(figure)


def plot_threshold_calibration(results: Sequence[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row_index, result in enumerate(results):
        binary = label_binarize(result["val_y"], classes=np.arange(len(LABELS)))
        for column_index, label in enumerate(RESTRICTED_LABELS):
            axis = axes[row_index, column_index]
            label_index = LABEL_TO_INDEX[label]
            scores = result["val_probabilities"][:, label_index]
            thresholds = np.linspace(0.0, 1.0, 201)
            precision_values = []
            recall_values = []
            for threshold in thresholds:
                metrics = binary_metrics(binary[:, label_index], scores, float(threshold))
                precision_values.append(metrics["precision"])
                recall_values.append(metrics["recall"])
            selected = result["thresholds"][label]
            axis.plot(thresholds, recall_values, label="recall", color="#B33F62", linewidth=2)
            axis.plot(thresholds, precision_values, label="precision", color="#2457A6", linewidth=2)
            axis.axhline(RECALL_TARGET, color="#6C757D", linestyle=":", linewidth=1.5)
            axis.axvline(selected, color="#111111", linestyle="--", linewidth=1.5)
            axis.set_title(f"{result['key']} | {label.replace('_', ' ')}\nt={selected:.3f}")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("Threshold")
            if column_index == 0:
                axis.set_ylabel("Validation metric")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    figure.suptitle("Validation-only threshold calibration")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "threshold_calibration.png", dpi=180)
    plt.close(figure)


def plot_model_comparison(results: Sequence[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    metrics = ("accuracy", "macro_f1", "macro_average_precision_ovr")
    labels = ("Accuracy", "Macro F1", "Macro AP")
    x = np.arange(len(metrics))
    width = 0.34
    for index, result in enumerate(results):
        values = [result["test_metrics"][metric] for metric in metrics]
        bars = axes[0].bar(x + (index - 0.5) * width, values, width, label=result["key"])
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.12)
    axes[0].set_title("Untouched test metrics")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    names = [result["key"] for result in results]
    medians = [result["cpu_batch1_benchmark"]["median_ms"] for result in results]
    p95s = [result["cpu_batch1_benchmark"]["p95_ms"] for result in results]
    latency_x = np.arange(len(names))
    axes[1].bar(latency_x - width / 2, medians, width, label="median", color="#2A9D73")
    axes[1].bar(latency_x + width / 2, p95s, width, label="p95", color="#E49B24")
    axes[1].set_xticks(latency_x, names)
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_title("Warm CPU batch-1 latency")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close(figure)


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    omitted = {"val_probabilities", "test_probabilities", "val_y", "test_y"}
    return {key: value for key, value in result.items() if key not in omitted}


def save_canonical_contract(
    results: Sequence[dict[str, Any]],
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Write the stable filenames consumed by the notebook and report builders."""

    primary = next(result for result in results if result["key"] == "vit")
    test_frame = frame.loc[masks["test"]].reset_index(drop=True)
    probabilities = primary["test_probabilities"]
    predicted_indices = probabilities.argmax(axis=1)
    predicted_labels = np.asarray([LABELS[index] for index in predicted_indices], dtype=object)
    true_labels = test_frame["label"].astype(str).to_numpy()

    prediction_columns = [
        "model",
        "model_version",
        "sample_id",
        "local_path",
        "source_group",
        "source_dataset",
        "true_label",
        "predicted_label",
        "correct",
        *[f"probability_{label}" for label in LABELS],
        *[f"threshold_flag_{label}" for label in RESTRICTED_LABELS],
    ]
    rows: list[dict[str, Any]] = []
    for row_index, record in test_frame.iterrows():
        row: dict[str, Any] = {
            "model": primary["key"],
            "model_version": primary["model_version"],
            "sample_id": record["sample_id"],
            "local_path": record["local_path"],
            "source_group": record["source_group"],
            "source_dataset": record["source_dataset"],
            "true_label": true_labels[row_index],
            "predicted_label": predicted_labels[row_index],
            "correct": bool(predicted_labels[row_index] == true_labels[row_index]),
        }
        for label_index, label in enumerate(LABELS):
            row[f"probability_{label}"] = float(probabilities[row_index, label_index])
        for label in RESTRICTED_LABELS:
            row[f"threshold_flag_{label}"] = bool(
                probabilities[row_index, LABEL_TO_INDEX[label]] >= primary["thresholds"][label]
            )
        rows.append(row)
    predictions = pd.DataFrame(rows, columns=prediction_columns)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    predictions.loc[~predictions["correct"]].to_csv(OUTPUT_DIR / "failure_cases.csv", index=False)

    matrix = np.asarray(primary["test_metrics"]["confusion_matrix"], dtype=int)
    matrix_frame = pd.DataFrame(
        matrix,
        index=pd.Index(LABELS, name="true_label"),
        columns=pd.Index(LABELS, name="predicted_label"),
    )
    matrix_frame.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    comparison_columns = [
        "model",
        "role",
        "backbone",
        "backbone_revision",
        "feature_dim",
        "total_backbone_parameters",
        "trainable_backbone_parameters",
        "validation_accuracy",
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_macro_average_precision_ovr",
        "cpu_batch1_median_ms",
        "cpu_batch1_p95_ms",
        "head_fit_seconds",
        "artifact_path",
        "artifact_sha256",
    ]
    comparison_rows = [
        {
            "model": result["key"],
            "role": result["role"],
            "backbone": result["backbone"]["name"],
            "backbone_revision": result["backbone"]["revision"],
            "feature_dim": result["backbone"]["feature_dim"],
            "total_backbone_parameters": result["backbone"]["total_parameters"],
            "trainable_backbone_parameters": result["backbone"]["trainable_backbone_parameters"],
            "validation_accuracy": result["validation_metrics"]["accuracy"],
            "validation_macro_f1": result["validation_metrics"]["macro_f1"],
            "test_accuracy": result["test_metrics"]["accuracy"],
            "test_macro_f1": result["test_metrics"]["macro_f1"],
            "test_macro_average_precision_ovr": result["test_metrics"]["macro_average_precision_ovr"],
            "cpu_batch1_median_ms": result["cpu_batch1_benchmark"]["median_ms"],
            "cpu_batch1_p95_ms": result["cpu_batch1_benchmark"]["p95_ms"],
            "head_fit_seconds": result["head"]["fit_seconds"],
            "artifact_path": result["artifact_path"],
            "artifact_sha256": result["artifact_sha256"],
        }
        for result in results
    ]
    pd.DataFrame(comparison_rows, columns=comparison_columns).to_csv(
        OUTPUT_DIR / "model_comparison.csv", index=False
    )

    save_json(
        OUTPUT_DIR / "latency.json",
        {
            "schema_version": "1.0.0",
            "model": primary["key"],
            "model_version": primary["model_version"],
            "hardware": hardware_name(),
            "device": "cpu",
            "protocol": primary["cpu_batch1_benchmark"]["scope"],
            "batch_size": primary["cpu_batch1_benchmark"]["batch_size"],
            "warmup_runs": primary["cpu_batch1_benchmark"]["warmup_runs"],
            "measured_runs": primary["cpu_batch1_benchmark"]["measured_runs"],
            "torch_cpu_threads": primary["cpu_batch1_benchmark"]["torch_cpu_threads"],
            "classifier_mean_ms": primary["cpu_batch1_benchmark"]["mean_ms"],
            "classifier_median_ms": primary["cpu_batch1_benchmark"]["median_ms"],
            "classifier_p95_ms": primary["cpu_batch1_benchmark"]["p95_ms"],
            "classifier_min_ms": primary["cpu_batch1_benchmark"]["min_ms"],
            "classifier_max_ms": primary["cpu_batch1_benchmark"]["max_ms"],
            "exclusions": [
                "file decoding",
                "OCR",
                "object detection",
                "network transfer",
                "application rendering",
            ],
        },
    )

    restricted_recall = float(
        np.mean([primary["test_metrics"]["per_class"][label]["recall"] for label in RESTRICTED_LABELS])
    )
    safe_false_positive_rate = float(1.0 - primary["test_metrics"]["per_class"]["safe"]["recall"])
    canonical_metrics = {
        "schema_version": "1.0.0",
        "model": primary["key"],
        "model_version": primary["model_version"],
        "split": "untouched_test",
        "samples": int(masks["test"].sum()),
        "accuracy": primary["test_metrics"]["accuracy"],
        "balanced_accuracy": primary["test_metrics"]["balanced_accuracy"],
        "macro_precision": primary["test_metrics"]["macro_precision"],
        "macro_recall": primary["test_metrics"]["macro_recall"],
        "macro_f1": primary["test_metrics"]["macro_f1"],
        "weighted_f1": primary["test_metrics"]["weighted_f1"],
        "macro_average_precision_ovr": primary["test_metrics"]["macro_average_precision_ovr"],
        "macro_roc_auc_ovr": primary["test_metrics"]["macro_roc_auc_ovr"],
        "log_loss": primary["test_metrics"]["log_loss"],
        "multiclass_brier": primary["test_metrics"]["multiclass_brier"],
        "restricted_recall": restricted_recall,
        "safe_false_positive_rate": safe_false_positive_rate,
        "group_bootstrap_95pct": primary["test_metrics"]["group_bootstrap_95pct"],
        "threshold_operating_points": primary["threshold_test_metrics"],
        "thresholds_selected_on_validation_only": primary["thresholds"],
    }
    save_json(OUTPUT_DIR / "metrics.json", canonical_metrics)
    return canonical_metrics


def checksum_entries(model_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded = {"output_checksums.csv", "evaluation_manifest.json"}
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file() and path.name not in excluded:
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for result in model_results:
        path = PROJECT_ROOT / result["artifact_path"]
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_output_checksums(model_results: Sequence[dict[str, Any]]) -> None:
    rows = checksum_entries(model_results)
    manifest_path = OUTPUT_DIR / "evaluation_manifest.json"
    rows.append(
        {
            "path": str(manifest_path.relative_to(PROJECT_ROOT)),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "output_checksums.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-embeddings", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    frame = pd.read_csv(REGISTRY_PATH, dtype=str, keep_default_na=False)
    registry_validation = validate_registry(frame)
    dataset_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    y = frame["label"].map(LABEL_TO_INDEX).to_numpy(dtype=int)
    masks = {split: frame["split"].eq(split).to_numpy() for split in SPLITS}
    print(json.dumps(registry_validation, indent=2, sort_keys=True), flush=True)

    save_split_summary(frame)
    plot_dataset_distribution(frame)
    make_contact_sheet(frame)

    benchmark_images: list[Image.Image] = []
    benchmark_frame = frame.loc[masks["val"]].sort_values("sample_id").head(20)
    for path in benchmark_frame["local_path"]:
        with Image.open(PROJECT_ROOT / path) as raw:
            benchmark_images.append(raw.convert("RGB"))

    requested_device = str(select_device(args.device))
    results: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        results.append(
            train_one_model(
                spec,
                frame,
                y,
                masks,
                registry_validation,
                requested_device,
                args.batch_size,
                args.reuse_embeddings,
                benchmark_images,
            )
        )

    save_prediction_tables(results, frame, masks)
    for result in results:
        plot_confusion(result)
    plot_precision_recall(results)
    plot_threshold_calibration(results)
    plot_model_comparison(results)
    canonical_metrics = save_canonical_contract(results, frame, masks)

    evaluation_payload = {
        "run": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "hardware": hardware_name(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": package_versions(),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "requested_feature_device": args.device,
            "actual_feature_device": requested_device,
            "cpu_threads": int(torch.get_num_threads()),
        },
        "protocol": {
            "primary_model_predeclared": "vit",
            "baseline_predeclared": "resnet50",
            "fit_split": "train only",
            "threshold_selection_split": "validation only",
            "threshold_rule": (
                f"restricted-class recall >= {RECALL_TARGET:.2f}; then maximize validation precision, F1, and threshold"
            ),
            "test_use": "single post-fit, post-calibration evaluation; never used for fitting or threshold selection",
            "frozen_backbones": True,
        },
        "dataset": registry_validation,
        "dataset_manifest": dataset_manifest,
        "models": [public_result(result) for result in results],
        "limitations": [
            "Every policy class is tied to one source or generator, so source-style confounding can inflate metrics.",
            "The untouched test set has only 12 images per class and cannot establish production performance.",
            "Validation has only 12 positives per class; meeting the recall target is not a population-level recall guarantee.",
            "ADautoGen safe images and all financial creatives are synthetic, while weapons_set1 is not declared synthetic.",
            "weapons_set1 lacks per-image provenance and rights; repository-level OSL-3.0 metadata is not enough for production reuse.",
            "The weapons images are object-centric, while real ads may contain tiny, occluded, stylized, or contextual objects.",
            "The financial creatives are rendered by one code path and do not cover real brands, layouts, languages, or jurisdictions.",
            "The four-way label is single-class and omits mixed-policy ads, alcohol, tobacco, nudity, violence, and other proposal categories.",
            "A classifier score is visual evidence, not proof that an ad violates law or platform policy.",
            "Latency excludes file decoding, OCR, object detection, network transfer, and application rendering.",
        ],
    }
    save_json(OUTPUT_DIR / "evaluation_metrics.json", evaluation_payload)

    validation_payload = {
        "all_checks_passed": True,
        "registry_rows": len(frame),
        "registry_hashes_recomputed": len(frame),
        "group_overlap_across_splits": registry_validation["group_overlap_across_splits"],
        "exact_hash_overlap_across_splits": registry_validation["exact_hash_overlap_across_splits"],
        "test_rows": int(masks["test"].sum()),
        "test_rows_per_class": frame.loc[masks["test"], "label"].value_counts().sort_index().to_dict(),
        "artifacts": {
            result["key"]: {
                "path": result["artifact_path"],
                "sha256": result["artifact_sha256"],
                "reload_max_abs_probability_difference": result[
                    "artifact_reload_max_abs_probability_difference"
                ],
                "trainable_backbone_parameters": result["backbone"]["trainable_backbone_parameters"],
            }
            for result in results
        },
    }
    save_json(OUTPUT_DIR / "independent_validation.json", validation_payload)

    evaluation_manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": evaluation_payload["run"]["created_at_utc"],
        "provenance": {
            "registry_path": str(REGISTRY_PATH.relative_to(PROJECT_ROOT)),
            "registry_sha256": registry_validation["registry_sha256"],
            "dataset_manifest_path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "dataset_manifest_sha256": sha256_file(MANIFEST_PATH),
            "dataset_sources": dataset_manifest["sources"],
            "test_protocol": evaluation_payload["protocol"],
            "models": [
                {
                    "model": result["key"],
                    "role": result["role"],
                    "backbone_repo_id": result["backbone"]["repo_id"],
                    "backbone_revision": result["backbone"]["revision"],
                    "backbone_weight_sha256": result["backbone"]["weight_sha256"],
                    "artifact_path": result["artifact_path"],
                    "artifact_sha256": result["artifact_sha256"],
                }
                for result in results
            ],
        },
        "canonical_metrics": canonical_metrics,
        "validation_result": validation_payload,
        "artifact_checksums_excluding_this_manifest": checksum_entries(results),
        "canonical_schemas": {
            "predictions.csv": [
                "model",
                "model_version",
                "sample_id",
                "local_path",
                "source_group",
                "source_dataset",
                "true_label",
                "predicted_label",
                "correct",
                *[f"probability_{label}" for label in LABELS],
                *[f"threshold_flag_{label}" for label in RESTRICTED_LABELS],
            ],
            "failure_cases.csv": "same columns as predictions.csv; rows where correct is false",
            "confusion_matrix.csv": {
                "row_index": "true_label",
                "columns": list(LABELS),
                "values": "integer counts",
            },
            "model_comparison.csv": list(pd.read_csv(OUTPUT_DIR / "model_comparison.csv", nrows=0).columns),
        },
    }
    save_json(OUTPUT_DIR / "evaluation_manifest.json", evaluation_manifest)
    write_output_checksums(results)

    concise = {
        result["key"]: {
            "test_accuracy": result["test_metrics"]["accuracy"],
            "test_macro_f1": result["test_metrics"]["macro_f1"],
            "test_macro_ap": result["test_metrics"]["macro_average_precision_ovr"],
            "cpu_batch1_median_ms": result["cpu_batch1_benchmark"]["median_ms"],
            "cpu_batch1_p95_ms": result["cpu_batch1_benchmark"]["p95_ms"],
            "thresholds": result["thresholds"],
        }
        for result in results
    }
    print(json.dumps(concise, indent=2, sort_keys=True), flush=True)
    print(f"Evaluation outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
