from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import timm
import torch
from PIL import Image

from .paths import PROJECT_ROOT


VIT_MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
CNN_MODEL_NAME = "resnet50.a1_in1k"
PINNED_BACKBONE_REPOS = {
    VIT_MODEL_NAME: f"timm/{VIT_MODEL_NAME}",
    CNN_MODEL_NAME: f"timm/{CNN_MODEL_NAME}",
}


class BackboneArtifactError(RuntimeError):
    """Raised when an offline backbone artifact is missing or inconsistent."""


def resolve_local_pretrained_file(model_name: str) -> Path:
    """Resolve a known backbone to its exact pinned local safetensors file."""

    repo_id = PINNED_BACKBONE_REPOS.get(model_name)
    if repo_id is None:
        supported = ", ".join(sorted(PINNED_BACKBONE_REPOS))
        raise BackboneArtifactError(
            f"Backbone '{model_name}' is not pinned for offline use. Supported backbones: {supported}."
        )

    manifest_path = PROJECT_ROOT / "models" / "pretrained_model_manifest.json"
    if not manifest_path.is_file():
        raise BackboneArtifactError(f"Pinned model manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackboneArtifactError(f"Pinned model manifest is not readable: {manifest_path}") from exc

    record = next((row for row in manifest.get("models", []) if row.get("repo_id") == repo_id), None)
    if record is None:
        raise BackboneArtifactError(f"Manifest has no pinned record for {repo_id}.")
    if not record.get("revision"):
        raise BackboneArtifactError(f"Manifest record for {repo_id} has no immutable revision.")

    snapshot = Path(str(record.get("local_snapshot", "")))
    if not snapshot.is_absolute():
        snapshot = PROJECT_ROOT / snapshot
    weights = snapshot / "model.safetensors"
    if not weights.is_file():
        raise BackboneArtifactError(
            f"Pinned weights for {repo_id} are missing: {weights}. Run scripts/download_models.py first."
        )
    # Preserve the manifest-facing snapshot path instead of resolving the
    # Hugging Face cache symlink to an opaque blob filename.
    return weights.absolute()


def select_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class EmbeddingBatch:
    features: np.ndarray
    elapsed_seconds: float
    device: str


class BackboneFeatureExtractor:
    """Frozen timm backbone used for transfer-learning feature extraction."""

    def __init__(self, model_name: str, device: str = "auto", allow_remote: bool = False) -> None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
        self.model_name = model_name
        self.device = select_device(device)
        if model_name in PINNED_BACKBONE_REPOS:
            weights = resolve_local_pretrained_file(model_name)
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                pretrained_cfg_overlay={"file": str(weights)},
                num_classes=0,
            )
        elif allow_remote:
            # Remote loading is opt-in so production and tests stay reproducible
            # and do not silently depend on network access.
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        else:
            supported = ", ".join(sorted(PINNED_BACKBONE_REPOS))
            raise BackboneArtifactError(
                f"Backbone '{model_name}' is not pinned. Use one of {supported}, or set allow_remote=True explicitly."
            )
        self.model.eval().to(self.device)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.feature_dim = int(getattr(self.model, "num_features"))

    @torch.inference_mode()
    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 8) -> EmbeddingBatch:
        if not images:
            return EmbeddingBatch(np.empty((0, self.feature_dim), dtype=np.float32), 0.0, str(self.device))
        started = perf_counter()
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            tensors = [self.transform(image.convert("RGB")) for image in images[start : start + batch_size]]
            batch = torch.stack(tensors).to(self.device)
            features = self.model(batch)
            outputs.append(features.detach().float().cpu().numpy())
        if self.device.type == "mps":
            torch.mps.synchronize()
        elapsed = perf_counter() - started
        return EmbeddingBatch(np.concatenate(outputs, axis=0), elapsed, str(self.device))

    def embed_paths(self, paths: Iterable[str | Path], batch_size: int = 8) -> EmbeddingBatch:
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as raw:
                images.append(raw.convert("RGB"))
        return self.embed_images(images, batch_size=batch_size)


def color_histogram_features(images: Sequence[Image.Image], bins: int = 16) -> np.ndarray:
    """Low-capacity visual baseline that captures color distribution only."""

    rows: list[np.ndarray] = []
    for image in images:
        arr = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.uint8)
        parts = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0, 256), density=True)
            parts.append(hist.astype(np.float32))
        rows.append(np.concatenate(parts))
    return np.vstack(rows)
