from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import joblib
import numpy as np
from PIL import Image

from .features import BackboneFeatureExtractor


@dataclass(frozen=True)
class ClassificationResult:
    scores: dict[str, float]
    latency_ms: float
    device: str
    model_version: str


class PolicyClassifier:
    """Frozen visual backbone plus a trained one-vs-rest policy head."""

    def __init__(self, artifact_path: str | Path, device: str = "auto") -> None:
        self.artifact_path = Path(artifact_path)
        artifact: dict[str, Any] = joblib.load(self.artifact_path)
        self.artifact = artifact
        self.class_names = list(artifact["class_names"])
        self.pipeline = artifact["pipeline"]
        self.extractor = BackboneFeatureExtractor(artifact["backbone_name"], device=device)
        self.model_version = str(artifact.get("model_version", self.artifact_path.stem))

    def predict(self, image: Image.Image) -> ClassificationResult:
        started = perf_counter()
        embedding = self.extractor.embed_images([image], batch_size=1)
        probabilities = self.pipeline.predict_proba(embedding.features)[0]
        elapsed_ms = (perf_counter() - started) * 1000.0
        return ClassificationResult(
            scores={name: float(probabilities[index]) for index, name in enumerate(self.class_names)},
            latency_ms=elapsed_ms,
            device=embedding.device,
            model_version=self.model_version,
        )

    def predict_many(self, images: Sequence[Image.Image], batch_size: int = 8) -> np.ndarray:
        embedding = self.extractor.embed_images(images, batch_size=batch_size)
        return np.asarray(self.pipeline.predict_proba(embedding.features), dtype=np.float32)

    def occlusion_heatmap(
        self,
        image: Image.Image,
        target_label: str,
        grid_size: int = 4,
        batch_size: int = 8,
    ) -> tuple[np.ndarray, float]:
        """Return a model-agnostic patch-occlusion sensitivity map."""

        if target_label not in self.class_names:
            raise ValueError(f"Unknown target label: {target_label}")
        target_index = self.class_names.index(target_label)
        base_score = self.predict(image).scores[target_label]
        width, height = image.size
        fill = tuple(int(value) for value in np.asarray(image).reshape(-1, 3).mean(axis=0))
        variants: list[Image.Image] = []
        for row in range(grid_size):
            for col in range(grid_size):
                masked = image.copy()
                x0 = int(col * width / grid_size)
                y0 = int(row * height / grid_size)
                x1 = int((col + 1) * width / grid_size)
                y1 = int((row + 1) * height / grid_size)
                patch = Image.new("RGB", (max(1, x1 - x0), max(1, y1 - y0)), fill)
                masked.paste(patch, (x0, y0))
                variants.append(masked)
        probabilities = self.predict_many(variants, batch_size=batch_size)[:, target_index]
        drops = np.maximum(0.0, base_score - probabilities).reshape(grid_size, grid_size)
        if float(drops.max()) > 0:
            drops = drops / float(drops.max())
        return drops.astype(np.float32), float(base_score)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.48) -> Image.Image:
    """Overlay a red-yellow sensitivity map without changing source dimensions."""

    normalized = np.clip(heatmap, 0.0, 1.0)
    small = Image.fromarray(np.uint8(normalized * 255), mode="L")
    resized = np.asarray(small.resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    red = np.full_like(resized, 255.0)
    green = 210.0 * (1.0 - resized)
    blue = 45.0 * (1.0 - resized)
    color = Image.fromarray(np.uint8(np.stack([red, green, blue], axis=-1)), mode="RGB")
    mask = Image.fromarray(np.uint8(resized * alpha * 255), mode="L")
    return Image.composite(color, image.convert("RGB"), mask)
