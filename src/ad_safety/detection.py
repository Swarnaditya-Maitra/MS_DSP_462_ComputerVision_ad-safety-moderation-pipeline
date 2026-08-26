from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .features import select_device
from .paths import MODEL_DIR, PROJECT_ROOT


GROUNDING_DINO_REPO = "IDEA-Research/grounding-dino-tiny"
DEFAULT_PROMPT = (
    "handgun. pistol. rifle. shotgun. ammunition. grenade. bomb. explosive device. "
    "fireworks."
)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundingDinoDetector:
    """Open-vocabulary detector used for visual evidence, not direct policy truth."""

    def __init__(self, device: str = "cpu") -> None:
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
        manifest_path = MODEL_DIR / "pretrained_model_manifest.json"
        revision = None
        local_snapshot: Path | None = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row = next(
                (item for item in manifest.get("models", []) if item.get("repo_id") == GROUNDING_DINO_REPO),
                None,
            )
            if row:
                revision = row.get("revision")
                local_snapshot = PROJECT_ROOT / str(row.get("local_snapshot", ""))
        cache_dir = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
        self.device = select_device(device)
        source = str(local_snapshot) if local_snapshot and local_snapshot.is_dir() else GROUNDING_DINO_REPO
        self.processor = AutoProcessor.from_pretrained(
            source,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            source,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        ).eval().to(self.device)
        self.revision = revision or "unpinned"

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> tuple[list[Detection], float]:
        started = perf_counter()
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
        if self.device.type == "mps":
            torch.mps.synchronize()
        detections = [
            Detection(
                label=str(label).strip().casefold(),
                score=float(score.detach().cpu()),
                box=tuple(float(value) for value in box.detach().cpu().tolist()),
            )
            for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"])
        ]
        return detections, (perf_counter() - started) * 1000.0
