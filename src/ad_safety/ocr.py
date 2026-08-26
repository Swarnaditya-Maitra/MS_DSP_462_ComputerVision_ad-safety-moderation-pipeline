from __future__ import annotations

import csv
import io
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OCRResult:
    text: str
    tokens: tuple[OCRToken, ...]
    latency_ms: float
    engine: str


def run_tesseract(image: Image.Image, minimum_confidence: float = 35.0) -> OCRResult:
    executable = shutil.which("tesseract")
    if executable is None:
        return OCRResult("", (), 0.0, "unavailable")
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="ad-safety-ocr-") as temp_dir:
        image_path = Path(temp_dir) / "input.png"
        image.convert("RGB").save(image_path)
        completed = subprocess.run(
            [executable, str(image_path), "stdout", "--psm", "11", "tsv"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    reader = csv.DictReader(io.StringIO(completed.stdout), delimiter="\t")
    tokens: list[OCRToken] = []
    for row in reader:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf", -1))
            left = int(row.get("left", 0))
            top = int(row.get("top", 0))
            width = int(row.get("width", 0))
            height = int(row.get("height", 0))
        except (TypeError, ValueError):
            continue
        if confidence < minimum_confidence:
            continue
        tokens.append(OCRToken(text, confidence, (left, top, left + width, top + height)))
    return OCRResult(
        text=" ".join(token.text for token in tokens),
        tokens=tuple(tokens),
        latency_ms=(perf_counter() - started) * 1000.0,
        engine="tesseract",
    )
