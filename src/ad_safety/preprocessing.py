from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be moderated safely."""


def _read_bytes(source: str | Path | bytes | BinaryIO | Image.Image) -> bytes | None:
    if isinstance(source, Image.Image):
        return None
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ImageValidationError(f"Image does not exist: {path}")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ImageValidationError("Image exceeds the 20 MB upload limit.")
        return path.read_bytes()
    # Bound the read itself so an oversized stream is rejected without first
    # copying an arbitrarily large upload into memory.
    data = source.read(MAX_IMAGE_BYTES + 1)
    if not isinstance(data, bytes):
        raise ImageValidationError("Uploaded object did not return bytes.")
    return data


def _validate_image_metadata(image: Image.Image) -> None:
    if image.format and image.format.upper() not in SUPPORTED_FORMATS:
        raise ImageValidationError(f"Unsupported image format: {image.format}")
    width, height = image.size
    if width < 32 or height < 32:
        raise ImageValidationError("Image dimensions must be at least 32 x 32 pixels.")
    if width * height > MAX_PIXELS:
        raise ImageValidationError("Image exceeds the 40 megapixel safety limit.")


def _reject_animation(image: Image.Image) -> None:
    if bool(getattr(image, "is_animated", False)) or int(getattr(image, "n_frames", 1)) > 1:
        raise ImageValidationError(
            "Animated images are unsupported. Upload one static JPEG, PNG, or WebP image."
        )


def load_image(source: str | Path | bytes | BinaryIO | Image.Image) -> Image.Image:
    """Load, orient, validate, and convert an input to an RGB PIL image."""

    try:
        if isinstance(source, Image.Image):
            _reject_animation(source)
            image = source.copy()
        else:
            data = _read_bytes(source)
            if data is None or len(data) == 0:
                raise ImageValidationError("The uploaded file is empty.")
            if len(data) > MAX_IMAGE_BYTES:
                raise ImageValidationError("Image exceeds the 20 MB upload limit.")
            image = Image.open(io.BytesIO(data))
            # Pillow exposes dimensions from the image header. Check them before
            # decoding pixel data to limit decompression-bomb memory pressure.
            _reject_animation(image)
            _validate_image_metadata(image)
            image.load()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("Image exceeds the 40 megapixel safety limit.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, ImageValidationError):
            raise
        raise ImageValidationError("The file is not a readable JPEG, PNG, or WebP image.") from exc

    _validate_image_metadata(image)

    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(canvas, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def letterbox(image: Image.Image, size: int = 640, fill: tuple[int, int, int] = (246, 248, 252)) -> Image.Image:
    """Resize without distorting the ad and center it on a square canvas."""

    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas
