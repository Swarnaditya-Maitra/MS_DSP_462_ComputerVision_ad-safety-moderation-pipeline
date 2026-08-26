from __future__ import annotations

import io

import pytest
from PIL import Image, PngImagePlugin

from ad_safety import preprocessing


def _encoded(image: Image.Image, image_format: str = "PNG", **save_kwargs: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **save_kwargs)
    return buffer.getvalue()


def test_transparent_pixels_are_composited_on_white() -> None:
    image = Image.new("RGBA", (40, 40), (255, 0, 0, 0))
    image.putpixel((20, 20), (12, 34, 56, 255))

    loaded = preprocessing.load_image(_encoded(image))

    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((20, 20)) == (12, 34, 56)


def test_exif_orientation_is_applied() -> None:
    image = Image.new("RGB", (80, 40), "navy")
    exif = Image.Exif()
    exif[274] = 6

    loaded = preprocessing.load_image(_encoded(image, "JPEG", exif=exif))

    assert loaded.size == (40, 80)


def test_letterbox_preserves_aspect_ratio_and_centers_content() -> None:
    image = Image.new("RGB", (100, 50), (220, 20, 60))

    boxed = preprocessing.letterbox(image, size=200, fill=(1, 2, 3))

    assert boxed.size == (200, 200)
    assert boxed.getpixel((100, 20)) == (1, 2, 3)
    assert boxed.getpixel((100, 100)) == (220, 20, 60)
    assert boxed.getpixel((100, 180)) == (1, 2, 3)


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\ntruncated"])
def test_corrupt_or_empty_input_is_rejected(payload: bytes) -> None:
    with pytest.raises(preprocessing.ImageValidationError):
        preprocessing.load_image(payload)


def test_oversized_stream_is_bounded_before_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preprocessing, "MAX_IMAGE_BYTES", 8)

    class TrackingStream(io.BytesIO):
        requested: int | None = None

        def read(self, size: int = -1) -> bytes:
            self.requested = size
            return super().read(size)

    stream = TrackingStream(b"123456789")

    with pytest.raises(preprocessing.ImageValidationError, match="20 MB"):
        preprocessing.load_image(stream)
    assert stream.requested == 9


def test_pixel_limit_is_checked_before_pixel_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _encoded(Image.new("RGB", (40, 40), "white"))
    monkeypatch.setattr(preprocessing, "MAX_PIXELS", 1_000)

    def fail_if_decoded(self: PngImagePlugin.PngImageFile, *args: object, **kwargs: object) -> None:
        raise AssertionError("pixel decoder was called before the metadata safety check")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", fail_if_decoded)

    with pytest.raises(preprocessing.ImageValidationError, match="megapixel"):
        preprocessing.load_image(payload)


def test_small_dimensions_are_rejected() -> None:
    payload = _encoded(Image.new("RGB", (31, 80), "white"))

    with pytest.raises(preprocessing.ImageValidationError, match="at least 32 x 32"):
        preprocessing.load_image(payload)


def test_unsupported_image_format_is_rejected() -> None:
    payload = _encoded(Image.new("RGB", (64, 64), "white"), "GIF")

    with pytest.raises(preprocessing.ImageValidationError, match="Unsupported image format"):
        preprocessing.load_image(payload)


@pytest.mark.parametrize("image_format", ["PNG", "WEBP"])
def test_animated_supported_format_is_rejected(image_format: str) -> None:
    first = Image.new("RGB", (64, 64), "white")
    second = Image.new("RGB", (64, 64), "red")
    payload = _encoded(
        first,
        image_format,
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )

    with pytest.raises(preprocessing.ImageValidationError, match="Animated images"):
        preprocessing.load_image(payload)
