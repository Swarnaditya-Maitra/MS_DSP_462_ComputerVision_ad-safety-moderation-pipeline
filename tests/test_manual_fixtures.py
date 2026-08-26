from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from scripts.create_manual_test_fixtures import MAX_UPLOAD_BYTES, create_fixtures


def test_create_manual_fixtures_matches_documented_contract(tmp_path: Path) -> None:
    paths = create_fixtures(tmp_path)

    assert [path.name for path in paths] == [
        "valid_static.png",
        "corrupt.png",
        "tiny.png",
        "animated.webp",
        "oversized.png",
    ]
    with Image.open(tmp_path / "valid_static.png") as image:
        assert image.size == (256, 256)
        assert image.mode == "RGB"
        assert getattr(image, "n_frames", 1) == 1
    with Image.open(tmp_path / "tiny.png") as image:
        assert image.size == (16, 16)
    with Image.open(tmp_path / "animated.webp") as image:
        assert image.n_frames == 2
    with pytest.raises(UnidentifiedImageError):
        Image.open(tmp_path / "corrupt.png").verify()
    assert (tmp_path / "oversized.png").stat().st_size > MAX_UPLOAD_BYTES
