#!/usr/bin/env python3
"""Create deterministic invalid-image fixtures for the user-manual checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tmp/manual_test_inputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corrupt = output_dir / "corrupt.png"
    corrupt.write_bytes(b"this is not an image")

    tiny = output_dir / "tiny.png"
    Image.new("RGB", (16, 16), "white").save(tiny, format="PNG")

    animated = output_dir / "animated.webp"
    frames = [Image.new("RGB", (64, 64), color) for color in ("red", "blue")]
    frames[0].save(
        animated,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

    oversized = output_dir / "oversized.png"
    with oversized.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(b"0" * (MAX_UPLOAD_BYTES + 1))

    for path in (corrupt, tiny, animated, oversized):
        print(f"{path.as_posix()}\t{path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
