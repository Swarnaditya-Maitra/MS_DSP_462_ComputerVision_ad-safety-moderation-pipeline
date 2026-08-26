#!/usr/bin/env python3
"""Build a small, licensed Wikimedia Commons pilot benchmark.

The benchmark is intentionally bounded. Search queries are split by concept so
the held-out test set uses different wording and visual sources than training.
Every row records its source page, license, query, split, and file hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "wikimedia_pilot"
REGISTRY_PATH = PROJECT_ROOT / "data" / "dataset_registry.csv"
CONTACT_SHEET_PATH = PROJECT_ROOT / "outputs" / "eda" / "wikimedia_diagnostic_contact_sheet.jpg"
REVIEW_MANIFEST_PATH = PROJECT_ROOT / "data" / "wikimedia_external_manifest.csv"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AdSafetyCapstone/1.0 (educational computer-vision project)"


@dataclass(frozen=True)
class QuerySpec:
    label: str
    split: str
    query: str
    target: int


QUERY_SPECS = (
    QuerySpec("safe", "train", "food advertisement", 8),
    QuerySpec("safe", "train", "shoe advertisement", 8),
    QuerySpec("safe", "val", "travel advertisement", 6),
    QuerySpec("safe", "test", "automobile advertisement", 6),
    QuerySpec("firearms", "train", "pistol advertisement", 8),
    QuerySpec("firearms", "train", "rifle advertisement", 8),
    QuerySpec("firearms", "val", "firearm display", 6),
    QuerySpec("firearms", "test", "revolver photograph", 6),
    QuerySpec("explosives", "train", "fireworks advertisement", 8),
    QuerySpec("explosives", "train", "grenade photograph", 8),
    QuerySpec("explosives", "val", "explosive ordnance museum", 6),
    QuerySpec("explosives", "test", "bomb disposal photograph", 6),
    QuerySpec("financial_promotion", "train", "stock broker advertisement", 8),
    QuerySpec("financial_promotion", "train", "investment advertisement", 8),
    QuerySpec("financial_promotion", "val", "Bitcoin advertisement", 6),
    QuerySpec("financial_promotion", "test", "cryptocurrency promotion", 6),
)


def clean_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(text).split())


def ext_value(metadata: dict[str, Any], key: str) -> str:
    raw = metadata.get(key, {})
    return clean_html(str(raw.get("value", ""))) if isinstance(raw, dict) else ""


def request_with_backoff(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 45,
    attempts: int = 6,
) -> requests.Response:
    delay = 2.0
    for attempt in range(attempts):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        if attempt == attempts - 1:
            response.raise_for_status()
        retry_after = float(response.headers.get("Retry-After", delay))
        time.sleep(max(delay, retry_after))
        delay = min(delay * 2.0, 30.0)
    raise RuntimeError("Unreachable retry loop")


def search_commons(session: requests.Session, spec: QuerySpec, limit: int = 100) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": spec.query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|sha1|extmetadata",
        "iiurlwidth": 768,
        "format": "json",
        "formatversion": 2,
        "origin": "*",
    }
    response = request_with_backoff(session, API_URL, params=params, timeout=45)
    return list(response.json().get("query", {}).get("pages", []))


def normalize_image(content: bytes) -> tuple[Image.Image, str]:
    raw_sha = hashlib.sha256(content).hexdigest()
    image = Image.open(io.BytesIO(content))
    image.load()
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < 160 or image.width * image.height > 50_000_000:
        raise ValueError("Image dimensions outside the pilot acceptance range")
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    return image, raw_sha


def build_dataset(refresh: bool = False) -> list[dict[str, Any]]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows: list[dict[str, Any]] = []
    seen_source_sha1: set[str] = set()

    for spec in QUERY_SPECS:
        pages = search_commons(session, spec)
        accepted = 0
        for page in pages:
            if accepted >= spec.target:
                break
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            mime = str(info.get("mime", ""))
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            source_sha1 = str(info.get("sha1", ""))
            if not source_sha1 or source_sha1 in seen_source_sha1:
                continue
            source_url = str(info.get("thumburl") or info.get("url") or "")
            if not source_url:
                continue
            try:
                response = request_with_backoff(session, source_url, timeout=60)
                image, download_sha256 = normalize_image(response.content)
            except (requests.RequestException, UnidentifiedImageError, OSError, ValueError):
                continue

            target_dir = DATA_ROOT / spec.split / spec.label
            target_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{spec.label}_{spec.split}_{source_sha1[:12]}.jpg"
            target_path = target_dir / file_name
            if refresh or not target_path.exists():
                image.save(target_path, "JPEG", quality=91, optimize=True)
            local_sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
            metadata = info.get("extmetadata") or {}
            rows.append(
                {
                    "label": spec.label,
                    "split": spec.split,
                    "query": spec.query,
                    "local_path": str(target_path.relative_to(PROJECT_ROOT)),
                    "file_name": file_name,
                    "source_title": str(page.get("title", "")),
                    "source_page_url": str(info.get("descriptionurl", "")),
                    "download_url": source_url,
                    "source_sha1": source_sha1,
                    "download_sha256": download_sha256,
                    "local_sha256": local_sha256,
                    "license": ext_value(metadata, "LicenseShortName") or ext_value(metadata, "UsageTerms"),
                    "artist": ext_value(metadata, "Artist"),
                    "credit": ext_value(metadata, "Credit"),
                    "mime": mime,
                    "width": image.width,
                    "height": image.height,
                    "audit_status": "pending_visual_review",
                }
            )
            seen_source_sha1.add(source_sha1)
            accepted += 1
            time.sleep(0.18)
        print(f"{spec.label:22s} {spec.split:5s}: {accepted:2d}/{spec.target:2d} from {spec.query}")
        time.sleep(0.65)

    fieldnames = list(rows[0].keys()) if rows else []
    with REGISTRY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_contact_sheet(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 180, 150
    columns = 8
    display_rows = sorted(rows, key=lambda row: (row["split"], row["label"], row["file_name"]))
    canvas = Image.new("RGB", (columns * tile_w, ((len(display_rows) + columns - 1) // columns) * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, row in enumerate(display_rows):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        with Image.open(PROJECT_ROOT / row["local_path"]) as raw:
            image = raw.convert("RGB")
            image.thumbnail((tile_w - 8, tile_h - 32), Image.Resampling.LANCZOS)
        px = x + (tile_w - image.width) // 2
        py = y + 3
        canvas.paste(image, (px, py))
        caption = f"{row['split']} | {row['label']}"
        draw.rectangle((x, y + tile_h - 27, x + tile_w, y + tile_h), fill=(245, 247, 250))
        draw.text((x + 5, y + tile_h - 22), caption[:28], fill=(20, 30, 45), font=font)
    canvas.save(CONTACT_SHEET_PATH, "JPEG", quality=90)


def verify_existing_dataset() -> dict[str, Any]:
    """Verify the retained, manually reviewed Commons diagnostic files."""

    with REGISTRY_PATH.open(encoding="utf-8", newline="") as handle:
        registry = list(csv.DictReader(handle))
    with REVIEW_MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))

    review_by_path = {row["local_path"]: row for row in review_rows}
    if set(review_by_path) != {row["local_path"] for row in registry}:
        raise RuntimeError("Registry and manual-review manifest paths do not match")

    active = 0
    excluded = 0
    optional_excluded_present = 0
    hashes: set[str] = set()
    for row in registry:
        path = PROJECT_ROOT / row["local_path"]
        include = review_by_path[row["local_path"]]["include_in_spot_check"].casefold() == "true"
        if not include:
            excluded += 1
            if not path.is_file():
                continue
            optional_excluded_present += 1
        elif not path.is_file():
            raise FileNotFoundError(f"Missing retained Wikimedia image: {row['local_path']}")
        else:
            active += 1

        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != row["local_sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {row['local_path']}")
        if actual_hash in hashes:
            raise RuntimeError(f"Duplicate retained image hash: {row['local_path']}")
        hashes.add(actual_hash)
        with Image.open(path) as image:
            image.load()
            if image.size != (int(row["width"]), int(row["height"])):
                raise RuntimeError(f"Dimension mismatch for {row['local_path']}")

    return {
        "registry_rows": len(registry),
        "retained_images": active,
        "excluded_rows": excluded,
        "optional_excluded_files_present": optional_excluded_present,
        "hash_mismatches": 0,
        "duplicate_hashes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Overwrite normalized local copies")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the 26 retained local images without querying Wikimedia Commons",
    )
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_existing_dataset(), indent=2, sort_keys=True))
        return
    rows = build_dataset(refresh=args.refresh)
    make_contact_sheet(rows)
    print(f"Registry: {REGISTRY_PATH}")
    print(f"Contact sheet: {CONTACT_SHEET_PATH}")
    print(f"Accepted images: {len(rows)}")


if __name__ == "__main__":
    main()
