#!/usr/bin/env python3
"""Build the bounded, provenance-first ad-safety capstone dataset.

The four-way benchmark is deliberately small. It uses source groups, rather
than individual images, as the unit of splitting so repeated product campaigns
cannot cross train, validation, and test. The test split is written once here;
the training script never uses it for fitting or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "capstone_dataset"
REGISTRY_PATH = PROJECT_ROOT / "data" / "capstone_registry.csv"
MANIFEST_PATH = DATA_ROOT / "dataset_manifest.json"

SEED = 462
BUILDER_VERSION = "1.0.0"
SPLITS = ("train", "val", "test")
LABELS = ("safe", "firearms", "explosives", "financial_promotion")
TARGETS = {"train": 48, "val": 12, "test": 12}

AD_REPO = "EthanGabis/ADautoGen-DS"
AD_REVISION = "fa7a7803265fc97926d7fab694bfc1e05c1fc7b4"
AD_PARQUET = "data/train-00000-of-00001.parquet"
AD_LICENSE = "MIT"

WEAPONS_REPO = "rajshivanshuu/weapons_set1"
WEAPONS_REVISION = "d4d9dbc8272958820a3f6757e4c48d8987300271"
WEAPONS_LICENSE = "OSL-3.0"

REGISTRY_FIELDS = (
    "sample_id",
    "label",
    "original_label",
    "split",
    "local_path",
    "source_dataset",
    "repository_url",
    "dataset_card_url",
    "source_revision",
    "source_repo_file",
    "source_url",
    "original_id",
    "source_group",
    "group_strategy",
    "split_method",
    "declared_license",
    "license_scope",
    "license_evidence_url",
    "is_synthetic",
    "synthetic_basis",
    "generator",
    "source_file_sha256",
    "raw_image_sha256",
    "local_sha256",
    "perceptual_dhash",
    "mime",
    "width",
    "height",
    "text_metadata",
    "selection_seed",
    "builder_version",
    "label_audit_status",
)

FINANCIAL_CAMPAIGNS = (
    ("alpha-growth", "ALPHA GROWTH FUND", "Build a long-term portfolio", "Stocks + ETFs", "+12.4% sample"),
    ("bright-bond", "BRIGHT BOND PLAN", "Steady income for tomorrow", "Bond portfolio", "4.8% sample"),
    ("cobalt-crypto", "COBALT CRYPTO", "Trade digital assets in one place", "Crypto exchange", "24/7 markets"),
    ("delta-dividend", "DELTA DIVIDEND", "Put dividends on repeat", "Income stocks", "Quarterly sample"),
    ("ember-etf", "EMBER ETF", "One fund, many companies", "Index investing", "Low-cost sample"),
    ("flux-forex", "FLUX FOREX", "Explore global currency markets", "FX trading", "Demo account"),
    ("green-retire", "GREEN RETIRE", "Plan the years ahead", "Retirement account", "Tax-aware sample"),
    ("harbor-savings", "HARBOR SAVINGS", "Make every deposit count", "Savings account", "5.1% sample APY"),
    ("ion-invest", "ION INVEST", "Start with a simple portfolio", "Managed investing", "$25 sample start"),
    ("jade-markets", "JADE MARKETS", "Research before you trade", "Market analytics", "Live sample chart"),
    ("kite-capital", "KITE CAPITAL", "Diversify beyond cash", "Mutual funds", "Risk-rated sample"),
    ("lumen-loans", "LUMEN LOANS", "Compare financing options", "Personal lending", "Sample 7.9% APR"),
    ("mint-money", "MINT MONEY", "Track spending and saving", "Finance app", "Free sample plan"),
    ("nova-options", "NOVA OPTIONS", "Learn options with a simulator", "Options trading", "Practice mode"),
    ("orbit-portfolio", "ORBIT PORTFOLIO", "See your assets in one view", "Portfolio tracker", "Daily sample view"),
    ("pulse-profit", "PULSE PROFIT", "Follow a disciplined strategy", "Trading signals", "Sample insights"),
    ("quartz-quant", "QUARTZ QUANT", "Data tools for market research", "Quant platform", "Model sample"),
    ("river-rewards", "RIVER REWARDS", "Earn points on daily spending", "Credit card", "3x sample points"),
    ("solar-stocks", "SOLAR STOCKS", "Explore renewable-energy firms", "Stock basket", "Sector sample"),
    ("terra-token", "TERRA TOKEN", "A simulated token campaign", "Digital token", "Sample presale"),
    ("unity-wealth", "UNITY WEALTH", "Advice for changing goals", "Wealth planning", "Sample consultation"),
    ("vector-venture", "VECTOR VENTURE", "Back emerging businesses", "Private markets", "Accredited sample"),
    ("wave-wallet", "WAVE WALLET", "Send and store digital value", "Crypto wallet", "Secure sample"),
    ("zenith-zero", "ZENITH ZERO FEE", "Compare commission-free trades", "Brokerage", "$0 sample fee"),
)

PALETTES = (
    ((11, 31, 58), (43, 197, 172), (235, 250, 247)),
    ((31, 20, 72), (143, 101, 255), (246, 242, 255)),
    ((53, 19, 42), (244, 114, 182), (255, 241, 248)),
    ((24, 50, 36), (139, 211, 113), (242, 252, 238)),
    ((55, 36, 9), (246, 183, 53), (255, 249, 230)),
    ((13, 45, 69), (64, 168, 224), (238, 248, 255)),
)


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:72] or "unknown"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def request_bytes(session: requests.Session, url: str, attempts: int = 6) -> bytes:
    delay = 1.0
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=120)
            if response.status_code == 429:
                raise requests.HTTPError("rate limited", response=response)
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 20.0)
    raise RuntimeError("unreachable request retry loop")


def hf_resolve_url(repo: str, revision: str, repo_file: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(repo_file, safe='/')}"


def dhash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{bits:016x}"


def normalize_image(raw: bytes) -> tuple[bytes, int, int, str]:
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("source is not a readable image") from exc
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < 32 or image.width * image.height > 50_000_000:
        raise ValueError(f"unsupported source dimensions: {image.size}")
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    fingerprint = dhash(image)
    output = io.BytesIO()
    image.save(output, "JPEG", quality=94, optimize=True, progressive=True)
    return output.getvalue(), image.width, image.height, fingerprint


def save_sample(label: str, split: str, raw: bytes) -> dict[str, Any]:
    normalized, width, height, fingerprint = normalize_image(raw)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    local_sha256 = hashlib.sha256(normalized).hexdigest()
    sample_id = f"{label}-{raw_sha256[:16]}"
    path = DATA_ROOT / split / label / f"{sample_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(normalized)
    return {
        "sample_id": sample_id,
        "local_path": str(path.relative_to(PROJECT_ROOT)),
        "raw_image_sha256": raw_sha256,
        "local_sha256": local_sha256,
        "perceptual_dhash": fingerprint,
        "mime": "image/jpeg",
        "width": width,
        "height": height,
    }


def partition_groups_exact(
    groups: dict[str, list[int]], targets: dict[str, int]
) -> dict[int, str]:
    """Select and partition whole groups to hit exact per-split row counts."""

    target_state = tuple(targets[split] for split in SPLITS)
    ordered = sorted(groups, key=lambda group: stable_rank(f"safe:{group}"))
    states: dict[tuple[int, int, int], tuple[tuple[str, str], ...]] = {(0, 0, 0): ()}
    for group in ordered:
        size = len(groups[group])
        updated = dict(states)
        for state, assignments in states.items():
            for split_index, split in enumerate(SPLITS):
                candidate = list(state)
                candidate[split_index] += size
                candidate_state = tuple(candidate)
                if candidate_state[split_index] > target_state[split_index]:
                    continue
                if candidate_state not in updated:
                    updated[candidate_state] = assignments + ((group, split),)
        states = updated
        if target_state in states:
            break
    if target_state not in states:
        sizes = Counter(len(rows) for rows in groups.values())
        raise RuntimeError(f"Cannot make exact grouped split targets {targets}; group sizes={dict(sizes)}")
    row_splits: dict[int, str] = {}
    for group, split in states[target_state]:
        for row_index in groups[group]:
            row_splits[row_index] = split
    return row_splits


def base_row(**overrides: Any) -> dict[str, Any]:
    row = {field: "" for field in REGISTRY_FIELDS}
    row.update(
        {
            "split_method": "deterministic_source_group_partition",
            "selection_seed": str(SEED),
            "builder_version": BUILDER_VERSION,
            "label_audit_status": "source-derived label; no independent expert image audit",
        }
    )
    row.update(overrides)
    return row


def build_safe_rows(session: requests.Session) -> tuple[list[dict[str, Any]], str]:
    url = hf_resolve_url(AD_REPO, AD_REVISION, AD_PARQUET)
    parquet_bytes = request_bytes(session, url)
    parquet_sha256 = hashlib.sha256(parquet_bytes).hexdigest()
    frame = pd.read_parquet(io.BytesIO(parquet_bytes))
    required = {
        "id",
        "product_name",
        "tagline",
        "ad_copy",
        "category",
        "target_audience",
        "source_model",
        "image",
    }
    if len(frame) != 120 or not required.issubset(frame.columns):
        raise RuntimeError(f"Unexpected ADautoGen schema/size: rows={len(frame)}, columns={list(frame.columns)}")

    groups: dict[str, list[int]] = defaultdict(list)
    for row_index, record in frame.iterrows():
        group = f"adautogen-product::{slugify(str(record['product_name']))}"
        groups[group].append(int(row_index))
    row_splits = partition_groups_exact(dict(groups), TARGETS)

    rows: list[dict[str, Any]] = []
    for row_index, split in sorted(row_splits.items(), key=lambda item: (SPLITS.index(item[1]), item[0])):
        record = frame.iloc[row_index]
        image_record = record["image"]
        if not isinstance(image_record, dict) or not image_record.get("bytes"):
            raise RuntimeError(f"ADautoGen row {row_index} has no embedded image bytes")
        raw = bytes(image_record["bytes"])
        saved = save_sample("safe", split, raw)
        group = f"adautogen-product::{slugify(str(record['product_name']))}"
        rows.append(
            base_row(
                **saved,
                label="safe",
                original_label=str(record["category"]),
                split=split,
                source_dataset=AD_REPO,
                repository_url=f"https://huggingface.co/datasets/{AD_REPO}",
                dataset_card_url=f"https://huggingface.co/datasets/{AD_REPO}/tree/{AD_REVISION}",
                source_revision=AD_REVISION,
                source_repo_file=f"{AD_PARQUET}#row={int(record['id'])}",
                source_url=url,
                original_id=str(record["id"]),
                source_group=group,
                group_strategy="exact product_name; repeated product campaigns stay together",
                declared_license=AD_LICENSE,
                license_scope="repository README statement; generated images have no per-row license field",
                license_evidence_url=f"https://huggingface.co/datasets/{AD_REPO}/blob/{AD_REVISION}/README.md#license",
                is_synthetic=bool_text(True),
                synthetic_basis="Dataset README states images were generated with Stable Diffusion v1.5",
                generator="Stable Diffusion v1.5 (declared by source README)",
                source_file_sha256=parquet_sha256,
                text_metadata=json.dumps(
                    {
                        "product_name": str(record["product_name"]),
                        "tagline": str(record["tagline"]),
                        "category": str(record["category"]),
                        "target_audience": str(record["target_audience"]),
                        "text_source_model": str(record["source_model"]),
                    },
                    sort_keys=True,
                ),
            )
        )
    return rows, parquet_sha256


def fetch_repo_files(session: requests.Session, repo: str, revision: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}"
    payload = json.loads(request_bytes(session, url))
    return [str(item["rfilename"]) for item in payload.get("siblings", []) if item.get("rfilename")]


def source_identity(repo_file: str) -> str:
    path = Path(repo_file)
    stem = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", path.stem, flags=re.IGNORECASE)
    return slugify(stem)


def split_queue(targets: dict[str, int], salt: str) -> list[str]:
    queue = [split for split in SPLITS for _ in range(targets[split])]
    random.Random(int(stable_rank(salt)[:16], 16)).shuffle(queue)
    return queue


def build_weapon_subset(
    session: requests.Session,
    repo_files: Iterable[str],
    source_label: str,
    policy_label: str,
    targets: dict[str, int],
) -> list[dict[str, Any]]:
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    candidates = [
        path
        for path in repo_files
        if path.startswith(f"{source_label}/") and Path(path).suffix.casefold() in allowed
    ]
    candidates.sort(key=lambda path: stable_rank(f"weapon:{source_label}:{path}"))
    queue = split_queue(targets, f"weapon-queue:{source_label}")
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for repo_file in candidates:
        if not queue:
            break
        identity = source_identity(repo_file)
        source_group = f"weapons-set1::{source_label}::{identity}"
        if source_group in seen_groups:
            continue
        url = hf_resolve_url(WEAPONS_REPO, WEAPONS_REVISION, repo_file)
        try:
            raw = request_bytes(session, url)
            split = queue[0]
            saved = save_sample(policy_label, split, raw)
        except (requests.RequestException, ValueError):
            continue
        queue.pop(0)
        seen_groups.add(source_group)
        rows.append(
            base_row(
                **saved,
                label=policy_label,
                original_label=source_label,
                split=split,
                source_dataset=WEAPONS_REPO,
                repository_url=f"https://huggingface.co/datasets/{WEAPONS_REPO}",
                dataset_card_url=f"https://huggingface.co/datasets/{WEAPONS_REPO}/tree/{WEAPONS_REVISION}",
                source_revision=WEAPONS_REVISION,
                source_repo_file=repo_file,
                source_url=url,
                original_id=identity,
                source_group=source_group,
                group_strategy="source class plus pre-Roboflow source basename",
                declared_license=WEAPONS_LICENSE,
                license_scope="repository-level card metadata only; per-image provenance and rights are absent",
                license_evidence_url=f"https://huggingface.co/datasets/{WEAPONS_REPO}/blob/{WEAPONS_REVISION}/README.md",
                is_synthetic=bool_text(False),
                synthetic_basis="Repository card does not declare synthesis; false is not an authenticity guarantee",
                generator="",
                source_file_sha256=hashlib.sha256(raw).hexdigest(),
                text_metadata=json.dumps({"source_folder": source_label}, sort_keys=True),
            )
        )
    if queue:
        raise RuntimeError(
            f"Could not obtain enough valid {source_label} images: missing {Counter(queue)} from {len(candidates)} candidates"
        )
    return rows


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_chars: int,
    spacing: int = 8,
) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    draw.multiline_text(xy, "\n".join(lines), font=font, fill=fill, spacing=spacing)


def render_financial_creative(campaign_index: int, variant: int) -> tuple[bytes, dict[str, str]]:
    campaign_id, brand, headline, product, offer = FINANCIAL_CAMPAIGNS[campaign_index]
    rng = random.Random(SEED * 10_000 + campaign_index * 101 + variant)
    dark, accent, pale = PALETTES[(campaign_index + variant) % len(PALETTES)]
    width, height = 640, 640
    image = Image.new("RGB", (width, height), pale)
    draw = ImageDraw.Draw(image)

    for _ in range(16):
        radius = rng.randint(16, 72)
        x = rng.randint(-radius, width)
        y = rng.randint(-radius, height)
        color = tuple(min(255, int(channel + rng.randint(10, 45))) for channel in accent)
        draw.ellipse((x, y, x + radius, y + radius), fill=color)

    draw.rounded_rectangle((30, 28, 610, 612), radius=30, fill=(250, 252, 252), outline=dark, width=3)
    draw.rounded_rectangle((30, 28, 610, 145), radius=30, fill=dark)
    draw.rectangle((30, 108, 610, 145), fill=dark)

    small = ImageFont.load_default(size=20)
    medium = ImageFont.load_default(size=29)
    large = ImageFont.load_default(size=46)
    draw.text((57, 55), brand, font=medium, fill=(255, 255, 255))
    draw.text((57, 101), product.upper(), font=small, fill=accent)
    draw_wrapped_text(draw, (57, 177), headline, large, dark, max_chars=22, spacing=10)

    chart_box = (58, 330, 394, 500)
    draw.rounded_rectangle(chart_box, radius=18, fill=(239, 244, 246), outline=(198, 210, 215), width=2)
    for grid_y in range(360, 490, 32):
        draw.line((75, grid_y, 375, grid_y), fill=(207, 217, 221), width=1)
    points: list[tuple[int, int]] = []
    current_y = rng.randint(448, 474)
    for x in range(78, 376, 42):
        current_y = max(352, min(475, current_y + rng.randint(-42, 18)))
        points.append((x, current_y))
    draw.line(points, fill=accent, width=8, joint="curve")
    for point in points:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=dark)

    draw.rounded_rectangle((418, 330, 576, 500), radius=22, fill=accent)
    draw.ellipse((452, 353, 542, 443), outline=(255, 255, 255), width=7)
    currency = ("$", "USD", "EUR", "BTC")[(campaign_index + variant) % 4]
    currency_font = large if currency == "$" else medium
    currency_box = draw.textbbox((0, 0), currency, font=currency_font)
    currency_width = currency_box[2] - currency_box[0]
    draw.text((497 - currency_width // 2, 378), currency, font=currency_font, fill=(255, 255, 255))
    draw.text((435, 455), offer, font=small, fill=dark)

    cta = ("LEARN MORE", "VIEW SAMPLE", "EXPLORE NOW")[(campaign_index + variant) % 3]
    draw.rounded_rectangle((58, 525, 270, 574), radius=16, fill=dark)
    draw.text((80, 539), cta, font=small, fill=(255, 255, 255))
    draw.text((291, 532), "SIMULATED CREATIVE", font=small, fill=dark)
    draw.text((291, 558), "NOT FINANCIAL ADVICE", font=small, fill=dark)

    output = io.BytesIO()
    image.save(output, "PNG", optimize=True)
    return output.getvalue(), {
        "campaign_id": campaign_id,
        "brand": brand,
        "headline": headline,
        "product": product,
        "offer": offer,
        "variant": str(variant),
    }


def build_financial_rows() -> list[dict[str, Any]]:
    campaign_order = sorted(range(len(FINANCIAL_CAMPAIGNS)), key=lambda index: stable_rank(f"finance:{index}"))
    split_by_campaign: dict[int, str] = {}
    cursor = 0
    for split in SPLITS:
        group_count = TARGETS[split] // 3
        for campaign_index in campaign_order[cursor : cursor + group_count]:
            split_by_campaign[campaign_index] = split
        cursor += group_count
    if len(split_by_campaign) != 24:
        raise RuntimeError("Financial campaign targets must select 24 groups of three variants")

    rows: list[dict[str, Any]] = []
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for campaign_index, split in sorted(split_by_campaign.items(), key=lambda item: (SPLITS.index(item[1]), item[0])):
        campaign_id = FINANCIAL_CAMPAIGNS[campaign_index][0]
        source_group = f"pil-financial-campaign::{campaign_id}"
        for variant in range(3):
            raw, metadata = render_financial_creative(campaign_index, variant)
            saved = save_sample("financial_promotion", split, raw)
            rows.append(
                base_row(
                    **saved,
                    label="financial_promotion",
                    original_label="financial_promotion",
                    split=split,
                    source_dataset="project_generated_financial_creatives_v1",
                    repository_url="",
                    dataset_card_url="",
                    source_revision=script_sha256,
                    source_repo_file=f"scripts/build_capstone_dataset.py#financial/{campaign_id}/{variant}",
                    source_url="",
                    original_id=f"{campaign_id}:{variant}",
                    source_group=source_group,
                    group_strategy="campaign id; all three visual variants stay together",
                    declared_license="project-generated; release terms not declared",
                    license_scope="no third-party visual assets; team must choose distribution terms",
                    license_evidence_url="",
                    is_synthetic=bool_text(True),
                    synthetic_basis="Deterministic PIL renderer in the dataset builder",
                    generator="Pillow drawing primitives; seed 462; generator v1",
                    source_file_sha256=script_sha256,
                    text_metadata=json.dumps(metadata, sort_keys=True),
                    label_audit_status="controlled generator label; text says simulated creative",
                )
            )
    return rows


def verify_registry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 72 * len(LABELS):
        raise RuntimeError(f"Expected 288 rows, found {len(rows)}")
    counts = Counter((row["label"], row["split"]) for row in rows)
    for label in LABELS:
        for split, expected in TARGETS.items():
            if counts[(label, split)] != expected:
                raise RuntimeError(f"Bad count for {label}/{split}: {counts[(label, split)]} != {expected}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    sha_splits: dict[str, set[str]] = defaultdict(set)
    ids: set[str] = set()
    local_paths: set[str] = set()
    for row in rows:
        group_splits[row["source_group"]].add(row["split"])
        sha_splits[row["local_sha256"]].add(row["split"])
        if row["sample_id"] in ids or row["local_path"] in local_paths:
            raise RuntimeError(f"Duplicate sample identity/path: {row['sample_id']}")
        ids.add(row["sample_id"])
        local_paths.add(row["local_path"])
        path = PROJECT_ROOT / row["local_path"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != row["local_sha256"]:
            raise RuntimeError(f"Hash mismatch: {path}")
    leaking_groups = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    leaking_hashes = {sha: splits for sha, splits in sha_splits.items() if len(splits) > 1}
    if leaking_groups or leaking_hashes:
        raise RuntimeError(f"Split leakage detected: groups={leaking_groups}, hashes={leaking_hashes}")
    return {
        "rows": len(rows),
        "groups": len(group_splits),
        "counts": {
            label: {split: counts[(label, split)] for split in SPLITS}
            for label in LABELS
        },
        "group_overlap_across_splits": 0,
        "exact_hash_overlap_across_splits": 0,
    }


def write_registry(rows: list[dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    split_order = {name: index for index, name in enumerate(SPLITS)}
    label_order = {name: index for index, name in enumerate(LABELS)}
    rows.sort(key=lambda row: (split_order[row["split"]], label_order[row["label"]], row["sample_id"]))
    with REGISTRY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify the existing registry and local files")
    args = parser.parse_args()

    if args.verify_only:
        with REGISTRY_PATH.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(json.dumps(verify_registry(rows), indent=2, sort_keys=True))
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "MS-DSP-462-AdSafetyCapstone/1.0 educational-research"})

    print(f"Downloading pinned safe source {AD_REPO}@{AD_REVISION}")
    safe_rows, ad_parquet_sha256 = build_safe_rows(session)
    print(f"Built safe rows: {len(safe_rows)}")

    print(f"Listing pinned weapons source {WEAPONS_REPO}@{WEAPONS_REVISION}")
    repo_files = fetch_repo_files(session, WEAPONS_REPO, WEAPONS_REVISION)
    firearms_rows = build_weapon_subset(
        session,
        repo_files,
        source_label="pistol",
        policy_label="firearms",
        targets={"train": 36, "val": 9, "test": 9},
    )
    firearms_rows += build_weapon_subset(
        session,
        repo_files,
        source_label="rifile",
        policy_label="firearms",
        targets={"train": 12, "val": 3, "test": 3},
    )
    explosive_rows = build_weapon_subset(
        session,
        repo_files,
        source_label="grenade",
        policy_label="explosives",
        targets=TARGETS,
    )
    if any(path.startswith("knife/") for path in (row["source_repo_file"] for row in firearms_rows + explosive_rows)):
        raise RuntimeError("Knife images must not enter this policy taxonomy")
    print(f"Built firearms rows: {len(firearms_rows)}; explosives rows: {len(explosive_rows)}")

    financial_rows = build_financial_rows()
    print(f"Built deterministic financial-promotion rows: {len(financial_rows)}")

    rows = safe_rows + firearms_rows + explosive_rows + financial_rows
    verification = verify_registry(rows)
    write_registry(rows)
    manifest = {
        "builder_version": BUILDER_VERSION,
        "seed": SEED,
        "labels": list(LABELS),
        "split_targets_per_label": TARGETS,
        "verification": verification,
        "sources": [
            {
                "repo_id": AD_REPO,
                "revision": AD_REVISION,
                "url": f"https://huggingface.co/datasets/{AD_REPO}",
                "license": AD_LICENSE,
                "license_scope": "README-level",
                "parquet_sha256": ad_parquet_sha256,
                "declared_synthetic": True,
            },
            {
                "repo_id": WEAPONS_REPO,
                "revision": WEAPONS_REVISION,
                "url": f"https://huggingface.co/datasets/{WEAPONS_REPO}",
                "license": WEAPONS_LICENSE,
                "license_scope": "repository metadata only; per-image provenance absent",
                "declared_synthetic": False,
                "synthetic_flag_caveat": "absence of a declaration is not proof that every image is real",
            },
            {
                "repo_id": "project_generated_financial_creatives_v1",
                "revision": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "license": "release terms not declared",
                "declared_synthetic": True,
            },
        ],
        "known_limitations": [
            "Class labels are confounded with data source and generator style.",
            "weapons_set1 provides no per-image origin or rights metadata.",
            "The financial class is code-generated and does not represent real ad-platform diversity.",
            "ADautoGen images are Stable Diffusion v1.5 outputs, not real submitted ads.",
            "Source labels were not independently expert-audited image by image.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(verification, indent=2, sort_keys=True))
    print(f"Registry: {REGISTRY_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
