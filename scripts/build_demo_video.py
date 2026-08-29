#!/usr/bin/env python3
"""Build and validate the narrated Ad Safety capstone demo video.

The macOS speech synthesizer creates narration fully offline. This script
prepares the storyboard, generates per-scene AIFF and WAV files, composes
evidence-led frames, assembles the MP4, creates captions, and validates media.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs" / "video"
FRAMES = OUTPUT / "frames"
AUDIO_SEGMENTS = OUTPUT / "audio" / "segments"
AUDIO_AIFF = OUTPUT / "audio" / "aiff"
SCENE_VIDEOS = OUTPUT / "scenes"
QA_FRAMES = OUTPUT / "qa_frames"
TMP = OUTPUT / ".build" / "speech"
FINAL_VIDEO = OUTPUT / "ad_safety_demo.mp4"
NO_SUBS_VIDEO = OUTPUT / "ad_safety_demo_no_subtitles.mp4"
CAPTIONS = OUTPUT / "ad_safety_demo.srt"
FULL_NARRATION = OUTPUT / "audio" / "narration_full.m4a"
NARRATION_PROVENANCE = OUTPUT / "narration_provenance.json"

WIDTH, HEIGHT, FPS = 1280, 720, 30
OFFLINE_ENGINE = "macOS say"
OFFLINE_VOICE = "Aman"
OFFLINE_LOCALE = "en_IN"
OFFLINE_RATE_WPM = 180


def slide(number: int) -> str:
    return f"outputs/presentation/rendered/slide-{number:02d}.png"


SCENES: list[dict[str, Any]] = [
    {
        "id": "01_executive_summary",
        "section": "Executive Summary",
        "presenter": "Swarnaditya Maitra",
        "title": "Ad Safety Moderation: Pilot results and next steps",
        "visual": {"layout": "single", "items": [slide(1)]},
        "narration": (
            "This video uses offline macOS system-generated narration with the Aman voice. "
            "I am presenting a bounded ad safety decision-support prototype. I turn one ad image into an "
            "APPROVE, REVIEW, or BLOCK recommendation while keeping the evidence visible. The formal result is "
            "strong, but I do not treat this pilot as approval for autonomous deployment."
        ),
    },
    {
        "id": "02_team",
        "section": "Team",
        "presenter": "Vijay Agnihotri",
        "title": "Team and presentation plan",
        "visual": {"layout": "single", "items": [slide(2)]},
        "narration": (
            "I divide the story across four presenters. Swarnaditya covers the decision and results. Vijay covers "
            "data and failure analysis. Myetchae covers policy, architecture, and audit. Bickramjit covers method, "
            "literature, and workflow."
        ),
    },
    {
        "id": "03_business_problem",
        "section": "Business Problem",
        "presenter": "Swarnaditya Maitra",
        "title": "Ambiguous ads are hard to review consistently",
        "visual": {"layout": "single", "items": [slide(3)]},
        "narration": (
            "I start from a queue problem. A creative can hide a restricted object, subtle text, or misleading "
            "context inside an ordinary image. Manual review is slow and inconsistent. I therefore use computer "
            "vision to prioritize risky images and show evidence, while a person remains the final decision maker."
        ),
    },
    {
        "id": "04_policy_boundary",
        "section": "Policy Boundary",
        "presenter": "Myetchae Thu",
        "title": "Four labels are not a full ad policy",
        "visual": {"layout": "single", "items": [slide(4)]},
        "narration": (
            "My modeled boundary has four image labels: safe, firearms, explosives, and financial promotion. "
            "Financial content always routes to REVIEW because an image cannot prove a misleading claim or a legal "
            "violation. The prototype does not model explicit content, violence, alcohol, tobacco, geography, age "
            "restrictions, landing pages, advertiser identity, or the full policy rulebook."
        ),
    },
    {
        "id": "05_data_profile",
        "section": "Data Profile",
        "presenter": "Vijay Agnihotri",
        "title": "Balanced counts, unbalanced sources",
        "visual": {"layout": "single", "items": [slide(5)]},
        "narration": (
            "I built a 288-image pilot dataset. Each class has 48 training images, 12 validation images, and 12 "
            "untouched test images. I split 215 source groups so the same campaign or exact source group stays in one "
            "partition. This blocks a simple duplicate leak, but balanced counts do not remove source bias."
        ),
    },
    {
        "id": "06_eda",
        "section": "EDA",
        "presenter": "Vijay Agnihotri",
        "title": "The classes look different for reasons unrelated to policy",
        "visual": {"layout": "single", "items": [slide(6)]},
        "narration": (
            "The contact sheet exposes the main weakness. Safe ads are synthetic product creatives. Financial ads "
            "are synthetic templates. Both weapon classes come from one weapons source. A classifier can therefore "
            "learn photographic style instead of policy meaning. I call this source confounding, and it limits every "
            "formal score that follows."
        ),
    },
    {
        "id": "07_architecture",
        "section": "Architecture",
        "presenter": "Myetchae Thu",
        "title": "How the system reaches a decision",
        "visual": {"layout": "single", "items": [slide(7)]},
        "narration": (
            "My pipeline keeps three evidence streams separate. A frozen vision transformer scores the whole image. "
            "Grounding DINO can localize weapon and explosive cues. Tesseract extracts visible words. A versioned "
            "policy layer then combines classifier, detector, and text evidence into APPROVE, REVIEW, or BLOCK. This "
            "design makes each reason inspectable instead of hiding the final action inside one score."
        ),
    },
    {
        "id": "08_methodology",
        "section": "Methodology",
        "presenter": "Bickramjit Basu",
        "title": "We kept the test set out of training and tuning",
        "visual": {"layout": "single", "items": [slide(8)]},
        "narration": (
            "I froze both the V-I-T and ResNet-fifty backbones, cached their embeddings, and fit logistic heads only "
            "on training data. I selected conservative class thresholds on validation data. The recorded protocol "
            "evaluates the 48-image test set once after calibration. I also benchmarked warm batch-one C-P-U inference with one thread so "
            "the timing protocol stays reproducible."
        ),
    },
    {
        "id": "09_literature",
        "section": "Literature",
        "presenter": "Bickramjit Basu",
        "title": "What prior research contributed",
        "visual": {"layout": "single", "items": [slide(9)]},
        "narration": (
            "I use published work to justify component choices, not to claim this pilot is production-ready. The V-I-T "
            "paper supports transferable image representations. Grounding DINO supports text-conditioned object "
            "localization. Tesseract supports visible-text extraction. Only this project's saved evaluation supports "
            "the numbers I report."
        ),
    },
    {
        "id": "10_formal_results",
        "section": "Formal Results",
        "presenter": "Swarnaditya Maitra",
        "title": "ViT wins accuracy, not speed",
        "visual": {
            "layout": "hero_right_stack",
            "items": [
                slide(10),
                "outputs/evaluation/model_comparison.png",
                "outputs/evaluation/confusion_matrix_vit.png",
            ],
            "labels": ["Management result", "Model comparison", "ViT confusion matrix"],
        },
        "narration": (
            "On the untouched 48-image test, my V-I-T reached macro F one of 0.97913 and accuracy of 0.97917. The "
            "frozen ResNet-fifty baseline reached macro F one of 0.87315. Safe precision 1.0 means the proposal's "
            "greater-than-0.98 Safe-class precision benchmark passed. The Safe-ad false-flag rate, defined as one "
            "minus Safe recall, was zero. Multiclass argmax restricted-class mean recall 0.97222 differs from the "
            "threshold-operating mean 0.94444 and minimum 0.91667. The proposal's per-class greater-than-0.95 target "
            "failed for firearms and explosives. The 0.90 validation-recall tuning "
            "floor in code was only for calibration, not that acceptance target. Warm C-P-U batch-one P ninety-five "
            "latency was 71.527 milliseconds for V-I-T and 40.064 for ResNet. The V-I-T classifier path exceeded the "
            "50-millisecond target. Full end-to-end P ninety-five remains unassessed because this timing covers "
            "preprocessing and classification only. It excludes file decoding, O-C-R, object detection, network "
            "transfer, and app rendering."
        ),
    },
    {
        "id": "11_known_formal_error",
        "section": "Class and Failure Analysis",
        "presenter": "Vijay Agnihotri",
        "title": "One grenade was classified as a firearm",
        "visual": {
            "layout": "hero_right_stack",
            "items": [
                slide(11),
                "outputs/demo_cases/formal_known_failure/evidence_overlay.jpg",
                "outputs/demo_cases/formal_known_failure/occlusion_heatmap.jpg",
            ],
            "labels": ["Formal analysis", "Detected evidence", "Occlusion explanation"],
        },
        "narration": (
            "The test had one classification error. An explosives image was predicted as firearms, so 47 of 48 "
            "labels were correct. In the saved demo, the fused policy score for firearms is 0.88998. Neither the calibrated firearms "
            "threshold nor the explosives threshold fires, so the policy returns REVIEW. The detector still localizes "
            "shotgun and grenade-like cues. This is a class mistake, not an automatic approval."
        ),
    },
    {
        "id": "12_external_audit",
        "section": "External Generalization Audit",
        "presenter": "Myetchae Thu",
        "title": "External images exposed weak generalization",
        "visual": {
            "layout": "hero_right_stack",
            "items": [
                slide(12),
                "outputs/demo_cases/external_safe_domain_shift/evidence_overlay.jpg",
                "outputs/demo_cases/external_safe_domain_shift/occlusion_heatmap.jpg",
            ],
            "labels": ["Diagnostic summary", "False detector evidence", "Model sensitivity"],
        },
        "narration": (
            "I then tested a manually reviewed 26-image Wikimedia diagnostic set. Classifier macro F one fell to "
            "0.55489, and the detector's nonweapon false-positive rate reached 0.90. This historical shoe ad is labeled "
            "safe. The classifier predicts explosives at 0.32411, detector evidence raises the fused policy score to "
            "0.45302, and the policy returns BLOCK. The detector boxes "
            "and heatmap concentrate on the shoe. This is direct evidence of domain shift and source confounding. The "
            "small diagnostic sample is not a second formal test set."
        ),
    },
    {
        "id": "13_demo_workflow",
        "section": "Demo Workflow",
        "presenter": "Bickramjit Basu",
        "title": "A reviewer can follow one case from upload to audit",
        "visual": {"layout": "single", "items": [slide(13)]},
        "narration": (
            "The workspace has four steps. I upload one J-P-E-G, P-N-G, or WebP image, configure optional object "
            "context, O-C-R, and occlusion, review raw and fused scores with reasons and timing, then export a case-level "
            "audit record. The policy returns APPROVE, REVIEW, or BLOCK. I use four saved formal examples next."
        ),
    },
    {
        "id": "14_safe_approve",
        "section": "Demo Case",
        "presenter": "Bickramjit Basu",
        "title": "Safe creative routes to APPROVE",
        "visual": {
            "layout": "hero_right_stack",
            "items": [
                "outputs/app/app_safe_result.png",
                "data/capstone_dataset/test/safe/safe-77c35d499df2417f.jpg",
                "outputs/demo_cases/formal_safe/evidence_overlay.jpg",
            ],
            "labels": ["App decision", "Source creative", "Saved evidence"],
            "crops": [[360, 420, 1200, 660], None, None],
        },
        "narration": (
            "For the safe formal example, the classifier score is 0.99735. O-C-R finds three tokens, the optional "
            "detector is off, and the policy returns APPROVE. I can trace the verdict to the saved input and audit "
            "record. This confirms the intended path for an ordinary product creative inside the pilot domain."
        ),
    },
    {
        "id": "15_firearm_block",
        "section": "Demo Case",
        "presenter": "Bickramjit Basu",
        "title": "Firearm evidence routes to BLOCK",
        "visual": {
            "layout": "hero_right_stack",
            "items": [
                "outputs/app/app_firearm_result.png",
                "outputs/demo_cases/formal_firearms/evidence_overlay.jpg",
                "outputs/demo_cases/formal_firearms/occlusion_heatmap.jpg",
            ],
            "labels": ["App decision", "Detected objects", "Occlusion explanation"],
            "crops": [[360, 420, 1200, 660], None, None],
        },
        "narration": (
            "For the firearm example, the fused policy score is 0.99907. Grounding DINO returns three detections, the "
            "evidence overlay localizes weapon cues, and the policy returns BLOCK. The optional detector and heatmap "
            "take seconds, so their latency must not be confused with the 71.527-millisecond classifier benchmark."
        ),
    },
    {
        "id": "16_explosive_block",
        "section": "Demo Case",
        "presenter": "Bickramjit Basu",
        "title": "Explosive evidence routes to BLOCK",
        "visual": {
            "layout": "two_column",
            "items": [
                "data/capstone_dataset/test/explosives/explosives-808678dfe3e2b9d6.jpg",
                "outputs/demo_cases/formal_explosives/evidence_overlay.jpg",
            ],
            "labels": ["Source creative", "Three detector boxes"],
        },
        "narration": (
            "For the explosives example, the fused policy score is 0.99981 and the detector returns three boxes. The "
            "overlay identifies grenade and bomb-like cues among the surrounding objects. The policy returns BLOCK. "
            "This is the clean in-domain success case, and it contrasts with the earlier grenade image that crossed "
            "into the firearms class."
        ),
    },
    {
        "id": "17_financial_review",
        "section": "Demo Case",
        "presenter": "Bickramjit Basu",
        "title": "Financial promotion routes to REVIEW",
        "visual": {
            "layout": "hero_right_stack",
            "items": [
                "outputs/app/app_financial_result.png",
                "outputs/demo_cases/formal_financial_promotion/evidence_overlay.jpg",
                "outputs/demo_cases/formal_financial_promotion/occlusion_heatmap.jpg",
            ],
            "labels": ["App decision", "Visible text evidence", "Occlusion explanation"],
            "crops": [[360, 420, 1200, 660], None, None],
        },
        "narration": (
            "For the financial promotion example, the classifier score is 0.99921 and O-C-R extracts 17 tokens. The "
            "policy still returns REVIEW, not BLOCK, because this prototype cannot determine whether a financial "
            "claim is lawful or misleading from pixels alone. The text and heatmap give a reviewer useful context "
            "without replacing legal or policy judgment."
        ),
    },
    {
        "id": "18_risks_controls",
        "section": "Risks and Controls",
        "presenter": "Myetchae Thu",
        "title": "What must change before a pilot",
        "visual": {"layout": "single", "items": [slide(14)]},
        "narration": (
            "The largest risks are source-confounded training data, an uncalibrated detector outside the pilot domain, "
            "and incomplete policy coverage. The formal test has only 48 images. The external audit has only 26. "
            "Full end-to-end latency and concurrent throughput were not measured. M-P-S resource use was not measured, "
            "including C-P-U, G-P-U, and memory use. "
            "Before an M-V-P trial, I need diverse real ads, local detector calibration, out-of-distribution checks, "
            "audit logs, human escalation, and monitoring for drift and subgroup failures."
        ),
    },
    {
        "id": "19_next_steps",
        "section": "Next Steps to MVP",
        "presenter": "Swarnaditya Maitra",
        "title": "What we would do next: a 10-week plan",
        "visual": {"layout": "single", "items": [slide(15)]},
        "narration": (
            "My recommendation is only a human-gated M-V-P, not autonomous enforcement. The recommended ten-week "
            "roadmap is a planning assumption, not a measured commitment. In weeks one through three, a data lead, "
            "policy reviewer, and two annotators build a campaign-grouped real-ad holdout. In weeks four through six, "
            "one M-L engineer and a reviewer recalibrate the system against the declared quality gates. In weeks seven "
            "through ten, product, M-L-Ops, and reviewers run a shadow pilot. The scale gate tests ten concurrent "
            "requests and measures end-to-end p ninety-five plus C-P-U, G-P-U, and memory before broader rollout."
        ),
    },
]


PUBLIC_SOURCES = [
    "https://support.google.com/adspolicy/answer/6008942",
    "https://huggingface.co/datasets/EthanGabis/ADautoGen-DS",
    "https://huggingface.co/datasets/rajshivanshuu/weapons_set1",
    "https://arxiv.org/abs/2010.11929",
    "https://arxiv.org/abs/2303.05499",
    "https://huggingface.co/IDEA-Research/grounding-dino-tiny",
    "https://github.com/tesseract-ocr/tesseract",
    "https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia",
]


LOCAL_SOURCES = [
    "outputs/evaluation/metrics.json",
    "outputs/evaluation/model_comparison.csv",
    "outputs/evaluation/latency.json",
    "outputs/evaluation/external_spot_check.json",
    "outputs/demo_cases/case_summary.json",
    "outputs/presentation/ad_safety_management_presentation.pptx",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w.-]+\b", text))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def load_crop(path: Path, crop: list[int] | None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if crop:
        image = image.crop(tuple(crop))
    return image


def paste_contain(canvas: Image.Image, source: Image.Image, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    scale = min(w / source.width, h / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    px = x + (w - resized.width) // 2
    py = y + (h - resized.height) // 2
    canvas.paste(resized, (px, py))


def draw_panel(
    canvas: Image.Image,
    item: Path,
    box: tuple[int, int, int, int],
    label: str,
    crop: list[int] | None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = box
    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill="#ffffff", outline="#d8e0eb", width=2)
    label_h = 30
    draw.rounded_rectangle((x, y, x + w, y + label_h), radius=10, fill="#eaf3ff")
    draw.rectangle((x, y + 18, x + w, y + label_h), fill="#eaf3ff")
    draw.text((x + 12, y + 6), label, fill="#185abc", font=font(15, bold=True))
    source = load_crop(item, crop)
    paste_contain(canvas, source, (x + 8, y + label_h + 8, w - 16, h - label_h - 16))


def compose_frame(scene: dict[str, Any]) -> Path:
    frame_path = FRAMES / f"{scene['id']}.png"
    visual = scene["visual"]
    items = [PROJECT / path for path in visual["items"]]
    for item in items:
        if not item.exists():
            raise FileNotFoundError(item)

    if visual["layout"] == "single":
        source = load_crop(items[0], None)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
        paste_contain(canvas, source, (0, 0, WIDTH, HEIGHT))
        canvas.save(frame_path, quality=95)
        return frame_path

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 76), fill="#ffffff")
    draw.rectangle((28, 21, 35, 55), fill="#3286ff")
    draw.text((52, 18), scene["title"], fill="#15191f", font=font(31, bold=True))
    draw.text((52, 52), scene["section"].upper(), fill="#58677a", font=font(13, bold=True))
    draw.rectangle((0, HEIGHT - 30, WIDTH, HEIGHT), fill="#ffffff")
    draw.text(
        (28, HEIGHT - 23),
        "Offline macOS system-generated narration  |  evidence from saved project artifacts",
        fill="#64748b",
        font=font(13),
    )

    labels = visual.get("labels", [f"Evidence {i + 1}" for i in range(len(items))])
    crops = visual.get("crops", [None] * len(items))
    if visual["layout"] == "two_column":
        boxes = [(28, 94, 600, 574), (652, 94, 600, 574)]
    elif visual["layout"] == "hero_right_stack":
        boxes = [(28, 94, 790, 574), (842, 94, 410, 275), (842, 393, 410, 275)]
    else:
        raise ValueError(f"Unknown layout: {visual['layout']}")

    for item, box, label, crop in zip(items, boxes, labels, crops):
        draw_panel(canvas, item, box, label, crop)
    canvas.save(frame_path, quality=95)
    return frame_path


def write_prepare_outputs() -> None:
    for directory in (OUTPUT, FRAMES, AUDIO_SEGMENTS, AUDIO_AIFF, SCENE_VIDEOS, QA_FRAMES, TMP):
        directory.mkdir(parents=True, exist_ok=True)

    for scene in SCENES:
        compose_frame(scene)

    narration_lines = [
        "# Ad Safety Capstone Narration Script",
        "",
        f"Voice disclosure: This script uses offline macOS system-generated narration with the {OFFLINE_VOICE} voice.",
        "",
    ]
    for index, scene in enumerate(SCENES, start=1):
        narration_lines.extend(
            [
                f"## {index:02d}. {scene['title']}",
                "",
                f"Presenter section: {scene['presenter']}",
                "",
                scene["narration"],
                "",
            ]
        )
    narration_lines.extend(["## Source index", "", "Public sources:", ""])
    narration_lines.extend([f"- {url}" for url in PUBLIC_SOURCES])
    narration_lines.extend(["", "Saved local evidence:", ""])
    narration_lines.extend([f"- `{path}`" for path in LOCAL_SOURCES])
    narration_lines.append("")
    narration_text = "\n".join(narration_lines)
    (OUTPUT / "narration_script.md").write_text(narration_text, encoding="utf-8")

    story_rows = []
    for index, scene in enumerate(SCENES, start=1):
        story_rows.append(
            {
                "sequence": index,
                "scene_id": scene["id"],
                "section": scene["section"],
                "presenter": scene["presenter"],
                "title": scene["title"],
                "frame": str((FRAMES / f"{scene['id']}.png").relative_to(PROJECT)),
                "visual_assets": " | ".join(scene["visual"]["items"]),
                "narration_words": word_count(scene["narration"]),
            }
        )
    with (OUTPUT / "storyboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(story_rows[0].keys()))
        writer.writeheader()
        writer.writerows(story_rows)
    storyboard_text = json.dumps(story_rows, indent=2) + "\n"
    (OUTPUT / "storyboard.json").write_text(storyboard_text, encoding="utf-8")

    summary = {
        "scene_count": len(SCENES),
        "narration_words": sum(word_count(scene["narration"]) for scene in SCENES),
        "max_scene_characters": max(len(scene["narration"]) for scene in SCENES),
        "narration_engine": OFFLINE_ENGINE,
        "narration_voice": OFFLINE_VOICE,
        "narration_locale": OFFLINE_LOCALE,
        "narration_rate_wpm": OFFLINE_RATE_WPM,
        "external_speech_api_used": False,
    }
    print(json.dumps(summary, indent=2))


def generate_offline_narration(force_scenes: set[str] | None = None) -> None:
    """Generate each narration segment locally with macOS say."""
    force_scenes = force_scenes or set()
    known_scene_ids = {scene["id"] for scene in SCENES}
    unknown = sorted(force_scenes - known_scene_ids)
    if unknown:
        raise ValueError(f"Unknown forced narration scene IDs: {unknown}")
    ffmpeg = ffmpeg_path()
    for directory in (AUDIO_AIFF, AUDIO_SEGMENTS, TMP):
        directory.mkdir(parents=True, exist_ok=True)
    provenance_rows: list[dict[str, Any]] = []
    for scene in SCENES:
        text_path = TMP / f"{scene['id']}.txt"
        aiff_path = AUDIO_AIFF / f"{scene['id']}.aiff"
        wav_path = AUDIO_SEGMENTS / f"{scene['id']}.wav"
        text_path.write_text(scene["narration"] + "\n", encoding="utf-8")
        # macOS say needs access to the local speech service. Preserve a
        # previously generated nonempty AIFF so a later sandboxed conversion
        # cannot replace valid narration with an empty container.
        forced = scene["id"] in force_scenes
        if forced or not aiff_path.exists() or aiff_path.stat().st_size <= 4096:
            fresh_aiff = TMP / f"{scene['id']}.fresh.aiff"
            fresh_aiff.unlink(missing_ok=True)
            run(
                [
                    "/usr/bin/say",
                    "-v",
                    OFFLINE_VOICE,
                    "-r",
                    str(OFFLINE_RATE_WPM),
                    "-o",
                    str(fresh_aiff),
                    "-f",
                    str(text_path),
                ]
            )
            if not fresh_aiff.exists() or fresh_aiff.stat().st_size <= 4096:
                raise RuntimeError(f"Forced narration generation produced no usable audio: {fresh_aiff}")
            fresh_aiff.replace(aiff_path)
        run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(aiff_path),
                "-c:a",
                "pcm_s16le",
                "-ar",
                "24000",
                "-ac",
                "1",
                str(wav_path),
            ]
        )
        provenance_rows.append(
            {
                "scene_id": scene["id"],
                "forced_regeneration": forced,
                "narration_text": scene["narration"],
                "narration_text_sha256": hashlib.sha256(scene["narration"].encode("utf-8")).hexdigest(),
                "text_file_sha256": sha256(text_path),
                "aiff_sha256": sha256(aiff_path),
                "wav_sha256": sha256(wav_path),
                "audio_duration_seconds": round(wav_duration(wav_path), 6),
            }
        )
    NARRATION_PROVENANCE.write_text(json.dumps(provenance_rows, indent=2) + "\n", encoding="utf-8")
    durations = [wav_duration(AUDIO_SEGMENTS / f"{scene['id']}.wav") for scene in SCENES]
    total = sum(durations)
    result = {
        "engine": OFFLINE_ENGINE,
        "voice": OFFLINE_VOICE,
        "locale": OFFLINE_LOCALE,
        "rate_wpm": OFFLINE_RATE_WPM,
        "external_speech_api_used": False,
        "scene_count": len(SCENES),
        "duration_seconds": round(total, 6),
    }
    print(json.dumps(result, indent=2))
    if not 300.0 <= total <= 480.0:
        raise SystemExit(f"Offline narration duration is outside 300-480 seconds: {total:.3f}")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def timestamp(seconds: float, srt: bool = False) -> str:
    millis = max(0, round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def split_caption_units(text: str) -> list[str]:
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= 150:
            units.append(sentence)
            continue
        clauses = [chunk.strip() for chunk in re.split(r"(?<=[,;:])\s+", sentence) if chunk.strip()]
        buffer = ""
        for clause in clauses:
            proposed = f"{buffer} {clause}".strip()
            if buffer and len(proposed) > 150:
                units.append(buffer)
                buffer = clause
            else:
                buffer = proposed
        if buffer:
            units.append(buffer)
    return units


def write_captions(durations: list[float]) -> list[dict[str, Any]]:
    cues: list[tuple[float, float, str]] = []
    actual_rows: list[dict[str, Any]] = []
    cursor = 0.0
    for index, (scene, duration) in enumerate(zip(SCENES, durations), start=1):
        start = cursor
        units = split_caption_units(scene["narration"])
        weights = [max(1, len(re.findall(r"\w+", unit))) for unit in units]
        total_weight = sum(weights)
        local = start
        for unit_index, (unit, weight) in enumerate(zip(units, weights)):
            if unit_index == len(units) - 1:
                cue_end = start + duration
            else:
                cue_end = local + duration * weight / total_weight
            cues.append((local, cue_end, unit))
            local = cue_end
        cursor += duration
        actual_rows.append(
            {
                "sequence": index,
                "scene_id": scene["id"],
                "section": scene["section"],
                "presenter": scene["presenter"],
                "title": scene["title"],
                "narration_words": word_count(scene["narration"]),
                "audio_duration_seconds": round(duration, 6),
                "start_seconds": round(start, 6),
                "end_seconds": round(cursor, 6),
            }
        )

    with CAPTIONS.open("w", encoding="utf-8") as handle:
        for index, (start, end, text) in enumerate(cues, start=1):
            handle.write(f"{index}\n{timestamp(start, srt=True)} --> {timestamp(end, srt=True)}\n{text}\n\n")
    with (OUTPUT / "storyboard_actual.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(actual_rows[0].keys()))
        writer.writeheader()
        writer.writerows(actual_rows)
    (OUTPUT / "storyboard_actual.json").write_text(json.dumps(actual_rows, indent=2) + "\n", encoding="utf-8")
    return actual_rows


def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio-ffmpeg is required to assemble the video") from exc


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=capture, check=False)
    if result.returncode != 0:
        detail = result.stderr if capture else f"command exited {result.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}")
    return result


def assemble() -> None:
    ffmpeg = ffmpeg_path()
    audio_paths = [AUDIO_SEGMENTS / f"{scene['id']}.wav" for scene in SCENES]
    missing = [str(path) for path in audio_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing TTS audio: {missing}")
    durations = [wav_duration(path) for path in audio_paths]
    actual_rows = write_captions(durations)

    scene_paths: list[Path] = []
    for scene, audio_path, duration in zip(SCENES, audio_paths, durations):
        frame_path = FRAMES / f"{scene['id']}.png"
        scene_path = SCENE_VIDEOS / f"{scene['id']}.mp4"
        scene_paths.append(scene_path)
        fade_out = max(0.0, duration - 0.25)
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-i",
            str(frame_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            (
                f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=white,format=yuv420p,"
                f"fade=t=in:st=0:d=0.20,fade=t=out:st={fade_out:.6f}:d=0.20[v]"
            ),
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-t",
            f"{duration:.6f}",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(scene_path),
        ]
        run(command)

    concat_file = TMP / "video_concat.txt"
    concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in scene_paths), encoding="utf-8")
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(NO_SUBS_VIDEO),
        ]
    )

    audio_concat = TMP / "audio_concat.txt"
    audio_concat.write_text("".join(f"file '{path.as_posix()}'\n" for path in audio_paths), encoding="utf-8")
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(audio_concat),
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            str(FULL_NARRATION),
        ]
    )

    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(NO_SUBS_VIDEO),
            "-i",
            str(CAPTIONS),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-map",
            "1:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
            "-metadata:s:s:0",
            "title=English",
            "-movflags",
            "+faststart",
            str(FINAL_VIDEO),
        ]
    )

    build_manifest = {
        "schema_version": "1.0.0",
        "system_generated_voice_disclosure": True,
        "narration": {
            "engine": OFFLINE_ENGINE,
            "offline": True,
            "voice": OFFLINE_VOICE,
            "locale": OFFLINE_LOCALE,
            "rate_wpm": OFFLINE_RATE_WPM,
            "source_format": "AIFF-C generated by macOS say",
            "assembly_format": "WAV PCM 16-bit mono 24 kHz",
            "external_speech_api_used": False,
        },
        "scene_count": len(SCENES),
        "caption_cue_count": sum(len(split_caption_units(scene["narration"])) for scene in SCENES),
        "narration_word_count": sum(word_count(scene["narration"]) for scene in SCENES),
        "planned_audio_duration_seconds": round(sum(durations), 6),
        "media": {
            "video": str(FINAL_VIDEO.relative_to(PROJECT)),
            "captions": str(CAPTIONS.relative_to(PROJECT)),
            "narration": str(FULL_NARRATION.relative_to(PROJECT)),
        },
        "storyboard_actual": actual_rows,
        "public_sources": PUBLIC_SOURCES,
        "local_sources": LOCAL_SOURCES,
    }
    (OUTPUT / "video_manifest.json").write_text(json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video": str(FINAL_VIDEO), "audio_seconds": sum(durations)}, indent=2))


def parse_probe(text: str) -> dict[str, Any]:
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    duration = None
    if duration_match:
        duration = int(duration_match.group(1)) * 3600 + int(duration_match.group(2)) * 60 + float(duration_match.group(3))
    video_line = next((line.strip() for line in text.splitlines() if "Video:" in line), "")
    audio_line = next((line.strip() for line in text.splitlines() if "Audio:" in line), "")
    subtitle_line = next((line.strip() for line in text.splitlines() if "Subtitle:" in line), "")
    resolution_match = re.search(r"\b(\d{3,5})x(\d{3,5})\b", video_line)
    fps_match = re.search(r"([\d.]+) fps", video_line)
    return {
        "duration_seconds": duration,
        "video_stream": video_line,
        "audio_stream": audio_line,
        "subtitle_stream": subtitle_line,
        "width": int(resolution_match.group(1)) if resolution_match else None,
        "height": int(resolution_match.group(2)) if resolution_match else None,
        "fps": float(fps_match.group(1)) if fps_match else None,
        "video_codec": "h264" if "Video: h264" in video_line else None,
        "audio_codec": "aac" if "Audio: aac" in audio_line else None,
        "subtitle_codec": "mov_text" if "Subtitle: mov_text" in subtitle_line else None,
    }


def build_qa_montage(frame_paths: list[Path]) -> Path:
    """Build a fresh 3x2 contact sheet from the validated representative frames."""
    if len(frame_paths) != 6:
        raise ValueError(f"Expected 6 representative frames, found {len(frame_paths)}")
    montage = Image.new("RGB", (1328, 552), "#F3F4F6")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for index, frame_path in enumerate(frame_paths):
        column = index % 3
        row = index // 3
        left = 11 + column * 439
        top = 10 + row * 270
        with Image.open(frame_path) as source:
            frame = source.convert("RGB").resize((426, 240), Image.Resampling.LANCZOS)
        montage.paste(frame, (left, top))
        draw.rectangle((left, top, left + 425, top + 239), outline="#AEB4BD", width=1)
        label = frame_path.name
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        draw.text((left + (426 - label_width) / 2, top + 244), label, fill="#111111", font=font)
    montage_path = QA_FRAMES / "qa_montage.png"
    montage.save(montage_path, "PNG", optimize=True)
    return montage_path


def validate(manual_status: str) -> None:
    if not FINAL_VIDEO.exists():
        raise FileNotFoundError(FINAL_VIDEO)
    ffmpeg = ffmpeg_path()
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(FINAL_VIDEO)], text=True, capture_output=True, check=False)
    metadata = parse_probe(probe.stderr)

    try:
        import imageio_ffmpeg

        frame_count, scanned_seconds = imageio_ffmpeg.count_frames_and_secs(str(FINAL_VIDEO))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Could not count frames") from exc

    decode = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(FINAL_VIDEO), "-f", "null", "-"],
        text=True,
        capture_output=True,
        check=False,
    )
    duration = float(metadata["duration_seconds"] or scanned_seconds)
    provenance_rows = (
        json.loads(NARRATION_PROVENANCE.read_text(encoding="utf-8"))
        if NARRATION_PROVENANCE.exists()
        else []
    )
    workflow_scene = next(scene for scene in SCENES if scene["id"] == "13_demo_workflow")
    workflow_provenance = next(
        (row for row in provenance_rows if row.get("scene_id") == workflow_scene["id"]),
        {},
    )
    caption_text = CAPTIONS.read_text(encoding="utf-8") if CAPTIONS.exists() else ""
    workflow_text_sha256 = hashlib.sha256(workflow_scene["narration"].encode("utf-8")).hexdigest()
    workflow_narration_provenance_current = (
        workflow_provenance.get("forced_regeneration") is True
        and workflow_provenance.get("narration_text") == workflow_scene["narration"]
        and workflow_provenance.get("narration_text_sha256") == workflow_text_sha256
        and bool(workflow_provenance.get("aiff_sha256"))
        and bool(workflow_provenance.get("wav_sha256"))
    )
    representative_times = [5.0, duration * 0.20, duration * 0.40, duration * 0.60, duration * 0.80, max(5.0, duration - 5.0)]
    frame_files: list[Path] = []
    for index, seconds in enumerate(representative_times, start=1):
        out = QA_FRAMES / f"frame_{index:02d}_{seconds:07.2f}s.jpg"
        run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{seconds:.3f}",
                "-i",
                str(FINAL_VIDEO),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out),
            ]
        )
        frame_files.append(out)
    montage_path = build_qa_montage(frame_files)
    frame_paths = [str(path.relative_to(PROJECT)) for path in frame_files]

    checks = {
        "duration_300_to_480_seconds": 300.0 <= duration <= 480.0,
        "resolution_1280x720": metadata["width"] == WIDTH and metadata["height"] == HEIGHT,
        "video_codec_h264": metadata["video_codec"] == "h264",
        "audio_codec_aac": metadata["audio_codec"] == "aac",
        "caption_stream_mov_text": metadata["subtitle_codec"] == "mov_text",
        "frame_count_nontrivial": frame_count > FPS * 300,
        "decode_completed_without_errors": decode.returncode == 0 and not decode.stderr.strip(),
        "narration_disclosure_present": SCENES[0]["narration"].lower().startswith("this video uses offline macos system-generated"),
        "external_speech_api_not_used": True,
        "all_15_slide_renders_used": all(slide(number) in {item for scene in SCENES for item in scene["visual"]["items"]} for number in range(1, 16)),
        "captions_present": CAPTIONS.exists() and CAPTIONS.stat().st_size > 1000,
        "workflow_narration_declares_four_steps": workflow_scene["narration"].startswith("The workspace has four steps."),
        "workflow_caption_declares_four_steps": "The workspace has four steps." in caption_text,
        "workflow_caption_has_no_three_step_text": "three steps" not in caption_text.casefold(),
        "workflow_narration_provenance_current": workflow_narration_provenance_current,
    }
    artifacts = [
        FINAL_VIDEO,
        CAPTIONS,
        OUTPUT / "narration_script.md",
        OUTPUT / "storyboard.csv",
        OUTPUT / "storyboard.json",
        OUTPUT / "storyboard_actual.csv",
        OUTPUT / "storyboard_actual.json",
        OUTPUT / "video_manifest.json",
        NARRATION_PROVENANCE,
    ]
    report = {
        "schema_version": "1.0.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "manual_visual_inspection": manual_status,
        "checks": checks,
        "media": {
            **metadata,
            "duration_seconds_scanned": round(float(scanned_seconds), 6),
            "frame_count": int(frame_count),
            "file_bytes": FINAL_VIDEO.stat().st_size,
            "sha256": sha256(FINAL_VIDEO),
        },
        "representative_frames": frame_paths,
        "representative_frame_checksums": [
            {
                "path": str(path.relative_to(PROJECT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (PROJECT / relative for relative in frame_paths)
        ],
        "representative_frame_montage": {
            "path": str(montage_path.relative_to(PROJECT)),
            "bytes": montage_path.stat().st_size,
            "sha256": sha256(montage_path),
        },
        "artifact_checksums": [
            {
                "path": str(path.relative_to(PROJECT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
            if path.exists()
        ],
        "ffmpeg_binary": Path(ffmpeg).name,
        "probe_excerpt": {
            "video": metadata["video_stream"],
            "audio": metadata["audio_stream"],
            "subtitle": metadata["subtitle_stream"],
        },
        "decode_stderr": decode.stderr,
        "workflow_narration_provenance": workflow_provenance,
    }
    (OUTPUT / "qa_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "narrate-offline", "assemble", "validate"))
    parser.add_argument(
        "--manual-status",
        default="PENDING",
        help="Recorded visual-inspection status for the validate command.",
    )
    parser.add_argument(
        "--force-scene-audio",
        action="append",
        default=[],
        help="Regenerate only the named scene AIFF before WAV conversion; repeat for multiple scenes.",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        write_prepare_outputs()
    elif args.command == "narrate-offline":
        generate_offline_narration(set(args.force_scene_audio))
    elif args.command == "assemble":
        assemble()
    else:
        validate(args.manual_status)


if __name__ == "__main__":
    main()
