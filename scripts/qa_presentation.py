#!/usr/bin/env python3
"""Create reproducible montages and QA metadata for the management deck."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "presentation"
PPTX = OUT / "ad_safety_management_presentation.pptx"
BUILDER = ROOT / "scripts" / "build_presentation.mjs"
VALIDATION = OUT / "presentation_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xml_text(blob: bytes) -> str:
    root = ET.fromstring(blob)
    return "\n".join((node.text or "") for node in root.iter() if node.tag.endswith("}t"))


def numeric_xml_key(path: str) -> int:
    match = re.search(r"(\d+)\.xml$", path)
    return int(match.group(1)) if match else 0


def build_montage(paths: list[Path], output: Path) -> None:
    if len(paths) != 15:
        raise RuntimeError(f"Expected 15 slide images for {output.name}, found {len(paths)}")
    canvas = Image.new("RGB", (2096, 781), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        with Image.open(path) as raw:
            image = raw.convert("RGB").resize((400, 225), Image.Resampling.LANCZOS)
        column = index % 5
        row = index // 5
        left = 16 + column * 416
        top = 26 + row * 251
        canvas.paste(image, (left, top))
        draw.rectangle((left, top, left + 399, top + 224), outline=(184, 188, 196), width=1)
    canvas.save(output, "PNG", optimize=True)


def render_pptx_independently() -> list[Path]:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    pdftoppm = shutil.which("pdftoppm")
    if not soffice or not pdftoppm:
        raise FileNotFoundError("Independent rendering requires LibreOffice and Poppler on PATH")

    output_dir = OUT / "rendered_pptx"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_render in output_dir.glob("*.png"):
        old_render.unlink()

    with tempfile.TemporaryDirectory(prefix="ad-safety-pptx-render-") as temp_root:
        temp_dir = Path(temp_root)
        conversion = subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(temp_dir),
                str(PPTX),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if conversion.returncode != 0:
            raise RuntimeError(f"LibreOffice render failed: {conversion.stderr[-2000:]}")
        pdf_path = temp_dir / f"{PPTX.stem}.pdf"
        if not pdf_path.is_file():
            raise FileNotFoundError(f"LibreOffice did not create {pdf_path}")
        raster = subprocess.run(
            [pdftoppm, "-png", "-r", "144", str(pdf_path), str(temp_dir / "slide")],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if raster.returncode != 0:
            raise RuntimeError(f"Poppler render failed: {raster.stderr[-2000:]}")
        source_renders = sorted(temp_dir.glob("slide-*.png"), key=lambda path: int(path.stem.split("-")[-1]))
        if len(source_renders) != 15:
            raise RuntimeError(f"Expected 15 independently rendered slides, found {len(source_renders)}")
        rendered_paths: list[Path] = []
        for index, source in enumerate(source_renders, start=1):
            destination = output_dir / f"slide-{index:02d}.png"
            shutil.copy2(source, destination)
            rendered_paths.append(destination)
    return rendered_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-status",
        required=True,
        help="Recorded result after visual inspection of both montages and changed full-size slides.",
    )
    args = parser.parse_args()
    if not args.manual_status.upper().startswith(("PASS:", "PENDING:")):
        raise ValueError("Manual status must start with PASS: or PENDING:")

    artifact_paths = [OUT / "rendered" / f"slide-{index:02d}.png" for index in range(1, 16)]
    pptx_paths = render_pptx_independently()
    if any(not path.is_file() for path in artifact_paths + pptx_paths):
        raise FileNotFoundError("One or more slide renders are missing")
    artifact_montage = OUT / "artifact_tool_montage.png"
    pptx_montage = OUT / "pptx_montage.png"
    build_montage(artifact_paths, artifact_montage)
    build_montage(pptx_paths, pptx_montage)

    with zipfile.ZipFile(PPTX) as archive:
        names = set(archive.namelist())
        slide_names = sorted(
            (
                name
                for name in names
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ),
            key=numeric_xml_key,
        )
        note_names = sorted(
            (
                name
                for name in names
                if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")
            ),
            key=numeric_xml_key,
        )
        note_texts = [xml_text(archive.read(name)) for name in note_names]
        archive_error = archive.testzip()

    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    layout_paths = [OUT / "rendered" / f"slide-{index:02d}.layout.json" for index in range(1, 16)]
    layout_out_of_bounds: list[str] = []
    for layout_path in layout_paths:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        for element in payload.get("elements", []):
            left, top, width, height = element.get("bbox", [0, 0, 0, 0])
            if left < -0.01 or top < -0.01 or left + width > 1280.01 or top + height > 720.01:
                layout_out_of_bounds.append(f"{layout_path.name}:{element.get('id')}")

    presenters = [
        "Swarnaditya Maitra",
        "Vijay Agnihotri",
        "Myetchae Thu",
        "Bickramjit Basu",
    ]
    qa = {
        "schema_version": "1.2.0",
        "artifact": {
            "path": str(PPTX.relative_to(ROOT)),
            "bytes": PPTX.stat().st_size,
            "sha256": sha256_file(PPTX),
            "slide_count": len(slide_names),
            "reopened_slide_count": validation.get("reopened_slide_count"),
            "zip_integrity": (
                "PASS: unzip-compatible archive test reported no errors"
                if archive_error is None
                else f"FAIL: {archive_error}"
            ),
        },
        "builder": {
            "path": str(BUILDER.relative_to(ROOT)),
            "sha256": sha256_file(BUILDER),
            "tool": "@oai/artifact-tool",
            "selected_codex_grid_templates": validation.get("selected_grid_templates", []),
        },
        "validation": {
            "path": str(VALIDATION.relative_to(ROOT)),
            "sha256": sha256_file(VALIDATION),
            "recorded_pptx_sha256_matches_artifact": (
                validation.get("pptx_sha256") == sha256_file(PPTX)
            ),
        },
        "rendering": {
            "artifact_tool_png_count": len(artifact_paths),
            "reopened_pptx_png_count": len(pptx_paths),
            "artifact_tool_montage": {
                "path": str(artifact_montage.relative_to(ROOT)),
                "sha256": sha256_file(artifact_montage),
            },
            "independent_pptx_montage": {
                "path": str(pptx_montage.relative_to(ROOT)),
                "sha256": sha256_file(pptx_montage),
            },
            "manual_visual_inspection": args.manual_status,
        },
        "speaker_notes": {
            "notes_slide_count": len(note_names),
            "sources_block_count": sum("[Sources]" in text for text in note_texts),
            "all_slides_have_sources": all("[Sources]" in text for text in note_texts),
            "all_local_sources_portable": all(
                "/Users/" not in text
                and ("Project-relative:" in text or "https://" in text)
                for text in note_texts
            ),
            "absolute_project_path_count": sum(text.count("/Users/") for text in note_texts),
            "pipeline_py_mention_count": sum(
                text.count("src/ad_safety/pipeline.py") for text in note_texts
            ),
            "inference_py_source_slides": [
                index
                for index, text in enumerate(note_texts, start=1)
                if "src/ad_safety/inference.py" in text
            ],
            "presenters": [name for name in presenters if any(name in text for text in note_texts)],
        },
        "layout_tests": {
            "semantic_regions_checked": validation.get("semantic_overlap_and_bounds_test", {}).get(
                "checked_regions"
            ),
            "semantic_overlap_issues": len(
                [
                    issue
                    for issue in validation.get("semantic_overlap_and_bounds_test", {}).get("issues", [])
                    if issue.get("type") == "overlap"
                ]
            ),
            "semantic_out_of_bounds_issues": len(layout_out_of_bounds),
            "padded_export_overflow_test": (
                "PASS: no exported element bounding box exceeds the 1280x720 canvas"
                if not layout_out_of_bounds
                else f"FAIL: {layout_out_of_bounds}"
            ),
            "pptx_zip_integrity": "PASS" if archive_error is None else "FAIL",
            "artifact_tool_reopen_slide_count": validation.get("reopened_slide_count"),
        },
        "mvp_roadmap_compliance": validation.get("mvp_roadmap_plan", {}),
        "measured_facts_used": validation.get("measured_facts_used", {}),
        "semantic_corrections": [
            "Safe precision 1.0 is separate from the Safe-ad false-flag rate, which equals one minus Safe recall.",
            "The 0.90 validation-recall value is the tuning floor, not the proposal acceptance target.",
            "The proposal per-class restricted recall target above 0.95 fails on the untouched test.",
            "The 71.527 ms result is classifier-path p95; full end-to-end p95 remains unassessed.",
            "The 10-week roadmap and staffing are recommendations, not measured commitments.",
        ],
    }
    (OUT / "qa_report.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
