# Reproducibility contract

This project targets functional reproduction. In plain terms, a supported computer should be able to clone the repository, provision the pinned model snapshot, pass preflight, run a real analysis, and pass the automated tests. It does not promise that every machine will produce byte-for-byte identical caches, timing, or floating-point scores.

## Supported systems

| Item | Supported contract |
|---|---|
| Operating system | Apple Silicon macOS, or 64-bit Windows 10/11 |
| Python | 64-bit CPython 3.10 in an isolated virtual environment |
| Core memory | 16 GB RAM recommended |
| Core disk | Several hundred MB for the ViT snapshot, plus the Python environment |
| Full profile | Several GB and Tesseract OCR; adds ResNet-50 and Grounding DINO |
| First setup | Internet access to PyPI and Hugging Face unless the exact caches already exist |

Intel macOS and Linux are outside this tested contract. A clone may still work there, but this repository does not label those paths as verified.

## What is pinned

- Every top-level Python package has an exact version in `requirements-lock.txt`.
- Each pretrained model has an immutable Hugging Face revision in `scripts/download_models.py` and `models/pretrained_model_manifest.json`.
- The two trained policy heads have exact byte counts and SHA-256 digests in `models/trained_head_manifest.json`.
- The application repeats the trained-head integrity check before pickle-based deserialization.
- Thresholds and policy rules are versioned in `configs/policy.yaml`.
- The 288-image formal dataset and 27-image external candidate set have complete registries containing local SHA-256 values and source metadata. Public Git includes only the 72 project-generated formal images and 26 retained Wikimedia images cleared for this release.
- Saved metrics, predictions, embeddings, charts, demo evidence, and validation records are under the single canonical `outputs/` tree.
- The executed notebook, original proposal, final report, PowerPoint presentation, and narrated video are tracked with the project.

The Python file is a top-level compatibility lock, not a hash lock for every operating-system-specific transitive wheel. Pip can select different low-level wheels on macOS and Windows. Hardware libraries can also produce small floating-point differences. Those limits make a universal byte-for-byte promise false.

## One-command setup after activating the environment

```text
python scripts/bootstrap.py --profile core
python scripts/smoke_test.py
python -m pytest -q
```

The bootstrap script stops before installation on an unsupported interpreter or host. It installs the pinned packages, downloads only the exact ViT revision needed by the core app, and runs preflight. The smoke test sends a generated image through the real FastAPI and ViT inference path with detector, OCR, and explanation disabled.

For all optional components:

```text
python scripts/bootstrap.py --profile full
python scripts/preflight.py --profile full
```

The full preflight also requires the baseline head, ResNet-50 snapshot, Grounding DINO snapshot, and a working `tesseract` command.

For an already provisioned machine with no network access:

```text
python scripts/bootstrap.py --profile core --offline --skip-install
```

This succeeds only when the pinned dependencies are already installed and the exact model cache is present.

## Dataset verification and independent rebuild

The included trained heads make a new clone runnable without retraining or raw training images. First verify whether the checkout is the full canonical local project or the public release:

```text
python scripts/validate_release.py
```

The full canonical local project contains all 288 formal images. Verify every one against the tracked registry before retraining:

```text
python scripts/build_capstone_dataset.py --verify-only
```

To reproduce the training workflow instead, provision the two backbones and retrain on CPU:

```text
python scripts/download_models.py --skip-grounding-dino
python scripts/train_and_evaluate.py --device cpu --batch-size 8
```

The public release contains 72 project-generated formal images. Running `python scripts/build_capstone_dataset.py` without `--verify-only` reconstructs the other formal data from the recorded sources and then rebuilds the complete set. That path depends on third-party data still being available under suitable terms. It can reproduce the method, but a future download is not guaranteed to reproduce the tracked source bytes. Downloading an image does not grant redistribution rights. Do not report new metrics as the saved course-run metrics unless the checksums and validation records match.

## Course-deliverable verification

The repository includes the final executed notebook, technical synopsis PDF, PowerPoint deck, narrated MP4, captions, representative demo evidence, and saved validation records. A fresh public clone first runs the portable release audit, which does not require the 216 local-only third-party formal images:

```text
python scripts/validate_release.py
```

The extended validator is the full-local evidence audit. It retains the 288-image raw-pixel regeneration and full pinned-backbone checks, so reconstruct the local-only images and provision the full profile before running it:

```text
python scripts/bootstrap.py --profile full
python scripts/build_capstone_dataset.py
python scripts/validate_deliverables.py
```

The extended audit needs Poppler, LibreOffice, and FFmpeg. It builds the static book in a temporary checkout, freshly renders the tracked PDF and PPTX, and extracts representative frames from the tracked MP4. It validates page and slide counts, dimensions, nonblank output, structure, facts, hashes, captions, audio signal, and media decodability without depending on ignored book HTML, saved page or slide renders, narration segments, or QA-frame caches.

The tracked final MP4 and SRT can be validated cross-platform with FFmpeg. Rebuilding the original offline narration with `python scripts/build_demo_video.py narrate-offline` requires macOS `say`. Windows can validate and use the final video, but cannot reproduce identical macOS voice bytes unless a different TTS workflow is used and disclosed as a new build.

These final artifacts are reproducible at the functional level. Different office suites, fonts, TTS engines, encoders, and operating systems can change rendered pixels, audio timing, compression bytes, and machine-specific timings. The saved hashes identify the validated course run; a rebuild that differs must be reported as a new run.

## Tracked and generated boundary

Public Git tracks 72 project-generated financial images and 26 retained Wikimedia images as standalone dataset files, complete registries for all 315 local candidates, and all final coursework deliverables. The 216 standalone third-party formal dataset files remain present only in the canonical local project until their redistribution rights are cleared. Final course records may contain bounded derived views of selected examples; those views do not clear the source images for extraction or reuse. Pretrained backbone snapshots, package environments, Python caches, render scratch directories, duplicate aliases, temporary audio segments, and other rebuildable intermediates are not tracked. Bootstrap downloads pinned model snapshots into the ignored repository-local cache.

Dataset inclusion does not prove that every upstream item has complete redistribution evidence. The formal registry records repository-level licenses and explicitly identifies missing per-image provenance. The Wikimedia registry records item-level source and attribution fields. Contributor names also appear in the report, presentation, notebook, and video. See `NOTICE.md` before republishing any of those artifacts.

## Sources

- Python virtual environments: https://docs.python.org/3.10/library/venv.html
- Hugging Face Hub download API: https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- ViT model card: https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
- ResNet-50 model card: https://huggingface.co/timm/resnet50.a1_in1k
- Grounding DINO Tiny model card: https://huggingface.co/IDEA-Research/grounding-dino-tiny
