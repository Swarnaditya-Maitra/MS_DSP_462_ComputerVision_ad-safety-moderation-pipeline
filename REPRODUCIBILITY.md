# Reproducibility contract

This project targets functional reproduction. In plain terms, a supported computer should be able to clone the repository, provision the pinned model snapshot, pass preflight, run a real analysis, and pass the automated tests. It does not promise byte-for-byte identical caches, rendered files, timing, or floating-point scores on every machine.

[USER_MANUAL.md](USER_MANUAL.md) is the sole command-by-command operating guide. This document defines the narrower reproduction contract and the boundary between a normal public-clone check and the stronger full-local evidence audit.

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
- The formal and external diagnostic collections have complete registries containing local SHA-256 values and source metadata. [DATASETS.md](DATASETS.md) is the canonical record of image counts, provenance, release status, and attribution requirements.
- Saved metrics, predictions, embeddings, charts, demo evidence, and validation records are under the single canonical `outputs/` tree.
- The executed notebook, original proposal, final report, PowerPoint presentation, and narrated video are tracked with the project.

The Python file is a top-level compatibility lock, not a hash lock for every operating-system-specific transitive wheel. Pip can select different low-level wheels on macOS and Windows. Hardware libraries can also produce small floating-point differences. Those limits make a universal byte-for-byte promise false.

## Core functional check

```text
python scripts/bootstrap.py --profile core
python scripts/smoke_test.py
python -m pytest -q
```

The bootstrap script stops before installation on an unsupported interpreter or host. It installs the pinned packages, downloads only the exact ViT revision needed by the core app, and runs preflight. The smoke test sends a generated image through the real FastAPI and ViT inference path with detector, OCR, and explanation disabled. [USER_MANUAL.md](USER_MANUAL.md) provides the complete macOS and Windows setup, expected output, API, UI, negative-path, and shutdown steps.

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

## Dataset and training boundary

The included trained heads make a new clone runnable without retraining or raw training images. Exact training reproduction is a separate workflow that requires the recorded upstream sources, pinned classifier backbones, complete registry checks, and a fresh training run in a separate clone or worktree.

That workflow depends on third-party data remaining available under suitable terms. A successful download does not grant redistribution rights, and a future rebuild is not guaranteed to reproduce the saved source bytes. [DATASETS.md](DATASETS.md) is the sole detailed inventory and provenance record. [USER_MANUAL.md](USER_MANUAL.md) gives the rebuild and verification commands. A new run must not be reported as the saved course run unless its checksums and validation records match.

## Course-deliverable verification

The repository includes the final executed notebook, technical synopsis PDF, PowerPoint deck, narrated MP4, captions, representative demo evidence, and saved validation records. A fresh public clone first runs the portable release audit:

```text
python scripts/validate_release.py
```

The extended validator is the full-local evidence audit. It retains raw-pixel regeneration and full pinned-backbone checks, so the complete registered data and full model profile must exist before it runs:

```text
python scripts/bootstrap.py --profile full
python scripts/build_capstone_dataset.py
python scripts/validate_deliverables.py
```

The extended audit needs Poppler, LibreOffice, and FFmpeg. It builds the static book in a temporary checkout, freshly renders the tracked PDF and PPTX, and extracts representative frames from the tracked MP4. It validates page and slide counts, dimensions, nonblank output, structure, facts, hashes, captions, audio signal, and media decodability without depending on ignored book HTML, saved page or slide renders, narration segments, or QA-frame caches.

The tracked final MP4 and SRT can be validated cross-platform with FFmpeg. Rebuilding the original offline narration with `python scripts/build_demo_video.py narrate-offline` requires macOS `say`. Windows can validate and use the final video, but cannot reproduce identical macOS voice bytes unless a different TTS workflow is used and disclosed as a new build.

These final artifacts are reproducible at the functional level. Different office suites, fonts, TTS engines, encoders, and operating systems can change rendered pixels, audio timing, compression bytes, and machine-specific timings. The saved hashes identify the validated course run; a rebuild that differs must be reported as a new run.

## Tracked and generated boundary

Public Git tracks the source code, complete registries, cleared standalone dataset files, trained policy heads, and final coursework deliverables. Pretrained backbone snapshots, package environments, Python caches, render scratch directories, duplicate aliases, temporary audio segments, and other rebuildable intermediates are not tracked. Bootstrap stores pinned model snapshots in the ignored repository-local cache.

Final course records can contain bounded derived views of examples that are not released as standalone files. Those views do not clear the source material for extraction or reuse. Read [DATASETS.md](DATASETS.md) for the detailed data boundary and [NOTICE.md](NOTICE.md) before republishing any project artifact.

External technical and policy references are maintained once in [SOURCES.md](SOURCES.md).
