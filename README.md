# Ad Safety Visual Moderation Pilot

[![Cross-platform verification](https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline/actions/workflows/cross-platform.yml/badge.svg)](https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline/actions/workflows/cross-platform.yml)

This repository contains the complete MS_DSP_462 computer-vision project for triaging static ad creatives. It includes the runnable application, publicly releasable dataset artifacts and complete registries, trained policy heads, executed notebook, saved evidence, final report, PowerPoint presentation, narrated demonstration, and the scripts used to build and validate them. The inference pipeline combines:

1. a frozen ViT-B/16 visual backbone with a four-class policy head;
2. a ResNet-50 comparison baseline;
3. optional Grounding DINO Tiny object evidence;
4. local Tesseract OCR cues;
5. occlusion sensitivity for qualitative inspection; and
6. a deterministic policy layer that returns `APPROVE`, `REVIEW`, or `BLOCK`.

The four visual labels are `safe`, `firearms`, `explosives`, and `financial_promotion`. A financial-promotion label means visible promotion cues were found. It does not establish fraud, legality, or regulatory status. Financial cases route to human review.

## Supported reproduction contract

I support the clone-ready flow on these hosts:

- CPython 3.10 on Apple Silicon macOS (`arm64`); or
- 64-bit CPython 3.10 on Windows 10 or 11 (`AMD64` or `x86_64`).

Intel macOS, Linux, 32-bit Python, and other Python versions are outside this validated contract. Exact universal reproduction is not possible because Python wheels, upstream model hosting, source availability, operating systems, and hardware can change. The repository reduces those variables with pinned top-level Python packages, pinned model revisions, included trained heads, SHA-256 checks, an offline preflight, and a real model smoke test. It does not claim identical latency or floating-point results on every machine.

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the formal functional-reproduction contract. Use [USER_MANUAL.md](USER_MANUAL.md) for prerequisites, manual fallback commands, Tesseract, independent training, Streamlit, FastAPI, health and analysis checks, negative-path fixtures, troubleshooting, and shutdown.

## Clone-ready quick start

The `core` profile is the shortest runnable path. It uses the included, hash-pinned ViT policy head and downloads only the pinned ViT backbone snapshot. The download is several hundred MB.

### Apple Silicon macOS Terminal

```bash
git clone https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline.git
cd MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline
python3.10 -m venv .venv
source .venv/bin/activate
python scripts/bootstrap.py --profile core
python scripts/preflight.py --profile core
python scripts/smoke_test.py
```

Start Streamlit:

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Or start the API in that terminal instead:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### Windows PowerShell

```powershell
git clone https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline.git
Set-Location MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/bootstrap.py --profile core
python scripts/preflight.py --profile core
python scripts/smoke_test.py
```

Start Streamlit:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Or start the API in that PowerShell window instead:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

`bootstrap.py` installs `requirements-lock.txt`, resolves the selected pinned model snapshots, and runs preflight. The second preflight command gives an explicit final readiness result. `smoke_test.py` then performs a real detector-free analysis through the FastAPI application with a generated image, so it does not depend on a user-supplied file.

Bootstrap stores model files only in the ignored repository-local cache. It checks downloads against the immutable tracked model manifest and does not rewrite tracked project files.

## Setup profiles and no-network reuse

- `--profile core` prepares the primary ViT classifier path. The ViT trained head is already included and verified against [`models/trained_head_manifest.json`](models/trained_head_manifest.json).
- `--profile full` also resolves the ResNet-50 baseline and Grounding DINO snapshots. It requires the included baseline head, ResNet-50 snapshot, detector snapshot, and Tesseract. Tesseract remains optional for the core app path, but full readiness treats OCR as required. The full download is several GB.
- `--skip-install` skips only the locked `pip` installation. It still resolves models and runs preflight.
- `--offline` tells model provisioning to use only snapshots already stored in the repository-local cache. It does not make `pip` offline. Use `--offline --skip-install` only after packages and the selected model snapshots already exist.

Preflight never downloads anything. For a machine-readable result, run:

```text
python scripts/preflight.py --profile core --json
```

## Scope and limitations

This is a bounded course pilot, not an autonomous moderation service. The saved formal test reached 0.9791 ViT macro F1, but the result is strongly affected by source style. A separate 26-image Wikimedia diagnostic reached only 0.5549 macro F1, and the detector produced a 0.90 image-level false-positive rate on nonweapon examples. The minimum threshold-operating restricted-class recall was 0.9167, below the proposed target above 0.95. ViT classifier-path p95 was 71.5 ms on CPU; full end-to-end latency and concurrent throughput were not measured.

The canonical local project contains 288 formal images and 27 Wikimedia candidates. As standalone dataset files, public Git contains the 72 project-generated financial-promotion images and 26 retained Wikimedia diagnostic images. The other 216 formal images remain local-only because their upstream sources do not provide supported redistribution terms or item-level provenance; one irrelevant Wikimedia title-search collision is also excluded. Some final course records contain bounded screenshots or annotations of selected examples, but that does not clear the underlying images for reuse. The public registries preserve all source URLs, revisions, grouping rules, expected hashes, and known provenance limits. Read [DATASETS.md](DATASETS.md) and [NOTICE.md](NOTICE.md) before rebuilding, extracting, or redistributing any dataset image.

## Repository layout

```text
app.py                         Streamlit demonstration app
api.py                         FastAPI inference service
configs/policy.yaml            Versioned thresholds and policy rules
src/ad_safety/                 Reusable preprocessing, models, and policy package
scripts/                       Model, dataset, training, evaluation, and demo scripts
tests/                         Automated unit and API tests
.github/workflows/             macOS arm64 and Windows x64 verification
USER_MANUAL.md                 Detailed Windows and macOS run/test instructions
REPRODUCIBILITY.md             Functional-reproduction contract and boundaries
Capstone Project Idea - Ad Safety.pdf
                               Original project proposal
ad_safety_moderation_pipeline.ipynb
                               Final executed technical notebook
book/                          Static-book source and builder configuration
data/capstone_dataset/         Local: 288 formal images; public Git: 72 cleared images
data/wikimedia_pilot/          Local: 27 candidates; public Git: 26 retained images
data/*.csv                     Source, attribution, split, and hash registries
models/pretrained_model_manifest.json
models/trained_head_manifest.json
models/vit_policy_head.joblib  Included primary policy head
models/resnet50_policy_head.joblib
outputs/evaluation/            Canonical metrics, predictions, embeddings, and charts
outputs/demo_cases/            Representative inference audits and evidence media
outputs/report/                Final technical synopsis PDF
outputs/presentation/          Final PowerPoint and presentation QA evidence
outputs/video/                 Final narrated MP4, captions, storyboard, and QA evidence
outputs/validation/            Cross-artifact validation records
```

## Manual environment setup

The bootstrap flow above is the normal path. These commands are the manual dependency-installation fallback.

### Apple Silicon macOS Terminal

```bash
brew install python@3.10
python3.10 --version
brew install tesseract
tesseract --version
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.mpl-cache"
export XDG_CACHE_HOME="$PWD/.cache"
export KMP_DUPLICATE_LIB_OK=TRUE
```

Tesseract is required by the full profile. In the core profile, it is required only when text extraction is enabled.

### Windows PowerShell

Install 64-bit Python 3.10 and Git for Windows. Install the 64-bit Tesseract package documented in [USER_MANUAL.md](USER_MANUAL.md) only for OCR or the full profile. Then run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

$env:PYTHONPATH = "$PWD\src"
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:MPLCONFIGDIR = "$PWD\.mpl-cache"
$env:XDG_CACHE_HOME = "$PWD\.cache"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## Provision or independently rebuild artifacts

The included policy heads make a dataset rebuild unnecessary for normal app use. To provision only the required ViT snapshot without bootstrap:

```text
python scripts/download_models.py --core-only
python scripts/preflight.py --profile core
python scripts/smoke_test.py
```

First verify the checkout's exact release boundary. This passes both in the full canonical local project and in a fresh public clone:

```bash
python scripts/validate_release.py
```

The canonical local project already contains all 288 formal images. Verify those local files against the formal registry before training:

```bash
python scripts/build_capstone_dataset.py --verify-only
```

In a public clone, reconstruct the 216 local-only third-party formal images from their pinned upstream sources before running the 288-image verification or retraining commands. Download availability does not grant redistribution rights. The Wikimedia registry records all 27 candidates; the public diagnostic retains the 26 relevant, attributable images.

For a full independent training rebuild, download the two classifier backbones at their pinned revisions and retrain both policy heads in a separate clone or worktree. Running the dataset builder without `--verify-only` reconstructs the formal dataset from the recorded upstream sources:

```bash
python scripts/download_models.py --skip-grounding-dino
python scripts/build_capstone_dataset.py
python scripts/train_and_evaluate.py --device cpu --batch-size 8
```

The training step requires additional backbone downloads. Reconstructing the dataset also requires its upstream sources, whose availability and terms can change. Inspect the registries, upstream dataset cards, and [NOTICE.md](NOTICE.md) before redistributing the tracked or rebuilt images.

The included `joblib` heads use Python's pickle mechanism, which can execute code while loading. Run preflight before inference so their byte sizes and SHA-256 digests are checked against [`models/trained_head_manifest.json`](models/trained_head_manifest.json). The API, Streamlit app, and classifier runtime repeat the head check before deserialization. Do not load a substituted head or one obtained from another source. A retraining run can produce different bytes, so the published manifest will reject a changed head. That failure is intentional. I use a separate clone for training and keep the clone-ready app on the verified included heads.

## Run the Streamlit app

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>, upload a JPEG, PNG, or WebP image, choose the optional detector, OCR, and explanation controls, and run the analysis.

## Run the API

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Check readiness:

```text
python -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')), indent=2))"
```

Analyze an image:

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/analyze?run_detector=false&run_ocr=false&explain=false" \
  -F "file=@/path/to/creative.jpg;type=image/jpeg" \
  | python -m json.tool
```

The curl upload syntax differs between macOS and PowerShell. [USER_MANUAL.md](USER_MANUAL.md) gives both exact commands and expected response fields.

Interactive API documentation is served at <http://127.0.0.1:8000/docs>.

## Test

```bash
python -m pytest -q
python scripts/preflight.py --profile core
python scripts/smoke_test.py
python scripts/validate_release.py
```

I expect zero test failures. A snapshot-dependent check can skip before model provisioning; it should run after the selected profile is ready. The suite covers preprocessing safeguards, animated-image rejection, effective-threshold auditing, policy behavior, model readiness, bootstrap and preflight behavior, feature interfaces, Streamlit state and chart helpers, and FastAPI input handling. The smoke test adds one real in-process API analysis with the included head and downloaded ViT backbone.

The GitHub Actions workflow runs the automated suite on Apple Silicon macOS and 64-bit Windows. On pushes to `main` and manual runs, it also bootstraps the pinned ViT snapshot, runs preflight, performs the real smoke analysis, reruns the snapshot integration test, and confirms that setup did not modify tracked files.

`validate_release.py` is the portable public-clone audit. It checks the tracked release boundary, structured files, dataset policy, and final deliverables without requiring the 216 local-only third-party formal images.

The extended full-local audit preserves the stronger 288-image raw-pixel and full-backbone reproduction checks. First provision the full model profile and reconstruct the local-only dataset, then run the validator:

```bash
python scripts/bootstrap.py --profile full
python scripts/build_capstone_dataset.py
python scripts/validate_deliverables.py
```

That audit checks the executed notebook, builds the static book in a temporary checkout, renders the PDF and PowerPoint into temporary directories, regenerates representative frames from the final MP4, verifies the tracked SRT and provenance, reruns the test suite, and binds the saved evidence to all 288 local pixels and the pinned model snapshots. It requires Poppler, LibreOffice, and FFmpeg but does not require ignored saved renders, narration audio segments, or QA-frame caches.

Validation of the tracked final MP4 and SRT is cross-platform with FFmpeg. Recreating the original offline narration bytes with `python scripts/build_demo_video.py narrate-offline` requires the macOS `say` command. Windows users can validate and use the final video, but identical macOS voice bytes require macOS or a different TTS workflow that is clearly disclosed as a new build.

## Evidence and sources

- [`outputs/evaluation/`](outputs/evaluation/) is the single canonical location for machine-readable metrics, predictions, thresholds, embeddings, charts, failure cases, and the external diagnostic record.
- [`outputs/demo_cases/case_summary.json`](outputs/demo_cases/case_summary.json) records the saved representative verdicts and timings.
- [`outputs/report/ad_safety_technical_synopsis.pdf`](outputs/report/ad_safety_technical_synopsis.pdf), [`outputs/presentation/ad_safety_management_presentation.pptx`](outputs/presentation/ad_safety_management_presentation.pptx), and [`outputs/video/ad_safety_demo.mp4`](outputs/video/ad_safety_demo.mp4) are the final course deliverables.
- [`ad_safety_moderation_pipeline.ipynb`](ad_safety_moderation_pipeline.ipynb) is the final executed notebook.
- [`outputs/README.md`](outputs/README.md) documents the complete canonical output tree and the generated intermediates that remain untracked.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) defines the supported hosts, pinned assets, readiness checks, and independent rebuild boundary.
- [`SOURCES.md`](SOURCES.md) separates external papers, model cards, policy references, data sources, and locally measured results.

The absence of a `LICENSE` file is intentional pending contributor agreement. See [NOTICE.md](NOTICE.md).
