# Ad Safety Visual Moderation Pilot

[![Cross-platform verification](https://github.com/Swarnaditya-Maitra/ad-safety-moderation-pipeline/actions/workflows/cross-platform.yml/badge.svg)](https://github.com/Swarnaditya-Maitra/ad-safety-moderation-pipeline/actions/workflows/cross-platform.yml)

This repository contains the public release of an evidence-led computer-vision pipeline for triaging static ad creatives. It combines:

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
git clone https://github.com/Swarnaditya-Maitra/ad-safety-moderation-pipeline.git
cd ad-safety-moderation-pipeline
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
git clone https://github.com/Swarnaditya-Maitra/ad-safety-moderation-pipeline.git
Set-Location ad-safety-moderation-pipeline
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

Raw images and image-bearing coursework deliverables are not included because redistribution rights are not uniformly cleared. Curated aggregate charts and a path-free validation summary are available under [`outputs/`](outputs/). See [`outputs/README.md`](outputs/README.md), [NOTICE.md](NOTICE.md), and the registries under [`data/`](data/).

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
data/*.csv                     Source and provenance registries without image files
models/pretrained_model_manifest.json
models/trained_head_manifest.json
models/vit_policy_head.joblib  Included primary policy head
models/resnet50_policy_head.joblib
outputs/                       Curated aggregate charts and validation summary
results/                       Text and tabular evidence from the validated course run
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

For a full independent training rebuild, download the two classifier backbones at their pinned revisions, rebuild the dataset from the recorded sources, and retrain both policy heads in a separate clone or worktree:

```bash
python scripts/download_models.py --skip-grounding-dino
python scripts/build_capstone_dataset.py
python scripts/train_and_evaluate.py --device cpu --batch-size 8
```

These steps require additional backbone and source-image downloads. Source availability and licensing can change, so inspect the registries and upstream dataset cards before redistributing any rebuilt images.

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
```

I expect zero test failures. A snapshot-dependent check can skip before model provisioning; it should run after the selected profile is ready. The suite covers preprocessing safeguards, animated-image rejection, effective-threshold auditing, policy behavior, model readiness, bootstrap and preflight behavior, feature interfaces, Streamlit state and chart helpers, and FastAPI input handling. The smoke test adds one real in-process API analysis with the included head and downloaded ViT backbone.

The GitHub Actions workflow runs the automated suite on Apple Silicon macOS and 64-bit Windows. On pushes to `main` and manual runs, it also bootstraps the pinned ViT snapshot, runs preflight, performs the real smoke analysis, reruns the snapshot integration test, and confirms that setup did not modify tracked files.

## Evidence and sources

- [`results/evaluation/`](results/evaluation/) contains machine-readable metrics, predictions, thresholds, latency, failure cases, and the external diagnostic record.
- [`results/demo_cases/case_summary.json`](results/demo_cases/case_summary.json) records the saved representative verdicts and timings.
- [`outputs/README.md`](outputs/README.md) explains the curated public charts, validation summary, and excluded artifacts.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) defines the supported hosts, pinned assets, readiness checks, and independent rebuild boundary.
- [`SOURCES.md`](SOURCES.md) separates external papers, model cards, policy references, data sources, and locally measured results.

The absence of a `LICENSE` file is intentional pending contributor agreement. See [NOTICE.md](NOTICE.md).
