# Ad Safety Visual Moderation Pilot

This repository contains the public source release of an evidence-led computer-vision pipeline for triaging static ad creatives. It combines:

1. a frozen ViT-B/16 visual backbone with a four-class policy head;
2. a ResNet-50 comparison baseline;
3. optional Grounding DINO Tiny object evidence;
4. local Tesseract OCR cues;
5. occlusion sensitivity for qualitative inspection; and
6. a deterministic policy layer that returns `APPROVE`, `REVIEW`, or `BLOCK`.

The four visual labels are `safe`, `firearms`, `explosives`, and `financial_promotion`. A financial-promotion label means visible promotion cues were found. It does not establish fraud, legality, or regulatory status. Financial cases route to human review.

## Run and test guide

Use [USER_MANUAL.md](USER_MANUAL.md) for the complete 64-bit Windows 10/11 and Apple Silicon macOS workflow. It covers prerequisites, virtual environments, Tesseract, model and dataset provisioning, Streamlit, FastAPI, health checks, automated tests, reproducible negative-path fixtures, troubleshooting, and shutdown. Intel macOS is explicitly unsupported because the pinned PyTorch release has no macOS x86_64 wheel.

The commands below are the short path. I ran the validated project and current release checks on Apple Silicon macOS. The manual gives native PowerShell equivalents for Windows; it does not claim that the saved performance numbers are Windows benchmarks.

## Scope and limitations

This is a bounded course pilot, not an autonomous moderation service. The saved formal test reached 0.9791 ViT macro F1, but the result is strongly affected by source style. A separate 26-image Wikimedia diagnostic reached only 0.5549 macro F1, and the detector produced a 0.90 image-level false-positive rate on nonweapon examples. The minimum threshold-operating restricted-class recall was 0.9167, below the proposed target above 0.95. ViT classifier-path p95 was 71.5 ms on CPU; full end-to-end latency and concurrent throughput were not measured.

Raw images and image-bearing coursework deliverables are not included because redistribution rights are not uniformly cleared. See [NOTICE.md](NOTICE.md) and the registries under [`data/`](data/).

## Repository layout

```text
app.py                         Streamlit demonstration app
api.py                         FastAPI inference service
configs/policy.yaml            Versioned thresholds and policy rules
src/ad_safety/                 Reusable preprocessing, models, and policy package
scripts/                       Model, dataset, training, evaluation, and demo scripts
tests/                         Automated unit and API tests
USER_MANUAL.md                 Detailed Windows and macOS run/test instructions
data/*.csv                     Source and provenance registries without image files
models/pretrained_model_manifest.json
results/                       Text and tabular evidence from the validated course run
```

## Environment setup

### Apple Silicon macOS Terminal

```bash
brew install python@3.10 tesseract
python3.10 --version
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

Tesseract is required only when text extraction is enabled.

### Windows PowerShell

Install 64-bit Python 3.10, Git for Windows, and the 64-bit Tesseract package documented in [USER_MANUAL.md](USER_MANUAL.md). Then run:

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

## Rebuild the runnable artifacts

Download the three pretrained backbones at their pinned revisions, rebuild the dataset from the recorded sources, and train the policy heads:

```bash
python scripts/download_models.py
python scripts/build_capstone_dataset.py
python scripts/train_and_evaluate.py --device cpu --batch-size 8
```

These steps download several gigabytes. Source availability and licensing can change, so inspect the registries and upstream dataset cards before redistributing any rebuilt images.

Training writes local `joblib` classifier heads. `joblib` uses Python's pickle mechanism, which can execute code while loading. Never copy a classifier head from an untrusted repository or URL; rebuild it locally with the pinned script and source records.

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
  "http://127.0.0.1:8000/analyze?run_detector=false&run_ocr=true&explain=false" \
  -F "file=@/path/to/creative.jpg;type=image/jpeg" \
  | python -m json.tool
```

The curl upload syntax differs between macOS and PowerShell. [USER_MANUAL.md](USER_MANUAL.md) gives both exact commands and expected response fields.

Interactive API documentation is served at <http://127.0.0.1:8000/docs>.

## Test

```bash
python -m pytest -q
```

The current source-only release passes 49 tests and skips one pinned-snapshot integration check before model download. After the pinned snapshots are downloaded, all 50 tests run. The tests cover preprocessing safeguards, animated-image rejection, effective-threshold auditing, policy behavior, model-readiness checks, feature interfaces, Streamlit state and chart helpers, and FastAPI input handling.

## Evidence and sources

- [`results/evaluation/`](results/evaluation/) contains machine-readable metrics, predictions, thresholds, latency, failure cases, and the external diagnostic record.
- [`results/demo_cases/case_summary.json`](results/demo_cases/case_summary.json) records the saved representative verdicts and timings.
- [`SOURCES.md`](SOURCES.md) separates external papers, model cards, policy references, data sources, and locally measured results.

The absence of a `LICENSE` file is intentional pending contributor agreement. See [NOTICE.md](NOTICE.md).
