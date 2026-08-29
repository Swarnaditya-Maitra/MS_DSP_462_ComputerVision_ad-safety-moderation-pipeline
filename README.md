# MS_DSP_462 Computer Vision: Ad Safety Moderation Pipeline

[![Cross-platform verification](https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline/actions/workflows/cross-platform.yml/badge.svg)](https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline/actions/workflows/cross-platform.yml)
[![Published Jupyter Book](https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline/actions/workflows/pages.yml/badge.svg)](https://swarnaditya-maitra.github.io/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline/)

This repository contains our complete MS_DSP_462 computer-vision project for triaging static ad creatives. It includes the runnable app and API, trained policy heads, an executed notebook, saved evaluation evidence, the final report and presentation, a narrated demonstration, and the scripts used to build and validate them.

The pipeline combines a frozen ViT-B/16 classifier, a ResNet-50 comparison baseline, optional Grounding DINO object evidence, local Tesseract OCR, occlusion sensitivity, and a deterministic policy layer. It returns `APPROVE`, `REVIEW`, or `BLOCK` while keeping each evidence stream visible.

The modeled visual labels are `safe`, `firearms`, `explosives`, and `financial_promotion`. A financial-promotion label means the image contains promotion cues. It does not establish fraud, legality, or regulatory status, so those cases route to human review.

## Supported setup

The verified clone-ready paths use 64-bit CPython 3.10 on either Apple Silicon macOS or 64-bit Windows 10/11. Intel macOS, Linux, 32-bit Python, and other Python versions fall outside the tested contract.

- [USER_MANUAL.md](USER_MANUAL.md) is the command-by-command guide for prerequisites, setup, Streamlit, FastAPI, normal and error-path tests, full rebuilds, troubleshooting, and shutdown on both supported operating systems.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) defines what is pinned, what functional reproduction means, and where machine-specific results can differ.
- [Published Jupyter Book](https://swarnaditya-maitra.github.io/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline/) presents the executed technical notebook as a browsable static site.

## Clone-ready quick start

The `core` profile is the shortest runnable path. It installs the locked environment, verifies the included ViT policy head, downloads the pinned ViT backbone, runs preflight, and performs one real detector-free analysis with a generated image.

### Apple Silicon macOS Terminal

```bash
git clone https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline.git
cd MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline
python3.10 -m venv .venv
source .venv/bin/activate
python scripts/bootstrap.py --profile core
python scripts/preflight.py --profile core
python scripts/smoke_test.py
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
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
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>. The full profile, OCR installation, offline reuse, API upload commands, and negative-path checks are documented in [USER_MANUAL.md](USER_MANUAL.md).

## What the evidence supports

This is a bounded course pilot, not an autonomous moderation service. The saved formal test reached 0.9791 ViT macro F1, but the result is affected by source style. The separate Wikimedia diagnostic fell to 0.5549 macro F1, and the detector produced a 0.90 image-level false-positive rate on nonweapon examples. The minimum threshold-operating restricted-class recall was 0.9167, below the proposed target above 0.95. ViT classifier-path p95 was 71.5 ms on CPU; full end-to-end latency and concurrent throughput were not measured.

The exact dataset inventory, provenance, public-release boundary, and attribution requirements are maintained in [DATASETS.md](DATASETS.md). [NOTICE.md](NOTICE.md) explains reuse, contributor-identification, model-file, and repository-license limits.

## Repository map

```text
app.py                         Streamlit demonstration app
api.py                         FastAPI inference service
configs/policy.yaml            Versioned thresholds and policy rules
src/ad_safety/                 Reusable preprocessing, models, and policy package
scripts/                       Dataset, model, evaluation, build, and QA scripts
tests/                         Automated unit, app, API, and release tests
USER_MANUAL.md                 Detailed Windows and macOS run/test guide
REPRODUCIBILITY.md             Functional-reproduction contract
DATASETS.md                    Canonical dataset inventory and provenance record
SOURCES.md                     Canonical external source register
NOTICE.md                      Reuse, safety, and license notice
ad_safety_moderation_pipeline.ipynb
                               Final executed technical notebook
book/                          Static-book source and configuration
data/                          Released data plus complete source registries
models/                        Hash-pinned trained heads and model manifest
outputs/evaluation/            Metrics, predictions, embeddings, and charts
outputs/demo_cases/            Representative inference audits and media
outputs/report/                Final technical synopsis PDF
outputs/presentation/          Final deck, speaker script, and QA evidence
outputs/video/                 Narrated MP4, captions, storyboard, and QA evidence
outputs/validation/            Cross-artifact validation records
```

## Run the API

After the core setup passes, start the local API:

```text
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for the interactive endpoint documentation. Use [USER_MANUAL.md](USER_MANUAL.md) for macOS and Windows health checks, file-upload commands, expected response fields, and error-path tests.

## Test and validate

These commands provide the normal portable check:

```text
python -m pytest -q
python scripts/preflight.py --profile core
python scripts/smoke_test.py
python scripts/validate_release.py
```

I expect zero test failures, a final `READY` result from preflight, a passed smoke-test summary, and a passed release audit. The stronger full-local audit needs the complete reconstructed dataset, the full model profile, Poppler, LibreOffice, and FFmpeg. [USER_MANUAL.md](USER_MANUAL.md) gives the exact macOS and Windows steps. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) separates portable validation from the full-local evidence audit.

## Final deliverables and evidence

- [outputs/report/ad_safety_technical_synopsis.pdf](outputs/report/ad_safety_technical_synopsis.pdf) is the final two-page report.
- [outputs/presentation/ad_safety_management_presentation.pptx](outputs/presentation/ad_safety_management_presentation.pptx) is the final management deck.
- [outputs/presentation/ad_safety_speaker_script.md](outputs/presentation/ad_safety_speaker_script.md) is the slide-by-slide live presentation script.
- [outputs/video/ad_safety_demo.mp4](outputs/video/ad_safety_demo.mp4) is the final narrated demonstration.
- [ad_safety_moderation_pipeline.ipynb](ad_safety_moderation_pipeline.ipynb) is the final executed notebook.
- [outputs/README.md](outputs/README.md) maps the canonical output tree.
- [SOURCES.md](SOURCES.md) separates external evidence from measured project outputs.

The original proposal remains at [Capstone Project Idea - Ad Safety.pdf](Capstone%20Project%20Idea%20-%20Ad%20Safety.pdf). The repository intentionally has no top-level `LICENSE` pending contributor agreement; see [NOTICE.md](NOTICE.md).
