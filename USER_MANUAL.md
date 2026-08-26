# Ad Safety Studio user manual

This manual gives me one supported, clone-ready path for Apple Silicon macOS and one for 64-bit Windows 10 or 11. Both require 64-bit CPython 3.10. I keep the service on `127.0.0.1`, so it is reachable only from the same computer unless I intentionally change the host binding.

Exact universal reproduction is not possible. Upstream model hosting, source websites, package wheels, operating systems, and hardware can change. This repository controls the parts it can with pinned top-level Python packages, pinned model revisions, included trained heads, SHA-256 verification, offline preflight checks, and a real model smoke test. I do not claim identical latency or floating-point results on every machine.

I use [REPRODUCIBILITY.md](REPRODUCIBILITY.md) as the formal contract and this manual as the command-by-command operating guide.

## 1. What I need before setup

### Common requirements

- An Apple Silicon Mac (`arm64`) or a 64-bit Windows computer (`AMD64`). I recommend 16 GB RAM. The optional detector has a large model snapshot and works best with more memory.
- Enough free disk space for the pinned Python packages and selected model cache. The core model download is several hundred MB; the full profile uses several GB.
- Python 3.10 and Git.
- An internet connection for the first bootstrap, model provisioning, or independent data build. The app can run locally after packages and pinned assets exist.
- A modern browser such as Chrome, Edge, Firefox, or Safari.
- Tesseract OCR if I want text extraction. The classifier still works when Tesseract is absent.

The public repository includes the validated ViT and ResNet-50 `joblib` policy heads plus their expected byte sizes and SHA-256 digests in `models/trained_head_manifest.json`. It excludes raw dataset images and pretrained backbone caches. The core app can analyze an image after the pinned ViT snapshot is downloaded and preflight passes. I never load a substituted `joblib` file or one obtained from another source because Python pickle-based formats can execute code during loading.

Platform boundary:

- CPython 3.10 on Apple Silicon macOS (`arm64`) is the validated platform.
- 64-bit CPython 3.10 on Windows 10 or 11 (`AMD64` or `x86_64`) is the supported cross-platform path, but the saved latency numbers are not Windows benchmarks.
- Intel macOS is not supported by `requirements-lock.txt`. PyTorch stopped publishing macOS x86_64 binaries after the 2.2 series, while this project pins a newer PyTorch release. I do not substitute an older framework and claim an exact reproduction. I use an Apple Silicon Mac or 64-bit Windows instead.
- Linux, 32-bit Python, non-CPython runtimes, and other Python versions are outside the supported reproduction contract.

## 2. Get the code

### macOS Terminal

```bash
git clone https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline.git
cd MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline
```

### Windows PowerShell

```powershell
git clone https://github.com/Swarnaditya-Maitra/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline.git
Set-Location MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline
```

I run all later commands from the repository root, the directory that contains `app.py`, `api.py`, and `requirements-lock.txt`.

## 3. Install the operating-system prerequisites

### Apple Silicon macOS

Confirm the processor architecture first:

```bash
uname -m
```

Expected: `arm64`. If it prints `x86_64`, the pinned environment is not supported on that Intel Mac.

If command-line build tools are missing:

```bash
xcode-select --install
```

If `brew` is not installed, I follow the official Homebrew installation instructions before continuing.

Install Python 3.10 with Homebrew, then verify the exact interpreter:

```bash
brew install python@3.10
python3.10 --version
```

Install Tesseract with Homebrew if I want OCR or the full profile:

```bash
brew install tesseract
tesseract --version
```

### Windows 10 or 11

1. I install 64-bit Python 3.10 from python.org and enable the installer option that adds Python to `PATH`.
2. I install Git for Windows.
3. If I want OCR or the full profile, I install the 64-bit Tesseract package from the UB Mannheim release page.
4. If the installer does not update `PATH`, I add `C:\Program Files\Tesseract-OCR` to my user `Path`, then close and reopen PowerShell.

Verify the tools:

```powershell
py -3.10 --version
git --version
```

If I installed Tesseract, I verify it separately:

```powershell
tesseract --version
```

If PowerShell blocks virtual-environment activation, I change the policy only for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 4. Create an isolated Python environment

### macOS Terminal

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Confirm that the active interpreter belongs to `.venv`:

```text
python -c "import sys; print(sys.executable); print(sys.version)"
```

### Clone-ready bootstrap and verification

From the active `.venv`, use the same three commands on macOS Terminal or Windows PowerShell:

```text
python scripts/bootstrap.py --profile core
python scripts/preflight.py --profile core
python scripts/smoke_test.py
```

The first command installs `requirements-lock.txt`, downloads only the pinned ViT snapshot, and runs core preflight. The second command gives me an explicit final readiness report. The smoke test then sends a deterministic generated PNG through the real FastAPI analysis path with the detector, OCR, and explanation disabled. It exits successfully only after it receives a valid audit response and policy verdict. I do not need to supply an image for this test.

Bootstrap stores downloads only in the ignored repository-local cache. It verifies them against the immutable tracked pretrained-model manifest and leaves tracked project files unchanged.

Bootstrap refuses to continue unless it detects CPython 3.10, a 64-bit supported host, and an active virtual environment. `--allow-system-python` bypasses only the virtual-environment check, so I avoid it unless I accept modifying that interpreter.

Profiles and reuse flags:

- `python scripts/bootstrap.py --profile core` prepares the primary ViT classifier path from the included, hash-pinned ViT head and a pinned ViT backbone download. This is the normal app path and downloads several hundred MB.
- `python scripts/bootstrap.py --profile full` also downloads the ResNet-50 baseline and Grounding DINO snapshots. Full preflight requires the included baseline head, the ResNet-50 and detector snapshots, and Tesseract on `PATH`. Tesseract remains optional for core classification but is required by the full readiness definition. The full download is several GB.
- `--skip-install` skips only `pip install -r requirements-lock.txt`. Model resolution and preflight still run.
- `--offline` prevents model downloads and requires the selected snapshots to exist in the repository-local cache. It does not make `pip` offline. For a completely no-network bootstrap, packages and snapshots must already exist and I combine `--offline --skip-install`.

Preflight is read-only and does not contact the network or load model weights. For a machine-readable report:

```text
python scripts/preflight.py --profile core --json
```

The JSON exit status is `0` only when `ready` is `true`. Optional checks appear as warnings and do not make the core profile fail.

### Manual dependency-installation fallback

If I do not use bootstrap, I install the exact lock file myself before Section 5 and Section 6:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

## 5. Set the local runtime variables

These variables keep imports and caches inside the project. I set them in every new terminal before running a build, the app, or the API.

### macOS Terminal

```bash
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.mpl-cache"
export XDG_CACHE_HOME="$PWD/.cache"
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Windows PowerShell

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:MPLCONFIGDIR = "$PWD\.mpl-cache"
$env:XDG_CACHE_HOME = "$PWD\.cache"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## 6. Provision models or run a full independent rebuild

### Core app provisioning

Section 4 is the normal path. If I installed dependencies manually and want to provision only the app-critical model, I run:

```text
python scripts/download_models.py --core-only
python scripts/preflight.py --profile core
python scripts/smoke_test.py
```

The clone already contains `models/vit_policy_head.joblib`. Preflight checks its size and SHA-256 against `models/trained_head_manifest.json`, validates the pinned ViT snapshot, imports the runtime packages, and checks required policy and result files. It does not deserialize the head. The smoke test loads the model stack only after preflight succeeds.

To verify an existing core cache without any model download:

```text
python scripts/download_models.py --core-only --offline
python scripts/preflight.py --profile core
```

### Full optional-evidence profile

For the baseline model, detector, and OCR readiness contract:

```text
python scripts/bootstrap.py --profile full
python scripts/preflight.py --profile full
```

This downloads the pinned ViT, ResNet-50, and Grounding DINO snapshots and uses several GB. Full preflight requires both included trained heads, both classifier backbone snapshots, the detector snapshot, and Tesseract. The normal smoke test remains detector-free so it provides a bounded core analysis check.

### Full independent data and training rebuild

The included heads are sufficient for normal use. I can still rebuild the bounded dataset and retrain both heads independently. I use a separate clone or worktree so the public, verified heads remain untouched:

```text
python scripts/download_models.py --skip-grounding-dino
python scripts/build_capstone_dataset.py
python scripts/train_and_evaluate.py --device cpu --batch-size 8
```

Important behavior:

- `build_capstone_dataset.py` reconstructs the bounded dataset from the recorded sources and writes the grouped train, validation, and test split.
- `train_and_evaluate.py` freezes both backbones, fits the logistic policy heads on train data, selects thresholds on validation data, and evaluates test data once.
- `--device cpu` is the cross-platform setting used in the saved evidence. On Apple Silicon, `--device mps` can speed feature extraction, but that creates a separate run and is not the saved CPU benchmark. The detector remains CPU-bound in this pilot.
- A source can become unavailable or change its terms. I inspect `SOURCES.md`, the dataset cards, [NOTICE.md](NOTICE.md), and [`outputs/README.md`](outputs/README.md) before redistributing rebuilt material.
- The rebuild overwrites local trained-head files. A new run can produce different bytes, so the published manifest can reject the rebuilt files even when training completed correctly. That failure protects the clone-ready app from silently loading a substituted artifact. I evaluate rebuilt heads in the separate training checkout and keep the normal app checkout on the verified included heads.

After rebuilding the dataset, I can verify its local files without downloading them again:

```text
python scripts/build_capstone_dataset.py --verify-only
```

## 7. Run the Streamlit app

### macOS Terminal 1

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/.cache/huggingface"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

### Windows PowerShell 1

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
$env:HF_HOME = "$PWD\.cache\huggingface"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501).

The header should show:

- `MODEL READY` only after the trained ViT head is nonempty and the pinned ViT backbone snapshot is complete;
- `EVIDENCE LOADED` when the saved benchmark JSON is available; and
- `NOT PRODUCTION-READY`, which is an intentional scope warning.

### App workflow

1. Use `Analyze` for one creative.
2. Upload one static JPEG, PNG, or WebP. Animated WebP and APNG inputs are rejected because a single-frame check could miss later content.
3. Choose optional evidence in the sidebar:
   - `Object context`: Grounding DINO boxes and phrases. This is slow and diagnostic.
   - `Text extraction`: local Tesseract OCR. It is on by default, so I turn it off when Tesseract is not installed.
   - `Occlusion view`: a slower qualitative sensitivity map.
4. Select `Run policy analysis`.
5. Review `Decision`, `Evidence`, `Timing`, `Benchmark`, and `Audit`.
6. Download the audit JSON. It records the input SHA-256, options, versions, scores, exact applied thresholds, evidence, and stage timing.

Changing an inference option after a completed run keeps the prior result visible but marks it stale. I run the analysis again before acting on the changed configuration.

`Pilot evidence` shows the formal benchmark separately from the weak external diagnostic. `How it works` explains the signal-specific rules. Fused evidence is a triage signal, not a probability of illegality.

## 8. Run the FastAPI service

Keep Streamlit in Terminal 1. Open a second terminal in the repository root.

### macOS Terminal 2

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/.cache/huggingface"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### Windows PowerShell 2

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
$env:HF_HOME = "$PWD\.cache\huggingface"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Health check on macOS

```bash
curl -sS http://127.0.0.1:8000/health | python -m json.tool
```

### Health check on Windows PowerShell

```powershell
curl.exe -sS http://127.0.0.1:8000/health | python -m json.tool
```

Before the first valid analysis, I expect:

```json
{
  "status": "ok",
  "ready_for_analysis": true,
  "engine_loaded": false
}
```

The response also reports classifier, backbone, detector, and Tesseract readiness. `engine_loaded` becomes `true` after the first valid analysis. `detector_loaded` becomes `true` only after a request enables the detector.

### Analyze an image on macOS

```bash
curl -sS -X POST \
  "http://127.0.0.1:8000/analyze?run_detector=false&run_ocr=false&explain=false" \
  -F "file=@/absolute/path/to/creative.jpg;type=image/jpeg" \
  | python -m json.tool
```

### Analyze an image on Windows PowerShell

```powershell
curl.exe -sS -X POST "http://127.0.0.1:8000/analyze?run_detector=false&run_ocr=false&explain=false" -F "file=@C:\absolute\path\to\creative.jpg;type=image/jpeg" | python -m json.tool
```

Expected success fields:

- `audit_schema_version` is `1.1`;
- `input.sha256` has 64 hexadecimal characters;
- `options` matches the query parameters;
- `analysis.decision.verdict` is `APPROVE`, `REVIEW`, or `BLOCK`;
- `analysis.applied_thresholds` contains the exact loaded rule snapshot; and
- `analysis.latency_ms` separates classifier, detector, OCR, explanation, and total time.

## 9. Run automated tests

The fast suite does not need the app or API server to be running:

```text
python -m pytest -q
python scripts/preflight.py --profile core
python scripts/smoke_test.py
```

Expected result: zero test failures, a final `READY` line from core preflight, and a JSON smoke-test summary whose `smoke_test` field is `passed`. A snapshot-dependent test can skip before model provisioning. It should run after the selected profile is ready.

Run focused groups when troubleshooting:

```text
python -m pytest -q tests/test_preprocessing.py
python -m pytest -q tests/test_policy.py
python -m pytest -q tests/test_api.py
python -m pytest -q tests/test_app.py
python -m pytest -q tests/test_inference_contract.py
python -m pytest -q tests/test_bootstrap.py
python -m pytest -q tests/test_preflight.py
```

## 10. Manual end-to-end test matrix

After the independent dataset build in Section 6, these recorded cases should reproduce the three policy actions. They are not part of a fresh public clone because the source images are not redistributed:

| Test | File | Settings | Expected result |
|---|---|---|---|
| Safe | `data/capstone_dataset/test/safe/safe-77c35d499df2417f.jpg` | Detector off, OCR on, occlusion off | `APPROVE` |
| Financial promotion | `data/capstone_dataset/test/financial_promotion/financial_promotion-ff884f7cd86b4295.jpg` | Detector off, OCR on, occlusion on | `REVIEW`, heatmap visible |
| Firearm | `data/capstone_dataset/test/firearms/firearms-08b5814b3661e772.jpg` | Detector on, OCR off, occlusion on | `BLOCK`, detector cues visible |

For each case I verify:

1. The filename, size, dimensions, and SHA prefix appear before analysis.
2. The verdict, policy focus, and evidence score appear after analysis.
3. The raw and fused chart has four labels.
4. The timing chart separates enabled stages.
5. `Evidence` clearly says when OCR or the detector is disabled or returns no result.
6. `Audit` shows exact thresholds and downloads valid JSON.
7. Changing a sidebar option marks the old result stale without silently relabeling its audit.
8. Replacing the uploaded file clears the old result before another run.

### Negative-path UI checks

Create deterministic test inputs with the same command on macOS or Windows:

```text
python scripts/create_manual_test_fixtures.py
```

This writes a valid generated `valid_static.png`, `corrupt.png`, a 16-by-16 `tiny.png`, a two-frame `animated.webp`, and an encoded `oversized.png` above 20 MB under `tmp/manual_test_inputs/`. The `tmp/` directory is ignored by Git.

- Upload `tmp/manual_test_inputs/valid_static.png`. Expected: analysis completes with an `APPROVE`, `REVIEW`, or `BLOCK` verdict. This checks the inference path, not semantic accuracy for the generated gradient.
- Upload `tmp/manual_test_inputs/corrupt.png`. Expected: `The file is not a readable JPEG, PNG, or WebP image.`
- Upload `tmp/manual_test_inputs/tiny.png`. Expected: a minimum-dimension error before any model loads.
- Upload `tmp/manual_test_inputs/animated.webp`. Expected: `Animated images are unsupported.`
- Upload `tmp/manual_test_inputs/oversized.png`. Expected: a file-size rejection before image decoding.
- Stop Streamlit, temporarily move `models/vit_policy_head.joblib`, and restart Streamlit. Expected: the app shell still loads, shows `MODEL SETUP NEEDED`, and blocks analysis with a stable setup message. Restore the exact file afterward.

macOS move and restore commands:

```bash
mv models/vit_policy_head.joblib models/vit_policy_head.joblib.disabled
# Restart Streamlit and run the missing-model check, then stop it.
mv models/vit_policy_head.joblib.disabled models/vit_policy_head.joblib
```

Windows PowerShell move and restore commands:

```powershell
Rename-Item models\vit_policy_head.joblib vit_policy_head.joblib.disabled
# Restart Streamlit and run the missing-model check, then stop it.
Rename-Item models\vit_policy_head.joblib.disabled vit_policy_head.joblib
```

The API should return HTTP `422` for corrupt, undersized, or animated image bytes, `413` for an encoded upload above 20 MB, `415` for an unsupported media type, and `503` when its readiness check detects a missing classifier or required model snapshot before analysis. An unexpected runtime load or parse failure, such as a corrupt nonempty `joblib` head, returns `500` with an opaque error reference instead of exposing the internal exception.

Successful generated-image API check on macOS:

```bash
curl -sS -X POST \
  "http://127.0.0.1:8000/analyze?run_detector=false&run_ocr=false&explain=false" \
  -F "file=@tmp/manual_test_inputs/valid_static.png;type=image/png" \
  | python -m json.tool
```

Successful generated-image API check on Windows PowerShell:

```powershell
curl.exe -sS -X POST "http://127.0.0.1:8000/analyze?run_detector=false&run_ocr=false&explain=false" -F "file=@tmp\manual_test_inputs\valid_static.png;type=image/png" | python -m json.tool
```

Expected status: `200`, with the same stable success fields listed in Section 8.

Example corrupt-input API check on macOS:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -F "file=@README.md;type=image/png;filename=corrupt.png" \
  http://127.0.0.1:8000/analyze
```

Example on Windows PowerShell:

```powershell
curl.exe -sS -o NUL -w "%{http_code}`n" -F "file=@README.md;type=image/png;filename=corrupt.png" http://127.0.0.1:8000/analyze
```

Expected status: `422`.

## 11. Validate the course deliverables

The full course checkout includes the report, PowerPoint, video, notebook, and their builders. It can run the cross-artifact audit:

```text
python scripts/validate_deliverables.py
```

This extended audit also needs Poppler `pdftoppm`, LibreOffice for independent PowerPoint rendering, and FFmpeg for video checks. These coursework files and builders are intentionally outside the public release.

### macOS validator tools

```bash
brew install poppler ffmpeg
brew install --cask libreoffice
export PATH="/Applications/LibreOffice.app/Contents/MacOS:$PATH"
command -v pdftoppm
command -v soffice
command -v ffmpeg
```

### Windows validator tools

1. Install LibreOffice with `winget install --id TheDocumentFoundation.LibreOffice`.
2. Install FFmpeg with `winget install --id Gyan.FFmpeg`.
3. Download a current Windows Poppler archive from the community-maintained [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases), inspect its release checksum, extract it to `C:\Tools\poppler`, and add its `Library\bin` directory to the user `Path`.
4. Add `C:\Program Files\LibreOffice\program` to the user `Path` if the LibreOffice installer did not do so.
5. Close and reopen PowerShell, activate `.venv`, set the Section 5 variables again, and verify:

```powershell
Get-Command pdftoppm
Get-Command soffice
Get-Command ffmpeg
```

The public checkout does not contain the PDF, PowerPoint, video, or `validate_deliverables.py`, so it does not need these three tools.

## 12. Stop the services

In each terminal running Streamlit or Uvicorn, press `Ctrl+C`. Then deactivate the environment:

```text
deactivate
```

## 13. Troubleshooting

### `python`, `streamlit`, or `uvicorn` is not recognized

I activate `.venv` and use the module form, `python -m streamlit` or `python -m uvicorn`. On Windows I confirm that Python is on `PATH` or use `py -3.10` to create the environment.

### PowerShell cannot run `Activate.ps1`

I run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window, then activate again. This does not change the permanent system policy.

### The port is already in use

I choose another local port, such as `8502` for Streamlit or `8001` for the API, and open the matching URL.

### `Classifier artifact missing` or `MODEL SETUP NEEDED`

I run `python scripts/preflight.py --profile core` first. If the ViT head fails its size or SHA-256 check, I restore the exact tracked file from the trusted repository clone. If the ViT snapshot is missing, I run `python scripts/bootstrap.py --profile core`. I do not bypass the check or load a replacement `joblib` file from another source.

### Core bootstrap fails in offline mode

`--offline` requires the selected pinned model snapshot to exist already under the repository-local cache. On a new clone, I remove `--offline` and allow the pinned ViT download. If packages are already installed and I only want to avoid another `pip` run, I use `--skip-install` without `--offline`.

### Tesseract is unavailable

I run `tesseract --version`, correct `PATH`, reopen the terminal, and restart Streamlit or Uvicorn. I can also turn text extraction off and test classifier-only behavior.

### The first analysis is slow

The first request initializes the frozen backbone. Enabling Grounding DINO loads an additional large model and can take several seconds on CPU. Later requests usually reuse the in-memory models.

### A browser still shows old code

I hard-refresh with `Cmd+Shift+R` on macOS or `Ctrl+Shift+R` on Windows. I can also restart Streamlit on a new port.

### PyTorch or a model install fails on Windows

I confirm that Python is 64-bit and version 3.10, upgrade `pip`, and reinstall from `requirements-lock.txt`. I use CPU first. CUDA behavior depends on the installed PyTorch build and local NVIDIA drivers and is not part of the validated run.

## Sources

- Python virtual environments: https://docs.python.org/3.10/tutorial/venv.html
- Streamlit command-line installation and activation: https://docs.streamlit.io/get-started/installation/command-line
- Streamlit app startup: https://docs.streamlit.io/develop/concepts/architecture/run-your-app
- FastAPI and Uvicorn startup: https://fastapi.tiangolo.com/deployment/manually/
- Tesseract installation guidance: https://github.com/tesseract-ocr/tesseract/wiki
- Homebrew Tesseract formula: https://formulae.brew.sh/formula/tesseract
- Homebrew Python 3.10 formula: https://formulae.brew.sh/formula/python@3.10
- Windows Tesseract documentation: https://github.com/UB-Mannheim/Tesseract_Dokumentation/blob/main/Tesseract_Doku_Windows.md
- PyTorch macOS x86_64 binary deprecation: https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690
- LibreOffice download: https://www.libreoffice.org/download/download-libreoffice/
- Poppler project: https://poppler.freedesktop.org/
- Community Windows Poppler builds: https://github.com/oschwartz10612/poppler-windows/releases
- FFmpeg download: https://ffmpeg.org/download.html
