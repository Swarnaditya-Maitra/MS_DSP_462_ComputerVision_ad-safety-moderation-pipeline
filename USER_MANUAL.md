# Ad Safety Studio user manual

This manual gives me one reproducible path for Apple Silicon macOS and one for 64-bit Windows 10 or 11. I use Python 3.10 because that is the version used for the validated project run. I keep the service on `127.0.0.1`, so it is reachable only from the same computer unless I intentionally change the host binding.

## 1. What I need before setup

### Common requirements

- An Apple Silicon Mac (`arm64`) or a 64-bit Windows computer (`AMD64`) with at least 16 GB RAM. The optional detector has a large model snapshot and works best with more memory.
- At least 8 GB of free disk space for Python packages, three pinned model snapshots, generated data, trained heads, and caches.
- Python 3.10 and Git.
- An internet connection for the first model and data build. The app can run locally after the pinned assets exist.
- A modern browser such as Chrome, Edge, Firefox, or Safari.
- Tesseract OCR if I want text extraction. The classifier still works when Tesseract is absent.

The public source release intentionally excludes raw images and trained `joblib` heads. I must rebuild those local artifacts before the app can analyze an image. I never load a `joblib` file from an untrusted source because Python pickle-based formats can execute code during loading.

Platform boundary:

- Apple Silicon macOS is the validated platform.
- The Windows commands use CPU mode and are the supported cross-platform reproduction path, but the saved latency numbers are not Windows benchmarks.
- Intel macOS is not supported by `requirements-lock.txt`. PyTorch stopped publishing macOS x86_64 binaries after the 2.2 series, while this project pins a newer PyTorch release. I do not substitute an older framework and claim an exact reproduction. I use an Apple Silicon Mac or 64-bit Windows instead.

## 2. Get the code

### macOS Terminal

```bash
git clone https://github.com/Swarnaditya-Maitra/ad-safety-moderation-pipeline.git
cd ad-safety-moderation-pipeline
```

If I already have the course `Project` folder, I open Terminal and change into that folder instead.

### Windows PowerShell

```powershell
git clone https://github.com/Swarnaditya-Maitra/ad-safety-moderation-pipeline.git
Set-Location ad-safety-moderation-pipeline
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

Install Tesseract with Homebrew:

```bash
brew install tesseract
tesseract --version
```

### Windows 10 or 11

1. I install 64-bit Python 3.10 from python.org and enable the installer option that adds Python to `PATH`.
2. I install Git for Windows.
3. I install the 64-bit Tesseract package from the UB Mannheim release page.
4. If the installer does not update `PATH`, I add `C:\Program Files\Tesseract-OCR` to my user `Path`, then close and reopen PowerShell.

Verify the tools:

```powershell
py -3.10 --version
git --version
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
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Confirm that the active interpreter belongs to `.venv`:

```text
python -c "import sys; print(sys.executable); print(sys.version)"
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

## 6. Provision models, data, and the trained policy heads

First check whether the runnable classifier already exists:

```text
python -c "from pathlib import Path; p=Path('models/vit_policy_head.joblib'); print('READY' if p.is_file() and p.stat().st_size else 'BUILD REQUIRED')"
```

If the result is `BUILD REQUIRED`, run the complete local build:

```text
python scripts/download_models.py
python scripts/build_capstone_dataset.py
python scripts/train_and_evaluate.py --device cpu --batch-size 8
```

Important behavior:

- `download_models.py` downloads pinned ViT, ResNet-50, and Grounding DINO snapshots. This uses several gigabytes.
- `build_capstone_dataset.py` reconstructs the bounded dataset from the recorded sources and writes the grouped train, validation, and test split.
- `train_and_evaluate.py` freezes both backbones, fits the logistic policy heads on train data, selects thresholds on validation data, and evaluates test data once.
- `--device cpu` is the cross-platform reproducibility setting used in the saved evidence. On Apple Silicon, `--device mps` can speed feature extraction, but that produces a separate run and must not be reported as the saved CPU benchmark. The detector remains CPU-bound in this pilot.
- A source can become unavailable or change its terms. I inspect `SOURCES.md` and the dataset cards before redistributing rebuilt images. In the public source release, I also inspect `NOTICE.md`.

Verify the existing local assets without rebuilding:

```text
python scripts/download_models.py --offline
python scripts/build_capstone_dataset.py --verify-only
```

Expected result: both commands finish without a missing-file or checksum error.

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

- `MODEL READY` after the trained ViT head exists;
- `EVIDENCE LOADED` when the saved benchmark JSON is available; and
- `NOT PRODUCTION-READY`, which is an intentional scope warning.

### App workflow

1. Use `Analyze` for one creative.
2. Upload one static JPEG, PNG, or WebP. Animated WebP and APNG inputs are rejected because a single-frame check could miss later content.
3. Choose optional evidence in the sidebar:
   - `Object context`: Grounding DINO boxes and phrases. This is slow and diagnostic.
   - `Text extraction`: local Tesseract OCR. It is on by default.
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
  "http://127.0.0.1:8000/analyze?run_detector=false&run_ocr=true&explain=false" \
  -F "file=@/absolute/path/to/creative.jpg;type=image/jpeg" \
  | python -m json.tool
```

### Analyze an image on Windows PowerShell

```powershell
curl.exe -sS -X POST "http://127.0.0.1:8000/analyze?run_detector=false&run_ocr=true&explain=false" -F "file=@C:\absolute\path\to\creative.jpg;type=image/jpeg" | python -m json.tool
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
```

Expected result: zero failures. A pinned-snapshot integration test can be skipped in the source-only checkout before the models are downloaded. After model provisioning, it should run.

Run focused groups when troubleshooting:

```text
python -m pytest -q tests/test_preprocessing.py
python -m pytest -q tests/test_policy.py
python -m pytest -q tests/test_api.py
python -m pytest -q tests/test_app.py
python -m pytest -q tests/test_inference_contract.py
```

## 10. Manual end-to-end test matrix

After the pinned build, these recorded cases should reproduce the three policy actions:

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

This writes `corrupt.png`, a 16-by-16 `tiny.png`, a two-frame `animated.webp`, and an encoded `oversized.png` above 20 MB under `tmp/manual_test_inputs/`. The `tmp/` directory is ignored by Git.

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

This extended audit also needs Poppler `pdftoppm`, LibreOffice for independent PowerPoint rendering, and FFmpeg for video checks. These coursework files and builders are intentionally outside the public source-only release.

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

The public source-only checkout does not contain the PDF, PowerPoint, video, or `validate_deliverables.py`, so it does not need these three tools.

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

I run Section 6. The public repository does not distribute the trained `joblib` heads.

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
