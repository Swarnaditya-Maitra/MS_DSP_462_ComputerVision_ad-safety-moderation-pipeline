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
- Saved metrics and predictions are under `results/`.
- Aggregate charts and the path-free course-run validation summary are under `outputs/`.

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

## Independent rebuild

The included trained heads make a new clone runnable without retraining. To reproduce the training workflow instead, rebuild the recorded dataset and retrain on CPU:

```text
python scripts/download_models.py --skip-grounding-dino
python scripts/build_capstone_dataset.py
python scripts/train_and_evaluate.py --device cpu --batch-size 8
```

That path depends on third-party data still being available under suitable terms. It can reproduce the method, but future downloads are not guaranteed to reproduce the original source bytes. Do not report new metrics as the saved course-run metrics unless the checksums and validation records match.

## Public artifact boundary

The public repository excludes raw creatives, detector overlays, image-bearing notebooks, the report, the deck, and the narrated video. Some embedded media lacks item-level redistribution records, some deliverables identify contributors, and machine-specific logs contain private absolute paths. See `NOTICE.md` and `outputs/README.md` for the exact boundary.

## Sources

- Python virtual environments: https://docs.python.org/3.10/library/venv.html
- Hugging Face Hub download API: https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- ViT model card: https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
- ResNet-50 model card: https://huggingface.co/timm/resnet50.a1_in1k
- Grounding DINO Tiny model card: https://huggingface.co/IDEA-Research/grounding-dino-tiny
