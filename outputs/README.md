# Canonical project outputs

`outputs/` is the single canonical location for saved evaluation evidence, demo cases, final course deliverables, and validation records. There is no parallel `results/` tree. Normal application startup reads its saved dashboard evidence from this directory but does not regenerate or modify it.

These artifacts document the validated course run. They do not prove that a new machine is ready. Use [`../USER_MANUAL.md`](../USER_MANUAL.md) for the supported macOS and Windows setup and test sequence, and [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the validation contract.

Some outputs contain bounded views of source examples. [`../DATASETS.md`](../DATASETS.md) is the sole detailed inventory and provenance record; those views do not change the underlying reuse terms.

## Directory map

### `evaluation/`

This directory contains the complete saved model-evaluation record:

- `metrics.json`, `evaluation_metrics.json`, and `independent_validation.json` record the primary metrics, model comparisons, dataset checks, and independent validation;
- `predictions.csv`, `test_predictions.csv`, `failure_cases.csv`, `per_class_metrics.csv`, `thresholds.csv`, and `split_summary.csv` retain row-level and aggregate evidence;
- `embeddings_vit.npz` and `embeddings_resnet50.npz` are the saved frozen-backbone feature matrices used by the training workflow;
- `benchmark_cpu_batch1.csv`, `latency.json`, `metrics_summary.csv`, and `model_comparison.csv` retain the measured comparison data;
- `confusion_matrix_vit.png`, `confusion_matrix_resnet50.png`, `dataset_distribution.png`, `dataset_contact_sheet.jpg`, `model_comparison.png`, `precision_recall_curves.png`, and `threshold_calibration.png` are the canonical figures;
- `external_spot_check.csv`, `external_spot_check.json`, and `external_annotated/` contain the separate Wikimedia diagnostic record and annotated examples; and
- `evaluation_manifest.json` plus `output_checksums.csv` record the saved artifact contract and checksums.

The generic duplicate `confusion_matrix.png` alias is not retained. The primary matrix has the explicit name `confusion_matrix_vit.png`.

### `demo_cases/`

`case_summary.csv` and `case_summary.json` record six representative cases. Each case directory contains its audit JSON and derived evidence overlay, plus an occlusion heatmap where that option was enabled. Source-image copies are not duplicated here; each audit records its canonical path under `data/`. Paths for the local-only third-party formal cases are intentionally absent from a public clone until the dataset is reconstructed.

### `app/`

This directory contains the final browser-validation record and the canonical Streamlit screenshots used by the report, deck, and video. Duplicate `*_deck` and `*_redesign` aliases are not retained.

### Final deliverables

- `report/ad_safety_technical_synopsis.pdf` is the final two-page technical report.
- `presentation/ad_safety_management_presentation.pptx` is the final 15-slide deck. Its montage and JSON files retain the independent rendering and layout QA evidence.
- `presentation/ad_safety_speaker_script.md` follows the deck slide by slide and is written for a live team presentation.
- `video/ad_safety_demo.mp4` is the final narrated demonstration with an embedded subtitle stream. Its `narration_script.md` follows 19 timed video scenes and supports the saved narration workflow; it is not a duplicate of the live 15-slide speaker script. The adjacent SRT, storyboard, provenance, manifest, and QA report are the remaining canonical video support files.

The final executed technical notebook is tracked at [`../ad_safety_moderation_pipeline.ipynb`](../ad_safety_moderation_pipeline.ipynb), and the original proposal is [`../Capstone Project Idea - Ad Safety.pdf`](../Capstone%20Project%20Idea%20-%20Ad%20Safety.pdf).

### `validation/`

- `FINAL_VALIDATION_PASSED.txt` is the saved pass marker.
- `validation_summary.json` is the concise validation summary.
- `final_validation.json` is the detailed cross-artifact audit record.

Run `python scripts/validate_release.py` for the portable public-clone boundary. The stronger `python scripts/validate_deliverables.py` audit requires the complete registered dataset, the full pinned model profile, Poppler, LibreOffice, and FFmpeg. It builds and renders into temporary directories, validates the tracked final artifacts directly, and writes a path-free `final_validation.json`; ignored render, audio-segment, and QA-frame caches are not inputs.

## Intentionally untracked intermediates

Model download caches, Python environments, render scratch directories, duplicate aliases, presentation slide-render folders, temporary narration segments, video scene fragments, extracted QA frames, and other rebuildable intermediates remain ignored. They are not unique project inputs or final deliverables. The builders recreate them when needed.

The tracked final MP4 and SRT are cross-platform validation inputs. See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) and [`../USER_MANUAL.md`](../USER_MANUAL.md) for the platform boundary around regenerating the original offline narration.

See [`../NOTICE.md`](../NOTICE.md) for dataset rights, attribution, contributor-identification, and repository-license limits.
