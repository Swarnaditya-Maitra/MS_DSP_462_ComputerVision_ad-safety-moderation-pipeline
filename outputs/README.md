# Curated public outputs

This directory contains a deliberately small, public-safe subset of the validated course run. The evaluation charts contain aggregate measurements only. The validation files contain a path-free pass marker and summary.

The full local build also creates predictions, demo creatives, overlays, rendered slides, narration, video, and machine-specific validation logs. Those files are not published here because some source images lack item-level redistribution records, contributor consent has not been recorded for every deliverable, and several logs expose absolute local paths. Machine-readable aggregate results remain under [`../results/`](../results/).

These files are evidence from the saved course run. They are not regenerated during normal app startup and do not prove that a new machine is ready. Run `python scripts/preflight.py --profile core` after setup to verify the current clone.

## Included files

- `evaluation/confusion_matrix.png`: primary ViT confusion matrix.
- `evaluation/confusion_matrix_resnet50.png`: ResNet-50 baseline confusion matrix.
- `evaluation/confusion_matrix_vit.png`: explicitly named ViT confusion matrix.
- `evaluation/dataset_distribution.png`: grouped split and label counts.
- `evaluation/model_comparison.png`: saved ViT and ResNet-50 comparison.
- `evaluation/precision_recall_curves.png`: saved one-vs-rest curves.
- `evaluation/threshold_calibration.png`: validation-set threshold selection.
- `validation/FINAL_VALIDATION_PASSED.txt`: pass marker from the completed local validation.
- `validation/validation_summary.json`: public, path-free summary of that validation.

See [`../NOTICE.md`](../NOTICE.md) for the release and rights boundary.
