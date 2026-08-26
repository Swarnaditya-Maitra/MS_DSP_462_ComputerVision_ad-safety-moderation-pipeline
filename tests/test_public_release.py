from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_published_trained_heads_match_manifest() -> None:
    manifest_path = PROJECT_ROOT / "models" / "trained_head_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1.0.0"
    assert {row["role"] for row in manifest["artifacts"]} == {"primary", "baseline"}
    for row in manifest["artifacts"]:
        relative_path = Path(row["path"])
        assert not relative_path.is_absolute()
        artifact = PROJECT_ROOT / relative_path
        assert artifact.stat().st_size == row["bytes"]
        assert _sha256(artifact) == row["sha256"]


def test_curated_output_contract_is_complete_and_path_free() -> None:
    summary_path = PROJECT_ROOT / "outputs" / "validation" / "validation_summary.json"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_text)

    assert summary["all_checks_passed"] is True
    assert summary["check_count"] == len(summary["checks"]) == 15
    assert "/Users/" not in summary_text
    assert "C:\\Users\\" not in summary_text
    assert (PROJECT_ROOT / "outputs" / "validation" / "FINAL_VALIDATION_PASSED.txt").is_file()


def test_curated_evaluation_charts_are_readable_pngs() -> None:
    expected = {
        "confusion_matrix.png",
        "confusion_matrix_resnet50.png",
        "confusion_matrix_vit.png",
        "dataset_distribution.png",
        "model_comparison.png",
        "precision_recall_curves.png",
        "threshold_calibration.png",
    }
    evaluation_dir = PROJECT_ROOT / "outputs" / "evaluation"
    checksum_path = PROJECT_ROOT / "results" / "evaluation" / "output_checksums.csv"
    with checksum_path.open(newline="", encoding="utf-8") as handle:
        checksum_rows = {row["path"]: row for row in csv.DictReader(handle)}

    assert {path.name for path in evaluation_dir.glob("*.png")} == expected
    for path in evaluation_dir.glob("*.png"):
        row = checksum_rows[f"outputs/evaluation/{path.name}"]
        assert path.stat().st_size == int(row["bytes"])
        assert _sha256(path) == row["sha256"]
        with Image.open(path) as image:
            image.verify()
            assert image.format == "PNG"
