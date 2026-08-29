from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest
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


def test_canonical_output_contract_is_complete_and_path_free() -> None:
    if os.environ.get("AD_SAFETY_FINAL_VALIDATION_RUNNING") == "1":
        pytest.skip("The final validation record is written after the validator's embedded test run")

    summary_path = PROJECT_ROOT / "outputs" / "validation" / "validation_summary.json"
    detail_path = PROJECT_ROOT / "outputs" / "validation" / "final_validation.json"
    summary_text = summary_path.read_text(encoding="utf-8")
    detail_text = detail_path.read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    detail = json.loads(detail_text)

    assert summary["all_checks_passed"] is True
    assert detail["all_checks_passed"] is True
    assert summary["check_count"] == len(summary["checks"]) == detail["check_count"] == 16
    assert summary["checks"] == sorted(detail["checks"])
    assert summary["source_validation_sha256"] == _sha256(detail_path)
    for payload in (summary_text, detail_text):
        assert str(PROJECT_ROOT.resolve()) not in payload
        assert "/Users/" not in payload
        assert "C:\\Users\\" not in payload
    marker = PROJECT_ROOT / "outputs" / "validation" / "FINAL_VALIDATION_PASSED.txt"
    assert marker.read_text(encoding="utf-8") == "FINAL_VALIDATION_PASSED\n"


def test_published_jupyter_book_workflow_is_reproducible() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements-book.txt").read_text(encoding="utf-8")

    assert "python scripts/build_book.py --builder jupyter-book --no-execute" in workflow
    assert "actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d" in workflow
    assert "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9" in workflow
    assert "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128" in workflow
    assert "jupyter-book==1.0.4.post1" in requirements
    assert "nbformat==5.10.4" in requirements
    assert "nbconvert==7.17.0" in requirements


def test_canonical_evaluation_charts_are_readable_pngs() -> None:
    expected = {
        "confusion_matrix_resnet50.png",
        "confusion_matrix_vit.png",
        "dataset_distribution.png",
        "model_comparison.png",
        "precision_recall_curves.png",
        "threshold_calibration.png",
    }
    evaluation_dir = PROJECT_ROOT / "outputs" / "evaluation"
    checksum_path = evaluation_dir / "output_checksums.csv"
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
