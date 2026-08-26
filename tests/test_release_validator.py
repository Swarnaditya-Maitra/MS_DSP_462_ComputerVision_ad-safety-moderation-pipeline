from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

from scripts import validate_deliverables, validate_release


UNIX_USERS_PREFIX = "/" + "Users" + "/"
WINDOWS_USERS_PREFIX = "C:" + "\\" + "Users" + "\\"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_tracked_policy_finds_cache_secret_blocked_large_and_absolute_path(
    tmp_path: Path,
) -> None:
    tracked = [
        ".cache/state.txt",
        ".env",
        "data/capstone_dataset/train/safe/example.jpg",
        "large.bin",
        "notes.md",
    ]
    for relative_path in tracked:
        _write(tmp_path / relative_path, b"placeholder")
    (tmp_path / "large.bin").open("r+b").truncate(validate_release.LARGE_FILE_LIMIT)
    (tmp_path / "notes.md").write_text(
        f"machine path: {UNIX_USERS_PREFIX}example/private/project/file.txt\n",
        encoding="utf-8",
    )

    result = validate_release.tracked_release_policy(tmp_path, tracked)
    issue_types = {issue["type"] for issue in result["issues"]}

    assert result["passed"] is False
    assert issue_types == {
        "tracked_cache",
        "tracked_secret",
        "tracked_blocked_capstone_image",
        "tracked_file_at_least_100_mib",
        "absolute_users_path_in_tracked_text",
    }


def test_structured_validation_rejects_invalid_json_and_csv(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text('{"broken":', encoding="utf-8")
    (tmp_path / "bad.csv").write_text('a,b\n"unterminated,b\n', encoding="utf-8")
    (tmp_path / "good.json").write_text('{"ok": true}\n', encoding="utf-8")

    result = validate_release.validate_structured_files(
        tmp_path, ["bad.json", "bad.csv", "good.json"]
    )

    assert result["passed"] is False
    assert {(issue["type"], issue["path"]) for issue in result["issues"]} == {
        ("invalid_json", "bad.json"),
        ("invalid_csv", "bad.csv"),
    }


def test_notebook_validation_requires_execution_and_no_error_outputs(tmp_path: Path) -> None:
    notebook_path = tmp_path / "final.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "source": ["raise RuntimeError('boom')"],
                "outputs": [{"output_type": "error", "ename": "RuntimeError"}],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    failed = validate_release.validate_notebook(notebook_path)
    assert {issue["type"] for issue in failed["issues"]} == {
        "non_executed_code_cell",
        "notebook_error_output",
    }

    notebook["cells"][0]["execution_count"] = 1
    notebook["cells"][0]["outputs"] = []
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    passed = validate_release.validate_notebook(notebook_path)
    assert passed["passed"] is True
    assert passed["executed_code_cell_count"] == 1


def test_pptx_validation_requires_a_valid_15_slide_archive(tmp_path: Path) -> None:
    pptx_path = tmp_path / "deck.pptx"
    with zipfile.ZipFile(pptx_path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("ppt/presentation.xml", "<presentation/>")
        for index in range(1, 16):
            archive.writestr(f"ppt/slides/slide{index}.xml", "<slide/>")

    passed = validate_release.validate_pptx(pptx_path)
    assert passed["passed"] is True
    assert passed["slide_count"] == 15

    invalid_path = tmp_path / "invalid.pptx"
    invalid_path.write_bytes(b"not a zip")
    failed = validate_release.validate_pptx(invalid_path)
    assert failed["passed"] is False
    assert failed["issues"][0]["type"] == "invalid_pptx_zip"


def test_capstone_validation_accepts_public_and_full_checkout_modes(tmp_path: Path) -> None:
    expected_counts = {(label, "train"): 1 for label in validate_release.CAPSTONE_LABELS}
    rows: list[dict[str, str]] = []
    for label in validate_release.CAPSTONE_LABELS:
        relative_path = f"data/capstone_dataset/train/{label}/{label}.jpg"
        payload = label.encode("utf-8")
        rows.append(
            {
                "label": label,
                "split": "train",
                "local_path": relative_path,
                "local_sha256": _sha256(payload),
            }
        )
        if label == "financial_promotion":
            _write(tmp_path / relative_path, payload)
    registry = tmp_path / "data/capstone_registry.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    public_result = validate_release.validate_capstone_dataset(
        tmp_path, expected_counts=expected_counts
    )
    assert public_result["passed"] is True
    assert public_result["mode"] == "public_clone"

    for row in rows:
        _write(tmp_path / row["local_path"], row["label"].encode("utf-8"))
    full_result = validate_release.validate_capstone_dataset(
        tmp_path, expected_counts=expected_counts
    )
    assert full_result["passed"] is True
    assert full_result["mode"] == "full_local"

    (tmp_path / rows[0]["local_path"]).write_bytes(b"corrupt")
    corrupt_result = validate_release.validate_capstone_dataset(
        tmp_path, expected_counts=expected_counts
    )
    assert corrupt_result["passed"] is False
    assert any(
        issue["type"] == "capstone_file_hash_mismatch"
        for issue in corrupt_result["issues"]
    )


def test_wikimedia_validation_checks_only_retained_public_files(tmp_path: Path) -> None:
    registry_rows = []
    manifest_rows = []
    for index, include in enumerate((True, True, False)):
        relative_path = f"data/wikimedia_pilot/sample-{index}.jpg"
        payload = f"sample-{index}".encode("utf-8")
        registry_rows.append(
            {"local_path": relative_path, "local_sha256": _sha256(payload)}
        )
        manifest_rows.append(
            {
                "local_path": relative_path,
                "include_in_spot_check": str(include).lower(),
            }
        )
        if include:
            _write(tmp_path / relative_path, payload)

    for filename, rows in (
        ("dataset_registry.csv", registry_rows),
        ("wikimedia_external_manifest.csv", manifest_rows),
    ):
        path = tmp_path / "data" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    passed = validate_release.validate_wikimedia_dataset(
        tmp_path, expected_registry_count=3, expected_retained_count=2
    )
    assert passed["passed"] is True
    assert passed["retained_file_count"] == 2

    (tmp_path / registry_rows[0]["local_path"]).write_bytes(b"changed")
    failed = validate_release.validate_wikimedia_dataset(
        tmp_path, expected_registry_count=3, expected_retained_count=2
    )
    assert failed["passed"] is False
    assert any(
        issue["type"] == "wikimedia_file_hash_mismatch" for issue in failed["issues"]
    )


def test_stale_alias_and_path_free_report(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.txt"
    alias = tmp_path / "alias.txt"
    canonical.write_text("same", encoding="utf-8")
    alias.write_text("same", encoding="utf-8")

    aliases = validate_release.find_stale_duplicate_aliases(
        tmp_path, [("alias.txt", "canonical.txt")]
    )
    assert aliases["passed"] is False
    assert aliases["issues"][0]["type"] == "stale_exact_duplicate_alias"

    report = {
        "all_checks_passed": False,
        "detail": f"{UNIX_USERS_PREFIX}example/private/project/file.txt",
    }
    output = validate_release.write_report(tmp_path, report)
    payload = output.read_text(encoding="utf-8")
    assert UNIX_USERS_PREFIX not in payload
    assert "<absolute-path-redacted>" in payload


def test_deliverable_report_is_path_free_for_success_and_exception_payloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "final_validation.json"
    monkeypatch.setattr(validate_deliverables, "RESULT_PATH", output)
    root_text = str(validate_deliverables.ROOT)
    report = {
        "all_checks_passed": False,
        "project_root": root_text,
        "checks": {
            "success_detail": {
                "passed": True,
                "artifact": f"{root_text}/outputs/report/final.pdf",
            },
            "exception_detail": {
                "passed": False,
                "message": (
                    f"{UNIX_USERS_PREFIX}example/private/project/file.py failed"
                ),
                "traceback": (
                    f'  File "{WINDOWS_USERS_PREFIX}'
                    'example\\project\\validator.py", line 10\n'
                    f'  File "{root_text}/scripts/validate_deliverables.py", line 20'
                ),
            },
        },
    }

    validate_deliverables.write_result(report)
    payload = output.read_text(encoding="utf-8")
    parsed = json.loads(payload)

    assert parsed["project_root"] == "."
    assert root_text not in payload
    assert UNIX_USERS_PREFIX not in payload
    assert WINDOWS_USERS_PREFIX not in payload
    assert "<ABSOLUTE_PATH>" in payload
