#!/usr/bin/env python3
"""Validate the portable public release and the optional full local dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_RELATIVE_PATH = Path("outputs/validation/release_validation.json")
LARGE_FILE_LIMIT = 100 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".srt",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
ABSOLUTE_USERS_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+/[^\s\"'`<>]+")

REQUIRED_DELIVERABLES = {
    "final_notebook": "ad_safety_moderation_pipeline.ipynb",
    "proposal_pdf": "Capstone Project Idea - Ad Safety.pdf",
    "report_pdf": "outputs/report/ad_safety_technical_synopsis.pdf",
    "presentation_pptx": "outputs/presentation/ad_safety_management_presentation.pptx",
    "speaker_script_md": "outputs/presentation/ad_safety_speaker_script.md",
    "final_video": "outputs/video/ad_safety_demo.mp4",
    "final_subtitles": "outputs/video/ad_safety_demo.srt",
}

CACHE_DIRECTORY_NAMES = {
    ".cache",
    ".ipynb_checkpoints",
    ".mpl-cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".torch-cache",
    ".venv",
    "__pycache__",
    "htmlcov",
    "node_modules",
    "tmp",
    "venv",
}
SECRET_EXACT_PATHS = {
    ".streamlit/secrets.toml",
    "kaggle.json",
    "pkcs11.txt",
}
SECRET_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
BLOCKED_CAPSTONE_LABELS = {"safe", "firearms", "explosives"}

STALE_ALIAS_PAIRS = (
    ("book/ad_safety_moderation_pipeline.ipynb", "ad_safety_moderation_pipeline.ipynb"),
    ("outputs/video/captions.srt", "outputs/video/ad_safety_demo.srt"),
    ("outputs/video/demo_narration.md", "outputs/video/narration_script.md"),
    ("outputs/video/demo_storyboard.json", "outputs/video/storyboard.json"),
)

CAPSTONE_LABELS = ("safe", "firearms", "explosives", "financial_promotion")
CAPSTONE_SPLIT_COUNTS = {"train": 48, "val": 12, "test": 12}
EXPECTED_CAPSTONE_COUNTS = {
    (label, split): count
    for label in CAPSTONE_LABELS
    for split, count in CAPSTONE_SPLIT_COUNTS.items()
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_result(issues: list[dict[str, Any]], **details: Any) -> dict[str, Any]:
    return {"passed": not issues, "issue_count": len(issues), "issues": issues, **details}


def safe_repo_path(root: Path, relative_path: str) -> Path | None:
    parsed = PurePosixPath(relative_path)
    if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
        return None
    return root.joinpath(*parsed.parts)


def read_csv_dicts(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            if not reader.fieldnames or any(not field for field in reader.fieldnames):
                return [], ["missing_or_empty_header"]
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return [], ["unreadable_or_malformed_csv"]
    return rows, []


def git_tracked_files(root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        files = [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]
        return sorted(files), []
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return [], [{"type": "git_tracked_file_inventory_failed"}]


def is_cache_path(relative_path: str) -> bool:
    parts = {part.casefold() for part in PurePosixPath(relative_path).parts}
    return bool(parts & CACHE_DIRECTORY_NAMES)


def is_secret_path(relative_path: str) -> bool:
    normalized = relative_path.casefold()
    name = PurePosixPath(normalized).name
    if normalized in SECRET_EXACT_PATHS:
        return True
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    if "credential" in name or "secrets" in PurePosixPath(normalized).parts:
        return True
    return PurePosixPath(name).suffix in SECRET_SUFFIXES


def is_blocked_capstone_image(relative_path: str) -> bool:
    parsed = PurePosixPath(relative_path)
    parts = parsed.parts
    if len(parts) < 5 or parts[0:2] != ("data", "capstone_dataset"):
        return False
    return parts[3] in BLOCKED_CAPSTONE_LABELS and parsed.suffix.casefold() in IMAGE_SUFFIXES


def tracked_release_policy(
    root: Path,
    tracked_files: Sequence[str],
    *,
    large_file_limit: int = LARGE_FILE_LIMIT,
) -> dict[str, Any]:
    cache_paths: list[str] = []
    secret_paths: list[str] = []
    blocked_images: list[str] = []
    oversized_files: list[dict[str, Any]] = []
    absolute_path_files: list[str] = []

    for relative_path in tracked_files:
        normalized = PurePosixPath(relative_path).as_posix()
        path = safe_repo_path(root, normalized)
        if path is None or not path.exists():
            continue
        if is_cache_path(normalized):
            cache_paths.append(normalized)
        if is_secret_path(normalized):
            secret_paths.append(normalized)
        if is_blocked_capstone_image(normalized):
            blocked_images.append(normalized)
        if path.is_file() and path.stat().st_size >= large_file_limit:
            oversized_files.append({"path": normalized, "bytes": path.stat().st_size})
        if path.is_file() and path.suffix.casefold() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if ABSOLUTE_USERS_PATH.search(text):
                absolute_path_files.append(normalized)

    issues: list[dict[str, Any]] = []
    for issue_type, paths in (
        ("tracked_cache", cache_paths),
        ("tracked_secret", secret_paths),
        ("tracked_blocked_capstone_image", blocked_images),
        ("absolute_users_path_in_tracked_text", absolute_path_files),
    ):
        issues.extend({"type": issue_type, "path": item} for item in sorted(set(paths)))
    issues.extend(
        {"type": "tracked_file_at_least_100_mib", **item}
        for item in sorted(oversized_files, key=lambda row: row["path"])
    )
    return check_result(
        issues,
        tracked_file_count=len(tracked_files),
        large_file_limit_bytes=large_file_limit,
        cache_count=len(set(cache_paths)),
        secret_count=len(set(secret_paths)),
        blocked_image_count=len(set(blocked_images)),
        oversized_file_count=len(oversized_files),
        absolute_path_file_count=len(set(absolute_path_files)),
    )


def validate_structured_files(root: Path, tracked_files: Sequence[str]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    json_count = 0
    csv_count = 0
    for relative_path in tracked_files:
        if relative_path == RESULT_RELATIVE_PATH.as_posix():
            continue
        path = safe_repo_path(root, relative_path)
        if path is None or not path.is_file():
            continue
        suffix = path.suffix.casefold()
        if suffix in {".json", ".ipynb"}:
            json_count += 1
            try:
                json.loads(
                    path.read_text(encoding="utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"Invalid JSON constant: {value}")
                    ),
                )
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                issues.append({"type": "invalid_json", "path": relative_path})
        elif suffix == ".csv":
            csv_count += 1
            try:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    rows = list(csv.reader(handle, strict=True))
                width = len(rows[0]) if rows else 0
                if width == 0 or any(len(row) != width for row in rows):
                    raise csv.Error("empty or inconsistent CSV rows")
            except (OSError, UnicodeError, csv.Error):
                issues.append({"type": "invalid_csv", "path": relative_path})
    return check_result(issues, json_files_checked=json_count, csv_files_checked=csv_count)


def validate_required_deliverables(root: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    sizes: dict[str, int] = {}
    for role, relative_path in REQUIRED_DELIVERABLES.items():
        path = root / relative_path
        if not path.is_file():
            issues.append({"type": "missing_final_deliverable", "role": role, "path": relative_path})
            continue
        size = path.stat().st_size
        sizes[role] = size
        if size <= 0:
            issues.append({"type": "empty_final_deliverable", "role": role, "path": relative_path})
    for role in ("proposal_pdf", "report_pdf"):
        relative_path = REQUIRED_DELIVERABLES[role]
        path = root / relative_path
        if path.is_file() and path.stat().st_size > 0:
            try:
                if not path.read_bytes().startswith(b"%PDF-"):
                    issues.append({"type": "invalid_pdf_signature", "role": role, "path": relative_path})
            except OSError:
                issues.append({"type": "unreadable_final_deliverable", "role": role, "path": relative_path})
    return check_result(issues, required_count=len(REQUIRED_DELIVERABLES), sizes_bytes=sizes)


def validate_notebook(path: Path, relative_path: str | None = None) -> dict[str, Any]:
    display_path = relative_path or path.name
    issues: list[dict[str, Any]] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return check_result(
            [{"type": "invalid_notebook_json", "path": display_path}],
            code_cell_count=0,
            executed_code_cell_count=0,
        )
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return check_result(
            [{"type": "invalid_notebook_cells", "path": display_path}],
            code_cell_count=0,
            executed_code_cell_count=0,
        )

    code_cell_count = 0
    executed_count = 0
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source_text = "".join(source) if isinstance(source, list) else str(source)
        if not source_text.strip():
            continue
        code_cell_count += 1
        if cell.get("execution_count") is None:
            issues.append({"type": "non_executed_code_cell", "cell_index": cell_index})
        else:
            executed_count += 1
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            issues.append({"type": "invalid_notebook_outputs", "cell_index": cell_index})
            continue
        for output_index, output in enumerate(outputs):
            if isinstance(output, dict) and output.get("output_type") == "error":
                issues.append(
                    {
                        "type": "notebook_error_output",
                        "cell_index": cell_index,
                        "output_index": output_index,
                    }
                )
    return check_result(
        issues,
        code_cell_count=code_cell_count,
        executed_code_cell_count=executed_count,
    )


def validate_pptx(path: Path, expected_slide_count: int = 15) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    slide_count = 0
    if not path.is_file():
        return check_result(
            [{"type": "missing_pptx", "path": path.name}],
            slide_count=0,
            expected_slide_count=expected_slide_count,
        )
    if not zipfile.is_zipfile(path):
        return check_result(
            [{"type": "invalid_pptx_zip", "path": path.name}],
            slide_count=0,
            expected_slide_count=expected_slide_count,
        )
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            bad_member = archive.testzip()
            required_members = {"[Content_Types].xml", "ppt/presentation.xml"}
            missing_members = sorted(required_members - names)
            if bad_member is not None:
                issues.append({"type": "corrupt_pptx_member", "member": bad_member})
            if missing_members:
                issues.append({"type": "missing_pptx_members", "members": missing_members})
            slide_count = sum(
                bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", member)) for member in names
            )
    except (OSError, zipfile.BadZipFile):
        issues.append({"type": "invalid_pptx_zip", "path": path.name})
    if slide_count != expected_slide_count:
        issues.append(
            {
                "type": "incorrect_pptx_slide_count",
                "expected": expected_slide_count,
                "actual": slide_count,
            }
        )
    return check_result(
        issues,
        slide_count=slide_count,
        expected_slide_count=expected_slide_count,
    )


def validate_video(root: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    sizes: dict[str, int] = {}
    for role in ("final_video", "final_subtitles"):
        relative_path = REQUIRED_DELIVERABLES[role]
        path = root / relative_path
        if not path.is_file():
            issues.append({"type": "missing_video_artifact", "role": role, "path": relative_path})
            continue
        size = path.stat().st_size
        sizes[role] = size
        if size <= 0:
            issues.append({"type": "empty_video_artifact", "role": role, "path": relative_path})
    return check_result(issues, sizes_bytes=sizes)


def format_counts(counts: Mapping[tuple[str, str], int]) -> dict[str, int]:
    return {
        f"{label}:{split}": count
        for (label, split), count in sorted(counts.items())
    }


def validate_capstone_dataset(
    root: Path,
    *,
    expected_counts: Mapping[tuple[str, str], int] = EXPECTED_CAPSTONE_COUNTS,
    public_label: str = "financial_promotion",
) -> dict[str, Any]:
    registry_path = root / "data/capstone_registry.csv"
    rows, csv_issues = read_csv_dicts(registry_path)
    issues: list[dict[str, Any]] = [
        {"type": issue, "path": "data/capstone_registry.csv"} for issue in csv_issues
    ]
    expected_total = sum(expected_counts.values())
    required_fields = {"label", "split", "local_path", "local_sha256"}
    if rows and not required_fields.issubset(rows[0]):
        issues.append({"type": "capstone_registry_missing_fields"})
        rows = []
    if len(rows) != expected_total:
        issues.append(
            {"type": "capstone_registry_row_count", "expected": expected_total, "actual": len(rows)}
        )

    actual_counts = Counter((row.get("label", ""), row.get("split", "")) for row in rows)
    if dict(actual_counts) != dict(expected_counts):
        issues.append(
            {
                "type": "capstone_registry_class_split_counts",
                "expected": format_counts(expected_counts),
                "actual": format_counts(actual_counts),
            }
        )

    relative_paths = [row.get("local_path", "") for row in rows]
    if len(set(relative_paths)) != len(relative_paths):
        issues.append({"type": "duplicate_capstone_registry_paths"})
    expected_paths = set(relative_paths)
    present_rows: list[dict[str, str]] = []
    invalid_registry_paths: list[str] = []
    hash_mismatches: list[str] = []
    for row in rows:
        relative_path = row.get("local_path", "")
        path = safe_repo_path(root, relative_path)
        if path is None:
            invalid_registry_paths.append(relative_path)
            continue
        if not path.is_file():
            continue
        present_rows.append(row)
        if sha256_file(path) != row.get("local_sha256"):
            hash_mismatches.append(relative_path)
    issues.extend(
        {"type": "invalid_capstone_registry_path", "path": path}
        for path in sorted(invalid_registry_paths)
    )
    issues.extend(
        {"type": "capstone_file_hash_mismatch", "path": path}
        for path in sorted(hash_mismatches)
    )

    public_rows = [row for row in rows if row.get("label") == public_label]
    missing_public: list[str] = []
    for row in public_rows:
        relative_path = row.get("local_path", "")
        path = safe_repo_path(root, relative_path)
        if path is None or not path.is_file():
            missing_public.append(relative_path)
    issues.extend(
        {"type": "missing_public_capstone_artifact", "path": path}
        for path in sorted(missing_public)
    )

    data_root = root / "data/capstone_dataset"
    discovered = {
        path.relative_to(root).as_posix()
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    } if data_root.is_dir() else set()
    unexpected = sorted(discovered - expected_paths)
    issues.extend({"type": "unregistered_capstone_image", "path": path} for path in unexpected)

    all_present = len(rows) == expected_total and len(present_rows) == expected_total
    if all_present and discovered != expected_paths:
        issues.append(
            {
                "type": "formal_local_dataset_file_count",
                "expected": expected_total,
                "actual": len(discovered),
            }
        )

    blocked_present_count = sum(
        row.get("label") in BLOCKED_CAPSTONE_LABELS for row in present_rows
    )
    if all_present:
        mode = "full_local"
    elif blocked_present_count == 0:
        mode = "public_clone"
    else:
        mode = "partial_local"
    return check_result(
        issues,
        mode=mode,
        registry_row_count=len(rows),
        expected_registry_row_count=expected_total,
        registered_file_count=len(present_rows),
        discovered_image_count=len(discovered),
        public_artifact_count=len(public_rows) - len(missing_public),
        blocked_local_file_count=blocked_present_count,
        full_local_dataset_present=all_present,
        file_hashes_checked=len(present_rows),
    )


def validate_wikimedia_dataset(
    root: Path,
    *,
    expected_registry_count: int = 27,
    expected_retained_count: int = 26,
) -> dict[str, Any]:
    registry_rows, registry_csv_issues = read_csv_dicts(root / "data/dataset_registry.csv")
    manifest_rows, manifest_csv_issues = read_csv_dicts(
        root / "data/wikimedia_external_manifest.csv"
    )
    issues: list[dict[str, Any]] = [
        {"type": issue, "path": "data/dataset_registry.csv"}
        for issue in registry_csv_issues
    ]
    issues.extend(
        {"type": issue, "path": "data/wikimedia_external_manifest.csv"}
        for issue in manifest_csv_issues
    )
    if len(registry_rows) != expected_registry_count:
        issues.append(
            {
                "type": "wikimedia_registry_row_count",
                "expected": expected_registry_count,
                "actual": len(registry_rows),
            }
        )
    if len(manifest_rows) != expected_registry_count:
        issues.append(
            {
                "type": "wikimedia_manifest_row_count",
                "expected": expected_registry_count,
                "actual": len(manifest_rows),
            }
        )

    registry_by_path = {row.get("local_path", ""): row for row in registry_rows}
    if len(registry_by_path) != len(registry_rows):
        issues.append({"type": "duplicate_wikimedia_registry_paths"})
    retained_paths: list[str] = []
    for row in manifest_rows:
        include = row.get("include_in_spot_check", "").strip().casefold()
        if include not in {"true", "false"}:
            issues.append(
                {"type": "invalid_wikimedia_include_flag", "path": row.get("local_path", "")}
            )
        elif include == "true":
            retained_paths.append(row.get("local_path", ""))
    if len(retained_paths) != expected_retained_count:
        issues.append(
            {
                "type": "wikimedia_retained_count",
                "expected": expected_retained_count,
                "actual": len(retained_paths),
            }
        )
    if len(set(retained_paths)) != len(retained_paths):
        issues.append({"type": "duplicate_wikimedia_retained_paths"})

    hashes: list[str] = []
    present_count = 0
    for relative_path in retained_paths:
        registry_row = registry_by_path.get(relative_path)
        if registry_row is None:
            issues.append({"type": "retained_wikimedia_path_not_registered", "path": relative_path})
            continue
        path = safe_repo_path(root, relative_path)
        if path is None or not path.is_file():
            issues.append({"type": "missing_retained_wikimedia_file", "path": relative_path})
            continue
        present_count += 1
        actual_hash = sha256_file(path)
        expected_hash = registry_row.get("local_sha256", "")
        hashes.append(actual_hash)
        if actual_hash != expected_hash:
            issues.append({"type": "wikimedia_file_hash_mismatch", "path": relative_path})
    if len(set(hashes)) != len(hashes):
        issues.append({"type": "duplicate_retained_wikimedia_hash"})

    data_root = root / "data/wikimedia_pilot"
    discovered = {
        path.relative_to(root).as_posix()
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    } if data_root.is_dir() else set()
    unknown = sorted(discovered - set(registry_by_path))
    issues.extend({"type": "unregistered_wikimedia_image", "path": path} for path in unknown)
    return check_result(
        issues,
        registry_row_count=len(registry_rows),
        manifest_row_count=len(manifest_rows),
        expected_retained_count=expected_retained_count,
        retained_file_count=present_count,
        local_image_count=len(discovered),
        file_hashes_checked=present_count,
    )


def find_stale_duplicate_aliases(
    root: Path,
    alias_pairs: Sequence[tuple[str, str]] = STALE_ALIAS_PAIRS,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    checked = 0
    for alias, canonical in alias_pairs:
        alias_path = root / alias
        canonical_path = root / canonical
        if not alias_path.is_file() or not canonical_path.is_file():
            continue
        checked += 1
        if alias_path.stat().st_size == canonical_path.stat().st_size and sha256_file(
            alias_path
        ) == sha256_file(canonical_path):
            issues.append(
                {"type": "stale_exact_duplicate_alias", "alias": alias, "canonical": canonical}
            )
    return check_result(issues, existing_alias_pairs_checked=checked)


def scrub_path_values(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {str(key): scrub_path_values(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_path_values(item, root) for item in value]
    if isinstance(value, tuple):
        return [scrub_path_values(item, root) for item in value]
    if isinstance(value, str):
        cleaned = value.replace(str(root.resolve()), ".")
        return ABSOLUTE_USERS_PATH.sub("<absolute-path-redacted>", cleaned)
    return value


def run_validation(
    root: Path = ROOT,
    *,
    tracked_files_override: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if tracked_files_override is None:
        tracked_files, git_issues = git_tracked_files(root)
    else:
        tracked_files = sorted(PurePosixPath(path).as_posix() for path in tracked_files_override)
        git_issues = []

    checks: dict[str, dict[str, Any]] = {}
    checks["tracked_file_inventory"] = check_result(
        git_issues, tracked_file_count=len(tracked_files)
    )
    checks["required_deliverables"] = validate_required_deliverables(root)
    checks["tracked_release_policy"] = tracked_release_policy(root, tracked_files)
    checks["structured_files"] = validate_structured_files(root, tracked_files)
    checks["capstone_dataset"] = validate_capstone_dataset(root)
    checks["wikimedia_dataset"] = validate_wikimedia_dataset(root)
    checks["notebook_execution"] = validate_notebook(
        root / REQUIRED_DELIVERABLES["final_notebook"],
        REQUIRED_DELIVERABLES["final_notebook"],
    )
    checks["presentation_archive"] = validate_pptx(
        root / REQUIRED_DELIVERABLES["presentation_pptx"]
    )
    checks["video_artifacts"] = validate_video(root)
    checks["stale_duplicate_aliases"] = find_stale_duplicate_aliases(root)
    failed_checks = sorted(name for name, result in checks.items() if not result["passed"])
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": RESULT_RELATIVE_PATH.as_posix(),
        "all_checks_passed": not failed_checks,
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "failed_checks": failed_checks,
        "checks": checks,
    }
    return scrub_path_values(report, root)


def write_report(root: Path, report: Mapping[str, Any]) -> Path:
    output_path = root / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    path_free_report = scrub_path_values(dict(report), root)
    payload = json.dumps(path_free_report, indent=2, sort_keys=True) + "\n"
    if ABSOLUTE_USERS_PATH.search(payload):
        raise RuntimeError("Release validation output contains an absolute user path")
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to validate")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        report = run_validation(root)
    except Exception as error:  # pragma: no cover - last-resort path-free report
        report = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output": RESULT_RELATIVE_PATH.as_posix(),
            "all_checks_passed": False,
            "check_count": 1,
            "failed_check_count": 1,
            "failed_checks": ["validator_internal_error"],
            "checks": {
                "validator_internal_error": {
                    "passed": False,
                    "issue_count": 1,
                    "issues": [{"type": type(error).__name__}],
                }
            },
        }
    output_path = write_report(root, report)
    status = "PASSED" if report["all_checks_passed"] else "FAILED"
    print(f"Release validation {status}: {output_path.relative_to(root).as_posix()}")
    if not report["all_checks_passed"]:
        print("Failed checks: " + ", ".join(report["failed_checks"]))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
