#!/usr/bin/env python3
"""Offline readiness checks for the supported local ad-safety hosts.

The core profile verifies everything needed by the ViT classifier path.  The
full profile additionally requires the baseline ResNet-50 head and snapshot,
Grounding DINO snapshot, and Tesseract executable.  The checker only reads local
files and imports local packages; it never downloads models or contacts a
service.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import re
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = "1.0.0"
TRAINED_HEAD_MANIFEST = Path("models/trained_head_manifest.json")
PRETRAINED_MODEL_MANIFEST = Path("models/pretrained_model_manifest.json")
VIT_REPO_ID = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
RESNET_REPO_ID = "timm/resnet50.a1_in1k"
DETECTOR_REPO_ID = "IDEA-Research/grounding-dino-tiny"
PINNED_REVISIONS = {
    VIT_REPO_ID: "063c6c38a5d8510b2e57df480445e94b231dad2c",
    RESNET_REPO_ID: "767268603ca0cb0bfe326fa87277f19c419566ef",
    DETECTOR_REPO_ID: "a2bb814dd30d776dcf7e30523b00659f4f141c71",
}

# These are the direct imports used by the Streamlit/FastAPI inference path.
# scikit-learn is required when joblib reconstructs the trained policy head.
REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("FastAPI", "fastapi"),
    ("joblib", "joblib"),
    ("NumPy", "numpy"),
    ("Pillow", "PIL"),
    ("Plotly", "plotly"),
    ("python-multipart", "multipart"),
    ("PyTorch", "torch"),
    ("PyYAML", "yaml"),
    ("scikit-learn", "sklearn"),
    ("Streamlit", "streamlit"),
    ("timm", "timm"),
    ("torchvision", "torchvision"),
    ("Transformers", "transformers"),
    ("Uvicorn", "uvicorn"),
)

REQUIRED_PROJECT_FILES: tuple[tuple[str, str, bool], ...] = (
    ("policy_config", "configs/policy.yaml", False),
    ("formal_metrics", "outputs/evaluation/metrics.json", True),
    ("evaluation_metrics", "outputs/evaluation/evaluation_metrics.json", True),
)

OPTIONAL_EVIDENCE_FILES: tuple[tuple[str, str, bool], ...] = (
    ("external_spot_check", "outputs/evaluation/external_spot_check.json", True),
    ("demo_case_summary", "outputs/demo_cases/case_summary.json", True),
)

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class CheckResult:
    """One deterministic preflight result."""

    check_id: str
    category: str
    label: str
    required: bool
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightReport:
    """Complete readiness report for one requested profile."""

    profile: str
    project_root: str
    checks: tuple[CheckResult, ...]
    schema_version: str = REPORT_SCHEMA_VERSION

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks if check.required)

    def to_dict(self) -> dict[str, Any]:
        required_failures = [
            check.check_id for check in self.checks if check.required and not check.passed
        ]
        warnings = [
            check.check_id for check in self.checks if not check.required and not check.passed
        ]
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "project_root": self.project_root,
            "ready": self.ready,
            "summary": {
                "checks": len(self.checks),
                "passed": sum(check.passed for check in self.checks),
                "required_failures": required_failures,
                "warnings": warnings,
            },
            "checks": [check.to_dict() for check in self.checks],
        }


def _result(
    *,
    check_id: str,
    category: str,
    label: str,
    required: bool,
    ok: bool,
    success: str,
    failure: str,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        category=category,
        label=label,
        required=required,
        status="pass" if ok else ("fail" if required else "warn"),
        detail=success if ok else failure,
    )


def _one_line(value: object) -> str:
    return " ".join(str(value).split())


def _read_nonempty(path: Path) -> tuple[bool, str]:
    try:
        if not path.is_file():
            return False, "missing"
        size = path.stat().st_size
        if size <= 0:
            return False, "empty"
        with path.open("rb") as handle:
            if not handle.read(1):
                return False, "empty"
        return True, f"{size} bytes"
    except OSError as exc:
        return False, f"unreadable: {_one_line(exc)}"


def _read_json(path: Path) -> tuple[Any | None, str]:
    ready, detail = _read_nonempty(path)
    if not ready:
        return None, detail
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {_one_line(exc)}"
    return payload, detail


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    payload, detail = _read_json(path)
    if payload is None:
        return None, detail
    if not isinstance(payload, dict):
        return None, "JSON root must be an object"
    return payload, detail


def _check_project_file(
    root: Path,
    check_id: str,
    relative_path: str,
    *,
    json_file: bool,
    required: bool,
) -> CheckResult:
    path = root / relative_path
    if json_file:
        payload, detail = _read_json(path)
        ok = payload is not None
    else:
        ok, detail = _read_nonempty(path)
    return _result(
        check_id=f"file_{check_id}",
        category="project",
        label=relative_path,
        required=required,
        ok=ok,
        success=f"readable ({detail})",
        failure=detail,
    )


def _resolve_manifest_path(root: Path, raw_path: object) -> tuple[Path | None, str]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "manifest path is missing"
    normalized = raw_path.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts or (
        relative.parts and relative.parts[0].endswith(":")
    ):
        return None, f"manifest path must stay under the project root: {raw_path}"
    root_resolved = root.resolve()
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(root_resolved)
    except (OSError, ValueError):
        return None, f"manifest path escapes the project root: {raw_path}"
    return candidate, relative.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_fingerprint(snapshot_path: Path) -> tuple[str, int, int]:
    """Hash sorted relative paths and bytes exactly as download_models does."""

    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(item for item in snapshot_path.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot_path).as_posix()
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        file_count += 1
    return digest.hexdigest(), total_bytes, file_count


def _check_head_record(
    root: Path,
    role: str,
    record: dict[str, Any] | None,
    *,
    required: bool,
) -> CheckResult:
    label = f"trained head ({role})"
    if record is None:
        return _result(
            check_id=f"trained_head_{role}",
            category="models",
            label=label,
            required=required,
            ok=False,
            success="",
            failure=f"no '{role}' artifact is declared in {TRAINED_HEAD_MANIFEST.as_posix()}",
        )

    path, path_detail = _resolve_manifest_path(root, record.get("path"))
    if path is None:
        return _result(
            check_id=f"trained_head_{role}",
            category="models",
            label=label,
            required=required,
            ok=False,
            success="",
            failure=path_detail,
        )

    expected_path = f"models/{'vit' if role == 'primary' else 'resnet50'}_policy_head.joblib"
    if path_detail != expected_path:
        return _result(
            check_id=f"trained_head_{role}",
            category="models",
            label=label,
            required=required,
            ok=False,
            success="",
            failure=f"expected path {expected_path}, manifest declares {path_detail}",
        )

    ready, file_detail = _read_nonempty(path)
    if not ready:
        return _result(
            check_id=f"trained_head_{role}",
            category="models",
            label=label,
            required=required,
            ok=False,
            success="",
            failure=f"{path_detail}: {file_detail}",
        )

    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or not _SHA256_PATTERN.fullmatch(expected_sha):
        return _result(
            check_id=f"trained_head_{role}",
            category="models",
            label=label,
            required=required,
            ok=False,
            success="",
            failure=f"{path_detail}: manifest SHA-256 is missing or invalid",
        )

    expected_bytes = record.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        return _result(
            check_id=f"trained_head_{role}",
            category="models",
            label=label,
            required=required,
            ok=False,
            success="",
            failure=f"{path_detail}: manifest byte count is missing or invalid",
        )

    try:
        actual_bytes = path.stat().st_size
        actual_sha = _sha256_file(path)
    except OSError as exc:
        return _result(
            check_id=f"trained_head_{role}",
            category="models",
            label=label,
            required=required,
            ok=False,
            success="",
            failure=f"{path_detail}: could not hash file: {_one_line(exc)}",
        )

    problems: list[str] = []
    if actual_bytes != expected_bytes:
        problems.append(f"size {actual_bytes} does not match manifest {expected_bytes}")
    if actual_sha.casefold() != expected_sha.casefold():
        problems.append(f"SHA-256 {actual_sha} does not match manifest {expected_sha.casefold()}")
    return _result(
        check_id=f"trained_head_{role}",
        category="models",
        label=label,
        required=required,
        ok=not problems,
        success=f"{path_detail} ({actual_bytes} bytes; SHA-256 verified)",
        failure=f"{path_detail}: {'; '.join(problems)}",
    )


def _check_trained_heads(root: Path, profile: str) -> list[CheckResult]:
    path = root / TRAINED_HEAD_MANIFEST
    manifest, detail = _read_json_object(path)
    manifest_problem = ""
    artifacts: list[object] = []
    if manifest is None:
        manifest_problem = detail
    elif manifest.get("schema_version") != REPORT_SCHEMA_VERSION:
        manifest_problem = (
            f"unsupported schema_version {manifest.get('schema_version')!r}; "
            f"expected {REPORT_SCHEMA_VERSION!r}"
        )
    elif not isinstance(manifest.get("artifacts"), list):
        manifest_problem = "'artifacts' must be a list"
    else:
        artifacts = manifest["artifacts"]

    checks = [
        _result(
            check_id="trained_head_manifest",
            category="models",
            label=TRAINED_HEAD_MANIFEST.as_posix(),
            required=True,
            ok=not manifest_problem,
            success=f"schema {REPORT_SCHEMA_VERSION}; {len(artifacts)} artifact record(s)",
            failure=manifest_problem,
        )
    ]

    by_role: dict[str, dict[str, Any]] = {}
    duplicate_roles: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            continue
        role = item["role"]
        if role in by_role:
            duplicate_roles.add(role)
        else:
            by_role[role] = item
    for role in duplicate_roles:
        by_role.pop(role, None)

    for role, required in (("primary", True), ("baseline", profile == "full")):
        record = by_role.get(role)
        if role in duplicate_roles:
            record = None
        checks.append(_check_head_record(root, role, record, required=required))
    return checks


def _check_snapshot(
    root: Path,
    *,
    repo_id: str,
    expected_revision: str,
    label: str,
    required_files: Iterable[str],
    required: bool,
) -> CheckResult:
    required_file_names = tuple(required_files)
    manifest_path = root / PRETRAINED_MODEL_MANIFEST
    manifest, detail = _read_json_object(manifest_path)
    if manifest is None:
        return _result(
            check_id=f"snapshot_{label}",
            category="models",
            label=f"{label} snapshot",
            required=required,
            ok=False,
            success="",
            failure=f"{PRETRAINED_MODEL_MANIFEST.as_posix()}: {detail}",
        )
    models = manifest.get("models")
    if not isinstance(models, list):
        row = None
    else:
        row = next(
            (item for item in models if isinstance(item, dict) and item.get("repo_id") == repo_id),
            None,
        )
    if row is None:
        return _result(
            check_id=f"snapshot_{label}",
            category="models",
            label=f"{label} snapshot",
            required=required,
            ok=False,
            success="",
            failure=f"no manifest record for {repo_id}",
        )

    revision = row.get("revision")
    expected_sha = row.get("snapshot_sha256")
    expected_bytes = row.get("snapshot_bytes")
    expected_files = row.get("snapshot_files")
    metadata_problems: list[str] = []
    if revision != expected_revision:
        metadata_problems.append(
            f"revision {revision!r} does not match pinned revision {expected_revision!r}"
        )
    if not isinstance(expected_sha, str) or not _SHA256_PATTERN.fullmatch(expected_sha):
        metadata_problems.append("snapshot_sha256 is missing or invalid")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        metadata_problems.append("snapshot_bytes is missing or invalid")
    if isinstance(expected_files, bool) or not isinstance(expected_files, int) or expected_files <= 0:
        metadata_problems.append("snapshot_files is missing or invalid")

    snapshot, snapshot_detail = _resolve_manifest_path(root, row.get("local_snapshot"))
    if snapshot is None:
        return _result(
            check_id=f"snapshot_{label}",
            category="models",
            label=f"{label} snapshot",
            required=required,
            ok=False,
            success="",
            failure=snapshot_detail,
        )

    missing: list[str] = []
    for filename in required_file_names:
        ready, _ = _read_nonempty(snapshot / filename)
        if not ready:
            missing.append(filename)
    if missing:
        metadata_problems.append(f"missing or empty {', '.join(missing)}")

    if metadata_problems:
        return _result(
            check_id=f"snapshot_{label}",
            category="models",
            label=f"{label} snapshot",
            required=required,
            ok=False,
            success="",
            failure=f"{snapshot_detail}: {'; '.join(metadata_problems)}",
        )

    try:
        actual_sha, actual_bytes, actual_files = _snapshot_fingerprint(snapshot)
    except OSError as exc:
        return _result(
            check_id=f"snapshot_{label}",
            category="models",
            label=f"{label} snapshot",
            required=required,
            ok=False,
            success="",
            failure=f"{snapshot_detail}: could not fingerprint snapshot: {_one_line(exc)}",
        )

    fingerprint_problems: list[str] = []
    if actual_sha.casefold() != expected_sha.casefold():
        fingerprint_problems.append(
            f"snapshot_sha256 {actual_sha} does not match manifest {expected_sha.casefold()}"
        )
    if actual_bytes != expected_bytes:
        fingerprint_problems.append(
            f"snapshot_bytes {actual_bytes} does not match manifest {expected_bytes}"
        )
    if actual_files != expected_files:
        fingerprint_problems.append(
            f"snapshot_files {actual_files} does not match manifest {expected_files}"
        )
    return _result(
        check_id=f"snapshot_{label}",
        category="models",
        label=f"{label} snapshot",
        required=required,
        ok=not fingerprint_problems,
        success=(
            f"{snapshot_detail} (revision {revision}; {actual_files} files; "
            f"{actual_bytes} bytes; SHA-256 verified)"
        ),
        failure=f"{snapshot_detail}: {'; '.join(fingerprint_problems)}",
    )


def run_preflight(
    *,
    profile: str,
    project_root: str | Path = PROJECT_ROOT,
    python_version: Sequence[int] | None = None,
    operating_system: str | None = None,
    machine: str | None = None,
    pointer_bits: int | None = None,
    importer: Callable[[str], object] | None = None,
    executable_finder: Callable[[str], str | None] | None = None,
) -> PreflightReport:
    """Run local checks without loading model weights or using the network."""

    if profile not in {"core", "full"}:
        raise ValueError("profile must be 'core' or 'full'")
    root = Path(project_root).expanduser().resolve()
    runtime_version = tuple(python_version or sys.version_info[:3])
    system_name = operating_system or platform.system()
    machine_name = machine or platform.machine()
    bits = pointer_bits if pointer_bits is not None else struct.calcsize("P") * 8
    importer = importer or importlib.import_module
    executable_finder = executable_finder or shutil.which

    checks: list[CheckResult] = []
    python_ok = len(runtime_version) >= 2 and runtime_version[:2] == (3, 10)
    version_text = ".".join(str(value) for value in runtime_version)
    checks.append(
        _result(
            check_id="python_version",
            category="environment",
            label="Python 3.10",
            required=True,
            ok=python_ok,
            success=f"Python {version_text}",
            failure=f"Python {version_text}; install and use Python 3.10",
        )
    )

    system_key = system_name.casefold()
    machine_key = machine_name.casefold()
    supported_host = bits == 64 and (
        (system_key == "darwin" and machine_key in {"arm64", "aarch64"})
        or (system_key == "windows" and machine_key in {"amd64", "x86_64"})
    )
    checks.append(
        _result(
            check_id="host_platform",
            category="environment",
            label="supported 64-bit host",
            required=True,
            ok=supported_host,
            success=f"{system_name} {machine_name}, {bits}-bit",
            failure=(
                f"{system_name} {machine_name}, {bits}-bit; supported hosts are "
                "Apple Silicon macOS and AMD64 Windows"
            ),
        )
    )

    for display_name, module_name in REQUIRED_IMPORTS:
        try:
            importer(module_name)
        except Exception as exc:  # Import failures include binary-loader errors.
            ok = False
            detail = f"cannot import {module_name}: {type(exc).__name__}: {_one_line(exc)}"
        else:
            ok = True
            detail = f"import {module_name} succeeded"
        checks.append(
            _result(
                check_id=f"package_{module_name.replace('.', '_')}",
                category="packages",
                label=display_name,
                required=True,
                ok=ok,
                success=detail,
                failure=detail,
            )
        )

    for check_id, relative_path, json_file in REQUIRED_PROJECT_FILES:
        checks.append(
            _check_project_file(
                root,
                check_id,
                relative_path,
                json_file=json_file,
                required=True,
            )
        )
    for check_id, relative_path, json_file in OPTIONAL_EVIDENCE_FILES:
        checks.append(
            _check_project_file(
                root,
                check_id,
                relative_path,
                json_file=json_file,
                required=False,
            )
        )

    checks.extend(_check_trained_heads(root, profile))
    checks.append(
        _check_snapshot(
            root,
            repo_id=VIT_REPO_ID,
            expected_revision=PINNED_REVISIONS[VIT_REPO_ID],
            label="vit",
            required_files=("model.safetensors", "config.json"),
            required=True,
        )
    )
    checks.append(
        _check_snapshot(
            root,
            repo_id=RESNET_REPO_ID,
            expected_revision=PINNED_REVISIONS[RESNET_REPO_ID],
            label="resnet50",
            required_files=("model.safetensors", "config.json"),
            required=profile == "full",
        )
    )
    checks.append(
        _check_snapshot(
            root,
            repo_id=DETECTOR_REPO_ID,
            expected_revision=PINNED_REVISIONS[DETECTOR_REPO_ID],
            label="detector",
            required_files=(
                "model.safetensors",
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ),
            required=profile == "full",
        )
    )

    try:
        tesseract_path = executable_finder("tesseract")
    except OSError as exc:
        tesseract_path = None
        tesseract_failure = f"executable lookup failed: {_one_line(exc)}"
    else:
        tesseract_failure = "tesseract executable was not found on PATH"
    checks.append(
        _result(
            check_id="tesseract",
            category="optional_tools",
            label="Tesseract OCR",
            required=profile == "full",
            ok=bool(tesseract_path),
            success=f"found at {tesseract_path}",
            failure=tesseract_failure,
        )
    )

    return PreflightReport(profile=profile, project_root=str(root), checks=tuple(checks))


def render_text(report: PreflightReport) -> str:
    """Render a concise report suitable for a terminal."""

    lines = [
        f"Ad Safety preflight ({report.profile})",
        f"Project root: {report.project_root}",
        "",
    ]
    for check in report.checks:
        marker = check.status.upper()
        optional = " (optional)" if not check.required else ""
        lines.append(f"[{marker}] {check.label}{optional}: {check.detail}")
    required_failures = [
        check.label for check in report.checks if check.required and not check.passed
    ]
    lines.append("")
    if report.ready:
        lines.append(f"READY: the {report.profile} profile passed every required check.")
    else:
        lines.append(
            f"NOT READY: {len(required_failures)} required check(s) failed: "
            + ", ".join(required_failures)
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify offline readiness for the ad-safety moderation pipeline."
    )
    parser.add_argument(
        "--profile",
        choices=("core", "full"),
        default="core",
        help="core checks classifier readiness; full also requires optional evidence components",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable JSON document instead of terminal status lines",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(profile=args.profile, project_root=args.root)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
