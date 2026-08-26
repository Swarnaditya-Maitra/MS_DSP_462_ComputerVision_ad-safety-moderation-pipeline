from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import preflight


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact(role: str, relative_path: str, payload: bytes) -> dict[str, object]:
    return {
        "role": role,
        "path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _snapshot_record(
    snapshot: Path,
    *,
    repo_id: str,
    revision: str,
    relative_path: str,
) -> dict[str, object]:
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        digest.update(path.relative_to(snapshot).as_posix().encode("utf-8"))
        payload = path.read_bytes()
        digest.update(payload)
        total_bytes += len(payload)
        file_count += 1
    return {
        "repo_id": repo_id,
        "revision": revision,
        "local_snapshot": relative_path,
        "snapshot_sha256": digest.hexdigest(),
        "snapshot_bytes": total_bytes,
        "snapshot_files": file_count,
    }


def _build_root(
    root: Path,
    *,
    baseline: bool = False,
    detector: bool = False,
    optional_evidence: bool = False,
) -> None:
    _write(root / "configs/policy.yaml", b"model_labels: [safe]\n")
    _write_json(root / "outputs/evaluation/metrics.json", {"primary_model": "vit"})
    _write_json(root / "outputs/evaluation/evaluation_metrics.json", {"models": []})
    if optional_evidence:
        _write_json(root / "outputs/evaluation/external_spot_check.json", {"rows": 1})
        _write_json(root / "outputs/demo_cases/case_summary.json", [])

    primary_payload = b"trusted primary policy head"
    _write(root / "models/vit_policy_head.joblib", primary_payload)
    artifacts = [
        _artifact("primary", "models/vit_policy_head.joblib", primary_payload)
    ]
    if baseline:
        baseline_payload = b"trusted baseline policy head"
        _write(root / "models/resnet50_policy_head.joblib", baseline_payload)
        artifacts.append(
            _artifact("baseline", "models/resnet50_policy_head.joblib", baseline_payload)
        )
    _write_json(
        root / "models/trained_head_manifest.json",
        {"schema_version": "1.0.0", "artifacts": artifacts},
    )

    vit_snapshot = root / "cache/vit"
    _write(vit_snapshot / "model.safetensors", b"vit weights")
    _write(vit_snapshot / "config.json", b"{}")
    model_rows: list[dict[str, object]] = [
        _snapshot_record(
            vit_snapshot,
            repo_id=preflight.VIT_REPO_ID,
            revision=preflight.PINNED_REVISIONS[preflight.VIT_REPO_ID],
            relative_path="cache/vit",
        )
    ]
    if baseline:
        resnet_snapshot = root / "cache/resnet50"
        _write(resnet_snapshot / "model.safetensors", b"resnet weights")
        _write(resnet_snapshot / "config.json", b"{}")
        model_rows.append(
            _snapshot_record(
                resnet_snapshot,
                repo_id=preflight.RESNET_REPO_ID,
                revision=preflight.PINNED_REVISIONS[preflight.RESNET_REPO_ID],
                relative_path="cache/resnet50",
            )
        )
    if detector:
        detector_snapshot = root / "cache/detector"
        for filename in (
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ):
            _write(detector_snapshot / filename, b"ready")
        model_rows.append(
            _snapshot_record(
                detector_snapshot,
                repo_id=preflight.DETECTOR_REPO_ID,
                revision=preflight.PINNED_REVISIONS[preflight.DETECTOR_REPO_ID],
                relative_path="cache/detector",
            )
        )
    _write_json(root / "models/pretrained_model_manifest.json", {"models": model_rows})


def _run(
    root: Path,
    *,
    profile: str = "core",
    system: str = "Darwin",
    machine: str = "arm64",
    importer=lambda _name: object(),
    finder=lambda _name: None,
) -> preflight.PreflightReport:
    return preflight.run_preflight(
        profile=profile,
        project_root=root,
        python_version=(3, 10, 14),
        operating_system=system,
        machine=machine,
        pointer_bits=64,
        importer=importer,
        executable_finder=finder,
    )


def _by_id(report: preflight.PreflightReport) -> dict[str, preflight.CheckResult]:
    return {check.check_id: check for check in report.checks}


def test_core_profile_passes_without_optional_components(tmp_path: Path) -> None:
    _build_root(tmp_path)

    report = _run(tmp_path)
    checks = _by_id(report)

    assert report.ready
    assert checks["trained_head_primary"].status == "pass"
    assert checks["trained_head_baseline"].status == "warn"
    assert checks["snapshot_vit"].status == "pass"
    assert checks["snapshot_resnet50"].status == "warn"
    assert checks["snapshot_detector"].status == "warn"
    assert checks["tesseract"].status == "warn"
    assert "READY: the core profile" in preflight.render_text(report)


def test_windows_amd64_is_a_supported_host(tmp_path: Path) -> None:
    _build_root(tmp_path)

    report = _run(tmp_path, system="Windows", machine="AMD64")

    assert report.ready
    assert _by_id(report)["host_platform"].status == "pass"


@pytest.mark.parametrize(
    ("version", "system", "machine", "bits", "failed_check"),
    [
        ((3, 11, 9), "Darwin", "arm64", 64, "python_version"),
        ((3, 10, 14), "Darwin", "x86_64", 64, "host_platform"),
        ((3, 10, 14), "Windows", "AMD64", 32, "host_platform"),
        ((3, 10, 14), "Linux", "x86_64", 64, "host_platform"),
    ],
)
def test_unsupported_runtime_blocks_readiness(
    tmp_path: Path,
    version: tuple[int, int, int],
    system: str,
    machine: str,
    bits: int,
    failed_check: str,
) -> None:
    _build_root(tmp_path)

    report = preflight.run_preflight(
        profile="core",
        project_root=tmp_path,
        python_version=version,
        operating_system=system,
        machine=machine,
        pointer_bits=bits,
        importer=lambda _name: object(),
        executable_finder=lambda _name: None,
    )

    assert not report.ready
    assert _by_id(report)[failed_check].status == "fail"


def test_missing_package_import_blocks_core(tmp_path: Path) -> None:
    _build_root(tmp_path)

    def import_package(name: str) -> object:
        if name == "torch":
            raise ImportError("broken binary")
        return object()

    report = _run(tmp_path, importer=import_package)

    assert not report.ready
    failure = _by_id(report)["package_torch"]
    assert failure.status == "fail"
    assert "broken binary" in failure.detail


def test_primary_head_checksum_mismatch_blocks_core(tmp_path: Path) -> None:
    _build_root(tmp_path)
    (tmp_path / "models/vit_policy_head.joblib").write_bytes(b"tampered")

    report = _run(tmp_path)
    check = _by_id(report)["trained_head_primary"]

    assert not report.ready
    assert check.status == "fail"
    assert "does not match manifest" in check.detail


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("revision", "0" * 40),
        ("snapshot_sha256", "0" * 64),
        ("snapshot_bytes", 1),
        ("snapshot_files", 1),
    ],
)
def test_vit_snapshot_manifest_mismatch_blocks_core(
    tmp_path: Path, field: str, replacement: object
) -> None:
    _build_root(tmp_path)
    manifest_path = tmp_path / "models/pretrained_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"][0][field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _run(tmp_path)
    check = _by_id(report)["snapshot_vit"]

    assert not report.ready
    assert check.status == "fail"
    assert field in check.detail


def test_vit_snapshot_content_tampering_blocks_core(tmp_path: Path) -> None:
    _build_root(tmp_path)
    (tmp_path / "cache/vit/model.safetensors").write_bytes(b"substituted weights")

    report = _run(tmp_path)
    check = _by_id(report)["snapshot_vit"]

    assert not report.ready
    assert check.status == "fail"
    assert "snapshot_sha256" in check.detail


def test_manifest_path_cannot_escape_project_root(tmp_path: Path) -> None:
    _build_root(tmp_path)
    manifest_path = tmp_path / "models/trained_head_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../outside.joblib"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _run(tmp_path)

    assert not report.ready
    assert "must stay under" in _by_id(report)["trained_head_primary"].detail


def test_full_profile_requires_baseline_detector_and_tesseract(tmp_path: Path) -> None:
    _build_root(tmp_path)

    incomplete = _run(tmp_path, profile="full")
    incomplete_checks = _by_id(incomplete)

    assert not incomplete.ready
    assert incomplete_checks["trained_head_baseline"].status == "fail"
    assert incomplete_checks["snapshot_resnet50"].status == "fail"
    assert incomplete_checks["snapshot_detector"].status == "fail"
    assert incomplete_checks["tesseract"].status == "fail"

    complete_root = tmp_path / "complete"
    _build_root(complete_root, baseline=True, detector=True, optional_evidence=True)
    complete = _run(
        complete_root,
        profile="full",
        finder=lambda name: "/usr/local/bin/tesseract" if name == "tesseract" else None,
    )

    assert complete.ready
    assert _by_id(complete)["file_demo_case_summary"].status == "pass"
    assert all(
        _by_id(complete)[check_id].status == "pass"
        for check_id in (
            "trained_head_baseline",
            "snapshot_resnet50",
            "snapshot_detector",
            "tesseract",
        )
    )


@pytest.mark.parametrize(
    ("relative_path", "check_id"),
    [
        ("cache/resnet50/model.safetensors", "snapshot_resnet50"),
        ("cache/detector/config.json", "snapshot_detector"),
    ],
)
def test_full_profile_detects_snapshot_content_tampering(
    tmp_path: Path, relative_path: str, check_id: str
) -> None:
    _build_root(tmp_path, baseline=True, detector=True)
    (tmp_path / relative_path).write_bytes(b"substituted content")

    report = _run(
        tmp_path,
        profile="full",
        finder=lambda name: "/usr/local/bin/tesseract" if name == "tesseract" else None,
    )

    assert not report.ready
    assert _by_id(report)[check_id].status == "fail"
    assert "snapshot_sha256" in _by_id(report)[check_id].detail


def test_json_cli_is_machine_readable_and_exit_code_tracks_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _build_root(tmp_path)
    monkeypatch.setattr(preflight.sys, "version_info", (3, 10, 14))
    monkeypatch.setattr(preflight.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(preflight.struct, "calcsize", lambda _format: 8)
    monkeypatch.setattr(preflight.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: None)

    exit_code = preflight.main(
        ["--profile", "core", "--root", str(tmp_path), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["profile"] == "core"
    assert payload["schema_version"] == "1.0.0"
    assert isinstance(payload["checks"], list)

    (tmp_path / "models/vit_policy_head.joblib").unlink()
    failed_exit = preflight.main(
        ["--profile", "core", "--root", str(tmp_path), "--json"]
    )
    failed_payload = json.loads(capsys.readouterr().out)

    assert failed_exit == 1
    assert failed_payload["ready"] is False
    assert "trained_head_primary" in failed_payload["summary"]["required_failures"]
