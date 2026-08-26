from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts import bootstrap


def _runtime(**overrides: Any) -> bootstrap.RuntimeInfo:
    values: dict[str, Any] = {
        "implementation": "CPython",
        "version": (3, 10, 14),
        "pointer_bits": 64,
        "system": "Darwin",
        "machine": "arm64",
        "in_virtual_environment": True,
    }
    values.update(overrides)
    return bootstrap.RuntimeInfo(**values)


@pytest.mark.parametrize(
    ("system", "machine"),
    [("Darwin", "arm64"), ("Windows", "AMD64"), ("Windows", "x86_64")],
)
def test_assess_runtime_accepts_supported_hosts(system: str, machine: str) -> None:
    assert bootstrap.assess_runtime(_runtime(system=system, machine=machine)) == ()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"implementation": "PyPy"}, "CPython is required"),
        ({"version": (3, 11, 0)}, "Python 3.10 is required"),
        ({"pointer_bits": 32}, "64-bit Python interpreter is required"),
        ({"system": "Darwin", "machine": "x86_64"}, "Supported hosts are"),
        ({"system": "Linux", "machine": "x86_64"}, "Supported hosts are"),
        ({"in_virtual_environment": False}, "virtual environment is required"),
    ],
)
def test_assess_runtime_rejects_unsupported_runtime(
    overrides: dict[str, Any], expected: str
) -> None:
    problems = bootstrap.assess_runtime(_runtime(**overrides))

    assert any(expected in problem for problem in problems)


def test_assess_runtime_can_explicitly_allow_system_python() -> None:
    problems = bootstrap.assess_runtime(
        _runtime(in_virtual_environment=False), allow_system_python=True
    )

    assert problems == ()


def test_build_commands_for_core_profile(tmp_path: Path) -> None:
    commands = bootstrap.build_commands(
        profile="core",
        skip_install=False,
        offline=False,
        python_executable="/test/python",
        project_root=tmp_path,
    )

    assert [command.label for command in commands] == [
        "Install pinned Python dependencies",
        "Resolve pinned model snapshots",
        "Run the core readiness checks",
    ]
    assert commands[0].argv == (
        "/test/python",
        "-m",
        "pip",
        "install",
        "-r",
        str(tmp_path / "requirements-lock.txt"),
    )
    assert commands[1].argv == (
        "/test/python",
        str(tmp_path / "scripts" / "download_models.py"),
        "--core-only",
    )
    assert commands[2].argv == (
        "/test/python",
        str(tmp_path / "scripts" / "preflight.py"),
        "--profile",
        "core",
    )


def test_build_commands_for_offline_full_profile_skips_install(tmp_path: Path) -> None:
    commands = bootstrap.build_commands(
        profile="full",
        skip_install=True,
        offline=True,
        python_executable="python.exe",
        project_root=tmp_path,
    )

    assert [command.argv for command in commands] == [
        (
            "python.exe",
            str(tmp_path / "scripts" / "download_models.py"),
            "--offline",
        ),
        (
            "python.exe",
            str(tmp_path / "scripts" / "preflight.py"),
            "--profile",
            "full",
        ),
    ]


def test_build_commands_rejects_unknown_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported profile"):
        bootstrap.build_commands(
            profile="partial",
            skip_install=True,
            offline=True,
            python_executable="python",
            project_root=tmp_path,
        )


def test_run_bootstrap_returns_child_failure_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[tuple[str, ...], Path, bool]] = []
    return_codes = iter((0, 17))

    def fake_run(
        argv: tuple[str, ...], *, cwd: Path, check: bool
    ) -> Any:
        calls.append((argv, cwd, check))
        return type("Completed", (), {"returncode": next(return_codes)})()

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    exit_code = bootstrap.run_bootstrap(
        profile="core",
        skip_install=False,
        offline=True,
        allow_system_python=False,
        runtime=_runtime(),
        python_executable="/test/python",
        project_root=tmp_path,
    )

    assert exit_code == 17
    assert len(calls) == 2
    assert all(cwd == tmp_path and check is False for _, cwd, check in calls)
    assert calls[0][0][1:4] == ("-m", "pip", "install")
    assert calls[1][0][-2:] == ("--core-only", "--offline")


def test_run_bootstrap_rejects_runtime_before_child_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subprocess.run must not be called")

    monkeypatch.setattr(bootstrap.subprocess, "run", unexpected_run)

    exit_code = bootstrap.run_bootstrap(
        profile="core",
        skip_install=True,
        offline=True,
        allow_system_python=False,
        runtime=_runtime(version=(3, 12, 0)),
        python_executable="/test/python",
        project_root=tmp_path,
    )

    assert exit_code == 2


def test_parse_args_defaults_to_core_profile() -> None:
    args = bootstrap.parse_args([])

    assert args.profile == "core"
    assert args.skip_install is False
    assert args.offline is False
    assert args.allow_system_python is False
