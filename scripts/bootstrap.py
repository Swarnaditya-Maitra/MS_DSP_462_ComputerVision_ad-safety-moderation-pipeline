#!/usr/bin/env python3
"""Prepare and verify a supported clone of the ad-safety application.

This entry point intentionally uses only the Python standard library. It can
therefore reject an unsupported interpreter or host before package imports or
model downloads begin.
"""

from __future__ import annotations

import argparse
import platform
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_LOCK = PROJECT_ROOT / "requirements-lock.txt"
DOWNLOAD_MODELS = PROJECT_ROOT / "scripts" / "download_models.py"
PREFLIGHT = PROJECT_ROOT / "scripts" / "preflight.py"


@dataclass(frozen=True)
class RuntimeInfo:
    """Facts needed to decide whether this bootstrap environment is supported."""

    implementation: str
    version: tuple[int, int, int]
    pointer_bits: int
    system: str
    machine: str
    in_virtual_environment: bool


@dataclass(frozen=True)
class BootstrapCommand:
    """One child process in the setup workflow."""

    label: str
    argv: tuple[str, ...]


def current_runtime_info() -> RuntimeInfo:
    """Collect runtime facts without importing any project dependencies."""

    return RuntimeInfo(
        implementation=platform.python_implementation(),
        version=(sys.version_info.major, sys.version_info.minor, sys.version_info.micro),
        pointer_bits=struct.calcsize("P") * 8,
        system=platform.system(),
        machine=platform.machine(),
        in_virtual_environment=sys.prefix != sys.base_prefix,
    )


def assess_runtime(
    runtime: RuntimeInfo, *, allow_system_python: bool = False
) -> tuple[str, ...]:
    """Return actionable problems for a runtime, or an empty tuple if supported."""

    problems: list[str] = []
    if runtime.implementation != "CPython":
        problems.append(
            f"CPython is required; detected {runtime.implementation or 'unknown implementation'}."
        )
    if runtime.version[:2] != (3, 10):
        rendered_version = ".".join(str(part) for part in runtime.version)
        problems.append(f"Python 3.10 is required; detected {rendered_version}.")
    if runtime.pointer_bits != 64:
        problems.append(
            "A 64-bit Python interpreter is required; "
            f"detected {runtime.pointer_bits}-bit."
        )

    normalized_system = runtime.system.casefold()
    normalized_machine = runtime.machine.casefold()
    supported_host = (
        normalized_system == "darwin" and normalized_machine == "arm64"
    ) or (
        normalized_system == "windows"
        and normalized_machine in {"amd64", "x86_64"}
    )
    if not supported_host:
        problems.append(
            "Supported hosts are Apple Silicon macOS (Darwin arm64) and "
            "64-bit Windows (AMD64/x86_64); detected "
            f"{runtime.system or 'unknown OS'} {runtime.machine or 'unknown architecture'}."
        )

    if not runtime.in_virtual_environment and not allow_system_python:
        problems.append(
            "An active virtual environment is required. Activate .venv or rerun "
            "with --allow-system-python if you accept modifying that interpreter."
        )
    return tuple(problems)


def build_commands(
    *,
    profile: str,
    skip_install: bool,
    offline: bool,
    python_executable: str,
    project_root: Path = PROJECT_ROOT,
) -> tuple[BootstrapCommand, ...]:
    """Build the deterministic child-process sequence for a setup profile."""

    if profile not in {"core", "full"}:
        raise ValueError(f"Unsupported profile: {profile}")

    commands: list[BootstrapCommand] = []
    if not skip_install:
        commands.append(
            BootstrapCommand(
                "Install pinned Python dependencies",
                (
                    python_executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(project_root / "requirements-lock.txt"),
                ),
            )
        )

    download_argv = [
        python_executable,
        str(project_root / "scripts" / "download_models.py"),
    ]
    if profile == "core":
        download_argv.append("--core-only")
    if offline:
        download_argv.append("--offline")
    commands.append(BootstrapCommand("Resolve pinned model snapshots", tuple(download_argv)))

    commands.append(
        BootstrapCommand(
            f"Run the {profile} readiness checks",
            (
                python_executable,
                str(project_root / "scripts" / "preflight.py"),
                "--profile",
                profile,
            ),
        )
    )
    return tuple(commands)


def run_bootstrap(
    *,
    profile: str,
    skip_install: bool,
    offline: bool,
    allow_system_python: bool,
    runtime: RuntimeInfo | None = None,
    python_executable: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> int:
    """Run setup commands in order and return the first nonzero child status."""

    detected_runtime = runtime or current_runtime_info()
    problems = assess_runtime(
        detected_runtime, allow_system_python=allow_system_python
    )
    if problems:
        print("Bootstrap cannot continue:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    interpreter = python_executable or sys.executable
    commands = build_commands(
        profile=profile,
        skip_install=skip_install,
        offline=offline,
        python_executable=interpreter,
        project_root=project_root,
    )

    print(f"Bootstrap profile: {profile}")
    print(f"Python executable: {interpreter}")
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {command.label}", flush=True)
        completed = subprocess.run(command.argv, cwd=project_root, check=False)
        if completed.returncode != 0:
            print(
                f"Step failed with exit code {completed.returncode}: {command.label}",
                file=sys.stderr,
            )
            return completed.returncode

    print(f"Bootstrap complete. The {profile} profile passed preflight.")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install, download, and verify a supported project clone."
    )
    parser.add_argument(
        "--profile",
        choices=("core", "full"),
        default="core",
        help="core prepares the ViT app path; full also prepares optional models",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not install requirements-lock.txt before model setup",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require all selected pinned model snapshots to exist in the local cache",
    )
    parser.add_argument(
        "--allow-system-python",
        action="store_true",
        help="Allow setup outside a virtual environment (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_bootstrap(
        profile=args.profile,
        skip_install=args.skip_install,
        offline=args.offline,
        allow_system_python=args.allow_system_python,
    )


if __name__ == "__main__":
    raise SystemExit(main())
