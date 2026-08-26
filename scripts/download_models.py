#!/usr/bin/env python3
"""Download pinned pretrained model snapshots into the Project cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
MANIFEST_PATH = PROJECT_ROOT / "models" / "pretrained_model_manifest.json"

MODEL_SPECS = (
    {
        "repo_id": "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "revision": "063c6c38a5d8510b2e57df480445e94b231dad2c",
        "license": "apache-2.0",
    },
    {
        "repo_id": "timm/resnet50.a1_in1k",
        "revision": "767268603ca0cb0bfe326fa87277f19c419566ef",
        "license": "apache-2.0",
    },
    {
        "repo_id": "IDEA-Research/grounding-dino-tiny",
        "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "license": "apache-2.0",
    },
)


def snapshot_fingerprint(snapshot_path: Path) -> tuple[str, int, int]:
    """Hash paths and contents so a local snapshot can be independently checked."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-grounding-dino", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only the pinned local cache")
    args = parser.parse_args()

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in MODEL_SPECS:
        repo_id = str(spec["repo_id"])
        if args.skip_grounding_dino and repo_id == "IDEA-Research/grounding-dino-tiny":
            continue
        revision = str(spec["revision"])
        print(f"Resolving {repo_id} at pinned revision {revision}")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=HF_HOME / "hub",
            local_files_only=args.offline,
        )
        fingerprint, total_bytes, file_count = snapshot_fingerprint(Path(snapshot_path))
        rows.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "local_snapshot": str(Path(snapshot_path).relative_to(PROJECT_ROOT)),
                "license": spec["license"],
                "snapshot_sha256": fingerprint,
                "snapshot_bytes": total_bytes,
                "snapshot_files": file_count,
                "downloads_recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    MANIFEST_PATH.write_text(json.dumps({"models": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"Model manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
