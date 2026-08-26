from __future__ import annotations

from pathlib import Path

import pytest

from ad_safety import classifier


def test_policy_classifier_rejects_unverified_pickle_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "tampered.joblib"
    artifact.write_bytes(b"not safe to deserialize")
    monkeypatch.setattr(
        classifier,
        "trained_head_status",
        lambda _path: (False, "artifact SHA-256 does not match the manifest"),
    )
    monkeypatch.setattr(
        classifier.joblib,
        "load",
        lambda _path: pytest.fail("joblib.load must not run before integrity verification"),
    )

    with pytest.raises(RuntimeError, match="integrity check failed"):
        classifier.PolicyClassifier(artifact)
