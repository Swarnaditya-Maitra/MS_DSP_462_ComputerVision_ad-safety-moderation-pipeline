"""Ad Safety pilot package."""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPL_CACHE = _PROJECT_ROOT / ".mpl-cache"
_GENERAL_CACHE = _PROJECT_ROOT / ".cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_GENERAL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_GENERAL_CACHE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".cache" / "huggingface"))

from .policy import PolicyDecision, PolicyEngine  # noqa: E402

__all__ = ["PolicyDecision", "PolicyEngine"]
__version__ = "1.0.0"
