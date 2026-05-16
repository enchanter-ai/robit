"""Path constants for the inference substrate.

Override the state directory via ROBIT_INFERENCE_STATE env var so tests
(and CI sandboxes) never touch the production state. The legacy
ENCHANTER_INFERENCE_STATE name is still honored (with a deprecation
warning) via :mod:`robit._compat`.
"""

from __future__ import annotations

from pathlib import Path

from robit._compat import get_env

# Default: <repo>/robit/state/inference/
# One directory up from this file is robit/inference/; two up is robit/.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent  # robit/

DEFAULT_STATE_DIR: Path = _PACKAGE_DIR / "state" / "inference"

# Tests override this to a tmp_path so production state is never touched.
STATE_DIR: Path = Path(get_env("ROBIT_INFERENCE_STATE") or DEFAULT_STATE_DIR)

CATALOG_PATH: Path = STATE_DIR / "catalog.json"
ARTIFACTS_PATH: Path = STATE_DIR / "artifacts.jsonl"
BRIEFINGS_DIR: Path = STATE_DIR / "briefings"


def resolve_state_dir() -> Path:
    """Re-read env var at call time (allows per-test override without reimport)."""
    override = get_env("ROBIT_INFERENCE_STATE")
    return Path(override) if override else DEFAULT_STATE_DIR
