"""Path constants for the inference substrate.

Override the state directory via ENCHANTER_INFERENCE_STATE env var so tests
(and CI sandboxes) never touch the production state.
"""

from __future__ import annotations

import os
from pathlib import Path

# Default: <repo>/enchanter/state/inference/
# One directory up from this file is enchanter/inference/; two up is enchanter/.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent  # enchanter/

DEFAULT_STATE_DIR: Path = _PACKAGE_DIR / "state" / "inference"

# Tests override this to a tmp_path so production state is never touched.
STATE_DIR: Path = Path(
    os.environ.get("ENCHANTER_INFERENCE_STATE") or DEFAULT_STATE_DIR
)

CATALOG_PATH: Path = STATE_DIR / "catalog.json"
ARTIFACTS_PATH: Path = STATE_DIR / "artifacts.jsonl"
BRIEFINGS_DIR: Path = STATE_DIR / "briefings"


def resolve_state_dir() -> Path:
    """Re-read env var at call time (allows per-test override without reimport)."""
    override = os.environ.get("ENCHANTER_INFERENCE_STATE")
    return Path(override) if override else DEFAULT_STATE_DIR
