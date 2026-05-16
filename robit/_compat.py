"""robit._compat — backward-compatibility shims for the enchanter → robit rename.

Wave 19 renamed every public env-var (``ENCHANTER_HOME`` → ``ROBIT_HOME`` etc.)
and the on-disk config directory (``~/.enchanter`` → ``~/.robit``). Existing
installs already have real data on disk under the old names; this module is
the single place that lets the rest of the codebase read either the canonical
new name or the legacy name without each call site re-implementing the
fallback.

Two helpers:

* :func:`get_env` — read an env var, preferring the canonical ``ROBIT_*`` name
  but falling back to the legacy ``ENCHANTER_*`` name with a one-shot
  deprecation WARNING.
* :func:`resolve_user_dir` — return the active config dir. Prefers ``~/.robit``
  (or ``%APPDATA%\\robit`` on Windows). If only the legacy ``~/.enchanter``
  exists, returns the legacy path and emits a one-shot migration WARNING.

The one-shot dedup guarantees a long-running process logs each migration
prompt once per Python interpreter, not once per call. Tests reset the
``_warned_env`` set when they need to assert the warning fires.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

__all__ = ["get_env", "resolve_user_dir", "LEGACY_ENV_MAP"]


# canonical → legacy. Adding a new compat pair? Update this map only.
LEGACY_ENV_MAP: dict[str, str] = {
    "ROBIT_HOME": "ENCHANTER_HOME",
    "ROBIT_AGENT_MOCK": "ENCHANTER_AGENT_MOCK",
    "ROBIT_ALLOW_FASTPATH_BYPASS": "ENCHANTER_ALLOW_FASTPATH_BYPASS",
    "ROBIT_INFERENCE_ENABLED": "ENCHANTER_INFERENCE_ENABLED",
    "ROBIT_INFERENCE_STATE": "ENCHANTER_INFERENCE_STATE",
    "ROBIT_STATE_DIR": "ENCHANTER_STATE_DIR",
    "ROBIT_AUDIT_FSYNC": "ENCHANTER_AUDIT_FSYNC",
}

_logger = logging.getLogger("robit.compat")

# Set of legacy env names (or the sentinel ``"user-dir"``) we've already
# warned about in this process. Reset between tests via the fixture in
# ``tests/test_compat.py`` so warnings can be re-asserted.
_warned_env: set[str] = set()


def get_env(canonical_name: str, default: str | None = None) -> str | None:
    """Read an env var, with a one-shot deprecation WARNING for legacy names.

    Precedence:
      1. ``canonical_name`` (e.g. ``ROBIT_HOME``) — wins silently if set.
      2. The legacy alias from :data:`LEGACY_ENV_MAP` (e.g. ``ENCHANTER_HOME``)
         — wins with a one-shot WARNING.
      3. ``default``.

    Args:
        canonical_name: the ``ROBIT_*`` env var name. Pass an unknown name
            (no legacy alias) to short-circuit straight to a plain
            ``os.environ.get`` lookup.
        default: returned when neither the canonical nor the legacy name is
            present.

    Returns:
        The string value of whichever env var is set, or ``default``.
    """
    value = os.environ.get(canonical_name)
    if value is not None:
        return value

    legacy = LEGACY_ENV_MAP.get(canonical_name)
    if legacy is None:
        return default

    legacy_value = os.environ.get(legacy)
    if legacy_value is not None:
        if legacy not in _warned_env:
            _warned_env.add(legacy)
            _logger.warning(
                "%s is deprecated; rename to %s. Continuing with legacy value.",
                legacy,
                canonical_name,
            )
        return legacy_value

    return default


def _platform_dir(base_name: str) -> Path:
    """Return ``%APPDATA%\\<base>`` on Windows, ``~/.<base>`` on POSIX."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / base_name
        # APPDATA unset on Windows is unusual but possible (e.g. sandbox).
        return Path.home() / f".{base_name}"
    return Path.home() / f".{base_name}"


def resolve_user_dir() -> Path:
    """Return the active robit config dir.

    Precedence:

    1. ``ROBIT_HOME`` (or its deprecated alias ``ENCHANTER_HOME``) if set.
    2. The new path (``~/.robit`` on POSIX, ``%APPDATA%\\robit`` on Windows)
       if it exists on disk.
    3. The legacy path (``~/.enchanter`` / ``%APPDATA%\\enchanter``) if it
       exists. Emits a one-shot migration WARNING so the operator knows to
       rename the directory.
    4. The new path (uncreated) as the default — callers that intend to
       *write* will create it via ``mkdir(parents=True, exist_ok=True)``.
    """
    override = get_env("ROBIT_HOME")
    if override:
        return Path(override)

    new_dir = _platform_dir("robit")
    if new_dir.exists():
        return new_dir

    legacy_dir = _platform_dir("enchanter")
    if legacy_dir.exists():
        if "user-dir" not in _warned_env:
            _warned_env.add("user-dir")
            _logger.warning(
                "Reading config from legacy path %s. "
                "Migrate by running: mv %s %s",
                legacy_dir,
                legacy_dir,
                new_dir,
            )
        return legacy_dir

    return new_dir
