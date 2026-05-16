"""Durable, append-only audit log for sidecar event rejections.

Sibling module :mod:`robit.loader.runtimes.sidecar` calls
:func:`record_rejection` whenever a sidecar-emitted event is dropped (source
forgery, undeclared topic, phase-out-of-scope, malformed event, ...). This
module owns the on-disk JSONL audit log so an operator can later answer
"did any sidecar try to forge events yesterday?" by tailing the file.

Public surface:

    async def record_rejection(adapter_name, rejection_reason, raw_event, *, expected=None) -> None
    def       get_audit_path() -> Path
    async def read_records(since: float | None = None) -> list[dict]

Design notes:

* JSONL on disk, one record per line, opened in ``"a"`` mode -- never seek.
* In-process concurrency is handled by an ``asyncio.Lock``.  Cross-process
  serialisation is *not* attempted; operators running multiple enchanter
  processes against the same audit file is out of scope (document and
  move on).
* All failures are swallowed -- audit is best-effort.  A failure to record
  must never abort the caller.
* fsync is opt-in via ``ROBIT_AUDIT_FSYNC=1`` (legacy ``ENCHANTER_AUDIT_FSYNC``
  still honored via :mod:`robit._compat`) because per-line fsync is expensive
  on Windows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robit._compat import get_env

__all__ = ["record_rejection", "get_audit_path", "read_records"]

_LOG = logging.getLogger(__name__)

# Module-level lock guards the single shared audit-file append path.
_WRITE_LOCK = asyncio.Lock()

# One-shot WARNING gate so a persistently broken disk doesn't spam the log.
_FALLBACK_WARNED = False
_FALLBACK_WARN_LOCK = threading.Lock()

# Record kind constant used in every emitted line.
_KIND = "sidecar.derived_event.rejected"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path | None:
    """Walk upwards from *start* looking for a ``pyproject.toml`` marker.

    Returns the directory containing the marker, or ``None`` if none found
    before hitting the filesystem root.
    """
    try:
        start = start.resolve()
    except OSError:
        return None
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _platform_default_dir() -> Path:
    """Platform-default audit directory when no env / repo marker is found.

    Routes through :func:`robit._compat.resolve_user_dir` so both the new
    ``~/.robit`` path and the legacy ``~/.enchanter`` path are honored
    consistently with the rest of the runtime.
    """
    from robit._compat import resolve_user_dir

    return resolve_user_dir() / "audit"


def _resolve_state_audit_dir() -> Path:
    """Resolve the audit directory using the documented precedence.

    1. ``$ROBIT_STATE_DIR`` (legacy ``$ENCHANTER_STATE_DIR``) -> ``<state_dir>/audit``
    2. ``<repo_root>/state/audit`` if a ``pyproject.toml`` is found upwards
       from CWD.
    3. Platform default (``~/.robit/audit`` or ``%APPDATA%/robit/audit``,
       falling back to ``~/.enchanter/audit`` if only the legacy dir exists).
    """
    env = get_env("ROBIT_STATE_DIR")
    if env:
        return Path(env) / "audit"

    repo = _find_repo_root(Path.cwd())
    if repo is not None:
        return repo / "state" / "audit"

    return _platform_default_dir()


def _fallback_dir() -> Path:
    return Path(tempfile.gettempdir()) / "robit-audit"


def get_audit_path() -> Path:
    """Return the resolved JSONL audit file path.

    Does not create the file; only resolves the directory.  Callers that want
    to ensure the file exists should perform a write.
    """
    return _resolve_state_audit_dir() / "sidecar-rejections.jsonl"


def _ensure_dir(path: Path) -> Path:
    """Ensure *path*'s parent directory exists; fall back to tempdir on error.

    Emits a one-shot WARNING the first time the fallback fires per process.
    """
    global _FALLBACK_WARNED
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as exc:
        fb_dir = _fallback_dir()
        with _FALLBACK_WARN_LOCK:
            if not _FALLBACK_WARNED:
                _FALLBACK_WARNED = True
                _LOG.warning(
                    "audit dir %s unwritable (%s); falling back to %s",
                    path.parent,
                    exc,
                    fb_dir,
                )
        try:
            fb_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Even the fallback failed; the write that follows will raise
            # and be swallowed by the caller's try/except.
            pass
        return fb_dir / path.name


# ---------------------------------------------------------------------------
# JSON encoding with non-serialisable-object tolerance
# ---------------------------------------------------------------------------

def _safe_encode_event(raw_event: Any) -> Any:
    """JSON-friendly representation of *raw_event*.

    If a straight ``json.dumps`` fails on the object (a ``complex``, a custom
    instance, etc.) we substitute a placeholder dict so the audit line is
    still well-formed JSON.
    """
    try:
        json.dumps(raw_event)
        return raw_event
    except (TypeError, ValueError) as exc:
        return {"_encode_error": repr(exc)}


def _safe_encode_expected(expected: Any) -> Any:
    if expected is None:
        return None
    try:
        json.dumps(expected)
        return expected
    except (TypeError, ValueError) as exc:
        return {"_encode_error": repr(exc)}


def _build_record(
    adapter_name: str,
    rejection_reason: str,
    raw_event: Any,
    expected: Any,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "ts": now.timestamp(),
        "ts_iso": now.isoformat(),
        "kind": _KIND,
        "adapter_name": adapter_name,
        "rejection_reason": rejection_reason,
        "raw_event": _safe_encode_event(raw_event),
        "expected": _safe_encode_expected(expected),
    }


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def _fsync_enabled() -> bool:
    return get_env("ROBIT_AUDIT_FSYNC") == "1"


def _write_line_sync(path: Path, line: str) -> None:
    """Synchronous single-line append used inside the lock.

    Builds the full line (incl. trailing newline) up-front and writes it in
    a single ``write()`` call.  Always flushes; fsyncs only when opted in.
    """
    # Encode once -- the caller passes the trailing newline already attached.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        if _fsync_enabled():
            try:
                os.fsync(fh.fileno())
            except OSError as exc:  # pragma: no cover - rare
                _LOG.debug("fsync failed for %s: %s", path, exc)


async def record_rejection(
    adapter_name: str,
    rejection_reason: str,
    raw_event: dict,
    *,
    expected: dict[str, object] | None = None,
) -> None:
    """Append one rejection record to the audit log.

    Best-effort: any failure to write is swallowed (logged at WARNING) so the
    caller's request path is never aborted by an audit problem.
    """
    try:
        record = _build_record(adapter_name, rejection_reason, raw_event, expected)
        # Encode the full line up-front, in one allocation.
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("audit record build failed for %s: %s", adapter_name, exc)
        return

    try:
        async with _WRITE_LOCK:
            target = _ensure_dir(get_audit_path())
            try:
                await asyncio.to_thread(_write_line_sync, target, line)
            except OSError as exc:
                _LOG.warning("audit write to %s failed: %s", target, exc)
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("audit write to %s raised: %s", target, exc)
    except asyncio.CancelledError:
        # Caller cancelled mid-flight: surface cancellation but don't log
        # half-written state -- the to_thread call either completed or
        # didn't reach the disk.
        raise


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

def _read_lines_sync(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return fh.readlines()


async def read_records(since: float | None = None) -> list[dict]:
    """Return parsed records from the audit log.

    Skips lines that fail to parse (corrupt entries shouldn't blind the
    operator to good ones).  When *since* is provided, only records with
    ``ts >= since`` are returned.
    """
    path = get_audit_path()
    try:
        lines = await asyncio.to_thread(_read_lines_sync, path)
    except OSError as exc:
        _LOG.warning("audit read from %s failed: %s", path, exc)
        return []

    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is not None:
            ts = rec.get("ts")
            if not isinstance(ts, (int, float)) or ts < since:
                continue
        out.append(rec)
    return out
