"""Durable, append-only audit log for proxy security vetoes.

Sibling module :mod:`robit.proxy.pipeline` calls :func:`record_veto`
whenever a lifecycle gate refuses a request (destructive-op-gate,
cve-pattern-gate, or any required plugin acking ``veto``).  This module owns
the on-disk JSONL audit log so an operator can later answer "why did this
request get a 451 yesterday at 14:32? which pattern?" by tailing the file —
the delegation-of-authority audit's section 9 gap (4), its top
compliance-gap priority.

Public surface:

    async def record_veto(*, correlation_id, engine, phase, reason,
                          pattern_id=None, pattern_name=None,
                          http_status=None, mode="pre-dispatch") -> None
    def       get_audit_path() -> Path
    async def read_records(since: float | None = None) -> list[dict]

Design notes:

* Mirrors the sink convention established by
  :mod:`robit.loader.runtimes._audit` (sidecar-rejections.jsonl).
  Wave 14.4 observability consolidation may unify the two writers.
* Records carry pattern identifiers and reason strings only — the reason
  convention is ``"<plugin>:<pattern_id>"``; raw request content never
  lands in the audit line.
* JSONL on disk, one record per line, opened in ``"a"`` mode — never seek.
* In-process concurrency is handled by an ``asyncio.Lock``.  Cross-process
  serialisation is not attempted.
* All failures are swallowed — audit is best-effort.  A failure to record
  must never abort (or delay surfacing) the veto itself.
* fsync is opt-in via ``ROBIT_AUDIT_FSYNC=1``.
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

from robit._compat import get_env

__all__ = ["record_veto", "get_audit_path", "read_records"]

_LOG = logging.getLogger(__name__)

# Module-level lock guards the single shared audit-file append path.
_WRITE_LOCK = asyncio.Lock()

# One-shot WARNING gate so a persistently broken disk doesn't spam the log.
_FALLBACK_WARNED = False
_FALLBACK_WARN_LOCK = threading.Lock()

# Record kind constant used in every emitted line.
_KIND = "proxy.request.vetoed"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path | None:
    """Walk upwards from *start* looking for a ``pyproject.toml`` marker."""
    try:
        start = start.resolve()
    except OSError:
        return None
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _platform_default_dir() -> Path:
    """Platform-default audit directory when no env / repo marker is found."""
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "robit" / "audit"
        return Path.home() / "AppData" / "Roaming" / "robit" / "audit"
    return Path.home() / ".robit" / "audit"


def _resolve_state_audit_dir() -> Path:
    """Resolve the audit directory using the documented precedence.

    1. ``$ROBIT_STATE_DIR``  -> ``<state_dir>/audit``
    2. ``<repo_root>/state/audit`` if a ``pyproject.toml`` is found upwards
       from CWD.
    3. Platform default (``~/.robit/audit`` or ``%APPDATA%/robit/audit``).
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

    Does not create the file; only resolves the directory.
    """
    return _resolve_state_audit_dir() / "vetoes.jsonl"


def _ensure_dir(path: Path) -> Path:
    """Ensure *path*'s parent directory exists; fall back to tempdir on error."""
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
            pass
        return fb_dir / path.name


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _build_record(
    *,
    correlation_id: str,
    engine: str,
    phase: str,
    reason: str,
    pattern_id: str | None,
    pattern_name: str | None,
    http_status: int | None,
    mode: str,
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "ts": now.timestamp(),
        "ts_iso": now.isoformat(),
        "kind": _KIND,
        "correlation_id": correlation_id,
        "engine": engine,
        "phase": phase,
        "reason": reason,
        "pattern_id": pattern_id,
        "pattern_name": pattern_name,
        "http_status": http_status,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def _fsync_enabled() -> bool:
    return get_env("ROBIT_AUDIT_FSYNC") == "1"


def _write_line_sync(path: Path, line: str) -> None:
    """Synchronous single-line append used inside the lock."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        if _fsync_enabled():
            try:
                os.fsync(fh.fileno())
            except OSError as exc:  # pragma: no cover - rare
                _LOG.debug("fsync failed for %s: %s", path, exc)


async def record_veto(
    *,
    correlation_id: str,
    engine: str,
    phase: str,
    reason: str,
    pattern_id: str | None = None,
    pattern_name: str | None = None,
    http_status: int | None = None,
    mode: str = "pre-dispatch",
) -> None:
    """Append one veto record to the audit log.

    ``mode`` is ``"pre-dispatch"`` when the veto was surfaced as an HTTP 451
    before the upstream was called, or ``"mid-stream"`` when a later-phase
    required plugin vetoed (or timed out) after the stream had opened — in
    that case the stream is closed without a status change, so
    ``http_status`` is ``None``.

    Best-effort: any failure to write is swallowed (logged at WARNING) so
    the caller's veto path is never aborted by an audit problem.
    """
    try:
        record = _build_record(
            correlation_id=correlation_id,
            engine=engine,
            phase=phase,
            reason=reason,
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            http_status=http_status,
            mode=mode,
        )
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("veto audit record build failed for %s: %s", engine, exc)
        return

    try:
        async with _WRITE_LOCK:
            target = _ensure_dir(get_audit_path())
            try:
                await asyncio.to_thread(_write_line_sync, target, line)
            except OSError as exc:
                _LOG.warning("veto audit write to %s failed: %s", target, exc)
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("veto audit write to %s raised: %s", target, exc)
    except asyncio.CancelledError:
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

    Skips lines that fail to parse.  When *since* is provided, only records
    with ``ts >= since`` are returned.
    """
    path = get_audit_path()
    try:
        lines = await asyncio.to_thread(_read_lines_sync, path)
    except OSError as exc:
        _LOG.warning("veto audit read from %s failed: %s", path, exc)
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
