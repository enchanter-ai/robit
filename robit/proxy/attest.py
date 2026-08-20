"""Mimir attestation emit hook — signed provenance for proxy decisions.

Sibling module :mod:`robit.proxy.pipeline` calls :func:`attest_decision`
after every enforcement decision (pass or veto).  When enabled, the decision
is POSTed to a Mimir issuer (``POST /v1/attest``) which returns an Ed25519-
signed provenance envelope binding ``(request, result)`` under one signature;
the outcome — envelope or failure — is spooled to a local JSONL stream.

Why both pass AND veto attest: the inference substrate's SPRT liveness
detector consumes the spool as a proof heartbeat.  If envelopes only fired
on vetoes, an empty stream would be indistinguishable from "no attacks
today"; attesting every decision makes *absence of proof* a statistical
signal that the gate went dead (RFC-robit-control-plane § 5 ceiling).

Snowball guard #1 applies verbatim: an envelope proves what this process
*reported*, not that the gate *ran*.  The spool + issuer stream is the
self-reported half of the ceiling; SPRT absence detection is the other half.

Public surface:

    def       is_enabled() -> bool
    async def attest_decision(req, *, correlation_id, decision, phase,
                              engine=None, reason=None, pattern_id=None,
                              http_status=None) -> None
    def       get_spool_path() -> Path
    async def read_spool(since: float | None = None) -> list[dict]

Configuration (env):

    ROBIT_ATTEST_ENABLED=1     opt-in gate; unset -> attest_decision no-ops
    MIMIR_ISSUER_URL               issuer base URL (default http://localhost:8080)
    ROBIT_ATTEST_TOOL_ID       envelope tool_id (default
                                   did:web:enchanter.dev:robit:proxy-gate)
    ROBIT_ATTEST_TIMEOUT_S     issuer POST timeout, seconds (default 3)

Design notes:

* Privacy invariant: the issuer receives a SHA-256 digest of the canonical
  request plus decision metadata (engine, phase, pattern_id) — never prompt
  or response content.
* HTTP uses stdlib ``urllib.request`` wrapped in ``asyncio.to_thread``,
  matching the codebase convention (see ``llm/_chatgpt_auth.py``,
  ``transport/http.py``) — no new dependency.
* Best-effort end to end: issuer unreachable -> spool line with
  ``envelope: null`` and the error string; spool unwritable -> tempdir
  fallback; nothing ever aborts the request path.
* Spool lives at ``<state_dir>/attest/decisions.jsonl`` with the same path
  precedence as the audit sinks.  Wave 14.4 observability consolidation may
  unify the three JSONL writers.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robit._compat import get_env

__all__ = ["is_enabled", "attest_decision", "get_spool_path", "read_spool"]

_LOG = logging.getLogger(__name__)

_WRITE_LOCK = asyncio.Lock()
_FALLBACK_WARNED = False
_FALLBACK_WARN_LOCK = threading.Lock()

_KIND = "proxy.decision.attested"

_DEFAULT_ISSUER_URL = "http://localhost:8080"
_DEFAULT_TOOL_ID = "did:web:enchanter.dev:robit:proxy-gate"
_DEFAULT_TIMEOUT_S = 3.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """True when the operator opted into attestation."""
    return os.environ.get("ROBIT_ATTEST_ENABLED") == "1"


def _issuer_url() -> str:
    return os.environ.get("MIMIR_ISSUER_URL", _DEFAULT_ISSUER_URL).rstrip("/")


def _tool_id() -> str:
    return os.environ.get("ROBIT_ATTEST_TOOL_ID", _DEFAULT_TOOL_ID)


def _timeout_s() -> float:
    raw = os.environ.get("ROBIT_ATTEST_TIMEOUT_S")
    if raw:
        try:
            return max(0.1, float(raw))
        except ValueError:
            pass
    return _DEFAULT_TIMEOUT_S


def _tool_version() -> str:
    try:
        from robit import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive
        return "0.0.0"


# ---------------------------------------------------------------------------
# Spool path resolution (same precedence as the audit sinks)
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path | None:
    try:
        start = start.resolve()
    except OSError:
        return None
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _platform_default_dir() -> Path:
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "robit" / "attest"
        return Path.home() / "AppData" / "Roaming" / "robit" / "attest"
    return Path.home() / ".robit" / "attest"


def _resolve_spool_dir() -> Path:
    env = get_env("ROBIT_STATE_DIR")
    if env:
        return Path(env) / "attest"

    repo = _find_repo_root(Path.cwd())
    if repo is not None:
        return repo / "state" / "attest"

    return _platform_default_dir()


def _fallback_dir() -> Path:
    return Path(tempfile.gettempdir()) / "robit-attest"


def get_spool_path() -> Path:
    """Return the resolved JSONL spool file path (directory not created)."""
    return _resolve_spool_dir() / "decisions.jsonl"


def _ensure_dir(path: Path) -> Path:
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
                    "attest spool dir %s unwritable (%s); falling back to %s",
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
# Request digest — content never leaves the process
# ---------------------------------------------------------------------------

def _digest_request(req: Any) -> str:
    """Deterministic SHA-256 over the canonical request.

    Dataclasses are converted via :func:`dataclasses.asdict` and serialised
    with sorted keys so the digest is stable across processes.  Falls back
    to ``repr`` for anything unserialisable — a weaker but still stable bind.
    """
    try:
        if dataclasses.is_dataclass(req) and not isinstance(req, type):
            payload = dataclasses.asdict(req)
        else:
            payload = req
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    except Exception:  # pragma: no cover - defensive
        blob = repr(req)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Issuer HTTP call (sync helper, isolated so tests can patch it)
# ---------------------------------------------------------------------------

def _post_attest(body: dict) -> dict:
    """POST *body* to the issuer's /v1/attest; return the decoded response.

    Raises on any transport or HTTP error — the caller records the failure.
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    http_req = urllib.request.Request(
        f"{_issuer_url()}/v1/attest",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_req, timeout=_timeout_s()) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Spool write / read
# ---------------------------------------------------------------------------

def _fsync_enabled() -> bool:
    return get_env("ROBIT_AUDIT_FSYNC") == "1"


def _write_line_sync(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        if _fsync_enabled():
            try:
                os.fsync(fh.fileno())
            except OSError as exc:  # pragma: no cover - rare
                _LOG.debug("fsync failed for %s: %s", path, exc)


async def _spool(record: dict) -> None:
    try:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("attest spool record encode failed: %s", exc)
        return

    try:
        async with _WRITE_LOCK:
            target = _ensure_dir(get_spool_path())
            try:
                await asyncio.to_thread(_write_line_sync, target, line)
            except OSError as exc:
                _LOG.warning("attest spool write to %s failed: %s", target, exc)
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("attest spool write to %s raised: %s", target, exc)
    except asyncio.CancelledError:
        raise


def _read_lines_sync(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return fh.readlines()


async def read_spool(since: float | None = None) -> list[dict]:
    """Return parsed records from the decision spool.

    Skips corrupt lines.  When *since* is provided, only records with
    ``ts >= since`` are returned.  This is the stream the SPRT liveness
    reader consumes.
    """
    path = get_spool_path()
    try:
        lines = await asyncio.to_thread(_read_lines_sync, path)
    except OSError as exc:
        _LOG.warning("attest spool read from %s failed: %s", path, exc)
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


# ---------------------------------------------------------------------------
# Public emit hook
# ---------------------------------------------------------------------------

async def attest_decision(
    req: Any,
    *,
    correlation_id: str,
    decision: str,
    phase: str,
    engine: str | None = None,
    reason: str | None = None,
    pattern_id: str | None = None,
    http_status: int | None = None,
    session_id: str | None = None,
) -> None:
    """Attest one enforcement decision via the Mimir issuer.

    ``decision`` is ``"pass"`` (lifecycle completed, upstream response
    surfaced) or ``"veto"`` (a gate refused the request).  No-op unless
    ``ROBIT_ATTEST_ENABLED=1``.

    Best-effort end to end: issuer failure spools ``envelope: null`` plus
    the error; nothing raises into the caller's request path.
    """
    if not is_enabled():
        return

    now = datetime.now(timezone.utc)
    attest_request = {
        "kind": "robit.proxy.decision.request",
        "correlation_id": correlation_id,
        "session_id": session_id,
        "request_sha256": _digest_request(req),
        "model": getattr(req, "model", None),
    }
    attest_result = {
        "kind": "robit.proxy.decision",
        "decision": decision,
        "engine": engine,
        "phase": phase,
        "reason": reason,
        "pattern_id": pattern_id,
        "http_status": http_status,
        "decided_at": now.isoformat(),
    }
    body = {
        "tool_id": _tool_id(),
        "tool_version": _tool_version(),
        "request": attest_request,
        "result": attest_result,
    }

    envelope = None
    validation_level = None
    error: str | None = None
    try:
        resp = await asyncio.to_thread(_post_attest, body)
        envelope = resp.get("envelope")
        validation_level = resp.get("validation_level")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        _LOG.warning("mimir attestation failed for %s: %s", correlation_id, error)
    except Exception as exc:  # pragma: no cover - defensive
        error = f"{type(exc).__name__}: {exc}"
        _LOG.warning("mimir attestation raised for %s: %s", correlation_id, error)

    await _spool(
        {
            "ts": now.timestamp(),
            "ts_iso": now.isoformat(),
            "kind": _KIND,
            "correlation_id": correlation_id,
            "decision": decision,
            "engine": engine,
            "phase": phase,
            "pattern_id": pattern_id,
            "http_status": http_status,
            "issuer_url": _issuer_url(),
            "envelope": envelope,
            "validation_level": validation_level,
            "error": error,
        }
    )
