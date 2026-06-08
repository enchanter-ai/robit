"""robit.proxy.audit — durable veto audit sink (G8).

Every security veto is appended as one JSON line to
``<state>/audits/vetoes.jsonl``.  The audit line is sourced from the
structured :class:`~robit.core.verdict.Verdict` carried by the veto — never
by re-parsing the reason string.

State-dir resolution
--------------------

The audit directory lives under the same runtime state root the inference
substrate already resolves (:func:`robit.inference.paths.resolve_state_dir`,
keyed off the ``ROBIT_INFERENCE_STATE`` env var, defaulting to
``<repo>/robit/state/inference``).  We hang ``audits/`` off the *parent* of
that directory so the layout is::

    robit/state/
    ├── inference/        # inference substrate
    └── audits/
        └── vetoes.jsonl  # this sink

Tests override the path by setting ``ROBIT_INFERENCE_STATE`` (or
monkeypatching :func:`resolve_audits_dir`) to a tmp dir, so production state
is never polluted.

Best-effort contract
--------------------

Writing the audit line is **best-effort**: a write failure (permission
denied, full disk, racing rmtree) is logged and swallowed — it must never
block or crash the request that triggered the veto.  Losing an audit line is
strictly better than failing the security path that produced it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from robit.core.verdict import Verdict
from robit.inference.paths import resolve_state_dir

_log = logging.getLogger(__name__)


def resolve_audits_dir() -> Path:
    """Return the ``audits/`` directory under the runtime state root.

    Re-reads the state root at call time (via
    :func:`robit.inference.paths.resolve_state_dir`) so per-test env overrides
    take effect without a reimport.
    """
    # resolve_state_dir() points at ``<root>/inference``; audits is a sibling.
    return resolve_state_dir().parent / "audits"


def vetoes_log_path() -> Path:
    """Return the path of the JSONL veto audit log."""
    return resolve_audits_dir() / "vetoes.jsonl"


def _payload_summary(verdict: Verdict) -> dict[str, object]:
    """A tiny, content-free summary of the veto for the audit line.

    Mirrors the safety stance of :class:`~robit.proxy.pipeline._BusRecorder`:
    only pattern identifiers / severity — never raw request content.
    """
    summary: dict[str, object] = {"severity": verdict.severity}
    if verdict.pattern_id is not None:
        summary["pattern_id"] = verdict.pattern_id
    if verdict.pattern_name is not None:
        summary["pattern_name"] = verdict.pattern_name
    return summary


def record_veto(
    verdict: Verdict,
    *,
    correlation_id: str,
    http_status: int = 451,
) -> None:
    """Append one JSON line describing *verdict* to the veto audit log.

    Each line carries::

        {ts, correlation_id, engine, pattern_id, phase, payload_summary,
         http_status}

    sourced from the structured :class:`Verdict` (G8 — no reason re-parsing).

    Best-effort: any failure is logged at warning level and swallowed so the
    caller's request is never blocked by an audit-write problem.
    """
    try:
        line = {
            "ts": int(time.time() * 1000),
            "correlation_id": correlation_id,
            "engine": verdict.plugin,
            "pattern_id": verdict.pattern_id,
            "phase": str(verdict.phase),
            "payload_summary": _payload_summary(verdict),
            "http_status": http_status,
        }
        path = vetoes_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001 — audit writes must never break the request.
        _log.warning(
            "veto audit write failed (correlation_id=%s engine=%s); continuing",
            correlation_id,
            getattr(verdict, "plugin", "?"),
            exc_info=True,
        )


__all__ = ["record_veto", "resolve_audits_dir", "vetoes_log_path"]
