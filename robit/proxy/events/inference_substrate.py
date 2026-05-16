"""robit.proxy.events.inference_substrate — Wave 13.2 Agent F.

Bridges the proxy lifecycle to the cross-session learning substrate
(``robit.inference.engine``).  Two phases:

* :attr:`EmitPhase.PRE_DISPATCH` (once per request): if the substrate's
  opt-in gate is on, read the current plugin briefing and stash it in
  ``ctx.scratch["inference-substrate"]["briefing"]``.  We never mutate
  the request — the briefing is informational for audit logs and a
  future opt-in conduct injection step.

* :attr:`EmitPhase.POST_SESSION` (once per request): append an artifact
  to ``artifacts.jsonl`` describing what happened in this request.  The
  category and code are derived from observable state:

    - VetoResult observed in ``ctx.scratch["veto"]`` → ``proxy-veto``
    - mid-stream redactions non-empty → ``proxy-redaction``
    - Otherwise → ``proxy-success``

  The substrate's SPRT machinery elevates only patterns that recur
  across multiple sessions, so single-emit observations are honest data
  contributions, not forcing functions.

Opt-in gate
-----------

The underlying engine uses :envvar:`ENCHANTER_INFERENCE_ENABLED` (see
``enchanter/inference/engine.py``).  When unset or ``!= "1"`` BOTH phases
are no-ops.  This matches the substrate's "default off during rollout"
contract — see ``shared/conduct/inference-substrate.md`` in the wixie
sibling repo for the original semantics.

Subprocess avoidance
--------------------

``emit`` and ``render_briefing`` are in-process Python calls into
:mod:`robit.inference.engine`.  We never spawn the
``inference-engine.py`` CLI from the hot loop.  Reconcile is NOT
triggered from here (SPRT needs accumulated observations; per-emit
reconcile is noise — same contract as the wixie substrate).

State-dir safety
----------------

If the substrate's state directory is missing on a fresh checkout, the
emitter logs a single WARNING and degrades to no-op for the rest of the
process.  Setting :envvar:`ENCHANTER_INFERENCE_ENABLED=1` without
initialising the state directory is a user-side configuration error;
papering over it would hide that from operators.
"""

from __future__ import annotations

import logging
from typing import Any

from robit.inference import engine as _inference_engine
from robit.inference.engine import (
    append_jsonl_locked as _append_jsonl_locked,  # noqa: F401 — referenced in docs
)

from ._types import EmitContext, EmitPhase


_log = logging.getLogger(__name__)


# Sentinel — flipped True the first time we discover the state directory
# is missing.  We log the warning once and stay quiet afterwards so the
# proxy's request log does not fill with the same line every request.
_state_warning_emitted: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate_enabled() -> bool:
    """Re-read the opt-in gate at call time.

    Delegates to :func:`robit.inference.engine.env_enabled` so the
    contract stays in one place (a future rename of the env-var only
    needs to touch the engine module).
    """
    return _inference_engine.env_enabled()


def _state_dir_ok() -> bool:
    """Return True iff the substrate's state dir exists (or its parent does).

    The engine creates ``artifacts.jsonl`` on demand under the parent dir,
    so we only need the parent to be present.  A missing parent means a
    fresh checkout where Wave 13.4's ship step has not yet initialised
    state — degrade to no-op rather than crash.
    """
    global _state_warning_emitted

    sd = _inference_engine._state_dir()
    if sd.exists() or sd.parent.exists():
        return True

    if not _state_warning_emitted:
        _log.warning(
            "inference-substrate state dir missing (%s) — emitter degraded "
            "to no-op for this process.  Run the substrate init or unset "
            "ENCHANTER_INFERENCE_ENABLED to silence this warning.",
            sd,
        )
        _state_warning_emitted = True
    return False


def _derive_artifact(ctx: EmitContext) -> dict[str, Any]:
    """Build the artifact record from the observable state of *ctx*.

    Decision precedence:

      1. Vetoed → ``proxy-veto``
      2. Redactions fired → ``proxy-redaction``
      3. Else → ``proxy-success``

    The artifact follows the schema documented in
    ``shared/conduct/inference-substrate.md`` (code, category, title,
    cause, counter, signal, tags, scope, evidence).  ``ts`` and
    ``session_id`` are stamped by :func:`engine.emit_unconditional`; we
    do not set them by hand.
    """
    veto = ctx.scratch.get("veto")
    redactions = tuple(ctx.redactions or ())
    model = ctx.req.model

    # Wire-format tag is best-effort — the canonical request does not
    # carry a wire-format marker, so we tag by model-family prefix.
    if model.startswith("claude"):
        wire_format = "anthropic"
    elif model.startswith(("gpt", "o1", "o3")):
        wire_format = "openai"
    elif model.startswith("gemini"):
        wire_format = "gemini"
    else:
        wire_format = "unknown"

    base_evidence: dict[str, Any] = {
        "iterations": 1,
        "user_rounds_of_pushback": 0,
        "model": model,
    }

    if veto is not None:
        # VetoResult shape: phase, plugin, reason, pattern_id, pattern_name.
        # We accept either a dataclass or a duck-typed dict to keep the
        # emitter independent of the pipeline's internal type.
        phase = getattr(veto, "phase", None) or (
            veto.get("phase") if isinstance(veto, dict) else "unknown"
        )
        plugin = getattr(veto, "plugin", None) or (
            veto.get("plugin") if isinstance(veto, dict) else "unknown"
        )
        reason = getattr(veto, "reason", None) or (
            veto.get("reason") if isinstance(veto, dict) else ""
        )
        base_evidence["veto_phase"] = phase
        base_evidence["veto_plugin"] = plugin
        return {
            "code": "F-PROXY-VETO",
            "category": "proxy-veto",
            "title": f"proxy vetoed by {plugin} at {phase}"[:120],
            "cause": (
                f"Lifecycle gate '{plugin}' vetoed the request at phase "
                f"'{phase}': {reason}"
            )[:600],
            "counter": (
                "Inspect the prompt for the matched pattern before retry; "
                "do not re-issue verbatim."
            ),
            "signal": (
                f"When phase={phase} plugin={plugin} fires again, treat the "
                "request as adversarial input and route to red-team review."
            ),
            "tags": ["proxy", "veto", wire_format, model],
            "scope": "agent",
            "evidence": base_evidence,
        }

    if redactions:
        base_evidence["redaction_count"] = len(redactions)
        base_evidence["redaction_ids"] = list(redactions)
        return {
            "code": "S-MASK-FIRED",
            "category": "proxy-redaction",
            "title": f"secret-mask redacted {len(redactions)} pattern(s)"[:120],
            "cause": (
                "Mid-stream secret-mask redacted output before it reached "
                f"the client.  Patterns fired: {', '.join(redactions)}."
            )[:600],
            "counter": (
                "Audit the upstream system prompt or tool output for the "
                "leaking pattern source; do not relax the mask."
            ),
            "signal": (
                "Repeated redaction-firing on the same model/wire-format "
                "indicates an upstream leak, not a mask false-positive."
            ),
            "tags": ["proxy", "redaction", wire_format, model],
            "scope": "agent",
            "evidence": base_evidence,
        }

    # Benign success path.
    return {
        "code": "S-PROXY-OK",
        "category": "proxy-success",
        "title": f"proxy completed request on {model}"[:120],
        "cause": (
            "Request passed all lifecycle gates and returned without "
            "redactions or veto."
        ),
        "counter": "",  # success patterns have no counter
        "signal": (
            "Baseline observation — accumulating success counts lets SPRT "
            "distinguish genuine pattern recurrences from noise."
        ),
        "tags": ["proxy", "success", wire_format, model],
        "scope": "agent",
        "evidence": base_evidence,
    }


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class InferenceSubstrateEmitter:
    """Bridges the proxy lifecycle to the inference substrate.

    See the module docstring for the full contract.  The emitter is
    stateless across requests — per-request state lives in
    ``ctx.scratch["inference-substrate"]``.
    """

    name = "inference-substrate"
    phases = (EmitPhase.PRE_DISPATCH, EmitPhase.POST_SESSION)

    async def emit(self, phase: str, ctx: EmitContext) -> None:
        # Opt-in gate — both phases are NO-OPs when unset.
        if not _gate_enabled():
            return

        # State-dir presence check — degrade gracefully if absent.
        if not _state_dir_ok():
            return

        if phase == EmitPhase.PRE_DISPATCH:
            await self._emit_pre_dispatch(ctx)
        elif phase == EmitPhase.POST_SESSION:
            await self._emit_post_session(ctx)
        # Other phases are silently ignored — matches the BuiltinEmitter
        # contract; the pipeline only drives our two registered slots.

    async def _emit_pre_dispatch(self, ctx: EmitContext) -> None:
        """Read the active briefing and stash it for downstream observers.

        The briefing is informational — we never mutate ``ctx.req`` or
        inject the briefing into the system prompt.  Wave 14+ may add an
        operator-gated injection step; until then the briefing's presence
        is observable in audit logs via ``ctx.scratch``.
        """
        try:
            # render_briefing writes the markdown file and returns its
            # path; we then read the file contents.  The engine itself
            # is the only writer — going around it would corrupt the
            # atomic-write contract documented in inference-substrate.md.
            path = _inference_engine.render_briefing("agent")
            briefing_text = path.read_text(encoding="utf-8") if path.exists() else ""
        except Exception as exc:  # noqa: BLE001
            # Briefing-read failures must NOT crash the proxy.  Log and
            # continue with an empty briefing so the post-session emit
            # still records the request.
            _log.warning("inference-substrate briefing read failed: %s", exc)
            briefing_text = ""

        ctx.scratch.setdefault("inference-substrate", {})["briefing"] = briefing_text

    async def _emit_post_session(self, ctx: EmitContext) -> None:
        """Append one artifact describing the request outcome."""
        artifact = _derive_artifact(ctx)

        try:
            # In-process Python call; never spawn the CLI from the hot
            # loop.  emit_unconditional bypasses the engine's own gate
            # check — we've already checked it at the top of emit().
            _inference_engine.emit_unconditional(artifact)
        except Exception as exc:  # noqa: BLE001
            # Substrate I/O failures are non-fatal for the proxy.  Log
            # at WARNING so an operator can correlate disk/permission
            # issues without the proxy itself failing.
            _log.warning(
                "inference-substrate emit failed (%s): %s",
                artifact.get("code", "?"),
                exc,
            )
            return

        # Stash the artifact shape in scratch so other emitters /
        # debug tooling can inspect what was recorded without re-deriving.
        ctx.scratch.setdefault("inference-substrate", {})["last_artifact"] = artifact


emitter = InferenceSubstrateEmitter()


__all__ = ["InferenceSubstrateEmitter", "emitter"]
