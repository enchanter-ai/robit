"""IntentAnchor engine — Python port of djinn.adapter.ts (v0.3.1+).

Port of the TS djinn adapter which implements intent anchoring + LCS
drift detection + HMM forward labelling + EMA posterior tracking.

Phase:       anchor     — captures the first user_prompt as the session anchor
             post-session — computes LCS ratio against the anchor; emits drift
                            event when ratio < DRIFT_THRESHOLD (0.3)

Required:    False — advisory, fail-open.  Never vetoes.

Topics sub:  session.start, user.prompt.submit, compact.requested
Topics emit: intent-anchor.anchor.set
             intent-anchor.drift.detected

Design faithfulness to TS:
  • Anchor is immutable once set per session (first prompt wins).
  • Drift threshold = 0.3 (< 30% shared tokens → drift signal).
  • HMM always updates on post-session regardless of whether drift fires.
  • EMA always updates on post-session.
  • Payload of drift event includes hmm_state, hmm_posterior, hmm_observation,
    ema_posterior.
  • Per-instance state (no module-level singleton), matching the Python convention
    established by trust_scorer.

Deviations from TS:
  • TS topic prefix is "djinn."; Python uses "intent-anchor." to match the
    engine's module name in the Python registry.
  • TS has optional PersistentHmmStore (JSONL); Python store is in-memory
    (per the wave-2b scope).  Persistence can be added as a follow-on.
  • TS export name is djinnAdapter; Python instance is `adapter`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Awaitable, Callable, Protocol, runtime_checkable

from robit.core import EnchantedEvent, PluginAck, RequestContext
from robit.core.plugin import PluginTopics
from robit.core.bus import new_event_id

from .store import IntentAnchorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (mirroring TS)
# ---------------------------------------------------------------------------

_DRIFT_THRESHOLD = 0.3

# Default-OFF feature flag for the agent-backed drift path (audit §8).  The
# deterministic LCS+HMM path is the default and runs unless BOTH this env flag
# is truthy AND an llm_call seam is injected.
_AGENT_ENV_FLAG = "ROBIT_INTENT_ANCHOR_AGENT"

# Engine-authored prompt directory (sibling to this module).
_ENGINE_DIR = Path(__file__).resolve().parent

# The agent-shaped tier this engine routes its model call to (F3(b)). Mirrors
# the [agent] table in engine.toml (``tier = "executor"``). Kept as a module
# constant so the default real ``llm_call`` can select a model via
# ``TierRouter.route_chain(_AGENT_TIER)`` without re-parsing the manifest.
_AGENT_TIER = "executor"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _agent_enabled() -> bool:
    """True only when the operator explicitly opts in via the env flag."""
    return os.environ.get(_AGENT_ENV_FLAG, "").strip() in ("1", "true", "True", "yes")


# ---------------------------------------------------------------------------
# Durable agent-verdict audit (F3(c))
# ---------------------------------------------------------------------------
#
# Every agent-engine verdict appends one JSON line to
# ``<state>/agent-verdicts.jsonl``. The state root is resolved exactly the way
# the veto audit / inference substrate resolve it
# (:func:`robit.inference.paths.resolve_state_dir`, keyed off
# ``ROBIT_INFERENCE_STATE``), so tests redirect it by setting that env var to a
# tmp dir and production state is never polluted.
#
# Privacy: we do NOT store the raw drift prompt (it embeds the user's anchor +
# current prompt and could carry sensitive content). We store a SHA-256 of the
# system prompt plus a short truncated summary of the model response — mirroring
# how the veto audit keeps payloads content-free.


_RESPONSE_SUMMARY_LIMIT = 240


def agent_verdicts_log_path() -> Path:
    """Return the path of the JSONL agent-verdict audit log.

    Sibling of the inference state dir's ``audits``-style layout: we hang the
    file off the *parent* of ``resolve_state_dir()`` so it sits next to the
    veto audit sink (``state/audits/vetoes.jsonl``) rather than inside the
    inference substrate's own directory.
    """
    from robit.inference.paths import resolve_state_dir

    return resolve_state_dir().parent / "agent-verdicts.jsonl"


def _prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_agent_verdict(
    *,
    correlation_id: str,
    engine: str,
    tier: str,
    system_prompt: str,
    response: str,
    verdict: dict[str, object],
) -> None:
    """Append one JSON line describing an agent verdict (F3(c)).

    Line shape::

        {ts, correlation_id, engine, tier, prompt_sha, response_summary, verdict}

    Best-effort: any failure (permission denied, full disk, racing rmtree) is
    logged and swallowed so an audit-write problem never blocks the engine's
    ack.
    """
    try:
        response_summary = response.strip()
        if len(response_summary) > _RESPONSE_SUMMARY_LIMIT:
            response_summary = response_summary[:_RESPONSE_SUMMARY_LIMIT]
        line = {
            "ts": int(time.time() * 1000),
            "correlation_id": correlation_id,
            "engine": engine,
            "tier": tier,
            # sha + short summary, never the raw (possibly sensitive) prompt.
            "prompt_sha": _prompt_sha(system_prompt),
            "response_summary": response_summary,
            "verdict": verdict,
        }
        path = agent_verdicts_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001 — audit writes must never block the ack.
        logger.warning(
            "agent-verdict audit write failed (correlation_id=%s engine=%s); "
            "continuing",
            correlation_id,
            engine,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Model-call seam (audit §8)
# ---------------------------------------------------------------------------
#
# The agent path calls a model via a narrow injectable callable rather than a
# hard dependency on a live provider.  Tests pass a MOCK that returns canned
# text and records the prompt, so no network is hit.  In production this would
# be wired to robit.llm.LlmClient through call_upstream; that wiring is a
# follow-up and is NOT proven here.
#
#   llm_call(system: str, user: str) -> str   (the model's raw text reply)


@runtime_checkable
class LlmCall(Protocol):
    """Minimal async seam: system + user prompt in, raw model text out."""

    async def __call__(self, system: str, user: str) -> str:  # pragma: no cover - protocol
        ...


# A plain async callable also satisfies the seam.
LlmCallable = Callable[[str, str], Awaitable[str]]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class IntentAnchor:
    """Advisory intent-anchor + drift-detection engine."""

    name = "intent-anchor"
    phases = ("anchor", "post-session")
    required = False  # advisory — fail-open
    topics = PluginTopics(
        subscribes=(
            "session.start",
            "user.prompt.submit",
            "compact.requested",
        ),
        emits=(
            "intent-anchor.anchor.set",
            "intent-anchor.drift.detected",
        ),
    )
    budget_tier = "med-or-higher"

    def __init__(
        self,
        llm_call: LlmCallable | None = None,
        prompt_overlay: str | None = None,
    ) -> None:
        # Per-session stores keyed by session_id
        self._sessions: dict[str, IntentAnchorStore] = {}

        # Agent-path seams (audit §8).  Both default to None → pure
        # deterministic behaviour, byte-for-byte unchanged.
        #
        #   llm_call:       injectable async model-call seam (mocked in tests).
        #   prompt_overlay: operator's additive override, APPENDED after the
        #                   engine-authored prompt (never replaces it).  This is
        #                   the operator layer of PipelineOptions.prompt_overlay.
        self._llm_call = llm_call
        self._prompt_overlay = prompt_overlay

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, session_id: str) -> IntentAnchorStore:
        if session_id not in self._sessions:
            self._sessions[session_id] = IntentAnchorStore()
        return self._sessions[session_id]

    def _clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Public test seam helpers (mirrors TS clearAnchor / getAnchor)
    # ------------------------------------------------------------------

    def get_store(self, session_id: str) -> IntentAnchorStore | None:
        return self._sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        self._clear_session(session_id)

    # ------------------------------------------------------------------
    # Agent-path helpers (audit §8) — default-OFF
    # ------------------------------------------------------------------

    def _agent_path_active(self, ctx: RequestContext | None = None) -> bool:
        """The agent path runs only when the operator opts in via the env flag.

        Default-OFF: when ``ROBIT_INTENT_ANCHOR_AGENT`` is unset the
        deterministic LCS+HMM path runs UNCHANGED.

        F3(b): the path no longer requires an injected ``llm_call`` seam — when
        the flag is on and no seam was injected, :meth:`_resolve_llm_call`
        builds a REAL model call routed through ``call_upstream``. Tests still
        inject a mock seam so the network is never hit.
        """
        return _agent_enabled()

    def _resolve_llm_call(self, ctx: RequestContext | None) -> LlmCallable:
        """Return the effective model-call seam for the agent path (F3(b)).

        Precedence:

        1. An explicitly injected ``self._llm_call`` (tests pass a mock here —
           this is the seam that keeps the network out of the test suite).
        2. Otherwise a REAL implementation that routes through
           :func:`robit.proxy.upstream.call_upstream` so the request flows
           through the same cost-ledger-observed dispatch path as any other
           upstream call. The model is selected via
           ``TierRouter.route_chain(_AGENT_TIER)`` (the engine's AgentSpec
           tier); ``call_upstream`` then iterates that chain with the normal
           retryable-fallback semantics. The triggering request's
           ``correlation_id`` (from *ctx*) is stamped on the canonical request's
           metadata for trace correlation.

        Honesty note: the real path is wired but cannot be PROVEN here without
        live provider credentials — exercising it would hit the network, which
        the test suite forbids. Tests cover only the injected-seam path; the
        real builder is covered structurally (it constructs without calling).
        """
        if self._llm_call is not None:
            return self._llm_call
        return self._build_real_llm_call(ctx)

    @staticmethod
    def _build_real_llm_call(ctx: RequestContext | None) -> LlmCallable:
        """Construct the production ``call_upstream``-backed model-call seam.

        Imported lazily so the deterministic default path never pulls in the
        proxy/litellm stack, and so a missing optional dependency can never
        break engine import.
        """
        correlation_id = getattr(ctx, "correlation_id", None)

        async def _real_call(system: str, user: str) -> str:
            # Lazy imports: keep the deterministic path free of the proxy stack.
            from robit.proxy.canonical import (
                CanonicalRequest,
                Message,
                TextPart,
            )
            from robit.proxy.upstream import call_upstream
            from robit.runtime.models_registry import ModelsRegistry
            from robit.runtime.tier_router import TierRouter

            router = TierRouter(ModelsRegistry.load())
            chain = router.route_chain(_AGENT_TIER)  # type: ignore[arg-type]

            metadata: dict[str, object] = {}
            if correlation_id:
                metadata["correlation_id"] = correlation_id

            req = CanonicalRequest(
                model=chain[0],
                system=system,
                messages=(
                    Message(role="user", content=(TextPart(text=user),)),
                ),
                metadata=metadata,
            )
            resp = await call_upstream(req, models=chain)
            # Flatten the response's text parts into the raw string the verdict
            # parser expects.
            parts = [
                p.text for p in resp.content if isinstance(p, TextPart)
            ]
            return "".join(parts)

        return _real_call

    @staticmethod
    def _load_prompt(rel_path: str) -> str:
        """Load an engine-authored prompt .md from the engine directory."""
        return (_ENGINE_DIR / rel_path).read_text(encoding="utf-8")

    def build_drift_prompt(
        self,
        anchor_intent: str,
        current_prompt: str,
        prompt_overlay: str | None = None,
    ) -> tuple[str, str]:
        """Build the (system, user) prompt pair for the drift verdict.

        Precedence (audit §8 / F3): framework < engine-author prompt < operator
        overlay.  The engine-authored body (``prompts/drift.md``) is the system
        prompt; the operator's overlay is APPENDED after it (never replaces it).

        *prompt_overlay* is the effective overlay resolved by :meth:`on_phase`
        from ``ctx.prompt_overlay`` (the production path, delivered end-to-end
        through the orchestrator). It falls back to the constructor-injected
        ``self._prompt_overlay`` seam (kept for tests that build the adapter
        directly without a pipeline ctx).
        """
        overlay = prompt_overlay if prompt_overlay is not None else self._prompt_overlay
        system = self._load_prompt("prompts/drift.md")
        if overlay:
            system = f"{system}\n\n## Operator overlay\n\n{overlay}"
        user = (
            f"ANCHOR INTENT:\n{anchor_intent}\n\n"
            f"CURRENT PROMPT:\n{current_prompt}"
        )
        return system, user

    @staticmethod
    def _parse_verdict(text: str) -> tuple[bool, float, str]:
        """Parse the model's JSON drift verdict → (drift, confidence, rationale)."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        obj = json.loads(stripped)
        drift = bool(obj.get("drift", False))
        confidence = float(obj.get("confidence", 0.0))
        rationale = str(obj.get("rationale", ""))
        return drift, confidence, rationale

    async def _handle_post_session_agent(
        self,
        event: EnchantedEvent,
        prompt_overlay: str | None = None,
        ctx: RequestContext | None = None,
    ) -> PluginAck:
        """Agent-backed drift verdict via the llm_call seam.

        Mirrors the deterministic handler's event shape but sources the drift
        decision from a model.  Still advisory / fail-open: any error falls
        back to a clean ack so the orchestrator is never blocked.

        The model-call seam is resolved by :meth:`_resolve_llm_call` — the
        injected mock in tests, or a real ``call_upstream``-backed call in
        production (F3(b)).

        *prompt_overlay* is the effective operator overlay resolved by
        :meth:`on_phase` (``ctx.prompt_overlay`` first, then the constructor
        seam) and appended to the engine-authored drift prompt.

        On a non-empty drift verdict a durable audit line is appended to
        ``state/agent-verdicts.jsonl`` (F3(c)); the write is best-effort and
        never blocks the ack.
        """
        store = self._get_or_create(event.session_id)
        if not store.has_anchor:
            return PluginAck(status="ack")

        prompt = event.payload.get("user_prompt", "")
        if not isinstance(prompt, str):
            prompt = str(prompt)

        # Keep the deterministic state machine warm so HMM/EMA posteriors stay
        # populated even on the agent path (the substrate still reads them).
        ratio, hmm_step, ema_posterior = store.record_observation(prompt)

        anchor = store.anchor
        assert anchor is not None  # guaranteed by has_anchor check

        llm_call = self._resolve_llm_call(ctx)
        system, user = self.build_drift_prompt(
            anchor.intent, prompt, prompt_overlay=prompt_overlay
        )
        raw = await llm_call(system, user)
        drift, confidence, rationale = self._parse_verdict(raw)

        # F3(c) — durable audit of the agent verdict (best-effort).  Recorded
        # for BOTH outcomes (drift / no-drift) so the audit reflects every
        # agent decision, not only the ones that fired a drift event.
        _record_agent_verdict(
            correlation_id=event.correlation_id,
            engine=self.name,
            tier=_AGENT_TIER,
            system_prompt=system,
            response=raw,
            verdict={
                "drift": drift,
                "confidence": confidence,
                "rationale": rationale,
            },
        )

        if not drift:
            return PluginAck(status="ack")

        drift_payload: dict[str, object] = {
            "lcs_ratio": ratio,
            "threshold": _DRIFT_THRESHOLD,
            "anchor_intent": anchor.intent,
            "current_prompt": prompt,
            "hmm_state": hmm_step.state,
            "hmm_posterior": dict(hmm_step.posterior),
            "hmm_observation": hmm_step.observation,
            "ema_posterior": ema_posterior,
            # Agent-path verdict provenance (audit §8).
            "verdict_source": "agent",
            "agent_confidence": confidence,
            "agent_rationale": rationale,
        }

        return PluginAck(
            status="ack",
            derived_events=[
                EnchantedEvent(
                    id=new_event_id(),
                    correlation_id=event.correlation_id,
                    session_id=event.session_id,
                    phase=event.phase,
                    topic="intent-anchor.drift.detected",
                    source=self.name,
                    budget_tier=event.budget_tier,
                    ts=_now_ms(),
                    payload=drift_payload,
                )
            ],
        )

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_anchor(self, event: EnchantedEvent) -> PluginAck:
        """Capture the first prompt as the session anchor."""
        store = self._get_or_create(event.session_id)

        if store.has_anchor:
            # Anchor is immutable once set.
            return PluginAck(status="ack")

        prompt = event.payload.get("user_prompt", "")
        if not isinstance(prompt, str):
            prompt = str(prompt)

        anchor = store.set_anchor(intent=prompt, ts_ms=event.ts)

        return PluginAck(
            status="ack",
            derived_events=[
                EnchantedEvent(
                    id=new_event_id(),
                    correlation_id=event.correlation_id,
                    session_id=event.session_id,
                    phase=event.phase,
                    topic="intent-anchor.anchor.set",
                    source=self.name,
                    budget_tier=event.budget_tier,
                    ts=_now_ms(),
                    payload={
                        "intent": anchor.intent,
                        "token_count": len(anchor.tokens),
                    },
                )
            ],
        )

    def _handle_post_session(self, event: EnchantedEvent) -> PluginAck:
        """Compare current prompt against anchor; emit drift event if ratio < threshold."""
        store = self._get_or_create(event.session_id)

        if not store.has_anchor:
            # No anchor set — nothing to compare against.
            return PluginAck(status="ack")

        prompt = event.payload.get("user_prompt", "")
        if not isinstance(prompt, str):
            prompt = str(prompt)

        ratio, hmm_step, ema_posterior = store.record_observation(prompt)

        if ratio >= _DRIFT_THRESHOLD:
            return PluginAck(status="ack")

        # Drift detected — build event payload
        anchor = store.anchor
        assert anchor is not None  # guaranteed by has_anchor check above

        drift_payload: dict[str, object] = {
            "lcs_ratio": ratio,
            "threshold": _DRIFT_THRESHOLD,
            "anchor_intent": anchor.intent,
            "current_prompt": prompt,
            "hmm_state": hmm_step.state,
            "hmm_posterior": dict(hmm_step.posterior),
            "hmm_observation": hmm_step.observation,
            "ema_posterior": ema_posterior,
        }

        return PluginAck(
            status="ack",
            derived_events=[
                EnchantedEvent(
                    id=new_event_id(),
                    correlation_id=event.correlation_id,
                    session_id=event.session_id,
                    phase=event.phase,
                    topic="intent-anchor.drift.detected",
                    source=self.name,
                    budget_tier=event.budget_tier,
                    ts=_now_ms(),
                    payload=drift_payload,
                )
            ],
        )

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        try:
            if event.phase == "anchor":
                return self._handle_anchor(event)
            if event.phase == "post-session":
                # Agent-backed path is opt-in (env flag + a usable seam) and
                # default-OFF.  When inactive, the deterministic LCS+HMM path
                # below runs UNCHANGED.
                if self._agent_path_active(ctx):
                    # F3 — resolve the operator overlay end-to-end: prefer the
                    # one the orchestrator carried on ctx; fall back to the
                    # constructor seam (used by direct-construction tests).
                    overlay = getattr(ctx, "prompt_overlay", None)
                    if overlay is None:
                        overlay = self._prompt_overlay
                    return await self._handle_post_session_agent(
                        event, prompt_overlay=overlay, ctx=ctx
                    )
                return self._handle_post_session(event)
            return PluginAck(status="ack")
        except Exception as exc:  # noqa: BLE001
            # Fail-open: advisory plugin must not block the orchestrator.
            return PluginAck(
                status="ack",
                degraded=True,
                reason=f"intent-anchor error: {exc}",
            )


# Module-level default instance (mirrors djinnAdapter export in TS)
adapter = IntentAnchor()
