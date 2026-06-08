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


def _now_ms() -> int:
    return int(time.time() * 1000)


def _agent_enabled() -> bool:
    """True only when the operator explicitly opts in via the env flag."""
    return os.environ.get(_AGENT_ENV_FLAG, "").strip() in ("1", "true", "True", "yes")


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

    def _agent_path_active(self) -> bool:
        """The agent path runs only when the env flag is on AND a seam exists.

        Either condition false → deterministic path (the default).
        """
        return self._llm_call is not None and _agent_enabled()

    @staticmethod
    def _load_prompt(rel_path: str) -> str:
        """Load an engine-authored prompt .md from the engine directory."""
        return (_ENGINE_DIR / rel_path).read_text(encoding="utf-8")

    def build_drift_prompt(self, anchor_intent: str, current_prompt: str) -> tuple[str, str]:
        """Build the (system, user) prompt pair for the drift verdict.

        Precedence (audit §8): framework < engine-author prompt < operator
        overlay.  The engine-authored body (``prompts/drift.md``) is the system
        prompt; the operator's ``prompt_overlay`` is APPENDED after it (never
        replaces it).
        """
        system = self._load_prompt("prompts/drift.md")
        if self._prompt_overlay:
            system = f"{system}\n\n## Operator overlay\n\n{self._prompt_overlay}"
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

    async def _handle_post_session_agent(self, event: EnchantedEvent) -> PluginAck:
        """Agent-backed drift verdict via the injected llm_call seam.

        Mirrors the deterministic handler's event shape but sources the drift
        decision from a model.  Still advisory / fail-open: any error falls
        back to a clean ack so the orchestrator is never blocked.
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

        assert self._llm_call is not None  # guaranteed by _agent_path_active
        system, user = self.build_drift_prompt(anchor.intent, prompt)
        raw = await self._llm_call(system, user)
        drift, confidence, rationale = self._parse_verdict(raw)

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
                # Agent-backed path is opt-in (env flag + injected seam) and
                # default-OFF.  When inactive, the deterministic LCS+HMM path
                # below runs UNCHANGED.
                if self._agent_path_active():
                    return await self._handle_post_session_agent(event)
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
