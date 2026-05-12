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

import time

from enchanter.core import EnchantedEvent, PluginAck, RequestContext
from enchanter.core.plugin import PluginTopics
from enchanter.core.bus import new_event_id

from .store import IntentAnchorStore


# ---------------------------------------------------------------------------
# Constants (mirroring TS)
# ---------------------------------------------------------------------------

_DRIFT_THRESHOLD = 0.3


def _now_ms() -> int:
    return int(time.time() * 1000)


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

    def __init__(self) -> None:
        # Per-session stores keyed by session_id
        self._sessions: dict[str, IntentAnchorStore] = {}

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
