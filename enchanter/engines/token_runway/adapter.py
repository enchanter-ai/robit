"""TokenRunway engine — port of emu.adapter.ts (phase_1.emu).

Token economy monitoring with two algorithms:
  A1 Markov Drift Detection  — fires at post-response; emits token-runway.drift.pattern
  A2 Linear Runway Forecast  — fires at pre-dispatch; emits token-runway.runway.forecast

Required: False (advisory, fail-open). Budget tier: med-or-higher.
Phases: post-response (record + drift check), pre-dispatch (runway check).
"""

from __future__ import annotations

import time

from enchanter.core import EnchantedEvent, PluginAck, RequestContext
from enchanter.core.plugin import PluginTopics
from enchanter.core.bus import new_event_id

from .store import TokenRunwayStore


def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_drift_event(base: EnchantedEvent, pattern_name: str) -> EnchantedEvent:
    """Build the emu.drift.pattern derived event (faithful to TS makeDriftEvent)."""
    return EnchantedEvent(
        id=f"{base.correlation_id}::token-runway-drift-{pattern_name}",
        correlation_id=base.correlation_id,
        session_id=base.session_id,
        phase=base.phase,
        topic="token-runway.drift.pattern",
        source="token-runway",
        budget_tier=base.budget_tier,
        ts=_now_ms(),
        payload={"pattern_name": pattern_name},
    )


def _make_runway_event(base: EnchantedEvent, store: TokenRunwayStore) -> EnchantedEvent | None:
    """Build the emu.runway.forecast derived event, or None when cold-start."""
    forecast = store.compute_runway()
    if forecast is None:
        return None
    return EnchantedEvent(
        id=f"{base.correlation_id}::emu-runway",
        correlation_id=base.correlation_id,
        session_id=base.session_id,
        phase=base.phase,
        topic="token-runway.runway.forecast",
        source="token-runway",
        budget_tier=base.budget_tier,
        ts=_now_ms(),
        payload={
            "point_estimate": forecast.point_estimate,
            "ci_lower": forecast.ci_lower,
            "ci_upper": forecast.ci_upper,
            "mean_tokens_per_call": forecast.mean_tokens_per_call,
            "sigma": forecast.sigma,
            "observation_count": forecast.observation_count,
        },
    )


class TokenRunway:
    """Advisory emu engine at post-response + pre-dispatch.

    State is per-instance (TokenRunwayStore). Construct one adapter per
    session or per test — never share across sessions.
    """

    name = "token-runway"
    phases = ("post-response", "pre-dispatch")
    required = False  # advisory — fail-open per hooks.md
    topics = PluginTopics(
        subscribes=(
            "mcp.tool.call.requested",
            "mcp.tool.result.received",
        ),
        emits=(
            "token-runway.runway.forecast",
            "token-runway.compression.applied",
            "token-runway.drift.pattern",
        ),
    )
    budget_tier = "med-or-higher"

    def __init__(self, remaining_budget: int | None = None) -> None:
        from .store import DEFAULT_REMAINING_BUDGET
        self._store = TokenRunwayStore(
            remaining_budget=remaining_budget
            if remaining_budget is not None
            else DEFAULT_REMAINING_BUDGET
        )

    # ── Public accessor (for tests / introspection) ───────────────────────────

    @property
    def store(self) -> TokenRunwayStore:
        return self._store

    # ── Phase handlers ────────────────────────────────────────────────────────

    def _handle_post_response(self, event: EnchantedEvent) -> PluginAck:
        """Record token usage, then check for drift patterns.

        Token keys: canonical `tokens.input` / `tokens.output` first;
        legacy `input_tokens` / `output_tokens` second — same as TS source.
        """
        payload = dict(event.payload)
        tokens = payload.get("tokens")
        if isinstance(tokens, dict):
            input_tokens = int(tokens.get("input") or tokens.get("input_tokens") or 0)
            output_tokens = int(tokens.get("output") or tokens.get("output_tokens") or 0)
        else:
            input_tokens = int(payload.get("input_tokens") or 0)
            output_tokens = int(payload.get("output_tokens") or 0)

        tool_call_id: str = (
            str(payload.get("tool_call_id")) if payload.get("tool_call_id") else event.correlation_id
        )

        self._store.record_observation(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_call_id=tool_call_id,
            ts=event.ts,
        )

        pattern = self._store.drift_pattern()
        if pattern is not None:
            return PluginAck(
                status="ack",
                derived_events=[_make_drift_event(event, pattern)],
            )
        return PluginAck(status="ack")

    def _handle_pre_dispatch(self, event: EnchantedEvent) -> PluginAck:
        """Emit a runway forecast derived event when enough data exists."""
        runway_event = _make_runway_event(event, self._store)
        if runway_event is None:
            return PluginAck(status="ack")  # cold start — nothing to emit
        return PluginAck(status="ack", derived_events=[runway_event])

    # ── PluginAdapter protocol ────────────────────────────────────────────────

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        try:
            if event.phase == "post-response":
                return self._handle_post_response(event)
            if event.phase == "pre-dispatch":
                return self._handle_pre_dispatch(event)
            return PluginAck(status="ack")
        except Exception:
            # Fail-open per hooks.md and plugin-contract required:False.
            return PluginAck(status="ack", degraded=True, reason="token-runway-internal-error")


# Convenience module-level singleton for simple wiring (mirrors TS emuAdapter export).
adapter = TokenRunway()
