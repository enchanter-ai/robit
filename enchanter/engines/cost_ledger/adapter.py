"""CostLedger engine — port of pech.adapter.ts (v0.3).

Per-request token ledger, vendor budget tracking, and tier boundary events.

Phase:      post-response
Required:   True  (always-tier; fail-closed per architecture-spec phase_5)
Topics sub: mcp.tool.result.received, sampling.completed
Topics emit:
  cost-ledger.appended         — every call; token counts acknowledged
  cost-ledger.threshold.crossed — vendor tier crossed a budget waypoint
  cost-ledger.vendor.exhausted  — vendor budget fully consumed (remaining = 0)

Token-key conventions supported (both payload shapes):
  canonical   payload["tokens"]["input"]  / payload["tokens"]["output"]
  legacy flat payload["input_tokens"]     / payload["output_tokens"]

Python divergence from TS:
  - Per-instance state (CostLedgerStore) instead of module-level singletons.
    Tests construct a fresh CostLedger() per case — no reset() required.
  - Derived event IDs use new_event_id() (UUID4) rather than the TS pattern
    of ``${correlation_id}::pech-${topic}`` — avoids ID collisions when the
    same vendor crosses multiple thresholds in rapid succession.
  - topic prefix is "cost-ledger" (Python kebab convention) rather than "pech"
    to match the enchanter-agent bus topic namespace.
"""

from __future__ import annotations

import time
from typing import Optional

from enchanter.core import EnchantedEvent, PluginAck, RequestContext
from enchanter.core.plugin import PluginTopics
from enchanter.core.bus import new_event_id

from .store import CostLedgerStore, DEFAULT_THRESHOLDS


def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_derived(
    base: EnchantedEvent,
    topic: str,
    payload: dict,
) -> EnchantedEvent:
    """Construct a derived event inheriting correlation/session/phase from *base*."""
    return EnchantedEvent(
        id=new_event_id(),
        correlation_id=base.correlation_id,
        session_id=base.session_id,
        phase=base.phase,
        topic=topic,
        source="cost-ledger",
        budget_tier=base.budget_tier,
        ts=_now_ms(),
        payload=payload,
    )


def _extract_tokens(payload: dict) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from an event payload.

    Supports two shapes:
      1. payload["tokens"]["input"]  / payload["tokens"]["output"]  (canonical)
      2. payload["input_tokens"]     / payload["output_tokens"]     (legacy flat)
    Returns (0, 0) when no token fields are present.
    """
    tokens = payload.get("tokens")
    if isinstance(tokens, dict):
        input_tokens = int(tokens.get("input") or tokens.get("input_tokens") or 0)
        output_tokens = int(tokens.get("output") or tokens.get("output_tokens") or 0)
    else:
        input_tokens = int(payload.get("input_tokens") or 0)
        output_tokens = int(payload.get("output_tokens") or 0)
    return input_tokens, output_tokens


class CostLedger:
    """Required post-response engine — per-request token ledger and tier-boundary detector.

    Construct one instance per session (or per test). The instance owns its own
    CostLedgerStore and is never shared across sessions.

    Args:
        thresholds: Three descending remaining-fraction waypoints for tier
                    classification. Default: [0.7, 0.3, 0.1].
        ledger_path: Optional path to a JSONL file for durable persistence.
                     Absent → pure in-memory (default for tests).
    """

    name = "cost-ledger"
    phases = ("post-response",)
    required = True  # fail-closed; always-tier per architecture-spec phase_5
    topics = PluginTopics(
        subscribes=(
            "mcp.tool.result.received",
            "sampling.completed",
        ),
        emits=(
            "cost-ledger.appended",
            "cost-ledger.threshold.crossed",
            "cost-ledger.vendor.exhausted",
        ),
    )
    budget_tier = "always"

    def __init__(
        self,
        thresholds: Optional[list[float]] = None,
        ledger_path: Optional[str] = None,
    ) -> None:
        self._store = CostLedgerStore(
            thresholds=thresholds if thresholds is not None else list(DEFAULT_THRESHOLDS),
            ledger_path=ledger_path,
        )

    # ------------------------------------------------------------------
    # Public store accessor (used by tests and downstream consumers)
    # ------------------------------------------------------------------

    @property
    def store(self) -> CostLedgerStore:
        return self._store

    def set_budget(self, vendor: str, limit_tokens: int) -> None:
        """Register a vendor token budget.  Convenience delegation to store."""
        self._store.set_budget(vendor, limit_tokens)

    # ------------------------------------------------------------------
    # Phase handler
    # ------------------------------------------------------------------

    def _handle_post_response(self, event: EnchantedEvent) -> PluginAck:
        payload = dict(event.payload)
        input_tokens, output_tokens = _extract_tokens(payload)

        plugin = str(payload.get("plugin") or event.source)
        model = str(payload.get("model") or "unknown")
        vendor = str(payload.get("vendor") or "unknown")
        tool_call_cost_raw = payload.get("tool_call_cost")
        tool_call_cost: Optional[float] = (
            float(tool_call_cost_raw)
            if isinstance(tool_call_cost_raw, (int, float))
            else None
        )

        store_error = self._store.record(
            session_id=event.session_id,
            correlation_id=event.correlation_id,
            plugin=plugin,
            model=model,
            vendor=vendor,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_call_cost=tool_call_cost,
        )

        # Always emit cost-ledger.appended.
        derived: list[EnchantedEvent] = [
            _make_derived(
                event,
                "cost-ledger.appended",
                {
                    "plugin": plugin,
                    "model": model,
                    "vendor": vendor,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
        ]

        # Tier boundary / exhaustion check.
        crossing = self._store.check_threshold_crossed(vendor)
        if crossing is not None:
            if crossing["new_tier"] == "EXHAUSTED":
                derived.append(
                    _make_derived(
                        event,
                        "cost-ledger.vendor.exhausted",
                        {"vendor": vendor, "remaining_pct": 0.0},
                    )
                )
            else:
                derived.append(
                    _make_derived(
                        event,
                        "cost-ledger.threshold.crossed",
                        {
                            "vendor": vendor,
                            "old_tier": crossing["old_tier"],
                            "new_tier": crossing["new_tier"],
                            "remaining_pct": crossing["remaining_pct"],
                        },
                    )
                )

        if store_error is not None:
            return PluginAck(
                status="ack",
                degraded=True,
                reason=f"cost-ledger: persist failed — {store_error}",
                derived_events=derived,
            )

        return PluginAck(status="ack", derived_events=derived)

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        if event.phase == "post-response":
            return self._handle_post_response(event)
        return PluginAck(status="ack")


# Convenience module-level singleton (mirrors TS pechAdapter export).
adapter = CostLedger()
