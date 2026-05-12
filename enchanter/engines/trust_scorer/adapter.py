"""TrustScorer engine — port of the TS crow adapter (crow.adapter.ts).

Beta-Bernoulli per-(server_id, tool_name) trust scoring at trust-gate.
Advisory (fail-open, required=False).

Phase:      trust-gate
Required:   False  — advisory, never vetoes
Topics sub: mcp.tool.call.requested, lifecycle.trust-gate
Topics emit: trust-scorer.trust.scored, trust-scorer.review.ordered

Review thresholds (faithful to TS):
  REVIEW_MEAN_THRESHOLD     = 0.5   (posterior mean < this triggers review)
  REVIEW_MIN_OBSERVATIONS   = 3     (must have ≥ 3 observations)

On every trust-gate call:
  1. Read (server_id, tool_name) from payload.
  2. Emit trust-scorer.trust.scored derived event with posterior_mean / observation_count.
  3. If mean < 0.5 AND n >= 3 → also emit trust-scorer.review.ordered + ack degraded=True.
  4. If mean < 0.5 AND n < 3  → cold-start degraded=True on ack (no review event).
  5. Otherwise: ack clean.

Note: the TS adapter is a module-level singleton backed by a module-level Map.
Python port uses instance-level TrustStore so each TrustScorer() is fully
isolated — tests construct a fresh instance per test case.
"""

from __future__ import annotations

import time

from enchanter.core import EnchantedEvent, PluginAck, RequestContext
from enchanter.core.plugin import PluginTopics
from enchanter.core.bus import new_event_id

from .store import TrustStore


_REVIEW_MEAN_THRESHOLD = 0.5
_REVIEW_MIN_OBSERVATIONS = 3


def _now_ms() -> int:
    return int(time.time() * 1000)


def _extract_server_tool(event: EnchantedEvent) -> tuple[str, str]:
    """Extract (server_id, tool_name) from event payload; graceful fallback."""
    payload = event.payload or {}
    raw_tool = payload.get("tool")
    raw_server = payload.get("server_id") or event.source

    tool_name = raw_tool if isinstance(raw_tool, str) else str(raw_tool or "unknown")
    server_id = raw_server if isinstance(raw_server, str) else str(raw_server or "unknown")
    return server_id, tool_name


class TrustScorer:
    """Advisory trust-gate engine — Beta-Bernoulli posterior per (server_id, tool_name)."""

    name = "trust-scorer"
    phases = ("trust-gate",)
    required = False  # fail-open
    topics = PluginTopics(
        subscribes=(
            "mcp.tool.call.requested",
            "lifecycle.trust-gate",
        ),
        emits=("trust-scorer.trust.scored", "trust-scorer.review.ordered"),
    )
    budget_tier = "med-or-higher"

    def __init__(self) -> None:
        self._store = TrustStore()

    # ------------------------------------------------------------------
    # Public store accessors — used by tests and downstream consumers
    # ------------------------------------------------------------------

    @property
    def store(self) -> TrustStore:
        return self._store

    def record_success(self, server_id: str, tool_name: str) -> None:
        self._store.record_success((server_id, tool_name))

    def record_failure(self, server_id: str, tool_name: str) -> None:
        self._store.record_failure((server_id, tool_name))

    def score(self, server_id: str, tool_name: str) -> float:
        return self._store.score((server_id, tool_name))

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        if event.phase != "trust-gate":
            return PluginAck(status="ack")

        server_id, tool_name = _extract_server_tool(event)
        key = (server_id, tool_name)

        mean = self._store.score(key)
        n = self._store.observation_count(key)

        # --- Build the always-emitted crow.trust.scored event ----------
        ts_now = _now_ms()
        trust_scored = EnchantedEvent(
            id=new_event_id(),
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            phase=event.phase,
            topic="trust-scorer.trust.scored",
            source=self.name,
            budget_tier=event.budget_tier,
            ts=ts_now,
            payload={
                "server_id": server_id,
                "tool_name": tool_name,
                "posterior_mean": mean,
                "observation_count": n,
            },
        )

        # --- Review threshold check ------------------------------------
        if mean >= _REVIEW_MEAN_THRESHOLD or n < _REVIEW_MIN_OBSERVATIONS:
            # Clean ack or cold-start degraded (no review event)
            if mean < _REVIEW_MEAN_THRESHOLD and n < _REVIEW_MIN_OBSERVATIONS:
                return PluginAck(
                    status="ack",
                    degraded=True,
                    reason=(
                        f"trust-scorer: low mean {mean:.3f} but cold-start "
                        f"(n={n} < {_REVIEW_MIN_OBSERVATIONS})"
                    ),
                    derived_events=[trust_scored],
                )
            return PluginAck(status="ack", derived_events=[trust_scored])

        # --- Review triggered ------------------------------------------
        review_event = EnchantedEvent(
            id=new_event_id(),
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            phase=event.phase,
            topic="trust-scorer.review.ordered",
            source=self.name,
            budget_tier=event.budget_tier,
            ts=ts_now,
            payload={
                "server_id": server_id,
                "tool_name": tool_name,
                "trust_score": mean,
                "observation_count": n,
                "reason": (
                    f"posterior mean {mean:.3f} < {_REVIEW_MEAN_THRESHOLD} "
                    f"after {n} observations"
                ),
            },
        )

        return PluginAck(
            status="ack",
            degraded=True,
            reason=(
                f"trust-scorer.review.ordered: {server_id}.{tool_name} "
                f"trust={mean:.3f} n={n}"
            ),
            derived_events=[trust_scored, review_event],
        )


adapter = TrustScorer()
