"""RateLimiter engine — per-vendor token-bucket rate limiter.

Port of the pech rate-check / rate-shield advisory logic to the Python
enchanter-agent runtime.

Phase:    pre-dispatch
Required: False  — advisory, fail-open (mirroring pech rate-check.py)
Topics:
  subscribes: lifecycle.pre-dispatch, mcp.tool.call.requested
  emits:      rate-limiter.bucket-exhausted

On every pre-dispatch event:
  1. Extract vendor from event.payload["vendor"] (fallback: event.source).
  2. Call store.try_consume(vendor, session_id).
  3a. Allowed  → PluginAck(status="ack")
  3b. Exhausted → PluginAck(status="ack", degraded=True, reason=...,
                             derived_events=[rate_limiter.bucket.exhausted])

The engine is advisory (required=False, never vetoes) to match the
pech rate-check.py contract.  The TS rate-shield (blocking) is a separate
stricter enforcement layer; this port targets the advisory path.
"""

from __future__ import annotations

import time
from typing import Callable

from robit.core import EnchantedEvent, PluginAck, RequestContext
from robit.core.plugin import PluginTopics
from robit.core.bus import new_event_id

from .store import RateLimiterStore


def _now_ms() -> int:
    return int(time.time() * 1000)


def _extract_vendor(event: EnchantedEvent) -> str:
    """Extract vendor from payload or fall back to event.source."""
    payload = event.payload or {}
    vendor = payload.get("vendor")
    if isinstance(vendor, str) and vendor:
        return vendor
    return event.source or "unknown"


class RateLimiter:
    """Advisory pre-dispatch engine — token-bucket per (vendor, session_id).

    Args:
        capacity:    Maximum tokens per bucket. Default 60 (matches pech
                     rate-check.py DEFAULT_CONFIG).
        refill_rate: Tokens refilled per second. Default 1.0.
        clock:       Monotonic clock callable; injected for test isolation.
    """

    name = "rate-limiter"
    phases = ("pre-dispatch",)
    required = False  # advisory, fail-open
    topics = PluginTopics(
        subscribes=(
            "lifecycle.pre-dispatch",
            "mcp.tool.call.requested",
        ),
        emits=("rate-limiter.bucket-exhausted",),
    )
    budget_tier = "always"

    def __init__(
        self,
        capacity: float = 60.0,
        refill_rate: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = RateLimiterStore(
            capacity=capacity,
            refill_rate=refill_rate,
            clock=clock,
        )

    # ------------------------------------------------------------------
    # Public store accessor — used by tests
    # ------------------------------------------------------------------

    @property
    def store(self) -> RateLimiterStore:
        return self._store

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        if event.phase != "pre-dispatch":
            return PluginAck(status="ack")

        vendor = _extract_vendor(event)
        session_id = event.session_id

        allowed = self._store.try_consume(vendor, session_id)

        if allowed:
            return PluginAck(status="ack")

        # Bucket exhausted — emit advisory derived event, degrade ack.
        exhausted_event = EnchantedEvent(
            id=new_event_id(),
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            phase=event.phase,
            topic="rate-limiter.bucket-exhausted",
            source=self.name,
            budget_tier=event.budget_tier,
            ts=_now_ms(),
            payload={
                "vendor": vendor,
                "session_id": session_id,
                "capacity": self._store.capacity,
                "refill_rate": self._store.refill_rate,
            },
        )

        return PluginAck(
            status="ack",
            degraded=True,
            reason=(
                f"rate-limiter: bucket exhausted for vendor={vendor!r} "
                f"session={session_id!r}; "
                f"capacity={self._store.capacity}, "
                f"refill={self._store.refill_rate}/s"
            ),
            derived_events=[exhausted_event],
        )


adapter = RateLimiter()
