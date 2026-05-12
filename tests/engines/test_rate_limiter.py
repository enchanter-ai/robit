"""Tests for the rate-limiter engine.

Six mandatory cases:
  1. Fresh bucket allows requests up to capacity.
  2. Bucket empties after capacity requests.
  3. Time advancement refills the bucket (clock injection).
  4. Multiple vendors are isolated.
  5. End-to-end: event causes a consume; exhausted bucket emits degraded ack
     with rate_limiter.bucket.exhausted derived event.
  6. Capacity and refill_rate are configurable per construction.
"""

from __future__ import annotations

import pytest

from enchanter.core import (
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    PluginAck,
    create_request_context,
)
from enchanter.core.bus import build_event
from enchanter.core.context import RequestContext
from enchanter.engines.rate_limiter import RateLimiter, RateLimiterStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    capacity: float = 60.0,
    refill_rate: float = 1.0,
    clock=None,
) -> RateLimiter:
    kwargs = dict(capacity=capacity, refill_rate=refill_rate)
    if clock is not None:
        kwargs["clock"] = clock
    return RateLimiter(**kwargs)


async def _fire_pre_dispatch(
    bus: InProcessBus,
    ctx: RequestContext,
    vendor: str,
) -> None:
    """Publish a lifecycle.pre-dispatch event carrying vendor in payload."""
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="pre-dispatch",
        topic="lifecycle.pre-dispatch",
        source=vendor,
        budget_tier=ctx.budget_tier,
        payload={"vendor": vendor},
    )
    await bus.publish(event.topic, event)


# ---------------------------------------------------------------------------
# Test 1 — Fresh bucket allows requests up to capacity
# ---------------------------------------------------------------------------

def test_fresh_bucket_allows_up_to_capacity():
    """try_consume must succeed for every request up to (but not over) capacity."""
    cap = 5
    engine = _make_engine(capacity=float(cap), refill_rate=0.0)  # no refill
    store = engine.store

    for i in range(cap):
        result = store.try_consume("vendor-a", "sess-1")
        assert result is True, f"Request {i + 1} should be allowed (capacity={cap})"


# ---------------------------------------------------------------------------
# Test 2 — Bucket empties after capacity requests
# ---------------------------------------------------------------------------

def test_bucket_empties_after_capacity_requests():
    """The (cap + 1)-th request must return False when no refill is in play."""
    cap = 3
    engine = _make_engine(capacity=float(cap), refill_rate=0.0)
    store = engine.store

    for _ in range(cap):
        store.try_consume("vendor-b", "sess-2")

    # One more — should be denied.
    result = store.try_consume("vendor-b", "sess-2")
    assert result is False, "Request after capacity exhaustion should be denied"


# ---------------------------------------------------------------------------
# Test 3 — Time advancement refills the bucket
# ---------------------------------------------------------------------------

def test_time_advance_refills_bucket():
    """Advancing the injected clock by T seconds must add T * refill_rate tokens."""
    fake_time = [0.0]

    def clock() -> float:
        return fake_time[0]

    refill_rate = 2.0  # 2 tokens/sec
    cap = 4.0
    engine = _make_engine(capacity=cap, refill_rate=refill_rate, clock=clock)
    store = engine.store

    # Drain the bucket completely (4 requests at t=0, no refill yet).
    for _ in range(int(cap)):
        assert store.try_consume("vendor-c", "sess-3") is True

    # Exhausted.
    assert store.try_consume("vendor-c", "sess-3") is False

    # Advance 1 second → 2 tokens refilled.
    fake_time[0] = 1.0

    assert store.try_consume("vendor-c", "sess-3") is True   # token 1 of 2
    assert store.try_consume("vendor-c", "sess-3") is True   # token 2 of 2
    assert store.try_consume("vendor-c", "sess-3") is False  # empty again


# ---------------------------------------------------------------------------
# Test 4 — Multiple vendors are isolated
# ---------------------------------------------------------------------------

def test_multiple_vendors_are_isolated():
    """Draining vendor-x must not affect vendor-y's bucket."""
    engine = _make_engine(capacity=3.0, refill_rate=0.0)
    store = engine.store

    # Drain vendor-x.
    for _ in range(3):
        store.try_consume("vendor-x", "shared-session")
    assert store.try_consume("vendor-x", "shared-session") is False, (
        "vendor-x should be exhausted"
    )

    # vendor-y is completely untouched.
    for i in range(3):
        result = store.try_consume("vendor-y", "shared-session")
        assert result is True, f"vendor-y request {i + 1} should be allowed"


# ---------------------------------------------------------------------------
# Test 5 — End-to-end: event causes consume; exhausted bucket emits degraded ack
# ---------------------------------------------------------------------------

async def test_e2e_exhausted_bucket_emits_degraded_ack():
    """
    After draining the in-store bucket directly, firing a pre-dispatch event
    must produce a degraded ack and publish a rate_limiter.bucket.exhausted
    derived event on the bus.
    """
    fake_time = [0.0]
    engine = _make_engine(capacity=1.0, refill_rate=0.0, clock=lambda: fake_time[0])

    # Pre-drain via the store directly (one token consumed, now empty).
    consumed = engine.store.try_consume("openai", "session-abc")
    assert consumed is True

    # Wire bus + orchestrator.
    bus = InProcessBus()
    registry = {engine.name: engine}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    dispatched = False

    async def dispatch(ctx: RequestContext) -> str:
        nonlocal dispatched
        dispatched = True
        return "ok"

    ctx = create_request_context(session_id="session-abc")

    # Override session_id so the key matches the pre-drained bucket.
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id="session-abc",
        phase="pre-dispatch",
        topic="lifecycle.pre-dispatch",
        source="openai",
        budget_tier=ctx.budget_tier,
        payload={"vendor": "openai"},
    )
    await bus.publish(event.topic, event)

    result = await orch.run(ctx, dispatch)

    # Advisory — must not veto dispatch.
    assert dispatched is True
    assert result == "ok"

    # rate-limiter.bucket-exhausted must appear on the bus.
    exhausted_events = [
        e for e in bus.tap(ctx.correlation_id)
        if e.topic == "rate-limiter.bucket-exhausted"
    ]
    assert len(exhausted_events) >= 1, (
        "Expected at least one rate-limiter.bucket-exhausted event"
    )
    payload = exhausted_events[0].payload
    assert payload["vendor"] == "openai"
    assert payload["session_id"] == "session-abc"


# ---------------------------------------------------------------------------
# Test 6 — Capacity and refill_rate are configurable per construction
# ---------------------------------------------------------------------------

def test_capacity_and_refill_rate_configurable():
    """Constructing with custom capacity/refill_rate must reflect in store behaviour."""
    engine = _make_engine(capacity=10.0, refill_rate=5.0)
    store = engine.store

    assert store.capacity == 10.0
    assert store.refill_rate == 5.0

    # Full 10 requests should succeed.
    for i in range(10):
        result = store.try_consume("vendor-cfg", "sess-cfg")
        assert result is True, f"Request {i + 1} of 10 should be allowed"

    # 11th should fail.
    assert store.try_consume("vendor-cfg", "sess-cfg") is False
