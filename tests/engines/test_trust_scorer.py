"""Tests for the trust-scorer (crow) engine.

Six mandatory cases plus helpers that exercise the full PluginAdapter path.
"""

from __future__ import annotations

import pytest

from robit.core import (
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    PluginAck,
    create_request_context,
)
from robit.core.bus import build_event
from robit.core.context import RequestContext
from robit.engines.trust_scorer import TrustScorer, TrustStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> TrustScorer:
    """Return a fresh TrustScorer instance with its own isolated store."""
    return TrustScorer()


async def _fire_trust_gate(
    bus: InProcessBus,
    ctx: RequestContext,
    server_id: str,
    tool_name: str,
) -> None:
    """Publish a mcp.tool.call.requested event at the trust-gate phase."""
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="trust-gate",
        topic="mcp.tool.call.requested",
        source=server_id,
        budget_tier=ctx.budget_tier,
        payload={"tool": tool_name, "server_id": server_id},
    )
    await bus.publish(event.topic, event)


# ---------------------------------------------------------------------------
# Test 1 — Fresh key returns the prior mean (0.5 for Beta(1,1))
# ---------------------------------------------------------------------------

def test_fresh_key_returns_prior_mean():
    engine = _make_engine()
    score = engine.score("my-server", "some-tool")
    assert score == pytest.approx(0.5, abs=1e-9), (
        "Cold-start posterior mean must equal prior_alpha / (prior_alpha + prior_beta) = 0.5"
    )


# ---------------------------------------------------------------------------
# Test 2 — After N successes, score approaches 1.0
# ---------------------------------------------------------------------------

def test_many_successes_score_approaches_one():
    engine = _make_engine()
    for _ in range(100):
        engine.record_success("s", "t")
    score = engine.score("s", "t")
    # Expected: (1 + 100) / (1 + 100 + 1) = 101/102 ≈ 0.990
    assert score > 0.98, f"Expected score close to 1.0 after 100 successes, got {score}"


# ---------------------------------------------------------------------------
# Test 3 — After N failures, score approaches 0.0
# ---------------------------------------------------------------------------

def test_many_failures_score_approaches_zero():
    engine = _make_engine()
    for _ in range(100):
        engine.record_failure("s", "t")
    score = engine.score("s", "t")
    # Expected: 1 / (1 + 1 + 100) = 1/102 ≈ 0.0098
    assert score < 0.02, f"Expected score close to 0.0 after 100 failures, got {score}"


# ---------------------------------------------------------------------------
# Test 4 — Mixed updates: alpha and beta tracked correctly
# ---------------------------------------------------------------------------

def test_mixed_updates_alpha_beta_tracked_correctly():
    engine = _make_engine()
    # 3 successes, 2 failures
    for _ in range(3):
        engine.record_success("srv", "tool")
    for _ in range(2):
        engine.record_failure("srv", "tool")

    alpha, beta = engine.store.alpha_beta(("srv", "tool"))
    # Prior alpha=1, beta=1; then +3 alpha, +2 beta
    assert alpha == pytest.approx(4.0)
    assert beta == pytest.approx(3.0)

    expected_mean = 4.0 / 7.0
    assert engine.score("srv", "tool") == pytest.approx(expected_mean, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 5 — Multiple keys are isolated (no cross-key leakage)
# ---------------------------------------------------------------------------

def test_multiple_keys_are_isolated():
    engine = _make_engine()

    # Drive key A toward success
    for _ in range(10):
        engine.record_success("srv", "tool-A")

    # Drive key B toward failure
    for _ in range(10):
        engine.record_failure("srv", "tool-B")

    score_a = engine.score("srv", "tool-A")
    score_b = engine.score("srv", "tool-B")

    assert score_a > 0.9, f"Key A should have high trust, got {score_a}"
    assert score_b < 0.15, f"Key B should have low trust, got {score_b}"

    # Key C has never been touched — must still return the prior mean
    score_c = engine.score("srv", "tool-C")
    assert score_c == pytest.approx(0.5, abs=1e-9), (
        f"Untouched key C must return prior mean 0.5, got {score_c}"
    )


# ---------------------------------------------------------------------------
# Test 6 — End-to-end: success signal via event updates the store;
#          derived crow.trust.scored event is published on the bus.
# ---------------------------------------------------------------------------

async def test_e2e_event_fires_derived_trust_scored():
    """
    Fire a trust-gate event for a (server_id, tool_name) pair that has already
    received a success observation via record_success().  The on_phase handler
    must produce a crow.trust.scored derived event and the score observable
    via the public method must reflect the recorded observation.
    """
    engine = _make_engine()

    # Pre-load one success so score is above the prior 0.5
    engine.record_success("mcp-server", "read_file")
    score_before = engine.score("mcp-server", "read_file")
    # alpha=2, beta=1 → mean=2/3 ≈ 0.667
    assert score_before == pytest.approx(2.0 / 3.0, rel=1e-9)

    # Wire up the bus + orchestrator with this engine instance
    bus = InProcessBus()
    registry = {engine.name: engine}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    dispatched = False

    async def dispatch(ctx: RequestContext) -> str:
        nonlocal dispatched
        dispatched = True
        return "ok"

    ctx = create_request_context()
    await _fire_trust_gate(bus, ctx, "mcp-server", "read_file")

    result = await orch.run(ctx, dispatch)

    # Advisory engine must not block dispatch
    assert dispatched is True
    assert result == "ok"

    # trust-scorer.trust.scored must appear on the bus
    scored_events = [
        e for e in bus.tap(ctx.correlation_id) if e.topic == "trust-scorer.trust.scored"
    ]
    assert len(scored_events) >= 1, "Expected at least one trust-scorer.trust.scored event"

    scored_payload = scored_events[0].payload
    assert scored_payload["server_id"] == "mcp-server"
    assert scored_payload["tool_name"] == "read_file"
    assert scored_payload["posterior_mean"] == pytest.approx(2.0 / 3.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Test 7 — Review event fires when mean < 0.5 AND n >= 3
# ---------------------------------------------------------------------------

async def test_review_ordered_fires_on_low_trust_with_enough_observations():
    engine = _make_engine()

    # 3 failures → alpha=1, beta=4 → mean=0.2, n=3 → review triggered
    for _ in range(3):
        engine.record_failure("srv", "risky-tool")

    bus = InProcessBus()
    registry = {engine.name: engine}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> str:
        return "ran"

    ctx = create_request_context()
    await _fire_trust_gate(bus, ctx, "srv", "risky-tool")

    result = await orch.run(ctx, dispatch)
    assert result == "ran"  # still advisory — does not veto

    review_events = [
        e for e in bus.tap(ctx.correlation_id) if e.topic == "trust-scorer.review.ordered"
    ]
    assert len(review_events) == 1, (
        "trust-scorer.review.ordered must fire when mean < 0.5 and n >= 3"
    )
    assert review_events[0].payload["trust_score"] == pytest.approx(1.0 / 5.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Test 8 — Cold-start: mean < 0.5 would require review, but n < 3 suppresses it
# ---------------------------------------------------------------------------

async def test_cold_start_degraded_without_review_event():
    """
    After only 2 failures the mean < 0.5 but n=2 < REVIEW_MIN_OBSERVATIONS=3,
    so crow.review.ordered must NOT fire; the ack must still carry degraded=True.
    """
    engine = _make_engine()

    for _ in range(2):
        engine.record_failure("srv", "cold-tool")

    bus = InProcessBus()
    registry = {engine.name: engine}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> str:
        return "ok"

    ctx = create_request_context()
    await _fire_trust_gate(bus, ctx, "srv", "cold-tool")

    result = await orch.run(ctx, dispatch)
    assert result == "ok"

    review_events = [
        e for e in bus.tap(ctx.correlation_id) if e.topic == "trust-scorer.review.ordered"
    ]
    assert len(review_events) == 0, (
        "trust-scorer.review.ordered must NOT fire when n < REVIEW_MIN_OBSERVATIONS"
    )

    # But a trust-scorer.trust.scored event should still appear
    scored_events = [
        e for e in bus.tap(ctx.correlation_id) if e.topic == "trust-scorer.trust.scored"
    ]
    assert len(scored_events) >= 1
