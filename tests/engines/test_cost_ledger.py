"""Tests for the cost-ledger engine.

Eight mandatory cases covering the full PluginAdapter path and both token-key
payload shapes.
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
from robit.engines.cost_ledger import CostLedger, CostLedgerStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(**kwargs) -> CostLedger:
    """Return a fresh CostLedger instance with its own isolated store."""
    return CostLedger(**kwargs)


def _post_response_event(ctx: RequestContext, payload: dict):
    """Build a post-response event carrying the given payload."""
    return build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="post-response",
        topic="sampling.completed",
        source="test-plugin",
        budget_tier=ctx.budget_tier,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Test 1 — Fresh ledger has zero totals
# ---------------------------------------------------------------------------


def test_fresh_ledger_has_zero_totals():
    store = CostLedgerStore()
    assert store.total("session-1") == 0
    assert store.vendor_total("session-1", "anthropic") == 0


# ---------------------------------------------------------------------------
# Test 2 — One observation increments totals correctly
# ---------------------------------------------------------------------------


def test_one_observation_increments_correctly():
    store = CostLedgerStore()
    store.record(
        session_id="s1",
        correlation_id="c1",
        plugin="p",
        model="claude-sonnet",
        vendor="anthropic",
        input_tokens=100,
        output_tokens=50,
    )
    assert store.total("s1") == 150
    assert store.vendor_total("s1", "anthropic") == 150


# ---------------------------------------------------------------------------
# Test 3 — Multiple sessions are isolated
# ---------------------------------------------------------------------------


def test_multiple_sessions_are_isolated():
    store = CostLedgerStore()
    store.record(
        session_id="alpha",
        correlation_id="c1",
        plugin="p",
        model="m",
        vendor="v",
        input_tokens=200,
        output_tokens=100,
    )
    store.record(
        session_id="beta",
        correlation_id="c2",
        plugin="p",
        model="m",
        vendor="v",
        input_tokens=10,
        output_tokens=5,
    )
    assert store.total("alpha") == 300
    assert store.total("beta") == 15
    # A third session that received nothing
    assert store.total("gamma") == 0


# ---------------------------------------------------------------------------
# Test 4 — Threshold crossing emits a derived event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_threshold_crossing_emits_derived_event():
    """
    Register a vendor budget of 1000 tokens.
    After spending 701 tokens the remaining fraction drops below 0.7 (HIGH→MED
    boundary), which should produce a cost-ledger.threshold.crossed event.
    """
    engine = _make_engine()
    engine.set_budget("anthropic", 1_000)

    ctx = create_request_context()
    bus = InProcessBus()
    registry = {engine.name: engine}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(c: RequestContext) -> str:
        return "ok"

    # Spend 600 tokens in one shot (60% used → 40% remaining).
    # 40% is below the HIGH waypoint (0.7) but above MED (0.3) → tier is MED.
    event = _post_response_event(
        ctx,
        {"vendor": "anthropic", "input_tokens": 400, "output_tokens": 200},
    )
    await bus.publish(event.topic, event)
    result = await orch.run(ctx, dispatch)
    assert result == "ok"

    crossed = [
        e
        for e in bus.tap(ctx.correlation_id)
        if e.topic == "cost-ledger.threshold.crossed"
    ]
    assert len(crossed) == 1, "Expected exactly one threshold-crossed event"
    p = crossed[0].payload
    assert p["vendor"] == "anthropic"
    assert p["old_tier"] == "HIGH"
    assert p["new_tier"] == "MED"


# ---------------------------------------------------------------------------
# Test 5 — Threshold NOT crossed → no derived crossing event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_threshold_crossing_produces_no_derived_event():
    """
    After spending only 100 / 1000 tokens (10%) the tier stays HIGH → no
    cost-ledger.threshold.crossed event should appear.
    """
    engine = _make_engine()
    engine.set_budget("anthropic", 1_000)

    ctx = create_request_context()
    bus = InProcessBus()
    registry = {engine.name: engine}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(c: RequestContext) -> str:
        return "ok"

    event = _post_response_event(
        ctx,
        {"vendor": "anthropic", "input_tokens": 60, "output_tokens": 40},
    )
    await bus.publish(event.topic, event)
    await orch.run(ctx, dispatch)

    crossed = [
        e
        for e in bus.tap(ctx.correlation_id)
        if e.topic == "cost-ledger.threshold.crossed"
    ]
    assert len(crossed) == 0, "No threshold event expected when tier is unchanged"


# ---------------------------------------------------------------------------
# Test 6 — Vendor-specific totals tracked independently across two vendors
# ---------------------------------------------------------------------------


def test_vendor_specific_totals_are_tracked():
    store = CostLedgerStore()
    store.record(
        session_id="s1",
        correlation_id="c1",
        plugin="p",
        model="gpt-4o",
        vendor="openai",
        input_tokens=300,
        output_tokens=200,
    )
    store.record(
        session_id="s1",
        correlation_id="c2",
        plugin="p",
        model="claude-sonnet",
        vendor="anthropic",
        input_tokens=50,
        output_tokens=25,
    )
    assert store.vendor_total("s1", "openai") == 500
    assert store.vendor_total("s1", "anthropic") == 75
    # Total across all vendors
    assert store.total("s1") == 575


# ---------------------------------------------------------------------------
# Test 7 — Both payload shapes: canonical tokens.input/output AND legacy flat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dual_token_key_shapes_both_supported():
    """
    The canonical shape nests tokens under a "tokens" sub-dict;
    the legacy shape provides input_tokens / output_tokens at the top level.
    Both must be correctly parsed by the adapter.
    """
    engine_canonical = _make_engine()
    engine_legacy = _make_engine()

    # --- Canonical shape ---
    ctx_c = create_request_context()
    bus_c = InProcessBus()
    orch_c = Orchestrator(OrchestratorConfig(registry={engine_canonical.name: engine_canonical}, bus=bus_c))

    async def dispatch_c(c: RequestContext) -> str:
        return "ok"

    event_c = _post_response_event(
        ctx_c,
        {"vendor": "anthropic", "tokens": {"input": 120, "output": 80}},
    )
    await bus_c.publish(event_c.topic, event_c)
    await orch_c.run(ctx_c, dispatch_c)

    entries_c = engine_canonical.store.entries()
    assert len(entries_c) == 1
    assert entries_c[0].input_tokens == 120
    assert entries_c[0].output_tokens == 80

    # --- Legacy flat shape ---
    ctx_l = create_request_context()
    bus_l = InProcessBus()
    orch_l = Orchestrator(OrchestratorConfig(registry={engine_legacy.name: engine_legacy}, bus=bus_l))

    async def dispatch_l(c: RequestContext) -> str:
        return "ok"

    event_l = _post_response_event(
        ctx_l,
        {"vendor": "openai", "input_tokens": 200, "output_tokens": 100},
    )
    await bus_l.publish(event_l.topic, event_l)
    await orch_l.run(ctx_l, dispatch_l)

    entries_l = engine_legacy.store.entries()
    assert len(entries_l) == 1
    assert entries_l[0].input_tokens == 200
    assert entries_l[0].output_tokens == 100


# ---------------------------------------------------------------------------
# Test 8 — End-to-end: event → ledger update → threshold → cost-ledger.threshold.crossed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_event_feeds_ledger_and_emits_threshold_crossed():
    """
    Full end-to-end path:
      1. Register a 500-token budget for vendor "openai".
      2. Fire a post-response event with 400 tokens (80% spend).
      3. Ledger reflects the tokens.
      4. The remaining 20% is below the 0.3 MED threshold → MED tier → LOW tier...
         Actually at 80% spend (20% remaining) the tier is LOW (< 0.3 remaining),
         crossing from HIGH → LOW in a single shot.
      5. A cost-ledger.threshold.crossed event must appear on the bus.
    """
    engine = _make_engine()
    engine.set_budget("openai", 500)

    ctx = create_request_context()
    bus = InProcessBus()
    registry = {engine.name: engine}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(c: RequestContext) -> str:
        return "dispatched"

    # Fire 400 tokens against a 500-token budget → 20% remaining → LOW tier
    event = _post_response_event(
        ctx,
        {
            "vendor": "openai",
            "model": "gpt-4o",
            "input_tokens": 300,
            "output_tokens": 100,
        },
    )
    await bus.publish(event.topic, event)
    result = await orch.run(ctx, dispatch)
    assert result == "dispatched"

    # Ledger must reflect the tokens.
    assert engine.store.total(ctx.session_id) == 400
    assert engine.store.vendor_total(ctx.session_id, "openai") == 400

    # Remaining should be 100.
    assert engine.store.remaining("openai") == 100

    # A threshold-crossed event must have been emitted.
    crossed = [
        e
        for e in bus.tap(ctx.correlation_id)
        if e.topic == "cost-ledger.threshold.crossed"
    ]
    assert len(crossed) == 1, f"Expected 1 threshold event, got {len(crossed)}"
    p = crossed[0].payload
    assert p["vendor"] == "openai"
    # 20% remaining is below both 0.7 (HIGH) and 0.3 (MED); compute_tier_label
    # returns LOW (≥ 0.1).  The old tier was HIGH (cold start).
    assert p["new_tier"] == "LOW"
    assert p["old_tier"] == "HIGH"

    # cost-ledger.appended should also be present.
    appended = [
        e for e in bus.tap(ctx.correlation_id) if e.topic == "cost-ledger.appended"
    ]
    assert len(appended) == 1
    assert appended[0].payload["vendor"] == "openai"
    assert appended[0].payload["input_tokens"] == 300
    assert appended[0].payload["output_tokens"] == 100
