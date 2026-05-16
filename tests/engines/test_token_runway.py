"""Tests for the token-runway engine (A1 drift detection + A2 runway forecast).

Six required tests plus two extras covering edge behaviour.
"""

from __future__ import annotations

import math
import time

import pytest

from robit.core.bus import build_event
from robit.core import create_request_context
from robit.engines.token_runway import TokenRunway, TokenRunwayStore
from robit.engines.token_runway.store import FORECAST_WINDOW, WINDOW_CAP


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> int:
    return int(time.time() * 1000)


def _store_with_observations(
    costs: list[int],
    tool_ids: list[str] | None = None,
    remaining_budget: int = 200_000,
) -> TokenRunwayStore:
    """Build a store pre-loaded with len(costs) observations."""
    store = TokenRunwayStore(remaining_budget=remaining_budget)
    for i, cost in enumerate(costs):
        tid = (tool_ids[i] if tool_ids else f"call-{i}")
        store.record_observation(
            input_tokens=cost,
            output_tokens=0,
            tool_call_id=tid,
            ts=_ts(),
        )
    return store


# ── Test 1: empty window returns no forecast ──────────────────────────────────

def test_empty_window_returns_no_forecast():
    """A fresh store with 0 observations must return None from compute_runway()."""
    store = TokenRunwayStore()
    assert store.compute_runway() is None


def test_single_observation_returns_no_forecast():
    """One observation is still cold-start — CI is undefined with n=1."""
    store = _store_with_observations([500])
    assert store.compute_runway() is None


# ── Test 2: stable observations produce a clean forecast ─────────────────────

def test_stable_observations_produce_forecast():
    """10 identical observations → deterministic point estimate = budget / cost."""
    remaining = 200_000
    cost = 1_000
    store = _store_with_observations([cost] * 10, remaining_budget=remaining)
    forecast = store.compute_runway()

    assert forecast is not None
    assert forecast.observation_count == 10
    assert math.isclose(forecast.mean_tokens_per_call, cost, rel_tol=1e-9)
    # σ == 0 when all observations are identical → CI collapses to point estimate.
    assert math.isclose(forecast.sigma, 0.0, abs_tol=1e-9)
    expected_point = remaining / cost  # 200.0
    assert math.isclose(forecast.point_estimate, expected_point, rel_tol=1e-9)
    # CI lower == CI upper == point_estimate when σ=0.
    assert math.isclose(forecast.ci_lower, expected_point, rel_tol=1e-9)
    assert math.isclose(forecast.ci_upper, expected_point, rel_tol=1e-9)


# ── Test 3: increasing token usage → shorter runway ───────────────────────────

def test_increasing_usage_gives_shorter_runway():
    """Higher mean cost → fewer remaining calls (smaller point_estimate)."""
    remaining = 200_000
    low_store = _store_with_observations([500] * 10, remaining_budget=remaining)
    high_store = _store_with_observations([5_000] * 10, remaining_budget=remaining)

    low_fc = low_store.compute_runway()
    high_fc = high_store.compute_runway()

    assert low_fc is not None
    assert high_fc is not None
    assert low_fc.point_estimate > high_fc.point_estimate


# ── Test 4: drift detection fires on regime change ────────────────────────────

def test_drift_read_loop_fires_on_regime_change():
    """3 consecutive identical tool_call_ids → detect_read_loop() == True."""
    store = TokenRunwayStore()
    # Two different calls first (not a loop yet).
    store.record_observation(100, 0, "call-A", _ts())
    store.record_observation(100, 0, "call-B", _ts())
    # Now 3 consecutive observations with the SAME id — the regime change.
    for _ in range(3):
        store.record_observation(100, 0, "stuck-id", _ts())

    assert store.detect_read_loop() is True
    assert store.drift_pattern() == "read-loop"


def test_drift_edit_revert_fires_on_abab_pattern():
    """ABAB pattern across last 4 observations → detect_edit_revert() == True."""
    store = TokenRunwayStore()
    for _ in range(2):  # two full ABAB cycles
        store.record_observation(100, 0, "edit-A", _ts())
        store.record_observation(100, 0, "edit-B", _ts())

    assert store.detect_edit_revert() is True
    assert store.drift_pattern() == "edit-revert"


# ── Test 5: drift detection does NOT fire on stable observations ──────────────

def test_drift_does_not_fire_on_stable_observations():
    """10 observations with distinct ids → neither drift pattern fires."""
    store = _store_with_observations([1_000] * 10)  # distinct ids call-0…call-9
    assert store.detect_read_loop() is False
    assert store.detect_edit_revert() is False
    assert store.drift_pattern() is None


def test_read_loop_takes_priority_over_edit_revert():
    """When read-loop fires, drift_pattern() returns 'read-loop', not 'edit-revert'.

    Sequence A B A B A B A (last 3 = B A B → not a loop; last 4 = A B A B → edit-revert).
    Then we add two more 'X X' to make the tail [..., A, B, X, X, X] where last 3 are X X X.
    Simpler: just build a sequence whose last 3 are identical (unambiguous read-loop).
    """
    store = TokenRunwayStore()
    # Build ABAB (edit-revert eligible), then append a third 'B' so last-3 = BBB → read-loop.
    for tid in ["A", "B", "A", "B", "B", "B"]:
        store.record_observation(100, 0, tid, _ts())
    # Last 3 are all "B" → read-loop fires.
    assert store.detect_read_loop() is True
    # drift_pattern() must return "read-loop", not "edit-revert".
    assert store.drift_pattern() == "read-loop"


# ── Test 6: end-to-end via adapter.on_phase ───────────────────────────────────

@pytest.mark.asyncio
async def test_end_to_end_post_response_feeds_observation():
    """An event through on_phase(post-response) is recorded in the store."""
    engine = TokenRunway(remaining_budget=200_000)
    ctx = create_request_context()

    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="post-response",
        topic="mcp.tool.result.received",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={
            "input_tokens": 300,
            "output_tokens": 150,
            "tool_call_id": "e2e-call-1",
        },
    )
    ack = await engine.on_phase(event, ctx)

    assert ack.status == "ack"
    assert engine.store.observation_count() == 1
    obs = engine.store.observations()[0]
    assert obs.input_tokens == 300
    assert obs.output_tokens == 150
    assert obs.tool_call_id == "e2e-call-1"


@pytest.mark.asyncio
async def test_end_to_end_pre_dispatch_emits_runway_event():
    """After enough post-response observations, pre-dispatch emits emu.runway.forecast."""
    engine = TokenRunway(remaining_budget=200_000)
    ctx = create_request_context()

    # Feed FORECAST_WINDOW + 1 observations so compute_runway() has >= 2 points.
    for i in range(FORECAST_WINDOW):
        obs_event = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="post-response",
            topic="mcp.tool.result.received",
            source="test",
            budget_tier=ctx.budget_tier,
            payload={
                "input_tokens": 1_000,
                "output_tokens": 0,
                "tool_call_id": f"obs-{i}",
            },
        )
        await engine.on_phase(obs_event, ctx)

    # Now fire pre-dispatch.
    pre_event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="pre-dispatch",
        topic="mcp.tool.call.requested",
        source="test",
        budget_tier=ctx.budget_tier,
    )
    ack = await engine.on_phase(pre_event, ctx)

    assert ack.status == "ack"
    assert len(ack.derived_events) == 1
    runway_event = ack.derived_events[0]
    assert runway_event.topic == "token-runway.runway.forecast"
    payload = runway_event.payload
    # 200 000 budget / 1 000 per call = 200 calls remaining.
    assert math.isclose(float(payload["point_estimate"]), 200.0, rel_tol=1e-6)  # type: ignore[arg-type]
    assert payload["observation_count"] == FORECAST_WINDOW


@pytest.mark.asyncio
async def test_pre_dispatch_cold_start_emits_no_runway_event():
    """pre-dispatch with < 2 observations returns ack with no derived events."""
    engine = TokenRunway()
    ctx = create_request_context()

    pre_event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="pre-dispatch",
        topic="mcp.tool.call.requested",
        source="test",
        budget_tier=ctx.budget_tier,
    )
    ack = await engine.on_phase(pre_event, ctx)
    assert ack.status == "ack"
    assert ack.derived_events == []


@pytest.mark.asyncio
async def test_post_response_emits_drift_event_on_read_loop():
    """3rd consecutive same-id post-response observation fires emu.drift.pattern."""
    engine = TokenRunway()
    ctx = create_request_context()

    for _ in range(3):
        obs_event = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="post-response",
            topic="mcp.tool.result.received",
            source="test",
            budget_tier=ctx.budget_tier,
            payload={
                "input_tokens": 200,
                "output_tokens": 50,
                "tool_call_id": "loop-id",
            },
        )
        ack = await engine.on_phase(obs_event, ctx)

    # Third call should have triggered the drift signal.
    assert len(ack.derived_events) == 1
    drift_ev = ack.derived_events[0]
    assert drift_ev.topic == "token-runway.drift.pattern"
    assert drift_ev.payload["pattern_name"] == "read-loop"


# ── Test: window cap ──────────────────────────────────────────────────────────

def test_window_cap_evicts_oldest():
    """Loading WINDOW_CAP + 10 observations keeps exactly WINDOW_CAP entries."""
    store = _store_with_observations([100] * (WINDOW_CAP + 10))
    assert store.observation_count() == WINDOW_CAP


# ── Test: CI properties ───────────────────────────────────────────────────────

def test_ci_lower_never_negative():
    """ci_lower must be clamped to >= 0 even when σ >> mean."""
    # Very small budget, very high variance → formula would go negative.
    store = TokenRunwayStore(remaining_budget=10)
    costs = [1, 10_000, 1, 10_000, 1, 10_000, 1, 10_000, 1, 10_000]
    for i, c in enumerate(costs):
        store.record_observation(c, 0, f"call-{i}", _ts())
    forecast = store.compute_runway()
    assert forecast is not None
    assert forecast.ci_lower >= 0.0


# ── Test: per-instance isolation ─────────────────────────────────────────────

def test_store_is_per_instance():
    """Two TokenRunway instances must not share observation state."""
    engine_a = TokenRunway(remaining_budget=100_000)
    engine_b = TokenRunway(remaining_budget=100_000)

    engine_a.store.record_observation(500, 0, "call-a", _ts())
    assert engine_a.store.observation_count() == 1
    assert engine_b.store.observation_count() == 0
