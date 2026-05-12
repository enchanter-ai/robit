"""Tests for the intent-anchor engine (LCS + HMM + EMA + adapter).

Covers:
  LCS: identical, empty, partial overlap, ratio bounds
  HMM: deterministic small example, uniform emissions
  EMA: cold-start prior, convergence toward new observations
  Store: drift signal increases when LCS ratio drops
  Adapter end-to-end: anchor phase emits anchor.set;
                       post-session below threshold emits drift.detected
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
from enchanter.engines.intent_anchor import IntentAnchor, lcs_length, lcs_ratio
from enchanter.engines.intent_anchor.hmm import (
    HMM,
    DEFAULT_EMISSIONS,
    DEFAULT_PRIOR,
    DEFAULT_TRANSITIONS,
    ObservationBucket,
)
from enchanter.engines.intent_anchor.store import IntentAnchorStore, tokenize


# ===========================================================================
# LCS tests
# ===========================================================================


class TestLcsLength:
    def test_identical_inputs_length_equals_input_length(self):
        a = ["write", "unit", "tests"]
        assert lcs_length(a, a) == len(a)

    def test_empty_both_returns_zero(self):
        assert lcs_length([], []) == 0

    def test_empty_a_returns_zero(self):
        assert lcs_length([], ["hello"]) == 0

    def test_empty_b_returns_zero(self):
        assert lcs_length(["hello"], []) == 0

    def test_partial_overlap_correct_length(self):
        # a = [1, 3, 4, 5, 6, 7, 8], b = [1, 2, 4, 5, 6, 7, 9]
        # LCS = [1, 4, 5, 6, 7] → length 5
        a = ["1", "3", "4", "5", "6", "7", "8"]
        b = ["1", "2", "4", "5", "6", "7", "9"]
        assert lcs_length(a, b) == 5

    def test_no_overlap_returns_zero(self):
        assert lcs_length(["alpha", "beta"], ["gamma", "delta"]) == 0

    def test_single_common_element(self):
        assert lcs_length(["x", "y", "z"], ["a", "z", "b"]) == 1


class TestLcsRatio:
    def test_identical_returns_one(self):
        a = ["refactor", "module"]
        assert lcs_ratio(a, a) == pytest.approx(1.0)

    def test_both_empty_returns_one(self):
        assert lcs_ratio([], []) == pytest.approx(1.0)

    def test_one_empty_returns_zero(self):
        assert lcs_ratio([], ["hello"]) == pytest.approx(0.0)
        assert lcs_ratio(["hello"], []) == pytest.approx(0.0)

    def test_ratio_between_zero_and_one(self):
        a = ["fix", "bug", "parser"]
        b = ["fix", "regression", "parser"]
        r = lcs_ratio(a, b)
        assert 0.0 <= r <= 1.0

    def test_partial_overlap_ratio_correct(self):
        # ["1","3","4","5","6","7","8"] vs ["1","2","4","5","6","7","9"]
        # LCS len = 5, max len = 7 → ratio = 5/7
        a = ["1", "3", "4", "5", "6", "7", "8"]
        b = ["1", "2", "4", "5", "6", "7", "9"]
        assert lcs_ratio(a, b) == pytest.approx(5 / 7, rel=1e-9)

    def test_ratio_scaled_by_longer_list(self):
        # a = ["x"], b = ["x", "y", "z"] → LCS=1, max=3 → ratio=1/3
        a = ["x"]
        b = ["x", "y", "z"]
        assert lcs_ratio(a, b) == pytest.approx(1 / 3, rel=1e-9)


# ===========================================================================
# HMM tests
# ===========================================================================


class TestHmmViterbi:
    def test_all_high_observations_stays_on_task(self):
        """With all 'high' observations ON_TASK should dominate throughout."""
        hmm = HMM()
        path = hmm.decode(["high", "high", "high", "high"])
        assert all(s == "ON_TASK" for s in path), f"Expected all ON_TASK, got {path}"

    def test_all_low_observations_converges_to_lost(self):
        """With enough 'low' observations the path should end in LOST."""
        hmm = HMM()
        path = hmm.decode(["low"] * 10)
        # The last state(s) should be LOST
        assert path[-1] == "LOST", f"Expected final state LOST, got {path[-1]}"

    def test_deterministic_small_example(self):
        """
        Hand-verified 2-step example.

        At t=0, obs='high':
          delta[ON_TASK]   = log(0.90) + log(0.75)
          delta[SIDEQUEST] = log(0.08) + log(0.15)
          delta[LOST]      = log(0.02) + log(0.02)

        At t=1, obs='low':
          For each j, best_i = argmax(delta[i] + log(A[i][j]))
          Then delta_new[j] = best + log(B[j][low])

          ON_TASK (j=0, low=0.05):
            from ON_TASK:    delta[0] + log(0.85) ≈ -0.3367 + (-0.1625) = -0.4993
            from SIDEQUEST:  delta[1] + log(0.40) ≈ -4.232  + (-0.916)  = -5.148
            from LOST:       delta[2] + log(0.05) ≈ -8.726  + (-2.996)  = -11.72
            best = from ON_TASK; delta_new[0] = -0.4993 + log(0.05) ≈ -0.4993 + (-2.996) = -3.495
          SIDEQUEST (j=1, low=0.40):
            best = ON_TASK: delta[0] + log(0.149) ≈ -0.3367 + (-1.904) = -2.241
            delta_new[1] = -2.241 + log(0.40) ≈ -2.241 + (-0.916) = -3.157
          LOST (j=2, low=0.80):
            best = ON_TASK: delta[0] + log(0.001) ≈ -0.3367 + (-6.908) = -7.245
            but from SIDEQUEST: -4.232 + log(0.05) = -4.232 + (-2.996) = -7.228  ← marginally better
            from LOST: -8.726 + log(0.80) = -8.726 + (-0.223) = -8.949
            best = SIDEQUEST; delta_new[2] = -7.228 + log(0.80) = -7.228 + (-0.223) = -7.451

          argmax at t=1: SIDEQUEST (-3.157) > ON_TASK (-3.495) > LOST (-7.451)
          → path[1] = SIDEQUEST, backpointer from ON_TASK

        Expected path: ['ON_TASK', 'SIDEQUEST']
        """
        import math
        hmm = HMM()
        path = hmm.decode(["high", "low"])
        assert path == ["ON_TASK", "SIDEQUEST"], f"Got {path}"

    def test_uniform_emissions_stays_near_prior(self):
        """With uniform emissions the prior dominates; ON_TASK should appear most."""
        uniform_B = [[1 / 3, 1 / 3, 1 / 3]] * 3
        hmm = HMM(emission_prob=uniform_B)
        path = hmm.decode(["high", "mid", "low"])
        # With uniform emissions and sticky ON_TASK prior, ON_TASK should dominate
        on_task_count = sum(1 for s in path if s == "ON_TASK")
        assert on_task_count >= 2, f"Expected ON_TASK dominant, got {path}"

    def test_empty_observations_returns_empty_list(self):
        hmm = HMM()
        assert hmm.decode([]) == []

    def test_single_observation_returns_single_state(self):
        hmm = HMM()
        path = hmm.decode(["high"])
        assert len(path) == 1
        assert path[0] == "ON_TASK"


# ===========================================================================
# EMA tests
# ===========================================================================


class TestEma:
    def _make_store(self) -> IntentAnchorStore:
        s = IntentAnchorStore()
        s.set_anchor("fix the authentication bug")
        return s

    def test_ema_starts_at_initial_prior(self):
        """EMA should be 1.0 before any observation."""
        s = self._make_store()
        assert s.ema_posterior == pytest.approx(1.0)

    def test_ema_converges_toward_new_observations(self):
        """After many observations with ratio=0.0 the EMA should approach 0."""
        s = self._make_store()
        # Feed 200 completely-off-topic prompts
        for _ in range(200):
            s.record_observation("xyz abc def")
        # EMA should have drifted well below 0.5 by now
        # (1 - 0.05)^200 ≈ 0.000035; posterior ≈ 0.05 * 0 * ... ≈ ~0
        assert s.ema_posterior < 0.1, f"EMA did not converge; got {s.ema_posterior}"

    def test_ema_single_step_formula(self):
        """Manually verify one EMA step."""
        s = self._make_store()
        # Force a known ratio by using an identical prompt as anchor
        anchor_text = "fix authentication bug"
        s._anchor = None  # reset and re-anchor with controlled text
        s.set_anchor(anchor_text)
        tokens_anchor = tokenize(anchor_text)

        # Same prompt as anchor → lcs_ratio = 1.0
        # EMA after 1 step: 0.05 * 1.0 + 0.95 * 1.0 = 1.0
        _, _, ema = s.record_observation(anchor_text)
        assert ema == pytest.approx(1.0, rel=1e-6)


# ===========================================================================
# Store: drift signal increases when LCS ratio drops
# ===========================================================================


class TestStoreSignal:
    def test_drift_signal_increases_on_topic_shift(self):
        """
        After anchoring on a topic, recording a completely different prompt
        should produce a low LCS ratio — signalling drift.
        """
        s = IntentAnchorStore()
        s.set_anchor("implement binary search tree insert delete")

        # Same topic → high ratio
        ratio_same, _, _ = s.record_observation("implement binary search tree insert delete")
        assert ratio_same == pytest.approx(1.0, rel=1e-6)

        # Reset and re-anchor for a clean test
        s.clear()
        s.set_anchor("implement binary search tree insert delete")

        # Completely different topic → low ratio
        ratio_diff, _, _ = s.record_observation("summarize the quarterly revenue report")
        assert ratio_diff < 0.3, f"Expected low ratio on topic shift, got {ratio_diff}"

    def test_hmm_state_reflects_drift(self):
        """HMM should transition away from ON_TASK after sustained low-ratio inputs."""
        s = IntentAnchorStore()
        s.set_anchor("refactor the database connection pool")

        # Feed 10 completely off-topic prompts to push HMM toward LOST/SIDEQUEST
        for _ in range(10):
            s.record_observation("what is the weather in paris today")

        # The HMM's most-likely state should not be ON_TASK
        hmm_state = s.hmm.current().state
        assert hmm_state in ("SIDEQUEST", "LOST"), (
            f"Expected SIDEQUEST or LOST after drift, got {hmm_state}"
        )


# ===========================================================================
# End-to-end adapter tests
# ===========================================================================


def _make_engine() -> IntentAnchor:
    return IntentAnchor()


async def _fire_phase(
    bus: InProcessBus,
    ctx: RequestContext,
    phase: str,
    prompt: str,
) -> None:
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase=phase,  # type: ignore[arg-type]
        topic="user.prompt.submit",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={"user_prompt": prompt},
    )
    await bus.publish(event.topic, event)


class TestAdapterEndToEnd:
    async def test_anchor_phase_emits_anchor_set(self):
        """Anchor phase on the first prompt must emit intent-anchor.anchor.set."""
        engine = _make_engine()
        bus = InProcessBus()
        registry = {engine.name: engine}
        orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context()
        await _fire_phase(bus, ctx, "anchor", "build a cache eviction engine")
        result = await orch.run(ctx, dispatch)
        assert result == "ok"

        anchor_events = [
            e for e in bus.tap(ctx.correlation_id)
            if e.topic == "intent-anchor.anchor.set"
        ]
        assert len(anchor_events) >= 1, "Expected intent-anchor.anchor.set event"
        payload = anchor_events[0].payload
        assert "intent" in payload
        assert "token_count" in payload

    async def test_anchor_is_immutable_on_second_anchor_phase(self):
        """A second anchor phase must NOT overwrite the first anchor."""
        engine = _make_engine()
        bus = InProcessBus()
        registry = {engine.name: engine}
        orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context()
        await _fire_phase(bus, ctx, "anchor", "first intent")
        await orch.run(ctx, dispatch)

        store = engine.get_store(ctx.session_id)
        assert store is not None
        original_intent = store.anchor.intent  # type: ignore[union-attr]

        # Fire a second anchor on the same session — should be a no-op
        ctx2 = create_request_context(session_id=ctx.session_id)
        await _fire_phase(bus, ctx2, "anchor", "different intent — should not replace")
        await orch.run(ctx2, dispatch)

        assert store.anchor.intent == original_intent  # type: ignore[union-attr]

    async def test_post_session_below_threshold_emits_drift(self):
        """
        post-session with a completely unrelated prompt must emit
        intent-anchor.drift.detected when LCS ratio < 0.3.
        """
        engine = _make_engine()
        session_id = "sess-drift-test"

        # Manually set the anchor so we control the token baseline
        store = engine._get_or_create(session_id)
        store.set_anchor("implement the red-black tree insertion algorithm")

        bus = InProcessBus()
        registry = {engine.name: engine}
        orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context(session_id=session_id)

        await _fire_phase(bus, ctx, "post-session", "summarize quarterly finance results")
        result = await orch.run(ctx, dispatch)
        assert result == "ok"

        drift_events = [
            e for e in bus.tap(ctx.correlation_id)
            if e.topic == "intent-anchor.drift.detected"
        ]
        assert len(drift_events) >= 1, "Expected intent-anchor.drift.detected event"
        payload = drift_events[0].payload
        assert payload["lcs_ratio"] < 0.3
        assert "hmm_state" in payload
        assert "hmm_posterior" in payload
        assert "ema_posterior" in payload

    async def test_post_session_above_threshold_no_drift_event(self):
        """
        post-session with a closely related prompt must NOT emit a drift event.
        """
        engine = _make_engine()
        session_id = "sess-no-drift"

        store = engine._get_or_create(session_id)
        store.set_anchor("fix the authentication token expiry bug")

        bus = InProcessBus()
        registry = {engine.name: engine}
        orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context(session_id=session_id)

        # Very similar prompt — should stay above threshold
        await _fire_phase(
            bus, ctx, "post-session",
            "fix the authentication token expiry bug in the oauth service"
        )
        await orch.run(ctx, dispatch)

        drift_events = [
            e for e in bus.tap(ctx.correlation_id)
            if e.topic == "intent-anchor.drift.detected"
        ]
        assert len(drift_events) == 0, (
            f"Expected no drift event for similar prompt, got {len(drift_events)}"
        )

    async def test_post_session_no_anchor_returns_clean_ack(self):
        """post-session without an anchor must return a clean ack, no drift event."""
        engine = _make_engine()
        bus = InProcessBus()
        registry = {engine.name: engine}
        orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

        async def dispatch(ctx: RequestContext) -> str:
            return "ok"

        ctx = create_request_context()
        await _fire_phase(bus, ctx, "post-session", "any prompt without anchor")
        result = await orch.run(ctx, dispatch)
        assert result == "ok"

        drift_events = [
            e for e in bus.tap(ctx.correlation_id)
            if e.topic == "intent-anchor.drift.detected"
        ]
        assert len(drift_events) == 0

    async def test_adapter_is_advisory_does_not_veto(self):
        """Advisory engine must not block dispatch even when drift is detected."""
        engine = _make_engine()
        session_id = "sess-advisory"

        store = engine._get_or_create(session_id)
        store.set_anchor("deploy the microservice to production")

        bus = InProcessBus()
        registry = {engine.name: engine}
        orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

        dispatched = False

        async def dispatch(ctx: RequestContext) -> str:
            nonlocal dispatched
            dispatched = True
            return "dispatched"

        ctx = create_request_context(session_id=session_id)

        await _fire_phase(bus, ctx, "post-session", "what is the best pizza topping")
        result = await orch.run(ctx, dispatch)

        assert dispatched is True
        assert result == "dispatched"
