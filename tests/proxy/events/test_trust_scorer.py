"""Tests for robit.proxy.events.trust_scorer — Wave 13.1 emitter.

Covers:
  - Identity contract (name, phases, discovery ordering after builtin).
  - Phase-guard behaviour (non-POST_SESSION → no-op).
  - POST_SESSION: bus event published with expected topic + payload shape.
  - Two-emit cumulative posterior update (alpha increments twice).
  - End-to-end: pipeline.run with a mocked upstream surfaces the
    trust-scorer observation in result.fired.

The engine adapter is a module-level singleton; tests reset its store at
setup to avoid cross-test bleed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from robit.core import InProcessBus
from robit.engines.trust_scorer.adapter import adapter as trust_engine
from robit.proxy import upstream
from robit.proxy.canonical import CanonicalRequest, Message, TextPart
from robit.proxy.events import EmitPhase, load_emitters
from robit.proxy.events._types import EmitContext
from robit.proxy.events.trust_scorer import (
    TrustScorerEmitter,
    emitter as trust_scorer_emitter,
)
from robit.proxy.pipeline import PipelineOptions, PipelineResult, run


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_trust_store():
    """Reset the engine's posterior store before AND after every test.

    The adapter is a module-level singleton — without this, alpha/beta would
    carry across tests and break the "two emits → +2 alpha" assertion.
    """
    trust_engine.store.reset()
    yield
    trust_engine.store.reset()


def _req(model: str = "gpt-4o-mini", text: str = "hello") -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=(Message(role="user", content=(TextPart(text=text),)),),
        system=None,
    )


def _make_completion(text: str = "ok", model: str = "gpt-4o-mini"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


async def _make_emit_ctx(req: CanonicalRequest) -> tuple[EmitContext, list]:
    """Build a minimal EmitContext + a captured-events list subscribed to '*'."""
    bus = InProcessBus()
    captured: list = []

    async def _capture(event):
        captured.append(event)
        return None

    bus.subscribe("*", _capture)

    ctx = EmitContext(
        req=req,
        bus=bus,
        correlation_id="corr-1",
        session_id="sess-1",
    )
    ctx.scratch["budget_tier"] = "med-or-higher"
    return ctx, captured


# ---------------------------------------------------------------------------
# Identity / discovery contract
# ---------------------------------------------------------------------------


def test_emitter_identity_and_phases():
    assert trust_scorer_emitter.name == "trust-scorer"
    assert trust_scorer_emitter.phases == (EmitPhase.POST_SESSION,)
    assert isinstance(trust_scorer_emitter, TrustScorerEmitter)


def test_emitter_discovered_after_builtin():
    """Discovery order is alphabetical by module name; 'builtin' < 'trust_scorer'.

    The brief calls this out: emitters that depend on the trust-gate events
    landing first should sort alphabetically after 'builtin', and this one
    does.
    """
    emitters = load_emitters()
    names = [e.name for e in emitters]
    assert "builtin" in names
    assert "trust-scorer" in names
    assert names.index("builtin") < names.index("trust-scorer")


# ---------------------------------------------------------------------------
# Phase guard
# ---------------------------------------------------------------------------


async def test_emit_non_post_session_is_noop():
    """Calling emit with PRE_DISPATCH must not record an observation or
    publish a bus event — the emitter only fires on POST_SESSION."""
    ctx, captured = await _make_emit_ctx(_req(model="model-X"))

    await trust_scorer_emitter.emit(EmitPhase.PRE_DISPATCH, ctx)

    # No bus event published.
    assert captured == []
    # No observation recorded — observation_count stays at 0.
    assert trust_engine.store.observation_count(("model-X", "completion")) == 0


# ---------------------------------------------------------------------------
# POST_SESSION publish / record
# ---------------------------------------------------------------------------


async def test_emit_post_session_publishes_observation_event():
    ctx, captured = await _make_emit_ctx(_req(model="gpt-4o-mini"))

    await trust_scorer_emitter.emit(EmitPhase.POST_SESSION, ctx)

    # Exactly one bus event on the observation topic.
    obs_events = [e for e in captured if e.topic == "trust-scorer.observation.recorded"]
    assert len(obs_events) == 1, f"expected 1 observation event, got {captured!r}"

    ev = obs_events[0]
    assert ev.source == "trust-scorer"
    assert ev.correlation_id == "corr-1"
    assert ev.session_id == "sess-1"
    # Payload shape — matches the engine's (server_id, tool_name) key shape.
    payload = dict(ev.payload)
    assert payload["server_id"] == "gpt-4o-mini"
    assert payload["tool_name"] == "completion"
    assert payload["outcome"] == "success"
    assert payload["observation_count"] == 1
    # Posterior after one success on Beta(1,1) is 2/3.
    assert payload["score"] == pytest.approx(2.0 / 3.0)


async def test_two_successive_emits_increment_posterior_twice():
    """Each POST_SESSION fire must add one success observation."""
    ctx, _ = await _make_emit_ctx(_req(model="claude-3-5-sonnet-latest"))

    await trust_scorer_emitter.emit(EmitPhase.POST_SESSION, ctx)
    await trust_scorer_emitter.emit(EmitPhase.POST_SESSION, ctx)

    key = ("claude-3-5-sonnet-latest", "completion")
    alpha, beta = trust_engine.store.alpha_beta(key)
    # Prior is Beta(1, 1); two successes → alpha = 3, beta = 1.
    assert alpha == 3.0
    assert beta == 1.0
    assert trust_engine.store.observation_count(key) == 2
    # Posterior mean = 3/4.
    assert trust_engine.score(*key) == pytest.approx(0.75)


async def test_emits_for_distinct_models_keep_separate_posteriors():
    """Key isolation: different model ids must not bleed into one another."""
    ctx_a, _ = await _make_emit_ctx(_req(model="model-A"))
    ctx_b, _ = await _make_emit_ctx(_req(model="model-B"))

    await trust_scorer_emitter.emit(EmitPhase.POST_SESSION, ctx_a)
    await trust_scorer_emitter.emit(EmitPhase.POST_SESSION, ctx_a)
    await trust_scorer_emitter.emit(EmitPhase.POST_SESSION, ctx_b)

    assert trust_engine.store.observation_count(("model-A", "completion")) == 2
    assert trust_engine.store.observation_count(("model-B", "completion")) == 1


# ---------------------------------------------------------------------------
# Integration: full pipeline.run
# ---------------------------------------------------------------------------


async def test_pipeline_run_surfaces_trust_scorer_observation_in_fired():
    """End-to-end: pipeline.run with a mocked upstream should produce at
    least one bus observation sourced from trust-scorer.

    The proxy's ``_BusRecorder`` lists ``trust-scorer`` as an interesting
    source, so anything the engine emits at trust-gate AND anything this
    emitter publishes at POST_SESSION lands in ``result.fired``.
    """
    fake = _make_completion(text="hello", model="gpt-4o-mini")
    with patch.object(
        upstream.litellm, "acompletion", new=AsyncMock(return_value=fake)
    ):
        result = await run(_req(text="benign question"), PipelineOptions(conduct=False))

    assert isinstance(result, PipelineResult)
    trust_obs = [o for o in result.fired if o.source == "trust-scorer"]
    assert trust_obs, (
        f"expected at least one trust-scorer observation in result.fired, "
        f"got {result.fired!r}"
    )

    # And the engine actually recorded the success — observation_count on
    # the natural key is non-zero.
    assert trust_engine.store.observation_count(("gpt-4o-mini", "completion")) >= 1
