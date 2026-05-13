"""Tests for enchanter.proxy.events.rate_limiter — Wave 13.1 / Agent A.

Verifies:
  * Emitter shape (name, phases) matches the contract advertised in the
    module docstring.
  * Discovery via :func:`load_emitters` places ``rate-limiter`` after
    ``builtin`` (alphabetical) — important because builtin sets up
    ``ctx.scratch["budget_tier"]`` that our emitter reads.
  * At PRE_DISPATCH, the emitter publishes the rate-limiter-shaped event
    to ``mcp.tool.call.requested`` with ``vendor`` set to the model name.
  * At any other phase, the emitter is a documented no-op.
  * Integration: with a fake required-plugin in the registry that vetoes
    on ``mcp.tool.call.requested``, the pipeline returns a VetoResult —
    confirming that the emitter's publish is observed by the orchestrator's
    veto machinery.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from enchanter.core import EnchantedEvent, InProcessBus, PluginAck
from enchanter.core.plugin import PluginTopics
from enchanter.proxy import upstream
from enchanter.proxy.canonical import CanonicalRequest, Message, TextPart
from enchanter.proxy.events import EmitContext, EmitPhase, load_emitters
from enchanter.proxy.events.rate_limiter import RateLimiterEmitter, emitter
from enchanter.proxy.pipeline import PipelineOptions, VetoResult, run


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _req(text: str = "hello") -> CanonicalRequest:
    return CanonicalRequest(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text=text),)),),
    )


def _make_completion(text: str = "ok"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3)
    return SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o-mini")


async def _make_emit_ctx(req: CanonicalRequest, bus: InProcessBus) -> EmitContext:
    ctx = EmitContext(
        req=req,
        bus=bus,
        correlation_id="corr-test",
        session_id="sess-test",
    )
    ctx.scratch["budget_tier"] = "always"
    return ctx


# ---------------------------------------------------------------------------
# Shape contract.
# ---------------------------------------------------------------------------


def test_emitter_name_is_rate_limiter():
    assert emitter.name == "rate-limiter"
    assert RateLimiterEmitter().name == "rate-limiter"


def test_emitter_phases_is_pre_dispatch_only():
    assert emitter.phases == (EmitPhase.PRE_DISPATCH,)
    # Belt-and-braces: PRE_DISPATCH is the literal "pre-dispatch".
    assert emitter.phases == ("pre-dispatch",)


def test_emitter_discovered_after_builtin_alphabetically():
    """load_emitters() returns modules in alphabetical order — 'builtin'
    (b) must precede 'rate-limiter' (the module is ``rate_limiter`` which
    sorts after ``builtin``).  This ordering is load-bearing because the
    builtin populates ``ctx.scratch["budget_tier"]`` before we read it.
    """
    emitters = load_emitters()
    names = [em.name for em in emitters]
    assert "builtin" in names, f"missing 'builtin' in {names!r}"
    assert "rate-limiter" in names, f"missing 'rate-limiter' in {names!r}"
    assert names.index("builtin") < names.index("rate-limiter"), (
        f"builtin must precede rate-limiter; got {names!r}"
    )


# ---------------------------------------------------------------------------
# Publish behaviour.
# ---------------------------------------------------------------------------


async def test_emit_at_pre_dispatch_publishes_rate_limiter_event():
    """At PRE_DISPATCH the emitter publishes a single event to
    ``mcp.tool.call.requested`` with the rate-limiter-shaped payload.
    """
    captured: list[EnchantedEvent] = []

    async def _recorder(event: EnchantedEvent) -> None:
        captured.append(event)
        return None

    bus = InProcessBus()
    bus.subscribe("mcp.tool.call.requested", _recorder)

    ctx = await _make_emit_ctx(_req("hello world"), bus)
    await emitter.emit(EmitPhase.PRE_DISPATCH, ctx)

    assert len(captured) == 1, f"expected one event, got {len(captured)}"
    ev = captured[0]
    assert ev.topic == "mcp.tool.call.requested"
    assert ev.correlation_id == "corr-test"
    assert ev.session_id == "sess-test"
    assert ev.source == "proxy-rate-limiter"
    assert ev.budget_tier == "always"
    # The contract with the engine — vendor is what _extract_vendor reads.
    assert ev.payload["vendor"] == "gpt-4o-mini"
    assert ev.payload["model"] == "gpt-4o-mini"
    # Tool field mirrors the builtin shape so other engines re-keying on
    # this topic see something sane.
    assert ev.payload["tool"] == "llm.proxy"


async def test_emit_at_other_phases_is_noop():
    """The emitter only publishes at PRE_DISPATCH; other phases are
    silently ignored (defensive against future broader dispatch).
    """
    captured: list[EnchantedEvent] = []

    async def _recorder(event: EnchantedEvent) -> None:
        captured.append(event)

    bus = InProcessBus()
    bus.subscribe("*", _recorder)

    ctx = await _make_emit_ctx(_req(), bus)
    # Try every non-PRE_DISPATCH phase the EmitPhase class advertises.
    for phase in (EmitPhase.POST_DISPATCH, EmitPhase.POST_SESSION, EmitPhase.CROSS_SESSION):
        await emitter.emit(phase, ctx)

    assert captured == [], (
        f"emitter should be a no-op outside PRE_DISPATCH; got {captured!r}"
    )


async def test_emit_uses_budget_tier_from_scratch():
    """If the builtin already stashed ``budget_tier`` in ctx.scratch, the
    rate-limiter emitter must honour it (consistent event metadata across
    the emitter chain).
    """
    captured: list[EnchantedEvent] = []

    async def _recorder(event: EnchantedEvent) -> None:
        captured.append(event)

    bus = InProcessBus()
    bus.subscribe("mcp.tool.call.requested", _recorder)

    ctx = EmitContext(
        req=_req(),
        bus=bus,
        correlation_id="c",
        session_id="s",
    )
    ctx.scratch["budget_tier"] = "high-only"
    await emitter.emit(EmitPhase.PRE_DISPATCH, ctx)

    assert len(captured) == 1
    assert captured[0].budget_tier == "high-only"


# ---------------------------------------------------------------------------
# Integration — emitter delivers an event that the orchestrator vetoes on.
# ---------------------------------------------------------------------------


class _VetoingRateLimiterStub:
    """Stand-in plugin: required + vetoes on every pre-dispatch event.

    Used to verify that the emitter's publish lands on the bus in time for
    the orchestrator's lifecycle.pre-dispatch ack window — without
    exercising the real (stateful) rate-limiter engine.
    """

    name = "rate-limiter"
    phases = ("pre-dispatch",)
    required = True
    topics = PluginTopics(
        subscribes=("lifecycle.pre-dispatch", "mcp.tool.call.requested"),
        emits=(),
    )
    budget_tier = "always"

    async def on_phase(self, event, ctx) -> PluginAck:
        return PluginAck(
            status="veto",
            reason=f"rate-limit exceeded for vendor={event.payload.get('vendor')!r}",
        )


async def test_pipeline_integration_veto_propagates_through_emitter_wiring():
    """End-to-end: a fake required rate-limiter plugin in the registry
    vetoes when our emitter's event hits the bus.  The pipeline must
    return a :class:`VetoResult` and never call the upstream.

    NOTE: we patch ``load_engine_registry`` rather than the real
    rate-limiter so this test does not pollute the engine's per-process
    token-bucket state.
    """
    from enchanter.proxy import pipeline as pipeline_mod

    fake_plugin = _VetoingRateLimiterStub()
    fake_registry = {fake_plugin.name: fake_plugin}

    mock_acomp = AsyncMock(return_value=_make_completion())
    with patch.object(pipeline_mod, "load_engine_registry", return_value=fake_registry):
        with patch.object(upstream.litellm, "acompletion", new=mock_acomp):
            result = await run(_req("benign prompt"), PipelineOptions(conduct=False))

    assert isinstance(result, VetoResult), f"expected VetoResult, got {result!r}"
    assert result.plugin == "rate-limiter"
    assert result.phase == "pre-dispatch"
    # Upstream must NEVER be called on a pre-dispatch veto.
    assert mock_acomp.await_count == 0
