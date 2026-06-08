"""tests/test_production_wiring.py — F1–F4 production-wiring follow-ups.

Closes the genuinely-deferred items from
``docs/architecture/ROADMAP-bus-contract-hardening.md`` ("Follow-ups"):

F1  call_upstream / stream_upstream fallback activation in the live run path:
      * the run path passes a multi-element registry-backed chain to
        ``call_upstream`` (spied via monkeypatch);
      * ``stream_upstream`` falls through PRE-STREAM on a retryable first-attempt
        error, and does NOT fall through once the first chunk has been yielded
        (the hard mid-stream boundary).
F2  the cost-ledger emitter publishes ``cents`` directly; the ``score`` smuggle
      is gone from both the bus payload and the recorder ``payload_summary``.
F3  ``RequestContext.prompt_overlay`` is delivered end-to-end through the
      orchestrator and applied by intent-anchor's agent path (mock seam); the
      deterministic path stays the default when the flag is off; an agent
      verdict appends a durable line to ``state/agent-verdicts.jsonl``.
F4  pricing parity sanity bound — prices live in a plausible range and
      input <= output for the known models, so a jointly-wrong value is caught
      (the existing parity test only detects seed-vs-registry divergence).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from robit.proxy import pipeline, upstream
from robit.proxy.canonical import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    Message,
    TextPart,
)
from robit.proxy.upstream import UpstreamError, resolve_fallback_chain, stream_upstream


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A registry-known model whose family ("Claude 4.x") has >1 member, so the
# fallback chain genuinely has a tail. (gpt-4o-mini etc. are NOT in the bundled
# registry, so they correctly collapse to a single-model no-op chain.)
_KNOWN_MULTI_MODEL = "claude-opus-4-6"


def _req(model: str = _KNOWN_MULTI_MODEL, text: str = "hi") -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=(Message(role="user", content=(TextPart(text=text),)),),
        max_tokens=64,
    )


def _completion(text: str = "ok", *, model: str = _KNOWN_MULTI_MODEL):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _chunk(content: str | None = None, *, finish_reason: str | None = None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice], usage=None)


class _AsyncStream:
    """Async iterator over fabricated litellm chunks, optionally raising."""

    def __init__(self, chunks, *, raise_after=None):
        self._chunks = list(chunks)
        self._raise_after = raise_after
        self._yielded = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._raise_after is not None and self._yielded >= self._raise_after:
            raise self._raise_after_exc()
        if not self._chunks:
            raise StopAsyncIteration
        self._yielded += 1
        return self._chunks.pop(0)

    def _raise_after_exc(self):
        return RuntimeError("mid-stream upstream blew up")


def _retryable(provider: str = "anthropic") -> UpstreamError:
    return UpstreamError(provider=provider, status=529, message="overloaded")


# ===========================================================================
# F1 — fallback chain activation in the live run path
# ===========================================================================


def test_resolve_fallback_chain_known_model_has_tail():
    """A registry-known model yields a real multi-element chain (primary
    first, same-family siblings as the tail)."""
    chain = resolve_fallback_chain(_KNOWN_MULTI_MODEL)
    assert chain[0] == _KNOWN_MULTI_MODEL
    assert len(chain) >= 2, f"expected fallback alternatives, got {chain}"
    assert len(chain) == len(set(chain)), f"chain has duplicates: {chain}"


def test_resolve_fallback_chain_unknown_model_is_single_element():
    """A model the registry does not track collapses to the legacy no-op
    single-model chain — byte-for-byte the pre-fallback behaviour."""
    assert resolve_fallback_chain("gpt-4o-mini") == ("gpt-4o-mini",)


@pytest.mark.asyncio
async def test_run_path_passes_multi_element_chain_to_call_upstream():
    """The run path resolves a registry-backed chain and hands it to
    call_upstream via the ``models=`` arg (spied through a monkeypatch)."""
    seen: dict = {}

    async def _spy_call_upstream(req, models=None, **kwargs):
        seen["models"] = models
        return CanonicalResponse(
            model=req.model,
            content=(TextPart(text="ok"),),
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=10, output_tokens=5),
        )

    with patch.object(pipeline, "call_upstream", new=_spy_call_upstream):
        result = await pipeline.run(_req(model=_KNOWN_MULTI_MODEL))

    assert hasattr(result, "response")  # PipelineResult, not a veto.
    chain = seen["models"]
    assert chain is not None and len(chain) >= 2, (
        f"run path should pass a multi-element fallback chain; got {chain!r}"
    )
    assert chain[0] == _KNOWN_MULTI_MODEL


@pytest.mark.asyncio
async def test_stream_upstream_falls_through_pre_stream_on_retryable_error():
    """First model's connection fails with a retryable error BEFORE any chunk
    is yielded → stream_upstream falls through to the second model."""
    success = _AsyncStream([_chunk("hello", finish_reason="stop")])
    # First acompletion call raises a retryable (529); the second returns a
    # working stream. The pre-stream loop must swallow the first and retry.
    mocked = AsyncMock(side_effect=[_retryable(), success])

    req = _req(model="model-a")
    with patch.object(upstream.litellm, "acompletion", new=mocked):
        out = [c async for c in stream_upstream(req, models=("model-a", "model-b"))]

    # Two acompletion attempts: failed model-a, then model-b succeeds.
    assert mocked.await_count == 2
    # The canonical translation produced real chunks (message_start, etc.).
    kinds = [c.type for c in out]
    assert "message_start" in kinds
    assert any(c.type == "text_delta" and c.text == "hello" for c in out)


@pytest.mark.asyncio
async def test_stream_upstream_does_not_fall_through_mid_stream():
    """Once the FIRST chunk has been yielded, a mid-stream retryable failure
    must NOT fall through to another model — the boundary is hard. The error
    propagates and the second model is never tried."""
    # Stream yields one good chunk, then __anext__ raises on the 2nd pull.
    flaky = _AsyncStream(
        [_chunk("partial"), _chunk("never-seen")], raise_after=1
    )
    # Provide a perfectly good second model that MUST NOT be consulted.
    second = _AsyncStream([_chunk("from-model-b", finish_reason="stop")])
    mocked = AsyncMock(side_effect=[flaky, second])

    req = _req(model="model-a")
    collected: list = []
    with patch.object(upstream.litellm, "acompletion", new=mocked):
        with pytest.raises(UpstreamError):
            async for c in stream_upstream(req, models=("model-a", "model-b")):
                collected.append(c)

    # Only the first model was opened — mid-stream fallback is impossible.
    assert mocked.await_count == 1
    # The caller DID receive the committed model's early chunks before the blow-up.
    assert any(getattr(c, "text", None) == "partial" for c in collected)
    # Nothing from model-b ever reached the caller.
    assert not any(getattr(c, "text", None) == "from-model-b" for c in collected)


# ===========================================================================
# F2 — cost ``cents`` in payload_summary; ``score`` smuggle removed
# ===========================================================================


@pytest.mark.asyncio
async def test_cost_cents_surfaces_and_score_smuggle_is_gone():
    """pipeline.run surfaces a cost observation whose payload_summary carries
    ``cents`` directly, and the bus payload no longer mirrors it under
    ``score`` (F2)."""
    req = CanonicalRequest(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text="hi"),)),),
        max_tokens=64,
    )

    def _make_litellm_completion(model, prompt_tokens, completion_tokens):
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return SimpleNamespace(choices=[choice], usage=usage, model=model)

    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(
            return_value=_make_litellm_completion("gpt-4o-mini", 10_000, 5_000)
        ),
    ):
        result = await pipeline.run(req)

    cost_obs = [ob for ob in result.fired if ob.topic == "cost.ledger.recorded"]
    assert len(cost_obs) == 1
    summary = cost_obs[0].payload_summary
    assert summary.get("cents", 0) > 0
    assert "score" not in summary, "the score smuggle must be gone (F2)"


def test_recorder_whitelist_includes_cents():
    """The recorder whitelist surfaces ``cents`` directly (and ``score`` is no
    longer the carrier for cost)."""
    summarised = pipeline._BusRecorder._summarise_payload(
        {"cents": 42, "model": "x", "input_tokens": 1, "output_tokens": 2}
    )
    assert summarised == {"cents": 42}


def test_cost_emitter_payload_has_no_score_key():
    """Direct emitter check: the published bus payload carries ``cents`` and no
    longer carries ``score``."""
    import asyncio

    from robit.core import InProcessBus
    from robit.proxy.events import EmitPhase
    from robit.proxy.events._types import EmitContext
    from robit.proxy.events.cost_ledger import emitter as cost_emitter

    captured: list = []

    async def _run():
        bus = InProcessBus()

        async def _capture(event):
            captured.append(event)

        bus.subscribe("*", _capture)
        resp = CanonicalResponse(
            model="claude-3-5-sonnet-20241022",
            content=(TextPart(text="ok"),),
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=10_000, output_tokens=5_000),
        )
        ctx = EmitContext(
            req=_req(model="claude-3-5-sonnet-20241022"),
            bus=bus,
            correlation_id="cid",
            session_id="sid",
            response=resp,
        )
        await cost_emitter.emit(EmitPhase.POST_SESSION, ctx)

    asyncio.run(_run())
    cost = [e for e in captured if e.topic == "cost.ledger.recorded"]
    assert len(cost) == 1
    assert cost[0].payload["cents"] == 11
    assert "score" not in cost[0].payload


# ===========================================================================
# F3 — prompt_overlay end-to-end + agent path + verdict audit
# ===========================================================================


def test_request_context_carries_prompt_overlay():
    """The new field exists, defaults None, and create_request_context sets it."""
    from robit.core import create_request_context

    assert create_request_context().prompt_overlay is None
    ctx = create_request_context(prompt_overlay="tenant: never drift to billing")
    assert ctx.prompt_overlay == "tenant: never drift to billing"


def test_orchestrator_delivers_overlay_into_plugin_ctx():
    """The orchestrator carries the overlay onto the RequestContext it hands a
    plugin's on_phase (end-to-end delivery without changing on_phase's
    signature)."""
    from robit.core import InProcessBus, OrchestratorConfig
    from robit.core.lifecycle import Orchestrator

    orch = Orchestrator(
        OrchestratorConfig(
            registry={}, bus=InProcessBus(), prompt_overlay="OP-OVERLAY"
        )
    )
    event = SimpleNamespace(
        correlation_id="c", session_id="s", phase="post-session",
        budget_tier="HIGH", ts=0,
    )
    ctx = orch._context_from_event(event)  # type: ignore[arg-type]
    assert ctx.prompt_overlay == "OP-OVERLAY"


@pytest.mark.asyncio
async def test_intent_anchor_agent_applies_overlay_from_ctx(monkeypatch):
    """Agent path ON + a MOCK seam: intent-anchor reads ``ctx.prompt_overlay``
    and APPENDS it to the engine-authored drift prompt (precedence: framework <
    engine-author < operator overlay)."""
    monkeypatch.setenv("ROBIT_INTENT_ANCHOR_AGENT", "1")

    from robit.core import RequestContext
    from robit.core.events import EnchantedEvent
    from robit.engines.intent_anchor import IntentAnchor

    seen: dict = {}

    async def _mock_llm(system: str, user: str) -> str:
        seen["system"] = system
        return json.dumps({"drift": False, "confidence": 0.1, "rationale": "ok"})

    engine = IntentAnchor(llm_call=_mock_llm)  # NO constructor overlay.
    # Seed an anchor for the session so the post-session path runs.
    engine._handle_anchor(  # type: ignore[attr-defined]
        EnchantedEvent(
            id="e0", correlation_id="c", session_id="s", phase="anchor",
            topic="lifecycle.anchor", source="t", budget_tier="HIGH", ts=0,
            payload={"user_prompt": "implement red-black tree insertion"},
        )
    )

    overlay = "OPERATOR: treat finance topics as in-scope"
    ctx = RequestContext(
        correlation_id="c", session_id="s", phase="post-session",
        budget_tier="HIGH", sampling_depth=0, deadline_ms=30_000,
        started_ms=0, prompt_overlay=overlay,
    )
    event = EnchantedEvent(
        id="e1", correlation_id="c", session_id="s", phase="post-session",
        topic="lifecycle.post-session", source="t", budget_tier="HIGH", ts=0,
        payload={"user_prompt": "summarize quarterly finance results"},
    )

    ack = await engine.on_phase(event, ctx)
    assert ack.status == "ack"
    # The mock saw the engine-authored prompt WITH the operator overlay appended.
    assert "Operator overlay" in seen["system"]
    assert overlay in seen["system"]


@pytest.mark.asyncio
async def test_deterministic_path_is_default_when_flag_off(monkeypatch):
    """Flag OFF → the deterministic LCS+HMM path runs; a MOCK seam is never
    consulted (the agent path stays default-OFF)."""
    monkeypatch.delenv("ROBIT_INTENT_ANCHOR_AGENT", raising=False)

    from robit.core import RequestContext
    from robit.core.events import EnchantedEvent
    from robit.engines.intent_anchor import IntentAnchor

    called = {"n": 0}

    async def _mock_llm(system: str, user: str) -> str:
        called["n"] += 1
        return json.dumps({"drift": True, "confidence": 1.0, "rationale": "x"})

    engine = IntentAnchor(llm_call=_mock_llm)
    engine._handle_anchor(  # type: ignore[attr-defined]
        EnchantedEvent(
            id="e0", correlation_id="c", session_id="s2", phase="anchor",
            topic="lifecycle.anchor", source="t", budget_tier="HIGH", ts=0,
            payload={"user_prompt": "implement red-black tree insertion"},
        )
    )
    ctx = RequestContext(
        correlation_id="c", session_id="s2", phase="post-session",
        budget_tier="HIGH", sampling_depth=0, deadline_ms=30_000, started_ms=0,
    )
    event = EnchantedEvent(
        id="e1", correlation_id="c", session_id="s2", phase="post-session",
        topic="lifecycle.post-session", source="t", budget_tier="HIGH", ts=0,
        payload={"user_prompt": "summarize quarterly finance results"},
    )
    ack = await engine.on_phase(event, ctx)
    assert ack.status == "ack"
    assert called["n"] == 0, "deterministic path must not consult the model"
    # Deterministic drift fired (unrelated prompt) → a drift derived event.
    drift = [e for e in ack.derived_events if e.topic == "intent-anchor.drift.detected"]
    assert len(drift) == 1
    assert drift[0].payload.get("verdict_source") != "agent"


@pytest.mark.asyncio
async def test_agent_verdict_writes_audit_line(tmp_path, monkeypatch):
    """An agent verdict appends a content-free JSON line to
    ``state/agent-verdicts.jsonl`` under the resolved state dir (F3(c))."""
    monkeypatch.setenv("ROBIT_INTENT_ANCHOR_AGENT", "1")
    # Redirect the state root to a tmp dir (same knob the veto audit uses).
    monkeypatch.setenv("ROBIT_INFERENCE_STATE", str(tmp_path / "inference"))

    from robit.core import RequestContext
    from robit.core.events import EnchantedEvent
    from robit.engines.intent_anchor import IntentAnchor
    from robit.engines.intent_anchor.adapter import agent_verdicts_log_path

    async def _mock_llm(system: str, user: str) -> str:
        return json.dumps(
            {"drift": True, "confidence": 0.9, "rationale": "topic changed"}
        )

    engine = IntentAnchor(llm_call=_mock_llm)
    engine._handle_anchor(  # type: ignore[attr-defined]
        EnchantedEvent(
            id="e0", correlation_id="corr-xyz", session_id="s3", phase="anchor",
            topic="lifecycle.anchor", source="t", budget_tier="HIGH", ts=0,
            payload={"user_prompt": "implement red-black tree insertion"},
        )
    )
    ctx = RequestContext(
        correlation_id="corr-xyz", session_id="s3", phase="post-session",
        budget_tier="HIGH", sampling_depth=0, deadline_ms=30_000, started_ms=0,
    )
    event = EnchantedEvent(
        id="e1", correlation_id="corr-xyz", session_id="s3", phase="post-session",
        topic="lifecycle.post-session", source="t", budget_tier="HIGH", ts=0,
        payload={"user_prompt": "summarize quarterly finance results"},
    )
    await engine.on_phase(event, ctx)

    log = agent_verdicts_log_path()
    assert log.exists(), f"expected audit log at {log}"
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["correlation_id"] == "corr-xyz"
    assert rec["engine"] == "intent-anchor"
    assert rec["tier"] == "executor"
    assert rec["verdict"]["drift"] is True
    # Privacy: a sha + short summary, never the raw prompt.
    assert "prompt_sha" in rec and len(rec["prompt_sha"]) == 64
    assert "response_summary" in rec
    # The raw user/anchor prompt text must NOT be stored verbatim.
    assert "red-black tree" not in json.dumps(rec)


# ===========================================================================
# F4 — pricing parity sanity bound (catches a jointly-wrong price)
# ===========================================================================


def test_registry_prices_within_sanity_bounds():
    """The existing parity test only catches seed-vs-registry DIVERGENCE — a
    value that is wrong in BOTH would slip through. This bounds every price to
    a plausible range and asserts input <= output for the known models, so a
    jointly-wrong number is caught."""
    from robit.proxy.events.cost_ledger import _load_registry_prices

    prices = _load_registry_prices()
    assert prices, "registry pricing map must be non-empty"

    _MIN_CENTS_PER_MTOK = 1
    _MAX_CENTS_PER_MTOK = 100_000

    for prefix, (in_rate, out_rate) in prices.items():
        assert _MIN_CENTS_PER_MTOK <= in_rate <= _MAX_CENTS_PER_MTOK, (
            f"{prefix} input rate {in_rate} out of plausible range"
        )
        assert _MIN_CENTS_PER_MTOK <= out_rate <= _MAX_CENTS_PER_MTOK, (
            f"{prefix} output rate {out_rate} out of plausible range"
        )
        # For every known frontier model, output tokens cost >= input tokens.
        assert in_rate <= out_rate, (
            f"{prefix}: input rate {in_rate} should be <= output rate {out_rate}"
        )
