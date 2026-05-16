"""Tests for robit.proxy.events.cost_ledger — Wave 13.1 Agent B.

Coverage:

* Discovery: emitter loads via :func:`load_emitters`, alphabetically after
  ``builtin``; exposes the documented ``name`` / ``phases`` contract.
* Phase gating: emit at PRE_DISPATCH is a no-op (no bus publishes).
* Unary path: ``ctx.response.usage`` drives the cents calculation.
* Streaming path: ``accumulated_text`` chars/4 estimate when response is
  ``None``; documents the known under-count.
* Integration: full :func:`pipeline.run` produces a ``BusObservation``
  with ``topic="cost.ledger.recorded"`` and ``payload_summary["score"]``
  carrying the cents.
* HTTP integration: a real :class:`ProxyServer` surfaces
  ``X-Enchanter-Cost-Cents`` on a unary 200 response.
* Accumulation: two requests with the same model both produce non-zero
  cost observations without crashing.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from robit.core import InProcessBus
from robit.proxy.canonical import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    Message,
    TextPart,
)
from robit.proxy.events import EmitPhase, load_emitters
from robit.proxy.events.cost_ledger import (
    CostLedgerEmitter,
    _compute_cents,
    _price_for,
    emitter as cost_emitter,
)
from robit.proxy.events._types import EmitContext


# ---------------------------------------------------------------------------
# Helpers — build minimal canonical requests / responses for the tests.
# ---------------------------------------------------------------------------


def _req(model: str = "gpt-4o-mini", text: str = "hi") -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=(
            Message(role="user", content=(TextPart(text=text),)),
        ),
        max_tokens=64,
    )


def _resp(
    model: str = "gpt-4o-mini",
    text: str = "ok",
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> CanonicalResponse:
    return CanonicalResponse(
        model=model,
        content=(TextPart(text=text),),
        stop_reason="end_turn",
        usage=CanonicalUsage(
            input_tokens=input_tokens, output_tokens=output_tokens
        ),
    )


async def _make_ctx(
    *,
    response: CanonicalResponse | None = None,
    accumulated_text: str | None = None,
    model: str = "gpt-4o-mini",
) -> tuple[EmitContext, list]:
    """Build an EmitContext + a captured-events list bound to its bus.

    The list is appended to via a wildcard subscriber so tests can assert
    which topics the emitter actually published.
    """
    bus = InProcessBus()
    captured: list = []

    async def _capture(event):
        captured.append(event)
        return None

    bus.subscribe("*", _capture)

    ctx = EmitContext(
        req=_req(model=model),
        bus=bus,
        correlation_id="cid-test",
        session_id="sid-test",
        response=response,
        accumulated_text=accumulated_text,
    )
    return ctx, captured


# ---------------------------------------------------------------------------
# 1. Discovery + protocol contract.
# ---------------------------------------------------------------------------


def test_emitter_protocol_contract():
    assert cost_emitter.name == "cost-ledger"
    assert cost_emitter.phases == (EmitPhase.POST_SESSION,)
    # Sanity: the class is what the docstring claims.
    assert isinstance(cost_emitter, CostLedgerEmitter)


def test_load_emitters_includes_cost_ledger_after_builtin():
    emitters = load_emitters()
    names = [em.name for em in emitters]
    assert "builtin" in names
    assert "cost-ledger" in names
    # Discovery is alphabetical by MODULE name (builtin.py < cost_ledger.py).
    assert names.index("builtin") < names.index("cost-ledger")


# ---------------------------------------------------------------------------
# 2. Phase gating — only POST_SESSION publishes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_at_pre_dispatch_is_noop():
    ctx, captured = await _make_ctx(response=_resp())
    await cost_emitter.emit(EmitPhase.PRE_DISPATCH, ctx)
    assert captured == []
    # And nothing was stashed in scratch either.
    assert "cost-ledger" not in ctx.scratch


# ---------------------------------------------------------------------------
# 3. Unary path — response.usage drives the cents.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unary_post_session_publishes_cost_event():
    # gpt-4o-mini: 15c per M input, 60c per M output.
    # 100 input + 50 output → (100*15 + 50*60) / 1e6 = (1500 + 3000) / 1e6
    # = 4500 / 1_000_000 = 0 cents (4500 sub-cent) → ceiling → 1 cent.
    resp = _resp(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
    ctx, captured = await _make_ctx(response=resp)

    await cost_emitter.emit(EmitPhase.POST_SESSION, ctx)

    cost_events = [e for e in captured if e.topic == "cost.ledger.recorded"]
    assert len(cost_events) == 1
    ev = cost_events[0]
    assert ev.source == "proxy-pipeline"
    assert ev.payload["cents"] == 1
    assert ev.payload["score"] == 1  # cents mirrored under score for recorder.
    assert ev.payload["model"] == "gpt-4o-mini"
    assert ev.payload["input_tokens"] == 100
    assert ev.payload["output_tokens"] == 50
    # Scratch is populated for downstream emitters.
    assert ctx.scratch["cost-ledger"]["cents"] == 1
    assert ctx.scratch["cost-ledger"]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_unary_high_token_count_produces_proportional_cents():
    # claude-3-5-sonnet: 300c per M input, 1500c per M output.
    # 10_000 input + 5_000 output → (3_000_000 + 7_500_000) / 1e6 = 10.5
    # ceiling → 11 cents.
    resp = _resp(
        model="claude-3-5-sonnet-20241022",
        input_tokens=10_000,
        output_tokens=5_000,
    )
    ctx, captured = await _make_ctx(response=resp)

    await cost_emitter.emit(EmitPhase.POST_SESSION, ctx)

    cost_events = [e for e in captured if e.topic == "cost.ledger.recorded"]
    assert len(cost_events) == 1
    assert cost_events[0].payload["cents"] == 11


# ---------------------------------------------------------------------------
# 4. Streaming path — accumulated_text length / 4 estimate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_path_uses_char_estimate():
    # 12-char text → 12 // 4 = 3 estimated output tokens, 0 input.
    # gpt-4o-mini output rate 60c/M → 3*60 = 180 sub-cents → ceiling → 1 cent.
    ctx, captured = await _make_ctx(
        response=None,
        accumulated_text="hello world!",  # 12 chars
        model="gpt-4o-mini",
    )

    await cost_emitter.emit(EmitPhase.POST_SESSION, ctx)

    cost_events = [e for e in captured if e.topic == "cost.ledger.recorded"]
    assert len(cost_events) == 1
    ev = cost_events[0]
    assert ev.payload["input_tokens"] == 0
    assert ev.payload["output_tokens"] == 3
    assert ev.payload["cents"] == 1


@pytest.mark.asyncio
async def test_streaming_path_with_empty_text_emits_nothing():
    """Zero usage → no bus pollution (per emitter docstring contract)."""
    ctx, captured = await _make_ctx(
        response=None,
        accumulated_text="",
        model="gpt-4o-mini",
    )

    await cost_emitter.emit(EmitPhase.POST_SESSION, ctx)

    cost_events = [e for e in captured if e.topic == "cost.ledger.recorded"]
    assert cost_events == []


# ---------------------------------------------------------------------------
# 5. Pricing helpers — longest-prefix wins, unknown model falls back.
# ---------------------------------------------------------------------------


def test_price_for_longest_prefix_match():
    # ``claude-3-5-sonnet`` must beat the shorter ``claude-3-sonnet`` row.
    assert _price_for("claude-3-5-sonnet-20241022") == (300, 1500)
    assert _price_for("claude-3-sonnet-old") == (300, 1500)
    assert _price_for("claude-3-haiku-20240307") == (25, 125)


def test_price_for_unknown_model_uses_default():
    in_rate, out_rate = _price_for("zephyr-mystery-7b")
    # Defaults are non-zero per the docstring contract.
    assert in_rate > 0
    assert out_rate > 0


def test_compute_cents_floors_zero_usage():
    assert _compute_cents(0, 0, "gpt-4o-mini") == 0


def test_compute_cents_rounds_up_subcent_usage():
    # 1 token at 15c/M is 0.000015 cent — must round to 1.
    assert _compute_cents(1, 0, "gpt-4o-mini") == 1


# ---------------------------------------------------------------------------
# 6. Integration — full pipeline.run produces a BusObservation.
# ---------------------------------------------------------------------------


def _make_litellm_completion(
    text: str = "hello back",
    *,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


@pytest.mark.asyncio
async def test_pipeline_run_surfaces_cost_observation():
    from robit.proxy import pipeline, upstream

    req = _req(model="gpt-4o-mini")

    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(
            return_value=_make_litellm_completion(
                model="gpt-4o-mini",
                prompt_tokens=10_000,
                completion_tokens=5_000,
            )
        ),
    ):
        result = await pipeline.run(req)

    # No veto path — must be a PipelineResult.
    assert hasattr(result, "fired")
    cost_obs = [ob for ob in result.fired if ob.topic == "cost.ledger.recorded"]
    assert len(cost_obs) == 1, (
        f"expected 1 cost.ledger.recorded observation; got: "
        f"{[ob.topic for ob in result.fired]}"
    )
    ob = cost_obs[0]
    assert ob.source == "proxy-pipeline"
    # ``cents`` is dropped by the recorder whitelist; ``score`` carries it.
    assert ob.payload_summary.get("score", 0) > 0
    # Sanity: 10_000 input + 5_000 output at gpt-4o-mini rates
    # (15c, 60c per M) = 0.15 + 0.30 = 0.45 cents → ceiling → 1 cent.
    assert int(ob.payload_summary["score"]) == 1


@pytest.mark.asyncio
async def test_pipeline_run_accumulates_across_requests_without_crashing():
    """Two back-to-back requests with the same model both surface observations.

    The cost-ledger engine carries per-session state internally; the proxy
    builds a fresh bus + orchestrator per request, so accumulation is
    bounded to within a single request.  This test asserts only that
    repeated invocation does not crash and each request produces its own
    cost observation.
    """
    from robit.proxy import pipeline, upstream

    req = _req(model="claude-3-5-sonnet-20241022")

    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(
            return_value=_make_litellm_completion(
                model="claude-3-5-sonnet-20241022",
                prompt_tokens=10_000,
                completion_tokens=5_000,
            )
        ),
    ):
        r1 = await pipeline.run(req)
        r2 = await pipeline.run(req)

    for r in (r1, r2):
        cost_obs = [ob for ob in r.fired if ob.topic == "cost.ledger.recorded"]
        assert len(cost_obs) == 1
        assert int(cost_obs[0].payload_summary["score"]) == 11


# ---------------------------------------------------------------------------
# 7. HTTP integration — X-Enchanter-Cost-Cents header on a real response.
# ---------------------------------------------------------------------------


async def _post_raw(
    host: str,
    port: int,
    path: str,
    body: bytes,
) -> tuple[int, dict[str, str], bytes]:
    """Reuse the raw HTTP helper from tests/proxy/test_proxy_server.py."""
    header_lines = [
        f"POST {path} HTTP/1.1",
        f"Host: {host}:{port}",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
        "Connection: close",
    ]
    raw = ("\r\n".join(header_lines) + "\r\n\r\n").encode("latin-1") + body

    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(raw)
        await writer.drain()

        status_line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
        parts = status_line.split(" ", 2)
        status = int(parts[1]) if len(parts) >= 2 else 0

        headers: dict[str, str] = {}
        while True:
            line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
            if not line:
                break
            k, _, v = line.partition(":")
            headers[k.strip().lower()] = v.strip()

        resp_body = await reader.read()
        return status, headers, resp_body
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.asyncio
async def test_proxy_server_surfaces_x_enchanter_cost_cents_header():
    from robit.proxy import upstream
    from robit.proxy.server import ProxyServer

    server = ProxyServer(host="127.0.0.1", port=0)
    host, port = await server.start()
    task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0)

    try:
        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")
        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(
                return_value=_make_litellm_completion(
                    text="hello back",
                    model="claude-3-5-sonnet-20241022",
                    prompt_tokens=10_000,
                    completion_tokens=5_000,
                )
            ),
        ):
            status, headers, _resp_body = await _post_raw(
                host, port, "/v1/messages", body
            )
        assert status == 200
        assert "x-enchanter-cost-cents" in headers, (
            f"missing cost header; got: {sorted(headers.keys())}"
        )
        # Must parse to a positive int.
        cents = int(headers["x-enchanter-cost-cents"])
        assert cents > 0
        # Sanity: same math as above → 11 cents.
        assert cents == 11
    finally:
        await server.close()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
