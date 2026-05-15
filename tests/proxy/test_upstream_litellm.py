"""Tests for enchanter.proxy.upstream — LiteLLM bridge with mocked acompletion."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from enchanter.proxy import upstream
from enchanter.proxy.canonical import (
    CanonicalRequest,
    Message,
    TextPart,
    Tool,
    ToolResultPart,
    ToolUsePart,
)
from enchanter.proxy.upstream import UpstreamError, call_upstream, stream_upstream


# ---------------------------------------------------------------------------
# Helpers for fabricating LiteLLM-shaped responses.
# ---------------------------------------------------------------------------


def _make_completion(
    text: str | None = "hi",
    *,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
):
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _make_tool_call(call_id: str, name: str, arguments: str):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, function=function, type="function")


def _make_chunk(
    *,
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
    usage=None,
):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice], usage=usage)


class _AsyncStream:
    """A trivial async iterator over a list of fabricated chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _basic_req(**overrides):
    base = dict(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text="hello"),)),),
    )
    base.update(overrides)
    return CanonicalRequest(**base)


# ---------------------------------------------------------------------------
# Non-streaming completion.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_completion_returns_canonical_response():
    fake = _make_completion(text="hello there", finish_reason="stop")
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=fake),
    ) as mocked:
        resp = await call_upstream(_basic_req())

    assert mocked.await_count == 1
    kwargs = mocked.await_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["stream"] is False
    assert kwargs["messages"][0]["role"] == "user"

    assert resp.model == "gpt-4o-mini"
    assert len(resp.content) == 1
    assert resp.content[0].text == "hello there"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_completion_with_tool_use_response_maps_to_tool_use_part():
    fake = _make_completion(
        text=None,
        tool_calls=[
            _make_tool_call("call_1", "get_weather", '{"city": "Paris"}'),
        ],
        finish_reason="tool_calls",
    )
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=fake),
    ):
        resp = await call_upstream(_basic_req())

    assert len(resp.content) == 1
    part = resp.content[0]
    assert isinstance(part, ToolUsePart)
    assert part.id == "call_1"
    assert part.name == "get_weather"
    assert part.input == {"city": "Paris"}
    assert resp.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_completion_with_tool_result_input_expands_to_tool_role_messages():
    """A canonical user message containing a tool_result becomes a `tool`-role
    OpenAI message in the LiteLLM payload."""
    req = _basic_req(
        messages=(
            Message(
                role="user",
                content=(ToolResultPart(tool_use_id="call_1", content="sunny"),),
            ),
        ),
    )
    fake = _make_completion(text="ok")
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=fake),
    ) as mocked:
        await call_upstream(req)

    payload = mocked.await_args.kwargs["messages"]
    assert payload[0]["role"] == "tool"
    assert payload[0]["tool_call_id"] == "call_1"
    assert payload[0]["content"] == "sunny"


@pytest.mark.asyncio
async def test_request_with_tools_and_system_renders_correctly():
    req = _basic_req(
        system="be brief",
        tools=(Tool(name="t", description="d", input_schema={"type": "object"}),),
        tool_choice="any",
        temperature=0.1,
        top_p=0.7,
        max_tokens=64,
        stop_sequences=("END",),
    )
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_make_completion()),
    ) as mocked:
        await call_upstream(req)

    kwargs = mocked.await_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "be brief"}
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["function"]["name"] == "t"
    # "any" lowers to OpenAI's "required".
    assert kwargs["tool_choice"] == "required"
    assert kwargs["temperature"] == 0.1
    assert kwargs["top_p"] == 0.7
    assert kwargs["max_tokens"] == 64
    assert kwargs["stop"] == ["END"]


# ---------------------------------------------------------------------------
# Streaming completion.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_of_text_deltas_emits_lifecycle_events():
    chunks = [
        _make_chunk(content="hel"),
        _make_chunk(content="lo"),
        _make_chunk(finish_reason="stop"),
    ]
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_AsyncStream(chunks)),
    ):
        out = [c async for c in stream_upstream(_basic_req(stream=True))]

    types = [c.type for c in out]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert types.count("text_delta") == 2
    assert "content_block_stop" in types
    assert types[-1] == "message_stop"

    # The two text_delta events carry the right fragments in order.
    text_chunks = [c for c in out if c.type == "text_delta"]
    assert [c.text for c in text_chunks] == ["hel", "lo"]


@pytest.mark.asyncio
async def test_stream_finish_reason_maps_to_canonical_stop_reason():
    chunks = [
        _make_chunk(content="done"),
        _make_chunk(finish_reason="length"),
    ]
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_AsyncStream(chunks)),
    ):
        out = [c async for c in stream_upstream(_basic_req(stream=True))]

    message_deltas = [c for c in out if c.type == "message_delta"]
    assert len(message_deltas) == 1
    assert message_deltas[0].stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_stream_with_tool_call_deltas_opens_tool_block():
    tool_delta = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="get_weather", arguments='{"city":'),
    )
    tool_delta_2 = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="get_weather", arguments='"Paris"}'),
    )
    chunks = [
        _make_chunk(tool_calls=[tool_delta]),
        _make_chunk(tool_calls=[tool_delta_2]),
        _make_chunk(finish_reason="tool_calls"),
    ]
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_AsyncStream(chunks)),
    ):
        out = [c async for c in stream_upstream(_basic_req(stream=True))]

    json_deltas = [c for c in out if c.type == "input_json_delta"]
    assert [c.partial_json for c in json_deltas] == ['{"city":', '"Paris"}']
    message_delta = [c for c in out if c.type == "message_delta"][0]
    assert message_delta.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_stream_text_only_content_block_start_carries_block_kind_text():
    """A text-only stream's content_block_start carries block_kind='text'
    with no tool_id / tool_name (Wave 1 contract-fix)."""
    chunks = [
        _make_chunk(content="hello"),
        _make_chunk(finish_reason="stop"),
    ]
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_AsyncStream(chunks)),
    ):
        out = [c async for c in stream_upstream(_basic_req(stream=True))]

    starts = [c for c in out if c.type == "content_block_start"]
    assert len(starts) == 1
    assert starts[0].block_kind == "text"
    assert starts[0].tool_id is None
    assert starts[0].tool_name is None


@pytest.mark.asyncio
async def test_stream_single_tool_call_content_block_start_carries_id_and_name():
    """A single tool-call stream emits one content_block_start with
    block_kind='tool_use' and tool_id/tool_name populated from the
    upstream's first tool-call delta, *before* the corresponding
    input_json_delta (Wave 1 contract-fix)."""
    tool_delta = SimpleNamespace(
        index=0,
        id="call_abc",
        function=SimpleNamespace(name="search_web", arguments='{"q":"x"}'),
    )
    chunks = [
        _make_chunk(tool_calls=[tool_delta]),
        _make_chunk(finish_reason="tool_calls"),
    ]
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_AsyncStream(chunks)),
    ):
        out = [c async for c in stream_upstream(_basic_req(stream=True))]

    starts = [c for c in out if c.type == "content_block_start"]
    assert len(starts) == 1
    assert starts[0].block_kind == "tool_use"
    assert starts[0].tool_id == "call_abc"
    assert starts[0].tool_name == "search_web"

    # Ordering: content_block_start MUST precede input_json_delta on
    # the same index — so adapters can ship authentic metadata on the
    # opening wire event.
    types = [c.type for c in out]
    start_pos = types.index("content_block_start")
    delta_pos = types.index("input_json_delta")
    assert start_pos < delta_pos


@pytest.mark.asyncio
async def test_stream_mixed_text_then_tool_call_emits_two_starts_with_kinds():
    """A stream that emits text first, then a tool call, produces two
    content_block_start events: one with block_kind='text' (no tool
    metadata) and one with block_kind='tool_use' carrying the right
    id/name."""
    tool_delta = SimpleNamespace(
        index=0,
        id="call_mix",
        function=SimpleNamespace(name="compute", arguments='{"n":1}'),
    )
    chunks = [
        _make_chunk(content="thinking..."),
        _make_chunk(tool_calls=[tool_delta]),
        _make_chunk(finish_reason="tool_calls"),
    ]
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_AsyncStream(chunks)),
    ):
        out = [c async for c in stream_upstream(_basic_req(stream=True))]

    starts = [c for c in out if c.type == "content_block_start"]
    assert len(starts) == 2
    # The text block is opened first.
    assert starts[0].block_kind == "text"
    assert starts[0].tool_id is None
    assert starts[0].tool_name is None
    # Then the tool_use block.
    assert starts[1].block_kind == "tool_use"
    assert starts[1].tool_id == "call_mix"
    assert starts[1].tool_name == "compute"
    # They use distinct indices.
    assert starts[0].index != starts[1].index


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class _FakeProviderError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@pytest.mark.asyncio
async def test_upstream_error_wraps_provider_exception():
    boom = _FakeProviderError("rate limited", 429)
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(side_effect=boom),
    ):
        with pytest.raises(UpstreamError) as ei:
            await call_upstream(_basic_req(model="anthropic/claude-3-5-sonnet-20241022"))

    err = ei.value
    assert err.provider == "anthropic"
    assert err.status == 429
    assert "rate limited" in err.message


# ---------------------------------------------------------------------------
# Pass-through auth — Wave 16.1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passthrough_auth_absent_metadata_adds_no_extra_kwargs():
    """Without a stashed auth blob, kwargs match the legacy shape and
    LiteLLM resolves credentials from env vars as before."""
    fake = _make_completion(text="ok")
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=fake),
    ) as mocked:
        await call_upstream(_basic_req())

    kwargs = mocked.await_args.kwargs
    assert "api_key" not in kwargs
    assert "extra_headers" not in kwargs


@pytest.mark.asyncio
async def test_passthrough_auth_anthropic_api_key_forwards_api_key_kwarg():
    req = _basic_req(
        model="anthropic/claude-3-5-sonnet-20241022",
        metadata={
            "_enchanter_passthrough_auth": {
                "kind": "anthropic-api-key",
                "value": "sk-ant-test-123",
            }
        },
    )
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_make_completion()),
    ) as mocked:
        await call_upstream(req)

    kwargs = mocked.await_args.kwargs
    assert kwargs["api_key"] == "sk-ant-test-123"
    assert "extra_headers" not in kwargs
    # The internal sentinel must not leak into the LiteLLM metadata bag.
    assert "metadata" not in kwargs or (
        "_enchanter_passthrough_auth" not in kwargs.get("metadata", {})
    )


@pytest.mark.asyncio
async def test_passthrough_auth_openai_bearer_forwards_api_key_kwarg():
    req = _basic_req(
        model="gpt-4o-mini",
        metadata={
            "_enchanter_passthrough_auth": {
                "kind": "openai-bearer",
                "value": "sk-openai-test-xyz",
            }
        },
    )
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_make_completion()),
    ) as mocked:
        await call_upstream(req)

    kwargs = mocked.await_args.kwargs
    assert kwargs["api_key"] == "sk-openai-test-xyz"
    assert "extra_headers" not in kwargs


@pytest.mark.asyncio
async def test_passthrough_auth_gemini_api_key_forwards_api_key_kwarg():
    req = _basic_req(
        model="gemini/gemini-1.5-pro",
        metadata={
            "_enchanter_passthrough_auth": {
                "kind": "gemini-api-key",
                "value": "AIza-test-gemini-key",
            }
        },
    )
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_make_completion()),
    ) as mocked:
        await call_upstream(req)

    kwargs = mocked.await_args.kwargs
    assert kwargs["api_key"] == "AIza-test-gemini-key"
    assert "extra_headers" not in kwargs


@pytest.mark.asyncio
async def test_passthrough_auth_anthropic_oauth_sets_extra_headers_bearer():
    """Anthropic OAuth bearer tokens go on extra_headers; LiteLLM still
    requires an api_key kwarg, so we supply a placeholder.

    # TODO: verify LiteLLM extra_headers acceptance across versions.
    """
    req = _basic_req(
        model="anthropic/claude-3-5-sonnet-20241022",
        metadata={
            "_enchanter_passthrough_auth": {
                "kind": "anthropic-oauth",
                "value": "sk-ant-oat-XXXX",
            }
        },
    )
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(return_value=_make_completion()),
    ) as mocked:
        await call_upstream(req)

    kwargs = mocked.await_args.kwargs
    assert kwargs["api_key"] == "sk-ant-placeholder"
    assert kwargs["extra_headers"] == {
        "Authorization": "Bearer sk-ant-oat-XXXX"
    }


@pytest.mark.asyncio
async def test_upstream_error_falls_back_to_unknown_provider():
    boom = RuntimeError("nope")
    with patch.object(
        upstream.litellm,
        "acompletion",
        new=AsyncMock(side_effect=boom),
    ):
        with pytest.raises(UpstreamError) as ei:
            await call_upstream(_basic_req(model="weird-private-model"))
    assert ei.value.provider == "unknown"
    assert ei.value.status is None
