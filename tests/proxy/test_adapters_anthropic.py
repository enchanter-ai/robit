"""Tests for robit.proxy.adapters.anthropic — wire-format adapter."""

from __future__ import annotations

import json
import re
from typing import AsyncIterator

import pytest

from robit.proxy.adapters.anthropic import AdapterParseError, AnthropicAdapter
from robit.proxy.canonical import (
    CanonicalChunk,
    CanonicalResponse,
    CanonicalUsage,
    Message,
    TextPart,
    ToolResultPart,
    ToolUsePart,
)


# ---------------------------------------------------------------------------
# Routing.
# ---------------------------------------------------------------------------


def test_matches_post_v1_messages():
    assert AnthropicAdapter.matches("POST", "/v1/messages") is True
    assert AnthropicAdapter.matches("POST", "/v1/messages?beta=1") is True
    assert AnthropicAdapter.matches("GET", "/v1/messages") is False
    assert AnthropicAdapter.matches("POST", "/v1/completions") is False


def test_paths_class_var_lists_owned_routes():
    assert "/v1/messages" in AnthropicAdapter.paths


# ---------------------------------------------------------------------------
# parse_request.
# ---------------------------------------------------------------------------


def _body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def test_parse_request_normalises_plain_string_content():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    req = AnthropicAdapter.parse_request(body, "/v1/messages", {})
    assert req.model == "claude-3-5-sonnet-20241022"
    assert req.max_tokens == 256
    assert len(req.messages) == 1
    msg = req.messages[0]
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextPart)
    assert msg.content[0].text == "Hi"


def test_parse_request_list_of_parts_with_tool_use_and_tool_result():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "system": "Be brief.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's the weather?"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "Sunny.",
                            "is_error": False,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        },
                    ],
                },
            ],
        }
    )
    req = AnthropicAdapter.parse_request(body, "/v1/messages", {})
    assert req.system == "Be brief."
    assert len(req.messages) == 2

    user_msg = req.messages[0]
    assert user_msg.role == "user"
    assert isinstance(user_msg.content[0], TextPart)
    assert isinstance(user_msg.content[1], ToolResultPart)
    assert user_msg.content[1].tool_use_id == "toolu_123"
    assert user_msg.content[1].content == "Sunny."
    assert user_msg.content[1].is_error is False

    asst_msg = req.messages[1]
    assert isinstance(asst_msg.content[1], ToolUsePart)
    assert asst_msg.content[1].id == "toolu_123"
    assert asst_msg.content[1].name == "get_weather"
    assert asst_msg.content[1].input == {"city": "NYC"}


def test_parse_request_missing_max_tokens_raises():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    with pytest.raises(AdapterParseError, match="max_tokens"):
        AnthropicAdapter.parse_request(body, "/v1/messages", {})


def test_parse_request_tool_choice_auto():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "auto"},
        }
    )
    req = AnthropicAdapter.parse_request(body, "/v1/messages", {})
    assert req.tool_choice == "auto"


def test_parse_request_tool_choice_any():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "any"},
        }
    )
    req = AnthropicAdapter.parse_request(body, "/v1/messages", {})
    assert req.tool_choice == "any"


def test_parse_request_tool_choice_specific_tool():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
    )
    req = AnthropicAdapter.parse_request(body, "/v1/messages", {})
    assert req.tool_choice == {"type": "tool", "name": "get_weather"}


def test_parse_request_tool_choice_missing_is_none():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    req = AnthropicAdapter.parse_request(body, "/v1/messages", {})
    assert req.tool_choice is None


def test_parse_request_invalid_json_raises():
    with pytest.raises(AdapterParseError, match="invalid JSON"):
        AnthropicAdapter.parse_request(b"{not json", "/v1/messages", {})


def test_parse_request_propagates_optional_params():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.95,
            "stop_sequences": ["\n\n"],
            "stream": True,
            "metadata": {"user_id": "u1"},
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "input_schema": {"type": "object"},
                }
            ],
        }
    )
    req = AnthropicAdapter.parse_request(body, "/v1/messages", {})
    assert req.temperature == pytest.approx(0.7)
    assert req.top_p == pytest.approx(0.95)
    assert req.stop_sequences == ("\n\n",)
    assert req.stream is True
    assert req.metadata == {"user_id": "u1"}
    assert len(req.tools) == 1
    assert req.tools[0].name == "get_weather"
    assert req.tools[0].input_schema == {"type": "object"}


def test_parse_request_system_block_list_flattens_to_string():
    body = _body(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "system": [
                {"type": "text", "text": "Be "},
                {"type": "text", "text": "brief."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    req = AnthropicAdapter.parse_request(body, "/v1/messages", {})
    assert req.system == "Be brief."


# ---------------------------------------------------------------------------
# render_response.
# ---------------------------------------------------------------------------


def test_render_response_plain_text():
    resp = CanonicalResponse(
        model="claude-3-5-sonnet-20241022",
        content=(TextPart(text="Hi there!"),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=12, output_tokens=34),
    )
    body = AnthropicAdapter.render_response(resp)
    payload = json.loads(body)
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["model"] == "claude-3-5-sonnet-20241022"
    assert payload["stop_reason"] == "end_turn"
    assert payload["stop_sequence"] is None
    assert payload["content"] == [{"type": "text", "text": "Hi there!"}]
    assert payload["usage"] == {"input_tokens": 12, "output_tokens": 34}


def test_render_response_tool_use_with_stop_reason_tool_use():
    resp = CanonicalResponse(
        model="claude-3-5-sonnet-20241022",
        content=(
            TextPart(text="Let me check."),
            ToolUsePart(id="toolu_456", name="search", input={"q": "foo"}),
        ),
        stop_reason="tool_use",
        usage=CanonicalUsage(input_tokens=12, output_tokens=20),
    )
    body = AnthropicAdapter.render_response(resp)
    payload = json.loads(body)
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"] == [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_456",
            "name": "search",
            "input": {"q": "foo"},
        },
    ]


def test_render_response_synthesises_id_of_correct_shape():
    resp = CanonicalResponse(
        model="claude-3-5-sonnet-20241022",
        content=(TextPart(text="hi"),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=1, output_tokens=1),
    )
    body = AnthropicAdapter.render_response(resp)
    payload = json.loads(body)
    assert re.fullmatch(r"msg_[0-9a-f]{8}", payload["id"])


# ---------------------------------------------------------------------------
# render_stream.
# ---------------------------------------------------------------------------


async def _collect(it: AsyncIterator[bytes]) -> list[bytes]:
    out = []
    async for item in it:
        out.append(item)
    return out


def _parse_sse_block(raw: bytes) -> tuple[str, dict]:
    """Parse one SSE block into (event, data-dict)."""
    text = raw.decode("utf-8")
    assert text.endswith("\n\n"), f"missing SSE terminator: {text!r}"
    lines = text.rstrip("\n").split("\n")
    event = None
    data = None
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data = json.loads(line[len("data: ") :])
    assert event is not None
    assert data is not None
    return event, data


async def _to_async_iter(items: list[CanonicalChunk]) -> AsyncIterator[CanonicalChunk]:
    for item in items:
        yield item


async def test_render_stream_full_text_only_sequence():
    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(type="content_block_start", index=0),
        CanonicalChunk(type="text_delta", index=0, text="Hi"),
        CanonicalChunk(type="text_delta", index=0, text=" there"),
        CanonicalChunk(type="content_block_stop", index=0),
        CanonicalChunk(
            type="message_delta",
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=12, output_tokens=34),
        ),
        CanonicalChunk(type="message_stop"),
    ]
    out = await _collect(AnthropicAdapter.render_stream(_to_async_iter(chunks)))
    events = [_parse_sse_block(b) for b in out]
    names = [e[0] for e in events]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # content_block_start carries a text shell.
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}
    # First delta is text_delta with "Hi".
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "Hi"}
    # message_delta carries stop_reason + usage.
    assert events[5][1]["delta"]["stop_reason"] == "end_turn"
    assert events[5][1]["usage"] == {"input_tokens": 12, "output_tokens": 34}


async def test_render_stream_tool_use_uses_input_json_delta():
    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(type="content_block_start", index=0),
        CanonicalChunk(
            type="input_json_delta", index=0, partial_json='{"city":'
        ),
        CanonicalChunk(
            type="input_json_delta", index=0, partial_json='"NYC"}'
        ),
        CanonicalChunk(type="content_block_stop", index=0),
        CanonicalChunk(type="message_delta", stop_reason="tool_use"),
        CanonicalChunk(type="message_stop"),
    ]
    out = await _collect(AnthropicAdapter.render_stream(_to_async_iter(chunks)))
    events = [_parse_sse_block(b) for b in out]
    names = [e[0] for e in events]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # Block start was promoted to tool_use shell when first delta arrived.
    assert events[1][1]["content_block"]["type"] == "tool_use"
    # Both deltas carry input_json_delta wire shape.
    assert events[2][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"city":',
    }
    assert events[3][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '"NYC"}',
    }
    assert events[5][1]["delta"]["stop_reason"] == "tool_use"


async def test_render_stream_empty_block_emits_text_shell_on_close():
    """A content_block_start followed immediately by content_block_stop
    (no deltas in between) is resolved as an empty text block.  This is
    the conservative default — see comment in render_stream."""
    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(type="content_block_start", index=0),
        CanonicalChunk(type="content_block_stop", index=0),
        CanonicalChunk(type="message_stop"),
    ]
    out = await _collect(AnthropicAdapter.render_stream(_to_async_iter(chunks)))
    events = [_parse_sse_block(b) for b in out]
    names = [e[0] for e in events]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_stop",
    ]
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}


async def test_render_stream_bytes_are_valid_sse_blocks():
    """Each yielded bytes object is a complete event/data/blank triple."""
    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(type="message_stop"),
    ]
    out = await _collect(AnthropicAdapter.render_stream(_to_async_iter(chunks)))
    for block in out:
        text = block.decode("utf-8")
        assert text.startswith("event: ")
        assert "\ndata: " in text
        assert text.endswith("\n\n")


async def test_render_stream_tool_use_block_kind_carries_real_id_and_name():
    """When the canonical content_block_start carries
    block_kind='tool_use' with tool_id and tool_name, the Anthropic SSE
    event emits those values verbatim in content_block.id / .name
    (Wave 1 contract-fix)."""
    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(
            type="content_block_start",
            index=0,
            block_kind="tool_use",
            tool_id="toolu_X",
            tool_name="my_tool",
        ),
        CanonicalChunk(
            type="input_json_delta", index=0, partial_json='{"a":1}'
        ),
        CanonicalChunk(type="content_block_stop", index=0),
        CanonicalChunk(type="message_delta", stop_reason="tool_use"),
        CanonicalChunk(type="message_stop"),
    ]
    out = await _collect(AnthropicAdapter.render_stream(_to_async_iter(chunks)))
    events = [_parse_sse_block(b) for b in out]
    start_event = next(e for e in events if e[0] == "content_block_start")
    assert start_event[1]["content_block"]["type"] == "tool_use"
    assert start_event[1]["content_block"]["id"] == "toolu_X"
    assert start_event[1]["content_block"]["name"] == "my_tool"
