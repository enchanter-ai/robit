"""Tests for enchanter.proxy.adapters.openai — wire-format translation."""

from __future__ import annotations

import json
import re

import pytest

from enchanter.proxy.adapters.openai import AdapterParseError, OpenAIAdapter
from enchanter.proxy.canonical import (
    CanonicalChunk,
    CanonicalResponse,
    CanonicalUsage,
    TextPart,
    ToolResultPart,
    ToolUsePart,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _parse(body: dict) -> object:
    return OpenAIAdapter.parse_request(
        json.dumps(body).encode("utf-8"),
        "/v1/chat/completions",
        {},
    )


def _decode_sse(line: bytes) -> dict | str:
    """Decode one ``data: …\\n\\n`` line into either a dict or the [DONE] sentinel."""
    text = line.decode("utf-8")
    assert text.startswith("data: "), f"bad SSE line: {text!r}"
    assert text.endswith("\n\n"), f"missing SSE terminator: {text!r}"
    payload = text[len("data: ") : -2]
    if payload == "[DONE]":
        return "[DONE]"
    return json.loads(payload)


async def _collect_stream(chunks: list[CanonicalChunk]) -> list[bytes]:
    async def gen():
        for c in chunks:
            yield c

    out: list[bytes] = []
    async for raw in OpenAIAdapter.render_stream(gen()):
        out.append(raw)
    return out


# ---------------------------------------------------------------------------
# Routing.
# ---------------------------------------------------------------------------


def test_matches_post_chat_completions():
    assert OpenAIAdapter.matches("POST", "/v1/chat/completions") is True


def test_matches_strips_query_string():
    assert OpenAIAdapter.matches("POST", "/v1/chat/completions?debug=1") is True


def test_matches_rejects_other_method_and_path():
    assert OpenAIAdapter.matches("GET", "/v1/chat/completions") is False
    assert OpenAIAdapter.matches("POST", "/v1/messages") is False


def test_paths_classvar_present():
    assert OpenAIAdapter.paths == ("/v1/chat/completions",)


# ---------------------------------------------------------------------------
# parse_request — basic shape.
# ---------------------------------------------------------------------------


def test_parse_request_minimal_string_content():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    assert req.model == "gpt-4o-mini"
    assert req.system is None
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == (TextPart(text="Hi"),)
    assert req.stream is False
    assert req.stop_sequences == ()


def test_parse_request_list_content_normalises_text_parts():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's "},
                        {"type": "text", "text": "the weather?"},
                    ],
                }
            ],
        }
    )
    parts = req.messages[0].content
    assert parts == (TextPart(text="What's "), TextPart(text="the weather?"))


def test_parse_request_drops_image_url_parts_silently():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look:"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                }
            ],
        }
    )
    assert req.messages[0].content == (TextPart(text="Look:"),)


def test_parse_request_invalid_json_body():
    with pytest.raises(AdapterParseError):
        OpenAIAdapter.parse_request(b"not-json", "/v1/chat/completions", {})


def test_parse_request_missing_model():
    with pytest.raises(AdapterParseError):
        _parse({"messages": [{"role": "user", "content": "hi"}]})


def test_parse_request_missing_messages():
    with pytest.raises(AdapterParseError):
        _parse({"model": "gpt-4o-mini"})


# ---------------------------------------------------------------------------
# parse_request — system handling.
# ---------------------------------------------------------------------------


def test_parse_request_system_message_at_index_zero():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
    )
    assert req.system == "You are helpful."
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"


def test_parse_request_multiple_system_messages_concatenated():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "system", "content": "Cite sources."},
                {"role": "user", "content": "Hi"},
            ],
        }
    )
    assert req.system == "Be brief.\n\nCite sources."


def test_parse_request_system_message_after_user_rejected():
    with pytest.raises(AdapterParseError):
        _parse(
            {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "system", "content": "stale system"},
                ],
            }
        )


def test_parse_request_system_with_list_content_flattens():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Part A. "},
                        {"type": "text", "text": "Part B."},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
        }
    )
    assert req.system == "Part A. Part B."


# ---------------------------------------------------------------------------
# parse_request — tool_calls + tool messages.
# ---------------------------------------------------------------------------


def test_parse_request_assistant_with_tool_calls():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "Checking…",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"NYC"}',
                            },
                        }
                    ],
                },
            ],
        }
    )
    assistant = req.messages[1]
    assert assistant.role == "assistant"
    assert assistant.content[0] == TextPart(text="Checking…")
    assert assistant.content[1] == ToolUsePart(
        id="call_abc", name="get_weather", input={"city": "NYC"}
    )


def test_parse_request_assistant_tool_calls_with_null_content():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
            ],
        }
    )
    assistant = req.messages[1]
    assert len(assistant.content) == 1
    assert isinstance(assistant.content[0], ToolUsePart)
    assert assistant.content[0].input == {}


def test_parse_request_tool_role_message_becomes_tool_result_part():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "tool",
                    "tool_call_id": "call_abc",
                    "content": "Sunny.",
                },
            ],
        }
    )
    tool_msg = req.messages[1]
    assert tool_msg.role == "tool"
    assert tool_msg.content == (
        ToolResultPart(tool_use_id="call_abc", content="Sunny.", is_error=False),
    )


def test_parse_request_invalid_tool_call_arguments_raises():
    with pytest.raises(AdapterParseError):
        _parse(
            {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "not-json-{{",
                                },
                            }
                        ],
                    },
                ],
            }
        )


def test_parse_request_tool_role_requires_tool_call_id():
    with pytest.raises(AdapterParseError):
        _parse(
            {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "tool", "content": "result"},
                ],
            }
        )


# ---------------------------------------------------------------------------
# parse_request — tools, tool_choice, stop, user.
# ---------------------------------------------------------------------------


def test_parse_request_tools_list():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Lookup weather.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    assert len(req.tools) == 1
    assert req.tools[0].name == "get_weather"
    assert req.tools[0].description == "Lookup weather."
    assert req.tools[0].input_schema == {"type": "object"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "any"),
    ],
)
def test_parse_request_tool_choice_string_variants(raw, expected):
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": raw,
        }
    )
    assert req.tool_choice == expected


def test_parse_request_tool_choice_function_dict():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        }
    )
    assert req.tool_choice == {"type": "tool", "name": "get_weather"}


def test_parse_request_stop_string_to_tuple():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "\n\n",
        }
    )
    assert req.stop_sequences == ("\n\n",)


def test_parse_request_stop_list_to_tuple():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["\n\n", "END"],
        }
    )
    assert req.stop_sequences == ("\n\n", "END")


def test_parse_request_user_into_metadata():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "user-42",
        }
    )
    assert req.metadata == {"user": "user-42"}


def test_parse_request_temperature_top_p_max_tokens_stream():
    req = _parse(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 256,
            "stream": True,
        }
    )
    assert req.temperature == 0.5
    assert req.top_p == 0.9
    assert req.max_tokens == 256
    assert req.stream is True


# ---------------------------------------------------------------------------
# render_response.
# ---------------------------------------------------------------------------


def test_render_response_plain_text():
    resp = CanonicalResponse(
        model="gpt-4o-mini",
        content=(TextPart(text="Hi there!"),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=12, output_tokens=34),
    )
    body = json.loads(OpenAIAdapter.render_response(resp))
    assert body["model"] == "gpt-4o-mini"
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hi there!"
    assert "tool_calls" not in body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "total_tokens": 46,
    }


def test_render_response_tool_calls_null_content():
    resp = CanonicalResponse(
        model="gpt-4o-mini",
        content=(
            ToolUsePart(id="call_x", name="get_weather", input={"city": "NYC"}),
        ),
        stop_reason="tool_use",
        usage=CanonicalUsage(input_tokens=10, output_tokens=5),
    )
    body = json.loads(OpenAIAdapter.render_response(resp))
    msg = body["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"] == [
        {
            "id": "call_x",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
        }
    ]
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_render_response_mixed_text_and_tool_use():
    resp = CanonicalResponse(
        model="gpt-4o-mini",
        content=(
            TextPart(text="Let me check."),
            ToolUsePart(id="call_x", name="get_weather", input={"city": "NYC"}),
        ),
        stop_reason="tool_use",
        usage=CanonicalUsage(input_tokens=10, output_tokens=5),
    )
    body = json.loads(OpenAIAdapter.render_response(resp))
    msg = body["choices"][0]["message"]
    assert msg["content"] == "Let me check."
    assert len(msg["tool_calls"]) == 1


def test_render_response_id_and_created_shape():
    resp = CanonicalResponse(
        model="gpt-4o-mini",
        content=(TextPart(text="ok"),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=1, output_tokens=1),
    )
    body = json.loads(OpenAIAdapter.render_response(resp))
    assert re.fullmatch(r"chatcmpl-[0-9a-f]{8}", body["id"])
    assert isinstance(body["created"], int)
    assert body["created"] > 0


@pytest.mark.parametrize(
    "canonical_stop,expected",
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("stop_sequence", "stop"),
        ("tool_use", "tool_calls"),
        (None, "stop"),
    ],
)
def test_render_response_stop_reason_mapping(canonical_stop, expected):
    resp = CanonicalResponse(
        model="gpt-4o-mini",
        content=(TextPart(text="x"),),
        stop_reason=canonical_stop,
        usage=CanonicalUsage(input_tokens=1, output_tokens=1),
    )
    body = json.loads(OpenAIAdapter.render_response(resp))
    assert body["choices"][0]["finish_reason"] == expected


# ---------------------------------------------------------------------------
# render_stream — SSE.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_stream_text_delta_sequence():
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
    raw_lines = await _collect_stream(chunks)
    decoded = [_decode_sse(line) for line in raw_lines]

    # First chunk: assistant role.
    assert decoded[0]["choices"][0]["delta"] == {"role": "assistant"}
    # Two text-delta chunks.
    assert decoded[1]["choices"][0]["delta"] == {"content": "Hi"}
    assert decoded[2]["choices"][0]["delta"] == {"content": " there"}
    # message_delta → finish_reason + usage.
    final = decoded[3]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "total_tokens": 46,
    }
    # Final sentinel.
    assert decoded[4] == "[DONE]"


@pytest.mark.asyncio
async def test_render_stream_done_terminator_present():
    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(type="message_stop"),
    ]
    raw_lines = await _collect_stream(chunks)
    assert raw_lines[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_render_stream_tool_call_sequence():
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
    raw_lines = await _collect_stream(chunks)
    decoded = [_decode_sse(line) for line in raw_lines]

    # message_start, opening tool_calls envelope, two arg deltas,
    # message_delta with finish_reason=tool_calls, [DONE]
    assert decoded[0]["choices"][0]["delta"] == {"role": "assistant"}

    open_delta = decoded[1]["choices"][0]["delta"]
    assert "tool_calls" in open_delta
    first_tc = open_delta["tool_calls"][0]
    assert first_tc["index"] == 0
    assert first_tc["type"] == "function"
    assert re.fullmatch(r"call_[0-9a-f]+", first_tc["id"])

    # Arg fragments.
    assert decoded[2]["choices"][0]["delta"]["tool_calls"][0]["function"][
        "arguments"
    ] == '{"city":'
    assert decoded[3]["choices"][0]["delta"]["tool_calls"][0]["function"][
        "arguments"
    ] == '"NYC"}'

    # finish + DONE
    assert decoded[4]["choices"][0]["finish_reason"] == "tool_calls"
    assert decoded[5] == "[DONE]"


@pytest.mark.asyncio
async def test_render_stream_chunks_share_id():
    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(type="text_delta", index=0, text="hi"),
        CanonicalChunk(type="message_delta", stop_reason="end_turn"),
        CanonicalChunk(type="message_stop"),
    ]
    raw_lines = await _collect_stream(chunks)
    decoded = [_decode_sse(line) for line in raw_lines if _decode_sse(line) != "[DONE]"]
    ids = {d["id"] for d in decoded}
    assert len(ids) == 1
    only_id = next(iter(ids))
    assert re.fullmatch(r"chatcmpl-[0-9a-f]{8}", only_id)
    assert all(d["object"] == "chat.completion.chunk" for d in decoded)


@pytest.mark.asyncio
async def test_render_stream_tool_use_block_kind_emits_real_id_and_name():
    """When content_block_start carries block_kind='tool_use' with
    tool_id and tool_name, the OpenAI SSE chunk emits those values
    verbatim — not a synthesised call_<hex> id, not an empty name
    (Wave 1 contract-fix)."""
    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(
            type="content_block_start",
            index=0,
            block_kind="tool_use",
            tool_id="call_real",
            tool_name="real_function",
        ),
        CanonicalChunk(
            type="input_json_delta", index=0, partial_json='{"q":1}'
        ),
        CanonicalChunk(type="content_block_stop", index=0),
        CanonicalChunk(type="message_delta", stop_reason="tool_use"),
        CanonicalChunk(type="message_stop"),
    ]
    raw_lines = await _collect_stream(chunks)
    decoded = [
        _decode_sse(line) for line in raw_lines if _decode_sse(line) != "[DONE]"
    ]
    # Find the chunk that opens the tool_calls envelope (carries id + name).
    opens = [
        d
        for d in decoded
        if isinstance(d, dict)
        and d["choices"][0]["delta"].get("tool_calls")
        and d["choices"][0]["delta"]["tool_calls"][0].get("id") is not None
    ]
    assert len(opens) == 1
    tc = opens[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["id"] == "call_real"
    assert tc["function"]["name"] == "real_function"
    # Must NOT have synthesised an id of shape call_<hex>.
    assert not re.fullmatch(r"call_[0-9a-f]{12}", tc["id"])
