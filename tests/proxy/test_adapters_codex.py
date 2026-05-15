"""Tests for enchanter.proxy.adapters.codex — Responses-API wire translation."""

from __future__ import annotations

import json

import pytest

from enchanter.proxy.adapters.codex import AdapterParseError, CodexAdapter
from enchanter.proxy.canonical import (
    CanonicalChunk,
    CanonicalResponse,
    CanonicalUsage,
    TextPart,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _parse(body: dict) -> object:
    return CodexAdapter.parse_request(
        json.dumps(body).encode("utf-8"),
        "/v1/responses",
        {},
    )


def _decode_sse(raw: bytes) -> tuple[str, dict]:
    """Decode one Responses-API SSE event into (event_name, json_payload)."""
    text = raw.decode("utf-8")
    assert text.endswith("\n\n"), f"missing terminator: {text!r}"
    head, _, _blank = text.rpartition("\n\n")
    event_line, _, data_line = head.partition("\n")
    assert event_line.startswith("event: "), f"bad event line: {event_line!r}"
    assert data_line.startswith("data: "), f"bad data line: {data_line!r}"
    return event_line[len("event: ") :], json.loads(data_line[len("data: ") :])


async def _collect_stream(chunks: list[CanonicalChunk]) -> list[bytes]:
    async def gen():
        for c in chunks:
            yield c

    out: list[bytes] = []
    async for raw in CodexAdapter.render_stream(gen()):
        out.append(raw)
    return out


# ---------------------------------------------------------------------------
# Routing.
# ---------------------------------------------------------------------------


def test_matches_post_v1_responses():
    assert CodexAdapter.matches("POST", "/v1/responses") is True


def test_matches_strips_query_string():
    assert CodexAdapter.matches("POST", "/v1/responses?stream=1") is True


def test_matches_rejects_get():
    assert CodexAdapter.matches("GET", "/v1/responses") is False


def test_matches_rejects_chat_completions():
    assert CodexAdapter.matches("POST", "/v1/chat/completions") is False


def test_paths_classvar_present():
    assert CodexAdapter.paths == ("/v1/responses",)


# ---------------------------------------------------------------------------
# parse_request — happy paths.
# ---------------------------------------------------------------------------


def test_parse_request_minimal_body():
    req = _parse(
        {
            "model": "gpt-5-codex",
            "instructions": "You are helpful.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
    )
    assert req.model == "gpt-5-codex"
    assert req.system == "You are helpful."
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == (TextPart(text="Hello"),)
    assert req.stream is False


def test_parse_request_developer_role_collapses_into_system():
    req = _parse(
        {
            "model": "gpt-5-codex",
            "instructions": "Top instructions.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Dev policy."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                },
            ],
        }
    )
    # Developer text is folded into system block (concatenated after
    # top-level instructions). The user message is the only conversation
    # entry.
    assert req.system == "Top instructions.\n\nDev policy."
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"


def test_parse_request_stream_flag_round_trips():
    req = _parse(
        {
            "model": "gpt-5-codex",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                }
            ],
            "stream": True,
        }
    )
    assert req.stream is True


def test_parse_request_no_instructions_ok():
    # The proxy should not be stricter than upstream — instructions is
    # optional at the wire-translation layer.
    req = _parse(
        {
            "model": "gpt-5-codex",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                }
            ],
        }
    )
    assert req.system is None


def test_parse_request_max_output_tokens_maps_to_max_tokens():
    req = _parse(
        {
            "model": "gpt-5-codex",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                }
            ],
            "max_output_tokens": 256,
            "temperature": 0.7,
        }
    )
    assert req.max_tokens == 256
    assert req.temperature == 0.7


# ---------------------------------------------------------------------------
# parse_request — error paths.
# ---------------------------------------------------------------------------


def test_parse_request_missing_model_raises():
    with pytest.raises(AdapterParseError):
        _parse(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hi"}],
                    }
                ],
            }
        )


def test_parse_request_invalid_json_raises():
    with pytest.raises(AdapterParseError):
        CodexAdapter.parse_request(b"{not json", "/v1/responses", {})


def test_parse_request_empty_input_raises():
    with pytest.raises(AdapterParseError):
        _parse({"model": "gpt-5-codex", "input": []})


def test_parse_request_unknown_role_raises():
    with pytest.raises(AdapterParseError):
        _parse(
            {
                "model": "gpt-5-codex",
                "input": [
                    {
                        "type": "message",
                        "role": "robot",
                        "content": [{"type": "input_text", "text": "Hi"}],
                    }
                ],
            }
        )


# ---------------------------------------------------------------------------
# render_response.
# ---------------------------------------------------------------------------


def test_render_response_builds_responses_api_shape():
    resp = CanonicalResponse(
        model="gpt-5-codex",
        content=(TextPart(text="Hello, world."),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=10, output_tokens=5),
    )
    raw = CodexAdapter.render_response(resp)
    body = json.loads(raw)
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "gpt-5-codex"
    assert body["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["role"] == "assistant"
    assert body["output"][0]["content"][0] == {
        "type": "output_text",
        "text": "Hello, world.",
    }


# ---------------------------------------------------------------------------
# render_stream.
# ---------------------------------------------------------------------------


def test_render_stream_emits_text_delta_events():
    import asyncio

    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(
            type="content_block_start", index=0, block_kind="text"
        ),
        CanonicalChunk(type="text_delta", index=0, text="Hello"),
        CanonicalChunk(type="text_delta", index=0, text=", world"),
        CanonicalChunk(type="content_block_stop", index=0),
        CanonicalChunk(
            type="message_delta",
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=3, output_tokens=4),
        ),
        CanonicalChunk(type="message_stop"),
    ]
    raw = asyncio.run(_collect_stream(chunks))
    events = [_decode_sse(r) for r in raw]

    names = [e[0] for e in events]
    assert names[0] == "response.created"
    assert "response.output_text.delta" in names
    assert names[-1] == "response.completed"

    deltas = [
        payload["delta"]
        for (name, payload) in events
        if name == "response.output_text.delta"
    ]
    assert deltas == ["Hello", ", world"]


def test_render_stream_final_event_carries_usage():
    import asyncio

    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(
            type="content_block_start", index=0, block_kind="text"
        ),
        CanonicalChunk(type="text_delta", index=0, text="ok"),
        CanonicalChunk(type="content_block_stop", index=0),
        CanonicalChunk(
            type="message_delta",
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=2, output_tokens=1),
        ),
        CanonicalChunk(type="message_stop"),
    ]
    raw = asyncio.run(_collect_stream(chunks))
    events = [_decode_sse(r) for r in raw]
    final_name, final_payload = events[-1]
    assert final_name == "response.completed"
    usage = final_payload["response"]["usage"]
    assert usage == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }
    assert final_payload["response"]["status"] == "completed"


def test_render_stream_drops_tool_call_argument_deltas():
    """v1 limitation: input_json_delta chunks do not emit SSE events."""
    import asyncio

    chunks = [
        CanonicalChunk(type="message_start"),
        CanonicalChunk(
            type="content_block_start",
            index=0,
            block_kind="tool_use",
            tool_id="call_abc",
            tool_name="lookup",
        ),
        CanonicalChunk(
            type="input_json_delta", index=0, partial_json='{"q":'
        ),
        CanonicalChunk(
            type="input_json_delta", index=0, partial_json='"hi"}'
        ),
        CanonicalChunk(type="content_block_stop", index=0),
        CanonicalChunk(type="message_stop"),
    ]
    raw = asyncio.run(_collect_stream(chunks))
    names = [_decode_sse(r)[0] for r in raw]
    # Only response.created + response.completed survive — no text deltas,
    # no function_call_arguments.delta (v1 cuts that).
    assert names == ["response.created", "response.completed"]
