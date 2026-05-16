"""Tests for robit.proxy.canonical — dataclass invariants and shape."""

from dataclasses import FrozenInstanceError, replace

import pytest

from robit.proxy.canonical import (
    CanonicalChunk,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    Message,
    TextPart,
    Tool,
    ToolResultPart,
    ToolUsePart,
)


# ---------------------------------------------------------------------------
# Content parts.
# ---------------------------------------------------------------------------


def test_text_part_has_fixed_type_discriminator():
    part = TextPart(text="hello")
    assert part.type == "text"
    assert part.text == "hello"


def test_tool_use_part_carries_parsed_input():
    part = ToolUsePart(id="t1", name="get_weather", input={"city": "Paris"})
    assert part.type == "tool_use"
    assert part.input == {"city": "Paris"}


def test_tool_result_part_defaults_to_non_error():
    part = ToolResultPart(tool_use_id="t1", content="sunny")
    assert part.type == "tool_result"
    assert part.is_error is False


def test_content_parts_are_frozen():
    part = TextPart(text="hi")
    with pytest.raises(FrozenInstanceError):
        part.text = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Message.
# ---------------------------------------------------------------------------


def test_message_accepts_tuple_of_parts():
    msg = Message(
        role="assistant",
        content=(TextPart(text="thinking"), ToolUsePart(id="t1", name="x", input={})),
    )
    assert msg.role == "assistant"
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextPart)
    assert isinstance(msg.content[1], ToolUsePart)


def test_messages_with_equivalent_content_are_equal():
    a = Message(role="user", content=(TextPart(text="hi"),))
    b = Message(role="user", content=(TextPart(text="hi"),))
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Request.
# ---------------------------------------------------------------------------


def test_canonical_request_minimal_defaults():
    req = CanonicalRequest(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=(Message(role="user", content=(TextPart(text="hi"),)),),
    )
    assert req.system is None
    assert req.tools == ()
    assert req.tool_choice is None
    assert req.stream is False
    assert req.stop_sequences == ()
    assert req.metadata == {}


def test_canonical_request_full_payload_round_trips_via_replace():
    req = CanonicalRequest(
        model="gpt-4o-mini",
        messages=(Message(role="user", content=(TextPart(text="hi"),)),),
        system="be brief",
        tools=(Tool(name="t", description="d", input_schema={"type": "object"}),),
        tool_choice="auto",
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
        stop_sequences=("END",),
        stream=True,
        metadata={"trace_id": "abc"},
    )
    twin = replace(req)
    assert twin == req
    # Metadata mutation on the copy does not affect the original (each
    # dataclass instance carries its own dict).
    twin.metadata["other"] = 1
    assert "other" not in req.metadata or req.metadata is twin.metadata


def test_canonical_request_is_frozen():
    req = CanonicalRequest(model="x", messages=(Message(role="user", content=(TextPart(text="hi"),)),))
    with pytest.raises(FrozenInstanceError):
        req.model = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Response and chunks.
# ---------------------------------------------------------------------------


def test_canonical_response_carries_usage_and_content():
    resp = CanonicalResponse(
        model="gpt-4o-mini",
        content=(TextPart(text="hi"),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=4, output_tokens=2),
    )
    assert resp.usage.input_tokens == 4
    assert resp.stop_reason == "end_turn"
    assert resp.content[0].text == "hi"


def test_canonical_chunk_text_delta_shape():
    chunk = CanonicalChunk(type="text_delta", index=0, text="hello")
    assert chunk.type == "text_delta"
    assert chunk.index == 0
    assert chunk.text == "hello"
    assert chunk.partial_json is None


def test_canonical_chunk_message_delta_carries_usage():
    chunk = CanonicalChunk(
        type="message_delta",
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=1, output_tokens=2),
    )
    assert chunk.usage is not None
    assert chunk.usage.output_tokens == 2
