"""Tests for enchanter.proxy.streaming — StreamAccumulator + tee_stream."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from enchanter.proxy.canonical import CanonicalChunk
from enchanter.proxy.streaming import (
    SecretSanitizingStream,
    StreamAccumulator,
    tee_stream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _aiter(chunks) -> AsyncIterator[CanonicalChunk]:
    for c in chunks:
        yield c


def _text(text: str, index: int = 0) -> CanonicalChunk:
    return CanonicalChunk(type="text_delta", index=index, text=text)


def _json(fragment: str, index: int = 0) -> CanonicalChunk:
    return CanonicalChunk(type="input_json_delta", index=index, partial_json=fragment)


# ---------------------------------------------------------------------------
# StreamAccumulator unit tests
# ---------------------------------------------------------------------------


def test_accumulator_appends_text_chunks():
    acc = StreamAccumulator()
    acc.feed(_text("hello "))
    acc.feed(_text("world"))
    assert acc.text == "hello world"
    assert not acc.truncated


def test_accumulator_appends_partial_json_chunks():
    """input_json_delta chunks (tool-call streaming) are also captured."""
    acc = StreamAccumulator()
    acc.feed(_json('{"q":"'))
    acc.feed(_json("hi"))
    acc.feed(_json('"}'))
    assert acc.text == '{"q":"hi"}'
    assert not acc.truncated


def test_accumulator_cap_truncates_and_drops_subsequent_feeds():
    """After cap is hit, truncated=True and feed() becomes a no-op."""
    acc = StreamAccumulator(cap_bytes=10)
    acc.feed(_text("0123456789ABCDEF"))  # 16 bytes > 10-byte cap.
    assert acc.truncated
    assert acc.text == "0123456789"  # exactly cap_bytes worth

    # Subsequent feeds are dropped.
    acc.feed(_text("more"))
    assert acc.text == "0123456789"
    assert acc.truncated


def test_accumulator_cap_exactly_at_boundary():
    """Filling to exactly the cap does not (yet) flip truncated; the next
    write does."""
    acc = StreamAccumulator(cap_bytes=5)
    acc.feed(_text("hello"))  # exactly 5 bytes.
    # Not truncated yet — we wrote exactly cap.
    assert acc.text == "hello"
    # The next byte trips truncation.
    acc.feed(_text("!"))
    assert acc.truncated
    assert acc.text == "hello"


def test_accumulator_ignores_non_payload_chunks():
    """message_start / content_block_start / message_stop carry no text."""
    acc = StreamAccumulator()
    acc.feed(CanonicalChunk(type="message_start"))
    acc.feed(CanonicalChunk(type="content_block_start", index=0, block_kind="text"))
    acc.feed(CanonicalChunk(type="content_block_stop", index=0))
    acc.feed(CanonicalChunk(type="message_stop"))
    assert acc.text == ""
    assert not acc.truncated


# ---------------------------------------------------------------------------
# tee_stream integration tests
# ---------------------------------------------------------------------------


async def test_tee_stream_yields_each_chunk_exactly_once():
    chunks = [_text("a"), _text("b"), _text("c")]
    acc = StreamAccumulator()
    received: list[str] = []
    async for chunk in tee_stream(_aiter(chunks), acc):
        received.append(chunk.text or "")
    assert received == ["a", "b", "c"]
    assert acc.text == "abc"


async def test_tee_stream_feeds_accumulator_in_lockstep_with_yield():
    """Each chunk must be in the accumulator no later than the yield itself.

    We assert this by capturing the accumulator's text *immediately after*
    each yield — the chunk just yielded must already be reflected.
    """
    chunks = [_text("a"), _text("b"), _text("c")]
    acc = StreamAccumulator()
    snapshots: list[str] = []
    async for chunk in tee_stream(_aiter(chunks), acc):
        # The chunk just yielded must already be in the accumulator.
        snapshots.append(acc.text)
    assert snapshots == ["a", "ab", "abc"]


async def test_tee_stream_handles_empty_upstream():
    """No chunks → generator exits cleanly, accumulator stays empty."""
    acc = StreamAccumulator()
    received = []
    async for chunk in tee_stream(_aiter([]), acc):
        received.append(chunk)
    assert received == []
    assert acc.text == ""
    assert not acc.truncated


# ---------------------------------------------------------------------------
# SecretSanitizingStream — mid-stream secret redaction tests
# ---------------------------------------------------------------------------


async def test_sanitizer_empty_stream_yields_nothing_and_records_no_redactions():
    """Empty source -> no chunks emitted, redactions stay empty."""
    sanitizer = SecretSanitizingStream(buffer_bytes=64)
    received = []
    async for chunk in sanitizer.wrap(_aiter([])):
        received.append(chunk)
    assert received == []
    assert sanitizer.redactions == []


async def test_sanitizer_short_clean_stream_passes_through_at_final_flush():
    """A stream with no secret and total bytes <= buffer_bytes flushes
    everything verbatim at the final-flush step. No redactions."""
    sanitizer = SecretSanitizingStream(buffer_bytes=64)
    chunks = [_text("hello "), _text("world")]
    received = []
    async for chunk in sanitizer.wrap(_aiter(chunks)):
        received.append(chunk)
    text_deltas = [c for c in received if c.type == "text_delta"]
    joined = "".join(c.text or "" for c in text_deltas)
    assert joined == "hello world"
    assert sanitizer.redactions == []


async def test_sanitizer_redacts_single_secret_in_middle_of_stream():
    """An AWS key in the middle of streamed text is replaced with the
    pattern's redaction string; the pattern id is recorded."""
    sanitizer = SecretSanitizingStream(buffer_bytes=64)
    secret = "AKIAIOSFODNN7EXAMPLE"
    chunks = [
        _text(f"prefix text {secret} more text "),
        _text("X" * 200),
    ]
    received = []
    async for chunk in sanitizer.wrap(_aiter(chunks)):
        received.append(chunk)
    joined = "".join(c.text or "" for c in received if c.type == "text_delta")
    assert secret not in joined
    assert "AKIA****[REDACTED]" in joined
    assert "s-aws-key" in sanitizer.redactions


async def test_sanitizer_catches_secret_spanning_two_chunks():
    """The whole point of the rolling buffer: a secret broken across chunk
    boundaries must still be caught."""
    sanitizer = SecretSanitizingStream(buffer_bytes=64)
    chunks = [
        _text("leading content AKIAIOSF"),
        _text("ODNN7EXAMPLE trailing content " + "X" * 200),
    ]
    received = []
    async for chunk in sanitizer.wrap(_aiter(chunks)):
        received.append(chunk)
    joined = "".join(c.text or "" for c in received if c.type == "text_delta")
    assert "AKIAIOSFODNN7EXAMPLE" not in joined
    assert "AKIA****[REDACTED]" in joined
    assert "s-aws-key" in sanitizer.redactions


async def test_sanitizer_secret_at_very_end_caught_by_final_flush():
    """Secret entirely within the rolling buffer at stream exhaustion is
    still caught by the final flush."""
    sanitizer = SecretSanitizingStream(buffer_bytes=512)
    secret = "AKIAIOSFODNN7EXAMPLE"
    chunks = [
        _text(f"the leaked key is {secret}"),
        CanonicalChunk(type="message_stop"),
    ]
    received = []
    async for chunk in sanitizer.wrap(_aiter(chunks)):
        received.append(chunk)
    joined = "".join(c.text or "" for c in received if c.type == "text_delta")
    assert secret not in joined
    assert "AKIA****[REDACTED]" in joined
    assert "s-aws-key" in sanitizer.redactions
    # message_stop must still be yielded after the final flush.
    assert received[-1].type == "message_stop"


async def test_sanitizer_per_index_buffers_dont_bleed_across_content_blocks():
    """Two different secrets in two different content-block indices are
    redacted independently - the buffers don't merge."""
    sanitizer = SecretSanitizingStream(buffer_bytes=64)
    secret_a = "AKIAIOSFODNN7EXAMPLE"
    secret_b = "AKIAIOSFODNN7EXOTHER"
    chunks = [
        _text(f"block 0 has {secret_a} ", index=0),
        _text("X" * 200, index=0),
        _text(f"block 1 has {secret_b} ", index=1),
        _text("Y" * 200, index=1),
    ]
    received = []
    async for chunk in sanitizer.wrap(_aiter(chunks)):
        received.append(chunk)
    by_index: dict[int, str] = {}
    for c in received:
        if c.type == "text_delta":
            by_index.setdefault(c.index, "")
            by_index[c.index] += c.text or ""
    assert secret_a not in by_index.get(0, "")
    assert secret_b not in by_index.get(1, "")
    assert "AKIA****[REDACTED]" in by_index[0]
    assert "AKIA****[REDACTED]" in by_index[1]
    # Two redactions recorded (one per block).
    assert sanitizer.redactions.count("s-aws-key") == 2
