"""Tests for enchanter.proxy.streaming — StreamAccumulator + tee_stream."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from enchanter.proxy.canonical import CanonicalChunk
from enchanter.proxy.streaming import StreamAccumulator, tee_stream


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
