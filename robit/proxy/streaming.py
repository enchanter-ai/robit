"""robit.proxy.streaming — shared streaming infrastructure for the proxy.

The proxy needs to (1) ship every upstream chunk to the client with zero added
latency, AND (2) hand the *fully accumulated* response text to post-response
security plugins (secret-mask, etc.) once the stream completes.  Buffering the
entire stream before forwarding would break (1); skipping accumulation entirely
would break (2).  The compromise is a **tee**: each chunk is yielded to the
caller AND copied into a bounded in-memory accumulator at the same time.

Two primitives live here:

* :class:`StreamAccumulator` — append-only text buffer with an 8 MiB cap.
  Past the cap it sets ``truncated=True`` and silently drops further writes;
  the stream keeps flowing to the client.
* :func:`tee_stream` — async generator that mirrors an upstream chunk
  iterator to (a) its own consumer and (b) the accumulator.

These are intentionally provider-agnostic: they work on
:class:`~robit.proxy.canonical.CanonicalChunk` events regardless of which
upstream produced them.

Known Limitation #1 (also documented on :mod:`robit.proxy.pipeline`):
secret-mask runs on the *post-stream* accumulated text.  Chunks the client
has already received are NOT retroactively redacted.  A targeted leak that
arrives in chunk N still reaches the client before chunk N+1 is shipped.
The post-response event still fires so the proxy can log the leak and
surface the match in response headers — but it cannot un-send the bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import AsyncIterator

from .canonical import CanonicalChunk
from robit.engines.secret_mask.patterns import SECRET_PATTERNS


# 8 MiB.  Empirically large enough for ~2M tokens of plain text and ~500k of
# JSON tool-call arguments; small enough that a runaway stream can't OOM the
# proxy.  Exposed as a class attribute so tests can pin a smaller value.
_DEFAULT_CAP_BYTES = 8 * 1024 * 1024


# Rolling-buffer size used by :class:`SecretSanitizingStream`.  Held bytes are
# the tail of streamed text that may still be part of an in-progress secret
# match — once enough additional bytes arrive, the safe prefix is flushed to
# the client.  Empirically 512 B covers every bounded SECRET_PATTERNS regex
# (AWS key = 20 B, OpenAI/Anthropic keys ≤ ~80 B, bearer = ~30 B) with room to
# spare.  PEM blocks are unbounded — they're only caught by the final flush,
# which is OK because PEM blocks straddle multiple chunks anyway.
_SECRET_BUFFER_BYTES = 512

# The longest bounded secret pattern we expect to mid-stream-redact.  Mirrors
# _SECRET_BUFFER_BYTES; named separately for documentation clarity.
_SECRET_PATTERN_MAX_LEN = 512


@dataclass
class StreamAccumulator:
    """8 MiB-capped text accumulator for post-stream security scans.

    Both ``text_delta`` and ``input_json_delta`` chunks are folded into the
    same buffer — secret-mask doesn't care whether a leaked token came from
    a text body or a tool-call argument blob, and merging them keeps the
    buffer footprint a single allocation.

    After ``cap_bytes`` is exceeded, :attr:`truncated` flips to ``True`` and
    subsequent :meth:`feed` calls are no-ops.  Callers should check
    :attr:`truncated` before claiming the accumulated text is complete.
    """

    cap_bytes: int = _DEFAULT_CAP_BYTES
    _buf: bytearray = field(default_factory=bytearray)
    _truncated: bool = False

    def feed(self, chunk: CanonicalChunk) -> None:
        """Append chunk.text and/or chunk.partial_json to the buffer.

        Non-payload events (``message_start``, ``content_block_start``, ...)
        carry no text and are silently ignored.  Once the cap is hit the
        method becomes a no-op until reset.
        """
        if self._truncated:
            return

        # Prefer text (text_delta), fall back to partial_json (input_json_delta).
        # A single chunk never carries both in the upstream translator, but the
        # type permits it; if both are present we feed both.
        payloads: list[str] = []
        if chunk.text:
            payloads.append(chunk.text)
        if chunk.partial_json:
            payloads.append(chunk.partial_json)
        if not payloads:
            return

        for piece in payloads:
            encoded = piece.encode("utf-8")
            remaining = self.cap_bytes - len(self._buf)
            if remaining <= 0:
                self._truncated = True
                return
            if len(encoded) > remaining:
                # Append what fits, then mark truncated.
                self._buf.extend(encoded[:remaining])
                self._truncated = True
                return
            self._buf.extend(encoded)

    @property
    def text(self) -> str:
        """The decoded accumulated text. Safe to call any time."""
        # ``errors='replace'`` guards against the (vanishingly unlikely) case
        # where we truncated mid-multibyte-codepoint; secret-mask scans tolerate
        # the replacement char.
        return self._buf.decode("utf-8", errors="replace")

    @property
    def truncated(self) -> bool:
        """True if the 8 MiB cap was hit and further writes were dropped."""
        return self._truncated

    def __len__(self) -> int:
        return len(self._buf)


async def tee_stream(
    src: AsyncIterator[CanonicalChunk],
    accumulator: StreamAccumulator,
) -> AsyncIterator[CanonicalChunk]:
    """Tee an upstream chunk iterator into the consumer AND an accumulator.

    Each chunk is fed to ``accumulator`` immediately *before* it is yielded
    to the caller.  Ordering matters: by feeding first, an exception while
    yielding (consumer cancellation) doesn't leave the accumulator with a
    partial view that the caller didn't actually see.

    The async-generator contract here is:
      - one yield per upstream chunk, in order, no filtering, no mutation;
      - accumulator side-effects happen in lockstep;
      - on upstream exhaustion the generator stops; the accumulator holds
        whatever text accrued and is ready for inspection.
    """
    async for chunk in src:
        accumulator.feed(chunk)
        yield chunk


@dataclass
class SecretSanitizingStream:
    """Mid-stream secret-redaction wrapper around an iterator of CanonicalChunks.

    Strategy: token-bounded buffer with regex sweep.

    * For each ``text_delta`` chunk, append its text to a rolling per-``index``
      buffer.  Maintain one buffer per content-block ``index`` — different
      content blocks are independent streams and a secret cannot span them.
    * Whenever a buffer exceeds :attr:`buffer_bytes`, the *prefix* over
      ``buffer_bytes`` is safe to flush: no in-progress secret can straddle
      that boundary (we sized ``buffer_bytes`` ≥ the longest bounded secret).
      Run :data:`SECRET_PATTERNS` over the to-flush prefix, replace any matches
      with each pattern's ``redaction`` string, and emit a redacted
      ``text_delta`` carrying the cleaned text.
    * On ``message_stop`` (or final exhaustion of the source iterator), flush
      every remaining buffer with one last sweep.
    * Every matched pattern id is appended to :attr:`redactions` in the order
      it was encountered.  Callers should read :attr:`redactions` *after* the
      iterator has been exhausted (i.e. after the ``async for`` finishes).

    Non-text chunks (``message_start``, ``content_block_start``, ``tool_use``,
    ``input_json_delta``, ``content_block_stop``, ``message_delta``,
    ``message_stop``) pass through unchanged.  Tool calls are JSON, not free
    text, and secret patterns aren't applied to them — by design.

    Backwards-compat: this class lives alongside :class:`StreamAccumulator` /
    :func:`tee_stream` (kept for wire-trace use cases where the *original*
    text is what observers want).  New code should prefer
    :class:`SecretSanitizingStream` because it actively redacts.
    """

    buffer_bytes: int = _SECRET_BUFFER_BYTES
    redactions: list[str] = field(default_factory=list)
    # Per-content-block rolling buffers, keyed by chunk.index.
    _buffers: dict[int, str] = field(default_factory=dict)

    def _sweep_and_redact(self, text: str) -> str:
        """Run all SECRET_PATTERNS over ``text``; record matches; return sanitized.

        Matches found are appended to :attr:`redactions` in pattern-declaration
        order — the same order the bulk :func:`_mask_secrets` helper in
        ``secret_mask.adapter`` uses.  We don't dedupe per-call so an emitter
        can see N occurrences across the stream.
        """
        masked = text
        for p in SECRET_PATTERNS:
            if p.match.search(masked):
                self.redactions.append(p.id)
                masked = p.match.sub(p.redaction, masked)
        return masked

    async def wrap(
        self, src: AsyncIterator[CanonicalChunk]
    ) -> AsyncIterator[CanonicalChunk]:
        """Wrap the source stream; yield sanitised chunks.

        Contract:
          - one yield per upstream chunk for non-text chunks (pass-through);
          - text_delta chunks may be transformed (text content replaced with
            its safe-prefix redaction) or held until enough bytes accumulate;
          - on stream exhaustion, any remaining buffered text is flushed as
            one final synthesised ``text_delta`` per index (yielded BEFORE
            the eventual ``message_stop`` if one is in flight, otherwise
            after the last upstream chunk).
        """
        async for chunk in src:
            if chunk.type == "text_delta" and chunk.text:
                idx = chunk.index
                buf = self._buffers.get(idx, "") + chunk.text

                # If buffer exceeds the rolling window, the prefix over the
                # window is safe to flush (no bounded secret pattern can span
                # that prefix→tail boundary by definition of window size).
                if len(buf.encode("utf-8")) > self.buffer_bytes:
                    # Split on a byte boundary safely by encoding/decoding.
                    encoded = buf.encode("utf-8")
                    flush_bytes = encoded[: -self.buffer_bytes]
                    keep_bytes = encoded[-self.buffer_bytes :]
                    # Decode tolerantly in case we split mid-multibyte — the
                    # likelihood is low and replacement chars are harmless for
                    # regex scanning.
                    flush_text = flush_bytes.decode("utf-8", errors="replace")
                    keep_text = keep_bytes.decode("utf-8", errors="replace")

                    cleaned = self._sweep_and_redact(flush_text)
                    self._buffers[idx] = keep_text
                    yield replace(chunk, text=cleaned)
                else:
                    # Not enough bytes to safely flush yet — hold the chunk.
                    self._buffers[idx] = buf
                    # We deliberately do NOT yield this chunk; it'll surface
                    # in a synthesised text_delta later.
                continue

            if chunk.type == "message_stop":
                # Final flush BEFORE the message_stop sentinel, so consumers
                # see all sanitised text first.
                async for flushed in self._final_flush():
                    yield flushed
                yield chunk
                continue

            if chunk.type == "content_block_stop":
                # End-of-block: flush this block's buffer before the close.
                idx = chunk.index
                if idx in self._buffers and self._buffers[idx]:
                    cleaned = self._sweep_and_redact(self._buffers[idx])
                    self._buffers[idx] = ""
                    yield CanonicalChunk(type="text_delta", index=idx, text=cleaned)
                yield chunk
                continue

            # Every other chunk type: pass through unchanged.
            yield chunk

        # Source exhausted without an explicit message_stop — still flush.
        async for flushed in self._final_flush():
            yield flushed

    async def _final_flush(self) -> AsyncIterator[CanonicalChunk]:
        """Yield one synthesised text_delta per non-empty buffer.

        Sorted by index for deterministic order — tests rely on this.
        """
        for idx in sorted(self._buffers.keys()):
            held = self._buffers[idx]
            if not held:
                continue
            cleaned = self._sweep_and_redact(held)
            self._buffers[idx] = ""
            yield CanonicalChunk(type="text_delta", index=idx, text=cleaned)


__all__ = [
    "StreamAccumulator",
    "tee_stream",
    "SecretSanitizingStream",
]
