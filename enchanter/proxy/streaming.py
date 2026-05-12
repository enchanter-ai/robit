"""enchanter.proxy.streaming — shared streaming infrastructure for the proxy.

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
:class:`~enchanter.proxy.canonical.CanonicalChunk` events regardless of which
upstream produced them.

Known Limitation #1 (also documented on :mod:`enchanter.proxy.pipeline`):
secret-mask runs on the *post-stream* accumulated text.  Chunks the client
has already received are NOT retroactively redacted.  A targeted leak that
arrives in chunk N still reaches the client before chunk N+1 is shipped.
The post-response event still fires so the proxy can log the leak and
surface the match in response headers — but it cannot un-send the bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator

from .canonical import CanonicalChunk


# 8 MiB.  Empirically large enough for ~2M tokens of plain text and ~500k of
# JSON tool-call arguments; small enough that a runaway stream can't OOM the
# proxy.  Exposed as a class attribute so tests can pin a smaller value.
_DEFAULT_CAP_BYTES = 8 * 1024 * 1024


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


__all__ = [
    "StreamAccumulator",
    "tee_stream",
]
