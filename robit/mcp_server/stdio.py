"""Server-side stdio transport: read JSON-RPC lines from a StreamReader,
write responses to a StreamWriter.

Mirrors the framing rules of the client-side transport (newline-delimited
UTF-8 JSON, 8 MiB body cap, no embedded newlines in outgoing messages).

The default ``serve()`` wires the server's own ``sys.stdin`` / ``sys.stdout``
through asyncio stream readers, so the server can be spawned as a child
subprocess by any MCP client (Claude Code, the existing client transport in
``robit.transport.stdio``, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

#: 8 MiB — FM-5 body cap, mirrors the client transport.
PER_MESSAGE_BODY_MAX_BYTES: int = 8 * 1024 * 1024


class BodyTooLargeError(Exception):
    def __init__(self, bytes_seen: int) -> None:
        super().__init__(
            f"stdio server line exceeded cap "
            f"({bytes_seen} > {PER_MESSAGE_BODY_MAX_BYTES} bytes)"
        )
        self.bytes_seen = bytes_seen


# Dispatcher signature: raw string in, optional response string out.
RawHandler = Callable[[str], Awaitable[str | None]]


class ServerStdioTransport:
    """Read newline-framed JSON-RPC, hand each line to a dispatcher coroutine."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handler: RawHandler,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._handler = handler
        self._closed = False

    async def serve(self) -> None:
        """Read lines forever; respond to each as the handler returns."""
        while not self._closed:
            try:
                line_bytes = await self._read_line_capped()
            except BodyTooLargeError as exc:
                # Drop the connection — oversized client; cannot continue safely.
                logger.warning("ServerStdioTransport: %s", exc)
                return

            if line_bytes is None:
                return  # EOF

            line = line_bytes.decode("utf-8", errors="replace")
            try:
                response = await self._handler(line)
            except Exception:  # noqa: BLE001
                logger.exception("ServerStdioTransport: handler raised")
                continue

            if response is None:
                continue

            payload = response.encode("utf-8") + b"\n"
            if len(payload) > PER_MESSAGE_BODY_MAX_BYTES:
                logger.error(
                    "ServerStdioTransport: outgoing message %d > cap",
                    len(payload),
                )
                continue

            try:
                self._writer.write(payload)
                await self._writer.drain()
            except (BrokenPipeError, ConnectionResetError):
                return

    async def close(self) -> None:
        self._closed = True
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass

    async def _read_line_capped(self) -> bytes | None:
        """Read up to ``\\n``, enforcing the 8 MiB cap."""
        chunks: list[bytes] = []
        total = 0

        while True:
            try:
                chunk = await self._reader.read(65536)
            except Exception:  # noqa: BLE001
                return None

            if not chunk:
                return None

            nl_pos = chunk.find(b"\n")
            if nl_pos != -1:
                before = chunk[:nl_pos]
                remainder = chunk[nl_pos + 1:]
                total += len(before)
                if total > PER_MESSAGE_BODY_MAX_BYTES:
                    raise BodyTooLargeError(total)
                chunks.append(before)
                if remainder:
                    self._reader.feed_data(remainder)
                line = b"".join(chunks)
                if not line:
                    # empty line — skip, read the next one
                    return await self._read_line_capped()
                return line

            total += len(chunk)
            if total > PER_MESSAGE_BODY_MAX_BYTES:
                raise BodyTooLargeError(total)
            chunks.append(chunk)


async def attach_to_sys_streams() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Wire ``sys.stdin`` + ``sys.stdout`` into asyncio stream pairs.

    Uses the loop's ``connect_read_pipe`` / ``connect_write_pipe`` APIs.
    On Windows where these are unavailable for stdin, falls back to running
    a synchronous reader in the default executor.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)

    try:
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        transport, _ = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout)
        writer = asyncio.StreamWriter(transport, _, None, loop)
    except (NotImplementedError, ValueError, OSError):
        # Fall back: blocking reads in the executor.
        reader = _BlockingStdinReader()  # type: ignore[assignment]
        writer = _BlockingStdoutWriter()  # type: ignore[assignment]

    return reader, writer  # type: ignore[return-value]


class _BlockingStdinReader:
    """Adapter mimicking just enough StreamReader for ServerStdioTransport."""

    def __init__(self) -> None:
        self._buffer = b""

    async def read(self, n: int = -1) -> bytes:
        loop = asyncio.get_running_loop()
        if self._buffer:
            out = self._buffer[:n] if n > 0 else self._buffer
            self._buffer = self._buffer[len(out):]
            return out
        chunk = await loop.run_in_executor(None, sys.stdin.buffer.read1, 65536)
        return chunk or b""

    def feed_data(self, data: bytes) -> None:
        self._buffer = data + self._buffer


class _BlockingStdoutWriter:
    """Adapter mimicking just enough StreamWriter."""

    def write(self, data: bytes) -> None:
        sys.stdout.buffer.write(data)

    async def drain(self) -> None:
        sys.stdout.buffer.flush()

    def close(self) -> None:
        try:
            sys.stdout.buffer.flush()
        except Exception:  # noqa: BLE001
            pass

    async def wait_closed(self) -> None:
        return
