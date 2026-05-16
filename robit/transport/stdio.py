"""enchanter/transport/stdio.py — port of transport/stdio.ts.

Implements the MCP stdio transport: spawn a child MCP server process,
communicate over its stdin/stdout using newline-delimited JSON-RPC 2.0
(UTF-8, no embedded newlines per MCP spec), and drain stderr to an
in-memory ring for logging.

Failure-mode defences
---------------------
FM-5 (unbounded resources): any line that would exceed PER_MESSAGE_BODY_MAX_BYTES
bytes before the newline is rejected immediately by raising BodyTooLargeError
and closing the transport.  The cap is checked *before* JSON parsing so a
malformed oversized payload never touches the parser.

Design notes
------------
- asyncio.create_subprocess_exec is used throughout (non-blocking I/O).
- Stderr is drained by a background asyncio Task that writes lines to a
  capped deque (STDERR_RING_CAPACITY lines).  This prevents stderr output
  from blocking stdout reads regardless of how chatty the server is.
- send() / receive() are primitives; callers are responsible for routing
  responses to their waiting coroutines (PendingRequests pattern belongs
  in the protocol layer, not here).
- The transport is a single-use object; call close() exactly once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import deque
from typing import TYPE_CHECKING

from robit.transport.descriptor import TransportDescriptor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: 8 MiB — FM-5 body cap.  Checked *before* JSON parse.
PER_MESSAGE_BODY_MAX_BYTES: int = 8 * 1024 * 1024

#: How many stderr lines we retain in the ring buffer.
STDERR_RING_CAPACITY: int = 256

#: Seconds to wait for the child process to exit after sending shutdown signal
#: before forcibly killing it.
SHUTDOWN_TIMEOUT_SECS: float = 5.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BodyTooLargeError(Exception):
    """Raised when a received line exceeds PER_MESSAGE_BODY_MAX_BYTES."""

    def __init__(self, bytes_seen: int) -> None:
        super().__init__(
            f"stdio message body exceeded cap "
            f"({bytes_seen} > {PER_MESSAGE_BODY_MAX_BYTES} bytes)"
        )
        self.bytes_seen = bytes_seen


class TransportClosedError(Exception):
    """Raised when send() is called on an already-closed transport."""


# ---------------------------------------------------------------------------
# StdioTransport
# ---------------------------------------------------------------------------


class StdioTransport:
    """Async stdio transport for MCP servers.

    Lifecycle::

        transport = StdioTransport()
        await transport.start(descriptor)
        await transport.send({"jsonrpc": "2.0", "method": "initialize", ...})
        msg = await transport.receive()   # returns dict or None on EOF
        await transport.close()

    All public methods are coroutines and must be called from an event loop.
    The object is NOT thread-safe; use it from a single event-loop thread.
    """

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_ring: deque[str] = deque(maxlen=STDERR_RING_CAPACITY)
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, descriptor: TransportDescriptor) -> None:
        """Spawn the child MCP server described by *descriptor*.

        Parameters
        ----------
        descriptor:
            Must be a stdio-kind descriptor (``descriptor.kind == "stdio"``
            and ``descriptor.cmd`` is not None).

        Raises
        ------
        ValueError:
            If the descriptor is not a stdio descriptor or cmd is missing.
        OSError:
            If the subprocess cannot be spawned.
        """
        if descriptor.kind != "stdio":
            raise ValueError(
                f"StdioTransport requires a 'stdio' descriptor, got '{descriptor.kind}'"
            )
        if not descriptor.cmd:
            raise ValueError("StdioTransport: descriptor.cmd must not be empty")

        # Build child environment from the allowlist.
        child_env = self._build_env(descriptor.env_allowlist)

        self._proc = await asyncio.create_subprocess_exec(
            descriptor.cmd,
            *descriptor.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )

        # Start background task draining stderr so it never blocks stdout.
        self._stderr_task = asyncio.get_event_loop().create_task(
            self._drain_stderr(), name="stdio-stderr-drain"
        )

        logger.debug(
            "StdioTransport: spawned pid=%d cmd=%r args=%r",
            self._proc.pid,
            descriptor.cmd,
            descriptor.args,
        )

    async def send(self, message: dict) -> None:
        """Encode *message* as JSON and write it to the child's stdin.

        The line ``json + "\\n"`` is written atomically in a single
        ``drain()`` call.  The 8 MiB body cap is checked before the write
        so an oversized outgoing message is caught early.

        Parameters
        ----------
        message:
            Any JSON-serialisable dict representing a JSON-RPC 2.0 envelope.

        Raises
        ------
        TransportClosedError:
            If the transport has been closed.
        BodyTooLargeError:
            If the serialised message exceeds PER_MESSAGE_BODY_MAX_BYTES.
        """
        if self._closed:
            raise TransportClosedError("send() called on a closed StdioTransport")

        proc = self._require_proc()
        assert proc.stdin is not None  # guaranteed by PIPE flag

        payload = json.dumps(message, ensure_ascii=False) + "\n"
        encoded = payload.encode("utf-8")

        if len(encoded) > PER_MESSAGE_BODY_MAX_BYTES:
            raise BodyTooLargeError(len(encoded))

        proc.stdin.write(encoded)
        await proc.stdin.drain()

    async def receive(self) -> dict | None:
        """Read the next newline-terminated line from the child's stdout.

        Returns
        -------
        dict
            The parsed JSON-RPC message.
        None
            On clean EOF (child closed stdout / exited).

        Raises
        ------
        BodyTooLargeError:
            If a line exceeds PER_MESSAGE_BODY_MAX_BYTES bytes before the
            newline.  The transport is closed before raising.
        json.JSONDecodeError:
            If the line is not valid JSON.
        TransportClosedError:
            If the transport has been closed.
        """
        if self._closed:
            raise TransportClosedError("receive() called on a closed StdioTransport")

        proc = self._require_proc()
        assert proc.stdout is not None  # guaranteed by PIPE flag

        # asyncio.StreamReader.readline() reads up to the next \n.
        # We must enforce the 8 MiB cap ourselves since readline() will
        # happily buffer a line of arbitrary length.
        #
        # Strategy: read in chunks up to the cap + 1 byte.  If we ever
        # accumulate more than PER_MESSAGE_BODY_MAX_BYTES bytes without
        # finding a newline, raise BodyTooLargeError.
        line_bytes = await self._read_line_capped(proc.stdout)

        if line_bytes is None:
            return None  # EOF

        line = line_bytes.decode("utf-8")
        return json.loads(line)

    async def close(self) -> None:
        """Gracefully shut down the child process.

        1. Close the child's stdin (signals EOF to the server).
        2. Wait up to SHUTDOWN_TIMEOUT_SECS for the process to exit.
        3. Kill the process if it hasn't exited by then.
        4. Cancel and await the stderr drain task.
        """
        if self._closed:
            return
        self._closed = True

        proc = self._proc
        if proc is None:
            return

        # 1. Close stdin — tells the server no more input is coming.
        if proc.stdin and not proc.stdin.is_closing():
            proc.stdin.close()
            try:
                await proc.stdin.wait_closed()
            except Exception:  # noqa: BLE001
                pass

        # 2. Wait for exit with timeout.
        try:
            await asyncio.wait_for(proc.wait(), timeout=SHUTDOWN_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            logger.warning(
                "StdioTransport: child pid=%d did not exit within %.1fs; killing",
                proc.pid,
                SHUTDOWN_TIMEOUT_SECS,
            )
            # 3. Force-kill.
            try:
                proc.kill()
            except ProcessLookupError:
                pass  # already gone
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.error(
                    "StdioTransport: child pid=%d could not be killed; giving up",
                    proc.pid,
                )

        # 4. Cancel the stderr drain task.
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass

        logger.debug("StdioTransport: closed (pid=%d)", proc.pid)

    def recent_stderr(self, n: int = STDERR_RING_CAPACITY) -> list[str]:
        """Return the last *n* stderr lines captured from the child process.

        Lines are returned in order (oldest first).  At most
        STDERR_RING_CAPACITY lines are retained.

        Parameters
        ----------
        n:
            Maximum number of lines to return.  Defaults to the ring capacity.
        """
        lines = list(self._stderr_ring)
        return lines[-n:] if n < len(lines) else lines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_proc(self) -> asyncio.subprocess.Process:
        if self._proc is None:
            raise RuntimeError(
                "StdioTransport: start() has not been called"
            )
        return self._proc

    @staticmethod
    def _build_env(env_allowlist: tuple[str, ...]) -> dict[str, str] | None:
        """Build a child-process environment from the allowlist.

        If the allowlist is empty, return *None* (inherit nothing extra;
        the OS provides the minimal environment).  Otherwise return only
        the subset of the current process's environment whose keys appear
        in the allowlist.

        We always pass PATH and (on Windows) PATHEXT so the child can
        locate executables.
        """
        if not env_allowlist:
            # Pass a minimal env so the child can find itself on PATH.
            minimal: dict[str, str] = {}
            for key in ("PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP"):
                val = os.environ.get(key)
                if val is not None:
                    minimal[key] = val
            return minimal or None

        child: dict[str, str] = {}
        for key in env_allowlist:
            val = os.environ.get(key)
            if val is not None:
                child[key] = val
        # Ensure PATH is always available so the child finds its deps.
        if "PATH" not in child:
            path = os.environ.get("PATH")
            if path is not None:
                child["PATH"] = path
        return child

    async def _read_line_capped(
        self,
        stream: asyncio.StreamReader,
    ) -> bytes | None:
        """Read exactly one newline-terminated line, enforcing the 8 MiB cap.

        Returns the line's bytes *without* the trailing ``\\n``, or ``None``
        on EOF.  Empty lines (bare ``\\n``) return ``b""``.

        Raises BodyTooLargeError if the accumulated bytes before ``\\n``
        exceed PER_MESSAGE_BODY_MAX_BYTES.
        """
        chunks: list[bytes] = []
        total = 0

        while True:
            # Read a chunk up to the cap + 1 so we detect oversized lines.
            remaining = PER_MESSAGE_BODY_MAX_BYTES - total + 1
            try:
                chunk = await stream.read(min(remaining, 65536))
            except Exception:  # noqa: BLE001
                # Stream error — treat as EOF.
                return None

            if not chunk:
                # EOF.  If we have partial data, it's a truncated line;
                # mirror the TS behaviour of silently dropping it.
                return None

            # Check for newline.
            nl_pos = chunk.find(b"\n")
            if nl_pos != -1:
                # Found the line terminator.
                before_nl = chunk[:nl_pos]
                remainder = chunk[nl_pos + 1:]
                total += len(before_nl)

                if total > PER_MESSAGE_BODY_MAX_BYTES:
                    await self.close()
                    raise BodyTooLargeError(total)

                chunks.append(before_nl)

                # Put any bytes after the newline back into the stream
                # by prepending them to the internal buffer.
                if remainder:
                    stream.feed_data(remainder)

                line = b"".join(chunks)
                if len(line) == 0:
                    # Empty line — skip silently (mirror TS recv logic).
                    # Recurse to get the next real line.
                    return await self._read_line_capped(stream)

                return line

            # No newline yet — accumulate and check cap.
            total += len(chunk)
            if total > PER_MESSAGE_BODY_MAX_BYTES:
                await self.close()
                raise BodyTooLargeError(total)

            chunks.append(chunk)

    async def _drain_stderr(self) -> None:
        """Background task: continuously drain the child's stderr stream."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return

        try:
            async for line in proc.stderr:
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                self._stderr_ring.append(text)
                logger.debug("StdioTransport stderr: %s", text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("StdioTransport: stderr drain ended: %r", exc)
