"""Server-side Streamable-HTTP transport.

Spec subset implemented (sufficient for MCP):
- POST <path> with JSON body → JSON-RPC dispatch → JSON response (single shot).
- GET  <path> with Accept: text/event-stream → minimal SSE keep-alive
  (sends no server-initiated events in v1; just holds the connection until
  the client closes it).
- 8 MiB body cap (FM-5), enforced before JSON parse.
- HTTP/1.1 parsed manually over asyncio streams (stdlib only).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

PER_MESSAGE_BODY_MAX_BYTES: int = 8 * 1024 * 1024
MAX_HEADER_BYTES: int = 64 * 1024  # 64 KiB request-line + headers cap


class BodyTooLargeError(Exception):
    def __init__(self, size: int) -> None:
        super().__init__(f"HTTP request body exceeds cap ({size} bytes)")
        self.size = size


RawHandler = Callable[[str], Awaitable[str | None]]


class ServerHttpTransport:
    """Minimal HTTP/1.1 server bound to an MCP JSON-RPC dispatcher."""

    def __init__(self, handler: RawHandler, path: str = "/mcp") -> None:
        self._handler = handler
        self._path = path
        self._server: asyncio.base_events.Server | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        """Start listening on (host, port). Returns the actual (host, port).

        Port 0 means "OS assigns a free port" — the actual port is in the
        returned tuple. Useful for tests.
        """
        self._server = await asyncio.start_server(self._on_connection, host, port)
        sockets = self._server.sockets or []
        if not sockets:
            raise RuntimeError("ServerHttpTransport: start_server returned no sockets")
        sock = sockets[0]
        bound_host, bound_port = sock.getsockname()[:2]
        return bound_host, bound_port

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("ServerHttpTransport: start() must be called first")
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _on_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line, headers, body = await self._read_request(reader)
        except BodyTooLargeError:
            await self._send_simple(writer, 413, "Payload Too Large", b"")
            return
        except Exception:  # noqa: BLE001
            await self._send_simple(writer, 400, "Bad Request", b"")
            return

        if request_line is None:
            # Client closed without sending anything.
            await self._safe_close(writer)
            return

        method, path, _proto = request_line

        if path.split("?", 1)[0] != self._path:
            await self._send_simple(writer, 404, "Not Found", b"")
            await self._safe_close(writer)
            return

        if method == "POST":
            await self._handle_post(writer, headers, body)
        elif method == "GET":
            await self._handle_get(writer, headers)
        else:
            await self._send_simple(writer, 405, "Method Not Allowed", b"")

        await self._safe_close(writer)

    async def _handle_post(
        self,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        raw = body.decode("utf-8", errors="replace")
        try:
            response = await self._handler(raw)
        except Exception:  # noqa: BLE001
            logger.exception("ServerHttpTransport: handler raised")
            await self._send_simple(writer, 500, "Internal Server Error", b"")
            return

        if response is None:
            # Notification — 202 with empty body
            await self._send_simple(writer, 202, "Accepted", b"")
            return

        payload = response.encode("utf-8")
        await self._send_simple(
            writer,
            200,
            "OK",
            payload,
            content_type="application/json",
        )

    async def _handle_get(
        self,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
    ) -> None:
        accept = headers.get("accept", "")
        if "text/event-stream" not in accept:
            await self._send_simple(writer, 406, "Not Acceptable", b"")
            return

        # Minimal SSE response — we do not push events in v1; we keep the
        # connection open briefly and then close. Clients reconnect with
        # backoff per spec, which is fine for a passive server.
        header = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            b": keep-alive\n\n"
        )
        try:
            writer.write(header)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            return

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    async def _read_request(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[tuple[str, str, str] | None, dict[str, str], bytes]:
        # Read request-line + headers (terminated by CRLF CRLF).
        header_bytes = b""
        while b"\r\n\r\n" not in header_bytes:
            chunk = await reader.read(4096)
            if not chunk:
                if not header_bytes:
                    return None, {}, b""
                break
            header_bytes += chunk
            if len(header_bytes) > MAX_HEADER_BYTES:
                raise ValueError("header section too large")

        head, _, rest = header_bytes.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        if not lines:
            raise ValueError("empty request")

        request_line = lines[0].decode("latin-1")
        parts = request_line.split(" ", 2)
        if len(parts) < 3:
            raise ValueError(f"bad request line: {request_line!r}")
        method, path, proto = parts[0], parts[1], parts[2]

        headers: dict[str, str] = {}
        for line in lines[1:]:
            decoded = line.decode("latin-1")
            if ":" in decoded:
                k, _, v = decoded.partition(":")
                headers[k.strip().lower()] = v.strip()

        # Body: only present on POST (or any method with Content-Length).
        content_len_s = headers.get("content-length")
        body = rest
        if content_len_s is not None:
            try:
                content_len = int(content_len_s)
            except ValueError as exc:
                raise ValueError(f"bad Content-Length: {content_len_s}") from exc
            if content_len > PER_MESSAGE_BODY_MAX_BYTES:
                raise BodyTooLargeError(content_len)
            remaining = content_len - len(body)
            if remaining > 0:
                more = await reader.readexactly(remaining)
                body += more
            body = body[:content_len]
        # No Content-Length: assume zero body (we don't support chunked uploads).

        return (method, path, proto), headers, body

    async def _send_simple(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        reason: str,
        body: bytes,
        *,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        header = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            "\r\n"
        ).encode("latin-1")
        try:
            writer.write(header + body)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            return

    async def _safe_close(self, writer: asyncio.StreamWriter) -> None:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
