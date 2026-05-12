"""tests/transport/test_http.py — hermetic tests for StreamableHttpTransport.

All tests spin up a tiny asyncio HTTP/1.1 server in-process; no external
network calls are made.

Server helper
-------------
``_HttpServer`` starts on a random localhost port, serves one or more
requests through a user-supplied handler coroutine, then shuts down.

Test coverage (≥ 6 tests required)
-----------------------------------
1. POST request → response is decoded and enqueued
2. SSE GET → multiple events received in order
3. 8 MB body cap rejects oversized response
4. Session id captured from first response header, echoed on subsequent requests
5. SSE reconnect with backoff on disconnect (server closes after first event)
6. SSE resume disabled by default (no Last-Event-ID header sent)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Coroutine

import pytest

from enchanter.transport.descriptor import TransportDescriptor
from enchanter.transport.http import (
    BACKOFF_INITIAL_S,
    BodyTooLargeError,
    PER_MESSAGE_BODY_MAX_BYTES,
    StreamableHttpMaxRetriesError,
    StreamableHttpTransport,
)

# ---------------------------------------------------------------------------
# Tiny asyncio HTTP/1.1 server
# ---------------------------------------------------------------------------

RequestHandler = Callable[
    [str, str, dict[str, str], bytes],  # method, path, headers, body
    Coroutine[Any, Any, tuple[int, dict[str, str], bytes]],  # status, headers, body
]


class _HttpServer:
    """Minimal HTTP/1.1 server that handles one connection at a time.

    Parameters
    ----------
    handler:
        Async callable ``(method, path, headers, body) -> (status, headers, body)``.
    max_requests:
        Stop accepting after this many requests.  ``-1`` = unlimited.
    """

    def __init__(self, handler: RequestHandler, *, max_requests: int = -1) -> None:
        self._handler = handler
        self._max_requests = max_requests
        self._server: asyncio.Server | None = None
        self._requests_handled = 0
        self._host = "127.0.0.1"
        self._port = 0

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, 0
        )
        self._port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                if self._max_requests >= 0 and self._requests_handled >= self._max_requests:
                    break
                # Read request line
                req_line_bytes = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not req_line_bytes:
                    break
                req_line = req_line_bytes.decode("latin-1").rstrip("\r\n")
                if not req_line:
                    break
                parts = req_line.split(" ", 2)
                if len(parts) < 2:
                    break
                method = parts[0]
                path = parts[1]

                # Read headers
                req_headers: dict[str, str] = {}
                while True:
                    hline_bytes = await asyncio.wait_for(reader.readline(), timeout=5.0)
                    hline = hline_bytes.decode("latin-1").rstrip("\r\n")
                    if not hline:
                        break
                    if ":" in hline:
                        k, _, v = hline.partition(":")
                        req_headers[k.strip().lower()] = v.strip()

                # Read body
                body = b""
                content_len_s = req_headers.get("content-length")
                if content_len_s:
                    length = int(content_len_s)
                    if length > 0:
                        body = await asyncio.wait_for(
                            reader.readexactly(length), timeout=5.0
                        )

                # Dispatch handler
                try:
                    status, resp_headers, resp_body = await self._handler(
                        method, path, req_headers, body
                    )
                except Exception as exc:  # noqa: BLE001
                    status, resp_headers, resp_body = 500, {}, str(exc).encode()

                self._requests_handled += 1

                # Send response
                reason = {200: "OK", 202: "Accepted", 500: "Internal Server Error"}.get(
                    status, "OK"
                )
                resp_lines = [f"HTTP/1.1 {status} {reason}"]
                resp_headers.setdefault("Content-Length", str(len(resp_body)))
                resp_headers.setdefault("Connection", "close")
                for k, v in resp_headers.items():
                    resp_lines.append(f"{k}: {v}")
                resp_lines.append("")
                resp_lines.append("")
                header_bytes = "\r\n".join(resp_lines).encode("latin-1")
                writer.write(header_bytes + resp_body)
                await writer.drain()

                # If Connection: close, stop
                if resp_headers.get("Connection", "close").lower() == "close":
                    break
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionResetError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _descriptor(url: str) -> TransportDescriptor:
    return TransportDescriptor.for_http(url)


def _json_response(data: Any) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(data).encode()
    return 200, {"Content-Type": "application/json"}, body


def _sse_response(events: list[str]) -> tuple[int, dict[str, str], bytes]:
    """Build a complete SSE response body from a list of JSON strings."""
    lines = []
    for event in events:
        lines.append(f"data: {event}\n\n")
    body = "".join(lines).encode("utf-8")
    return 200, {"Content-Type": "text/event-stream"}, body


# ---------------------------------------------------------------------------
# Test 1: POST request → response decoded and enqueued
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_response_decoded() -> None:
    """POST returns a JSON-RPC response that is decoded and available via receive()."""
    expected = {"jsonrpc": "2.0", "id": 1, "result": {"value": 42}}

    async def handler(method, path, headers, body):
        assert method == "POST"
        return _json_response(expected)

    server = _HttpServer(handler, max_requests=1)
    await server.start()
    try:
        transport = StreamableHttpTransport(_descriptor(server.url))
        await transport.start()
        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        msg = await asyncio.wait_for(transport.receive(), timeout=3.0)
        assert msg == expected
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Test 2: SSE GET → multiple events received in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_get_multiple_events_in_order() -> None:
    """GET SSE stream delivers multiple events in order via open_get_stream."""
    events = [
        {"jsonrpc": "2.0", "method": "notify", "params": {"n": i}} for i in range(3)
    ]
    event_jsons = [json.dumps(e) for e in events]

    async def handler(method, path, headers, body):
        assert method == "GET"
        assert "text/event-stream" in headers.get("accept", "")
        return _sse_response(event_jsons)

    server = _HttpServer(handler, max_requests=1)
    await server.start()
    try:
        transport = StreamableHttpTransport(_descriptor(server.url))
        await transport.start()

        # Run open_get_stream in background; it will connect, drain, then return
        stop = asyncio.Event()
        task = asyncio.create_task(transport.open_get_stream(signal=stop))

        received = []
        for _ in range(len(events)):
            msg = await asyncio.wait_for(transport.receive(), timeout=5.0)
            received.append(msg)

        stop.set()
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except (asyncio.TimeoutError, StreamableHttpMaxRetriesError):
            pass

        assert received == events
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Test 3: 8 MB body cap rejects oversized response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_8mb_cap_rejects_oversized_response() -> None:
    """send() raises BodyTooLargeError when the server returns > 8 MB."""
    oversized_body = b"x" * (PER_MESSAGE_BODY_MAX_BYTES + 1)

    async def handler(method, path, headers, body):
        return (
            200,
            {"Content-Type": "application/json", "Content-Length": str(len(oversized_body))},
            oversized_body,
        )

    server = _HttpServer(handler, max_requests=1)
    await server.start()
    try:
        transport = StreamableHttpTransport(_descriptor(server.url))
        with pytest.raises(BodyTooLargeError):
            await transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Test 4: Session id captured and echoed on subsequent requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_id_captured_and_echoed() -> None:
    """Server sends mcp-session-id on first response; client echoes it on second."""
    session_id = "test-session-abc123"
    received_session_ids: list[str | None] = []

    request_count = 0

    async def handler(method, path, headers, body):
        nonlocal request_count
        request_count += 1
        received_session_ids.append(headers.get("mcp-session-id"))
        resp = {"jsonrpc": "2.0", "id": request_count, "result": {}}
        h: dict[str, str] = {"Content-Type": "application/json"}
        if request_count == 1:
            h["mcp-session-id"] = session_id
        return 200, h, json.dumps(resp).encode()

    server = _HttpServer(handler, max_requests=2)
    await server.start()
    try:
        transport = StreamableHttpTransport(_descriptor(server.url))

        # First request — no session id sent, server returns one
        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "init"})
        await asyncio.wait_for(transport.receive(), timeout=3.0)

        # Second request — client must echo the session id
        await transport.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        await asyncio.wait_for(transport.receive(), timeout=3.0)

        assert received_session_ids[0] is None, "First request should not echo a session id"
        assert received_session_ids[1] == session_id, (
            f"Second request should echo session id; got {received_session_ids[1]!r}"
        )
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Test 5: SSE reconnect with backoff on disconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_reconnect_on_disconnect() -> None:
    """SSE stream reconnects after server closes the connection mid-stream.

    Server pattern: first connection sends one event then closes immediately
    (simulating a disconnect).  Second connection sends one event and closes.
    We expect both events to be received across the reconnect.
    """
    connection_count = 0

    async def handler(method, path, headers, body):
        nonlocal connection_count
        connection_count += 1
        n = connection_count
        # Send one event and close
        event = json.dumps({"jsonrpc": "2.0", "method": "notify", "params": {"conn": n}})
        body_bytes = f"data: {event}\n\n".encode()
        return 200, {"Content-Type": "text/event-stream"}, body_bytes

    # max_requests=3 so we can handle up to 3 reconnect attempts
    server = _HttpServer(handler, max_requests=3)
    await server.start()
    try:
        transport = StreamableHttpTransport(_descriptor(server.url))

        stop = asyncio.Event()
        task = asyncio.create_task(transport.open_get_stream(signal=stop))

        # Collect first event from first connection
        msg1 = await asyncio.wait_for(transport.receive(), timeout=5.0)
        assert msg1["params"]["conn"] == 1

        # Collect event from second connection (after reconnect)
        msg2 = await asyncio.wait_for(transport.receive(), timeout=10.0)
        assert msg2["params"]["conn"] == 2

        stop.set()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.TimeoutError, StreamableHttpMaxRetriesError):
            pass

        assert connection_count >= 2, f"Expected at least 2 connections, got {connection_count}"
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Test 6: SSE resume disabled by default (no Last-Event-ID sent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_resume_disabled_by_default() -> None:
    """GET requests must NOT include Last-Event-ID unless allow_resume=True."""
    received_headers_list: list[dict[str, str]] = []

    async def handler(method, path, headers, body):
        received_headers_list.append(dict(headers))
        event = json.dumps({"jsonrpc": "2.0", "method": "notify", "params": {}})
        body_bytes = f"data: {event}\n\n".encode()
        return 200, {"Content-Type": "text/event-stream"}, body_bytes

    server = _HttpServer(handler, max_requests=1)
    await server.start()
    try:
        # Default: allow_resume=False
        transport = StreamableHttpTransport(_descriptor(server.url))

        stop = asyncio.Event()
        task = asyncio.create_task(transport.open_get_stream(signal=stop))

        await asyncio.wait_for(transport.receive(), timeout=5.0)
        stop.set()
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except (asyncio.TimeoutError, StreamableHttpMaxRetriesError):
            pass

        assert len(received_headers_list) >= 1
        headers_seen = received_headers_list[0]
        assert "last-event-id" not in headers_seen, (
            f"Last-Event-ID must not be sent when allow_resume=False; "
            f"headers: {headers_seen}"
        )
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Bonus test 7: oversized outgoing message raises BodyTooLargeError pre-send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_outgoing_message_raises_body_too_large() -> None:
    """send() raises BodyTooLargeError immediately for a > 8 MB outgoing message."""
    # We do not need a real server — the cap fires before the network call.
    transport = StreamableHttpTransport(_descriptor("http://127.0.0.1:19999"))
    big_data = "x" * (PER_MESSAGE_BODY_MAX_BYTES + 1)
    with pytest.raises(BodyTooLargeError):
        await transport.send({"jsonrpc": "2.0", "id": 0, "data": big_data})


# ---------------------------------------------------------------------------
# Bonus test 8: Accept header is always sent (MCP MUST)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accept_header_always_sent() -> None:
    """Every POST and GET must include Accept: application/json, text/event-stream."""
    accept_headers_seen: list[str] = []

    async def handler(method, path, headers, body):
        accept_headers_seen.append(headers.get("accept", ""))
        return _json_response({"jsonrpc": "2.0", "id": 1, "result": {}})

    server = _HttpServer(handler, max_requests=1)
    await server.start()
    try:
        transport = StreamableHttpTransport(_descriptor(server.url))
        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        await asyncio.wait_for(transport.receive(), timeout=3.0)

        assert accept_headers_seen, "No requests were received by the test server"
        for accept in accept_headers_seen:
            assert "application/json" in accept, f"Missing application/json in Accept: {accept}"
            assert "text/event-stream" in accept, f"Missing text/event-stream in Accept: {accept}"
    finally:
        await server.stop()
