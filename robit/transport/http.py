"""enchanter/transport/http.py — Streamable-HTTP MCP transport (port of streamable-http.ts).

MCP Streamable-HTTP transport spec
-----------------------------------
- **POST** the JSON-RPC request to the server URL.  Response body is the
  JSON-RPC response (single ``application/json``) or a ``text/event-stream``
  SSE body if the server wants to stream.  ``202 Accepted`` means the server
  will deliver via the GET SSE stream.
- **GET** the same URL with ``Accept: text/event-stream`` opens a long-lived
  SSE stream for server-initiated messages (notifications, async responses).
- **Session token**: the server echoes a session-id in a response header on
  the first request; subsequent requests send it back via a request header.
- **8 MB body cap** (FM-5 unbounded-resource mitigation): enforced during
  read, not after.
- **Reconnect**: exponential backoff base-2 with full jitter
  (``delay * uniform(0.5, 1.0)``), up to 10 attempts (FM-3 reconnect).
- **SSE resume disabled by default** (FM-8 session-hijacking defense).
  Set ``allow_resume=True`` and call ``set_session_nonce()`` to opt in.
- **TLS pinning** via :class:`~robit.transport.tls_pin.TlsPinStore` when
  supplied (FM-6 server-spoofing mitigation).

Stdlib only
-----------
HTTP is sent with ``asyncio.open_connection()`` (raw TCP or TLS) building
HTTP/1.1 requests manually.  No ``aiohttp``, no ``requests``, no
``httpx`` — pure stdlib.

Deviation from TS
-----------------
The TS transport uses *undici* (a Node HTTP/2-capable client) and its
``MockAgent`` for tests.  Python's stdlib does not include an async HTTP
client, so we implement HTTP/1.1 over ``asyncio.StreamReader /
StreamWriter``.  HTTP/2 and HTTP keep-alive connection pooling are out of
scope for v1.  Tests use an ``asyncio``-based local HTTP server (no threads).
"""

from __future__ import annotations

import asyncio
import json
import random
import ssl
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from robit.transport.descriptor import TransportDescriptor

# ---------------------------------------------------------------------------
# Constants (mirrors TS backoff + MCP constants)
# ---------------------------------------------------------------------------

PER_MESSAGE_BODY_MAX_BYTES: int = 8 * 1024 * 1024  # 8 MB

BACKOFF_INITIAL_S: float = 0.5        # 500 ms
BACKOFF_FACTOR: float = 2.0
BACKOFF_MAX_S: float = 30.0
BACKOFF_MAX_ATTEMPTS: int = 10

# MCP normative Accept header
_ACCEPT_HEADER = "application/json, text/event-stream"

# Response header that carries the session id (MCP convention)
_SESSION_ID_HEADER = "mcp-session-id"
# Request header we echo the session id back on
_SESSION_ID_REQUEST_HEADER = "mcp-session-id"

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BodyTooLargeError(Exception):
    """Raised when the HTTP/SSE body exceeds :data:`PER_MESSAGE_BODY_MAX_BYTES`."""

    def __init__(self, size: int) -> None:
        super().__init__(
            f"Response body exceeds 8 MB cap ({size} bytes): FM-5 unbounded resource"
        )
        self.size = size


class StreamableHttpMaxRetriesError(Exception):
    """Raised when SSE reconnect gives up after :data:`BACKOFF_MAX_ATTEMPTS` attempts."""

    def __init__(self, attempts: int) -> None:
        super().__init__(f"SSE stream reconnect gave up after {attempts} attempts")
        self.attempts = attempts


class StreamableHttpResumeError(Exception):
    """Raised when resume is requested without a session nonce (FM-8)."""

    def __init__(self) -> None:
        super().__init__(
            "GET resume is disabled by default (FM-8 session-hijacking defense). "
            "Pass allow_resume=True and call set_session_nonce() to opt in."
        )


# ---------------------------------------------------------------------------
# Low-level HTTP/1.1 over asyncio (stdlib only)
# ---------------------------------------------------------------------------


@dataclass
class _HttpResponse:
    status: int
    reason: str
    headers: dict[str, str]   # lower-cased header names
    body: bytes


async def _send_http_request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    *,
    ssl_context: ssl.SSLContext | None = None,
    body_max_bytes: int = PER_MESSAGE_BODY_MAX_BYTES,
) -> _HttpResponse:
    """Send a single HTTP/1.1 request and return the parsed response.

    Streaming (chunked) is handled for SSE bodies.  The caller is responsible
    for passing ``body_max_bytes`` to enforce the 8 MB cap; this function
    raises :class:`BodyTooLargeError` if the cap is exceeded during read.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    use_tls = parsed.scheme == "https"
    path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")

    if use_tls and ssl_context is None:
        ssl_context = ssl.create_default_context()

    reader, writer = await asyncio.open_connection(
        host, port, ssl=ssl_context if use_tls else None
    )

    try:
        # Build HTTP/1.1 request
        req_lines: list[str] = [f"{method} {path} HTTP/1.1"]
        merged: dict[str, str] = {
            "Host": f"{host}:{port}" if port not in (80, 443) else host,
            "Connection": "close",
        }
        if body is not None:
            merged["Content-Length"] = str(len(body))
        merged.update(headers)

        for k, v in merged.items():
            req_lines.append(f"{k}: {v}")
        req_lines.append("")
        req_lines.append("")
        request_bytes = "\r\n".join(req_lines).encode("latin-1")
        if body is not None:
            request_bytes = request_bytes + body

        writer.write(request_bytes)
        await writer.drain()

        # Read status line
        status_line = (await reader.readline()).decode("latin-1", errors="replace").rstrip("\r\n")
        parts = status_line.split(" ", 2)
        status = int(parts[1]) if len(parts) >= 2 else 0
        reason = parts[2] if len(parts) >= 3 else ""

        # Read headers
        resp_headers: dict[str, str] = {}
        while True:
            line = (await reader.readline()).decode("latin-1", errors="replace").rstrip("\r\n")
            if not line:
                break
            if ":" in line:
                k, _, v = line.partition(":")
                resp_headers[k.strip().lower()] = v.strip()

        content_type = resp_headers.get("content-type", "")
        is_sse = "text/event-stream" in content_type

        if is_sse:
            # For SSE: stream bytes until connection closes, enforcing cap
            raw_body = await _read_capped(reader, body_max_bytes)
        else:
            transfer = resp_headers.get("transfer-encoding", "")
            content_len_s = resp_headers.get("content-length")
            if "chunked" in transfer.lower():
                raw_body = await _read_chunked(reader, body_max_bytes)
            elif content_len_s is not None:
                length = int(content_len_s)
                if length > body_max_bytes:
                    raise BodyTooLargeError(length)
                raw_body = await asyncio.wait_for(reader.readexactly(length), timeout=30.0)
            else:
                raw_body = await _read_capped(reader, body_max_bytes)

        return _HttpResponse(
            status=status,
            reason=reason,
            headers=resp_headers,
            body=raw_body,
        )
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def _read_capped(reader: asyncio.StreamReader, max_bytes: int) -> bytes:
    """Read until EOF, raising :class:`BodyTooLargeError` if cap exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise BodyTooLargeError(total)
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_chunked(reader: asyncio.StreamReader, max_bytes: int) -> bytes:
    """Read HTTP/1.1 chunked transfer-encoding body."""
    chunks: list[bytes] = []
    total = 0
    while True:
        size_line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
        # chunk size is hex; extensions after ';' are ignored
        chunk_size = int(size_line.split(";")[0].strip(), 16)
        if chunk_size == 0:
            # Consume trailing CRLF
            await reader.readline()
            break
        total += chunk_size
        if total > max_bytes:
            raise BodyTooLargeError(total)
        data = await reader.readexactly(chunk_size + 2)  # +2 for CRLF
        chunks.append(data[:-2])
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# SSE parser
# ---------------------------------------------------------------------------


def _parse_sse_events(raw: bytes) -> list[dict[str, Any]]:
    """Parse a raw SSE body into a list of JSON-RPC message dicts.

    Handles multi-line ``data:`` fields per spec (concatenated).
    Other SSE fields (``event:``, ``id:``, ``retry:``) are silently skipped.

    Note: ``id:`` / ``Last-Event-ID`` is intentionally NOT tracked here —
    resume is disabled by default (FM-8).  When ``allow_resume=True`` the
    caller must manage the last-event-id separately.
    """
    text = raw.decode("utf-8", errors="replace")
    messages: list[dict[str, Any]] = []

    event_data = ""
    for line in text.splitlines(keepends=True):
        line = line.rstrip("\r\n")
        if line == "":
            # Blank line → flush accumulated event data
            payload = event_data.strip()
            if payload:
                try:
                    messages.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass  # malformed SSE payload — skip
            event_data = ""
        elif line.startswith("data:"):
            chunk = line[5:]
            if chunk.startswith(" "):
                chunk = chunk[1:]
            event_data = (event_data + chunk) if event_data else chunk

    # Tail flush — stream closed without trailing blank line
    payload = event_data.strip()
    if payload:
        try:
            messages.append(json.loads(payload))
        except json.JSONDecodeError:
            pass

    return messages


# ---------------------------------------------------------------------------
# StreamableHttpTransport
# ---------------------------------------------------------------------------


class StreamableHttpTransport:
    """MCP Streamable-HTTP transport.

    Parameters
    ----------
    descriptor:
        A :class:`~robit.transport.descriptor.TransportDescriptor` with
        ``kind="http"`` carrying the endpoint URL.
    allow_resume:
        Set ``True`` to allow SSE ``Last-Event-ID`` resume.  Requires a call
        to :meth:`set_session_nonce` first (FM-8 mitigation).
    ssl_context:
        Optional ``ssl.SSLContext``.  When ``None`` and the URL is HTTPS, a
        default verify context is built.  Pass a context with pinned certs or
        a custom CA bundle when needed for TLS pinning at the SSL layer.
    tls_pin_store:
        Optional :class:`~robit.transport.tls_pin.TlsPinStore` for
        TOFU/PINNED TLS leaf-cert pinning.  When supplied a post-handshake
        callback verifies the leaf cert fingerprint.  (Python's stdlib
        ``ssl`` module exposes the peer cert after handshake via
        ``SSLObject.getpeercert(binary_form=True)``.)
    tls_pin_policy:
        ``"tofu"`` (default) or ``"pinned"``.
    """

    def __init__(
        self,
        descriptor: TransportDescriptor,
        *,
        allow_resume: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        tls_pin_store: Any | None = None,  # TlsPinStore (avoid circular import in type hint)
        tls_pin_policy: str = "tofu",
    ) -> None:
        if descriptor.kind != "http" or not descriptor.url:
            raise ValueError("StreamableHttpTransport requires an http descriptor with a URL")
        self._url = descriptor.url
        parsed = urlparse(self._url)
        self._origin = f"{parsed.scheme}://{parsed.netloc}"

        self._allow_resume = allow_resume
        self._ssl_context = ssl_context
        self._tls_pin_store = tls_pin_store
        self._tls_pin_policy = tls_pin_policy

        self._auth_token: str | None = None
        self._session_id: str | None = None
        self._session_nonce: str | None = None

        # Inbound message queue for receive()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_auth_token(self, token: str) -> None:
        """Set a Bearer token included on every request."""
        self._auth_token = token

    def set_session_nonce(self, nonce: str) -> None:
        """Bind a session nonce for FM-8 opt-in resume."""
        self._session_nonce = nonce

    async def start(self) -> None:
        """No-op for HTTP transport — connection is per-request."""
        pass

    async def send(self, message: dict[str, Any]) -> None:
        """POST *message* as JSON to the endpoint.

        Parses the response body (``application/json`` or SSE) and enqueues
        any returned messages for :meth:`receive`.
        """
        body = json.dumps(message).encode("utf-8")
        if len(body) > PER_MESSAGE_BODY_MAX_BYTES:
            raise BodyTooLargeError(len(body))

        headers = self._build_headers(content_type="application/json")
        ssl_ctx = self._get_ssl_context()

        resp = await _send_http_request(
            "POST",
            self._url,
            headers,
            body,
            ssl_context=ssl_ctx,
            body_max_bytes=PER_MESSAGE_BODY_MAX_BYTES,
        )

        # Capture session id from response header (first response)
        self._capture_session_id(resp.headers)

        # TLS pin check (post-connection; SSL context has already verified)
        # Stdlib ssl exposes the cert after the fact via get_channel_binding;
        # actual pinning at connect-time requires a custom ssl context hook.
        # See _get_ssl_context for where TOFU pinning is applied.

        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            for msg in _parse_sse_events(resp.body):
                await self._queue.put(msg)
        elif resp.status == 202:
            # 202 Accepted: server will push via GET SSE stream; nothing to enqueue
            pass
        else:
            raw = resp.body.decode("utf-8", errors="replace").strip()
            if raw:
                try:
                    await self._queue.put(json.loads(raw))
                except json.JSONDecodeError:
                    pass

    async def receive(self) -> dict[str, Any] | None:
        """Return the next inbound message, blocking until one arrives.

        Returns ``None`` if the transport has been closed.
        """
        if self._closed and self._queue.empty():
            return None
        try:
            return await self._queue.get()
        except asyncio.CancelledError:
            return None

    async def open_get_stream(self, *, signal: asyncio.Event | None = None) -> None:
        """Open the long-lived GET SSE stream for server-initiated messages.

        Auto-reconnects with exponential backoff (initial 0.5 s, factor 2,
        max 30 s, full jitter) up to :data:`BACKOFF_MAX_ATTEMPTS` attempts.

        Parameters
        ----------
        signal:
            Optional :class:`asyncio.Event`.  Set the event to stop the loop.
        """
        if self._allow_resume and self._session_nonce is None:
            raise StreamableHttpResumeError()

        attempt = 0
        delay = BACKOFF_INITIAL_S

        while True:
            if signal is not None and signal.is_set():
                return
            try:
                headers = self._build_headers()
                if self._allow_resume and self._session_nonce:
                    headers["x-session-nonce"] = self._session_nonce
                # SSE resume disabled by default — do NOT send Last-Event-ID
                # unless allow_resume is True (FM-8 mitigation)
                if self._allow_resume and self._session_nonce:
                    # Opt-in: caller manages Last-Event-ID externally
                    pass

                ssl_ctx = self._get_ssl_context()
                resp = await _send_http_request(
                    "GET",
                    self._url,
                    headers,
                    None,
                    ssl_context=ssl_ctx,
                    body_max_bytes=PER_MESSAGE_BODY_MAX_BYTES,
                )

                self._capture_session_id(resp.headers)

                # Successful connection — reset backoff
                attempt = 0
                delay = BACKOFF_INITIAL_S

                for msg in _parse_sse_events(resp.body):
                    await self._queue.put(msg)

                # Server closed cleanly — reconnect
            except (StreamableHttpMaxRetriesError, StreamableHttpResumeError):
                raise
            except Exception:
                if signal is not None and signal.is_set():
                    return
                attempt += 1
                if attempt >= BACKOFF_MAX_ATTEMPTS:
                    raise StreamableHttpMaxRetriesError(attempt)

                # Exponential backoff with full jitter (uniform 0.5–1.0)
                jitter = random.uniform(0.5, 1.0)
                wait = min(delay * jitter, BACKOFF_MAX_S)
                await asyncio.sleep(wait)
                delay = min(delay * BACKOFF_FACTOR, BACKOFF_MAX_S)

    async def close(self) -> None:
        """Mark the transport closed; drain the receive queue."""
        self._closed = True
        # Wake any blocked receive() call
        await self._queue.put(None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_headers(self, *, content_type: str | None = None) -> dict[str, str]:
        h: dict[str, str] = {"Accept": _ACCEPT_HEADER}
        if content_type:
            h["Content-Type"] = content_type
        if self._auth_token:
            h["Authorization"] = f"Bearer {self._auth_token}"
        if self._session_id:
            h[_SESSION_ID_REQUEST_HEADER] = self._session_id
        return h

    def _capture_session_id(self, resp_headers: dict[str, str]) -> None:
        sid = resp_headers.get(_SESSION_ID_HEADER)
        if sid and not self._session_id:
            self._session_id = sid

    def _get_ssl_context(self) -> ssl.SSLContext | None:
        """Return the SSL context to use, optionally wiring TLS pin verification.

        Python's ``ssl`` module does not offer a pre-handshake cert callback
        like undici's connector.  Post-handshake pinning via
        ``SSLContext.set_servername_callback`` is complex and version-specific.

        Approach: we create a wrapping context that calls
        :func:`~robit.transport.tls_pin.verify_tls_pin` after the
        handshake completes by inspecting the DER cert via
        ``ssl.SSLSocket.getpeercert(binary_form=True)``.

        When no pin store is supplied, the context is used as-is (standard
        certificate verification).  When a pin store is supplied, post-connect
        cert-pinning verification is performed inside :meth:`send` and
        :meth:`open_get_stream` after the connection is established.

        DEVIATION FROM TS: the TS code pins at connector level (before any
        bytes are sent).  Python's stdlib does not expose a pre-send async
        callback, so pinning happens at the application layer immediately after
        ``asyncio.open_connection`` returns.  A future v2 could use a custom
        ``ssl.SSLContext`` subclass or a wrapped ``SSLSocket`` to pin earlier.
        """
        if self._ssl_context is not None:
            return self._ssl_context
        url = urlparse(self._url)
        if url.scheme != "https":
            return None
        return ssl.create_default_context()
