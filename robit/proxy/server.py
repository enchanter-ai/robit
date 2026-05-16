"""robit.proxy.server — stdlib asyncio HTTP/1.1 frontend for the proxy.

This module is the Wave 2 frontend: it binds a TCP socket, parses HTTP/1.1
request lines + headers manually (mirroring the idioms in
:mod:`robit.mcp_server.http`), routes the request to the matching
provider adapter, runs the canonical request through
:mod:`robit.proxy.pipeline`, and writes the rendered response back.

Design notes
------------

* **Stdlib only.**  No aiohttp / fastapi / uvicorn — the same constraint as
  ``mcp_server.http``.  HTTP is parsed by hand over ``asyncio.start_server``.
* **8 MiB body cap.**  Enforced before JSON parsing, identically to the MCP
  server.  Bodies larger than the cap return HTTP 413 with an empty body.
* **Per-adapter wire-format error envelopes.**  When ``parse_request``
  raises ``AdapterParseError`` we return a 400 with the matching family's
  native error shape (Anthropic: ``{"type":"error",...}``; OpenAI:
  ``{"error":{...}}``; Gemini: ``{"error":{"code":400,...}}``).
* **Family-aware ``accept`` filter.**  The ``accept`` constructor argument
  is a frozenset of family names (``{"anthropic","openai","gemini"}``).  If
  an adapter matches but its family is *not* in ``accept`` the response is
  404 — we treat the disabled family as "no such endpoint" rather than
  exposing a 403 leak.
* **Per-request conduct override.**  ``?conduct=off`` on the request URL
  disables the conduct injection for that one request, regardless of the
  server-wide ``conduct`` flag.
* **Bus headers.**  ``X-Enchanter-Bus-Events`` carries the number of post-
  response observations the pipeline emitted; ``X-Enchanter-Mask-Matched``
  is appended when any observation has ``topic == "secret-mask.matched"``.
  On veto: ``X-Enchanter-Veto`` and ``X-Enchanter-Veto-Pattern``.
* **Streaming caveat.**  Sibling D's contract is that bus observations fire
  *after* the stream iterator is exhausted.  By that point the HTTP
  response headers are already on the wire — so streaming responses do
  not get ``X-Enchanter-Mask-Matched`` headers.  Non-streaming responses
  do.  Documented here so callers don't expect parity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import replace
from typing import Any, AsyncIterator
from urllib.parse import parse_qs, urlparse

from . import fastpath
from .adapters import (
    AdapterParseError,
    AnthropicAdapter,
    CodexAdapter,
    GeminiAdapter,
    OpenAIAdapter,
)
from .canonical import CanonicalChunk, CanonicalRequest, CanonicalResponse

logger = logging.getLogger(__name__)

PER_MESSAGE_BODY_MAX_BYTES: int = 8 * 1024 * 1024
MAX_HEADER_BYTES: int = 64 * 1024  # 64 KiB request-line + headers cap


# ---------------------------------------------------------------------------
# Adapter → family-id mapping (we can't modify the adapter classes).
# ---------------------------------------------------------------------------

_ADAPTER_FAMILY: dict[type, str] = {
    AnthropicAdapter: "anthropic",
    OpenAIAdapter: "openai",
    GeminiAdapter: "gemini",
    CodexAdapter: "codex",
}

_ADAPTERS: tuple[type, ...] = (
    AnthropicAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    CodexAdapter,
)


class BodyTooLargeError(Exception):
    def __init__(self, size: int) -> None:
        super().__init__(f"HTTP request body exceeds cap ({size} bytes)")
        self.size = size


# ---------------------------------------------------------------------------
# Error-envelope builders (one per family — adapter-shaped).
# ---------------------------------------------------------------------------


def _anthropic_error_envelope(message: str) -> bytes:
    return json.dumps(
        {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
    ).encode("utf-8")


def _openai_error_envelope(message: str) -> bytes:
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": None,
            }
        }
    ).encode("utf-8")


def _gemini_error_envelope(message: str) -> bytes:
    return json.dumps(
        {"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}}
    ).encode("utf-8")


_FAMILY_ERROR_ENVELOPE = {
    "anthropic": _anthropic_error_envelope,
    "openai": _openai_error_envelope,
    "gemini": _gemini_error_envelope,
    # Codex talks the OpenAI error envelope shape (Responses API is OpenAI-shaped).
    "codex": _openai_error_envelope,
}


# ---------------------------------------------------------------------------
# ProxyServer.
# ---------------------------------------------------------------------------


class ProxyServer:
    """Asyncio-based HTTP/1.1 frontend for the canonical proxy pipeline.

    Parameters
    ----------
    host, port:
        Bind address.  ``port=0`` lets the OS assign a free port; the
        actual port is returned from :meth:`start`.
    accept:
        Frozenset of family names to accept.  When an adapter matches a
        request but its family is not in this set, the server returns 404
        (treating the family as "not enabled" without leaking that the
        endpoint exists).
    conduct:
        When ``True`` the server passes ``conduct=True`` to the pipeline.
        When ``False`` it disables conduct injection server-wide.  A
        ``?conduct=off`` query parameter on an individual request
        downgrades this to ``False`` for that request only.

    Streaming caveat
    ----------------
    Bus observations from the streaming pipeline fire *after* the
    iterator is exhausted.  HTTP response headers are already on the
    wire by then, so streaming responses do not surface
    ``X-Enchanter-Mask-Matched`` headers.  Non-streaming responses do.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        accept: frozenset[str] = frozenset({"anthropic", "openai", "gemini", "codex"}),
        conduct: bool = True,
        passthrough_auth: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.accept = frozenset(accept)
        self.conduct = conduct
        self.passthrough_auth = passthrough_auth
        self._server: asyncio.base_events.Server | None = None

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    async def start(self) -> tuple[str, int]:
        """Bind and start listening. Returns ``(host, actual_port)``."""
        self._server = await asyncio.start_server(
            self._on_connection, self.host, self.port
        )
        sockets = self._server.sockets or []
        if not sockets:
            raise RuntimeError("ProxyServer: start_server returned no sockets")
        sock = sockets[0]
        bound_host, bound_port = sock.getsockname()[:2]
        self.host = bound_host
        self.port = bound_port
        return bound_host, bound_port

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("ProxyServer: start() must be called first")
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._server = None

    # ------------------------------------------------------------------
    # Connection handler.
    # ------------------------------------------------------------------

    async def _on_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
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
                return

            method, path, _proto = request_line

            await self._dispatch(method, path, headers, body, writer)
        finally:
            await self._safe_close(writer)

    async def _dispatch(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        # 1. Locate the matching adapter.
        matched_cls: type | None = None
        for cls in _ADAPTERS:
            try:
                if cls.matches(method, path):
                    matched_cls = cls
                    break
            except Exception:  # noqa: BLE001
                continue

        if matched_cls is None:
            await self._send_simple(writer, 404, "Not Found", b"")
            return

        family = _ADAPTER_FAMILY[matched_cls]

        # 2. Accept-filter — if the family is disabled, hide it as 404.
        if family not in self.accept:
            await self._send_simple(writer, 404, "Not Found", b"")
            return

        # 2b. Fast-path bypass — only fires if ENCHANTER_ALLOW_FASTPATH_BYPASS=1
        # AND the caller's key SHA-256 is in <state_dir>/fastpath-allowlist.json.
        # SKIPS conduct injection + lifecycle gates. Audit-logged to JSONL.
        fp_config = fastpath.load_config()
        if fp_config.enabled:
            decision = await fastpath.evaluate(method, path, headers, body, fp_config)
            if decision.eligible:
                credential = fastpath._extract_auth_credential(headers, decision.upstream_provider or "")
                key_short = fastpath.short_key_hash(credential)
                logger.warning(
                    "fastpath bypass: provider=%s model=%s key=%s body=%d",
                    decision.upstream_provider, decision.model, key_short, len(body),
                )
                status, out_headers, out_body = await fastpath.passthrough(
                    method, path, headers, body,
                    upstream_provider=decision.upstream_provider or "",
                    model=decision.model or "",
                )
                await fastpath.record_bypass(
                    upstream_provider=decision.upstream_provider or "",
                    key_hash_short=key_short,
                    model=decision.model or "",
                    body_size=len(body),
                    upstream_status=status,
                )
                content_type = out_headers.get("Content-Type", "application/json")
                await self._send_simple(
                    writer, status, "OK" if 200 <= status < 300 else "Upstream Error",
                    out_body, content_type=content_type,
                    extra_headers=(("X-Enchanter-FastPath", "bypass"),),
                )
                return

        # 3. Parse the body into a CanonicalRequest.
        try:
            canonical_req = matched_cls.parse_request(body, path, headers)
        except AdapterParseError as exc:
            envelope_builder = _FAMILY_ERROR_ENVELOPE[family]
            payload = envelope_builder(str(exc))
            await self._send_simple(
                writer,
                400,
                "Bad Request",
                payload,
                content_type="application/json",
            )
            return
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.exception("ProxyServer: adapter.parse_request raised")
            envelope_builder = _FAMILY_ERROR_ENVELOPE[family]
            payload = envelope_builder(f"internal parse error: {exc}")
            await self._send_simple(
                writer,
                500,
                "Internal Server Error",
                payload,
                content_type="application/json",
            )
            return

        # 3b. Pass-through auth — stash inbound credential on metadata so
        # upstream.py can forward it to LiteLLM as api_key / extra_headers.
        # Only fires when explicitly enabled at the server level; default
        # behavior continues to rely on operator-set env vars.
        if self.passthrough_auth:
            inbound_auth = _extract_inbound_auth(headers, family)
            if inbound_auth is not None:
                canonical_req = replace(
                    canonical_req,
                    metadata={
                        **canonical_req.metadata,
                        "_enchanter_passthrough_auth": inbound_auth,
                    },
                )

        # 4. Resolve per-request conduct override (?conduct=off).
        per_request_conduct = self._resolve_conduct(path)

        # 5. Build pipeline options.  Import lazily so that test environments
        #    where Sibling D's pipeline hasn't landed yet can still import
        #    server.py without crashing.
        try:
            from . import pipeline as _pipeline  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            logger.exception("ProxyServer: robit.proxy.pipeline unavailable")
            payload = _FAMILY_ERROR_ENVELOPE[family](
                "proxy pipeline module unavailable"
            )
            await self._send_simple(
                writer,
                500,
                "Internal Server Error",
                payload,
                content_type="application/json",
            )
            return

        opts = _pipeline.PipelineOptions(conduct=per_request_conduct)

        # 6. Dispatch streaming vs non-streaming.
        if canonical_req.stream:
            await self._dispatch_stream(
                matched_cls, family, canonical_req, opts, _pipeline, writer
            )
        else:
            await self._dispatch_unary(
                matched_cls, family, canonical_req, opts, _pipeline, writer
            )

    # ------------------------------------------------------------------
    # Non-streaming path.
    # ------------------------------------------------------------------

    async def _dispatch_unary(
        self,
        adapter_cls: type,
        family: str,
        canonical_req: CanonicalRequest,
        opts: Any,
        pipeline_mod: Any,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            result = await pipeline_mod.run(canonical_req, opts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ProxyServer: pipeline.run raised")
            payload = _FAMILY_ERROR_ENVELOPE[family](f"upstream error: {exc}")
            await self._send_simple(
                writer,
                502,
                "Bad Gateway",
                payload,
                content_type="application/json",
            )
            return

        # Veto?
        if _is_veto(result):
            await self._send_veto(writer, result)
            return

        # Otherwise: result is a PipelineResult with .response + .fired.
        response: CanonicalResponse = result.response
        fired = tuple(result.fired or ())

        try:
            body = adapter_cls.render_response(response)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ProxyServer: adapter.render_response raised")
            payload = _FAMILY_ERROR_ENVELOPE[family](f"render error: {exc}")
            await self._send_simple(
                writer,
                500,
                "Internal Server Error",
                payload,
                content_type="application/json",
            )
            return

        extra_headers = _bus_headers(fired)
        await self._send_simple(
            writer,
            200,
            "OK",
            body,
            content_type="application/json",
            extra_headers=extra_headers,
        )

    # ------------------------------------------------------------------
    # Streaming path.
    # ------------------------------------------------------------------

    async def _dispatch_stream(
        self,
        adapter_cls: type,
        family: str,
        canonical_req: CanonicalRequest,
        opts: Any,
        pipeline_mod: Any,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            result = await pipeline_mod.stream(canonical_req, opts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ProxyServer: pipeline.stream raised")
            payload = _FAMILY_ERROR_ENVELOPE[family](f"upstream error: {exc}")
            await self._send_simple(
                writer,
                502,
                "Bad Gateway",
                payload,
                content_type="application/json",
            )
            return

        if _is_veto(result):
            await self._send_veto(writer, result)
            return

        # result is an async iterator of CanonicalChunk.  Open the SSE
        # response now; we cannot retro-add bus headers later because
        # observations fire post-iteration.
        sse_header = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"X-Accel-Buffering: no\r\n"
            b"Connection: close\r\n"
            b"\r\n"
        )
        try:
            writer.write(sse_header)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            return

        try:
            async for block in adapter_cls.render_stream(_as_async_iter(result)):
                try:
                    writer.write(block)
                    await writer.drain()
                except (BrokenPipeError, ConnectionResetError):
                    return
        except Exception:  # noqa: BLE001
            logger.exception("ProxyServer: render_stream raised mid-stream")
            # Body already started — best we can do is close.
            return

    # ------------------------------------------------------------------
    # HTTP framing helpers.
    # ------------------------------------------------------------------

    async def _read_request(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[tuple[str, str, str] | None, dict[str, str], bytes]:
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

        return (method, path, proto), headers, body

    async def _send_simple(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        reason: str,
        body: bytes,
        *,
        content_type: str = "text/plain; charset=utf-8",
        extra_headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        header_lines = [
            f"HTTP/1.1 {status} {reason}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(body)}",
            "Connection: close",
        ]
        for key, value in extra_headers:
            header_lines.append(f"{key}: {value}")
        header = ("\r\n".join(header_lines) + "\r\n\r\n").encode("latin-1")
        try:
            writer.write(header + body)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            return

    async def _send_veto(self, writer: asyncio.StreamWriter, veto: Any) -> None:
        phase = getattr(veto, "phase", "")
        plugin = getattr(veto, "plugin", "")
        reason = getattr(veto, "reason", "")
        pattern_id = getattr(veto, "pattern_id", None)
        pattern_name = getattr(veto, "pattern_name", None)
        body = json.dumps(
            {
                "type": "policy_veto",
                "phase": phase,
                "plugin": plugin,
                "reason": reason,
                "pattern_id": pattern_id,
                "pattern_name": pattern_name,
            }
        ).encode("utf-8")
        extras: list[tuple[str, str]] = [
            ("X-Enchanter-Veto", str(plugin or "")),
        ]
        if pattern_id:
            extras.append(("X-Enchanter-Veto-Pattern", str(pattern_id)))
        await self._send_simple(
            writer,
            451,
            "Unavailable For Legal Reasons",
            body,
            content_type="application/json",
            extra_headers=tuple(extras),
        )

    async def _safe_close(self, writer: asyncio.StreamWriter) -> None:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Conduct override.
    # ------------------------------------------------------------------

    def _resolve_conduct(self, path: str) -> bool:
        """Return the effective conduct flag for one request.

        Server-wide ``self.conduct`` is the baseline; a query string of
        ``conduct=off`` (or ``conduct=false``, ``conduct=0``) on the
        request URL flips it to ``False`` for this one request.
        """
        if not self.conduct:
            return False
        parsed = urlparse(path)
        qs = parse_qs(parsed.query)
        values = qs.get("conduct", [])
        if values and values[0].lower() in ("off", "false", "0", "no"):
            return False
        return True


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


# JWT shape: three base64url segments separated by dots, with the first
# segment starting with ``eyJ`` (base64url of ``{"`` — the opening of every
# JWS header). We do NOT validate the signature; we only shape-match to
# discriminate Codex's two auth modes (API key vs ChatGPT subscription JWT).
_JWT_SHAPE_RE = re.compile(
    r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$"
)


def _looks_like_jwt(token: str) -> bool:
    """Shape-match a string against the JWT (compact-serialization) form.

    True for ``eyJ…``.``…``.``…``; false otherwise. No signature or claim
    validation — we only need to discriminate inbound auth modes.
    """
    return bool(_JWT_SHAPE_RE.match(token or ""))


def _extract_inbound_auth(
    headers: dict[str, str], family: str
) -> dict[str, Any] | None:
    """Extract a host-agent's auth credential from inbound headers.

    Header keys are lowercased by the HTTP parser. Returns a dict with
    ``kind`` and ``value`` keys, or ``None`` if no recognizable auth
    header is present for the given family.

    Honesty note: the returned dict contains the credential verbatim.
    Callers must not log it. ``upstream.py`` hands it to LiteLLM and
    never includes it in error envelopes; ``server.py`` does not log
    metadata payloads.
    """
    if family == "anthropic":
        api_key = headers.get("x-api-key")
        if api_key:
            return {"kind": "anthropic-api-key", "value": api_key}
        bearer = headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            return {"kind": "anthropic-oauth", "value": bearer[7:].strip()}
        return None
    if family == "openai":
        bearer = headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            token = bearer[7:].strip()
            # Codex CLI may also POST /v1/responses through the OpenAI
            # adapter on some upstreams; if the token is JWT-shaped we
            # route through the ChatGPT-internal path.
            if _looks_like_jwt(token):
                return {
                    "kind": "chatgpt-jwt",
                    "value": token,
                    "account_id": headers.get("chatgpt-account-id"),
                }
            return {"kind": "openai-bearer", "value": token}
        return None
    if family == "codex":
        # Codex CLI sends either an API key (`sk-…`) or a ChatGPT JWT
        # (`eyJ…`) under Authorization: Bearer. ChatGPT-login mode is
        # routed via the non-LiteLLM path in upstream.py
        # (`_call_chatgpt_internal`); the JWT shape is the only honest
        # signal we have to discriminate — Codex CLI sends an `sk-…` key
        # for API-key mode and a ``eyJ…``-shaped JWT for ChatGPT mode.
        bearer = headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            token = bearer[7:].strip()
            if _looks_like_jwt(token):
                return {
                    "kind": "chatgpt-jwt",
                    "value": token,
                    # ChatGPT-Account-ID is required by the upstream for
                    # subscription-auth mode; if missing we let the
                    # upstream return the right error (do not synthesise).
                    "account_id": headers.get("chatgpt-account-id"),
                }
            return {"kind": "openai-bearer", "value": token}
        return None
    if family == "gemini":
        api_key = headers.get("x-goog-api-key")
        if api_key:
            return {"kind": "gemini-api-key", "value": api_key}
        return None
    return None


def _is_veto(value: Any) -> bool:
    """Duck-type a :class:`robit.proxy.pipeline.VetoResult`.

    The pipeline module may not be importable in some test states; we
    avoid a hard ``isinstance`` against ``pipeline.VetoResult`` so this
    helper works as long as the value has the expected dataclass shape
    (``phase`` + ``plugin`` + ``reason``).
    """
    if value is None:
        return False
    return (
        hasattr(value, "phase")
        and hasattr(value, "plugin")
        and hasattr(value, "reason")
        and not hasattr(value, "response")
    )


def _bus_headers(fired: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    """Build the X-Enchanter-* headers from a tuple of BusObservation."""
    headers: list[tuple[str, str]] = [
        ("X-Enchanter-Bus-Events", str(len(fired)))
    ]
    mask_matches = sum(
        1
        for ob in fired
        if getattr(ob, "topic", "") == "secret-mask.matched"
    )
    if mask_matches:
        headers.append(("X-Enchanter-Mask-Matched", str(mask_matches)))
    # Wave 13.1 cost-ledger: surface per-request spend.  The cost-ledger
    # emitter publishes ``cost.ledger.recorded`` and parks cents in the
    # ``score`` field (the recorder whitelists ``score``, not ``cents``).
    cost_cents = sum(
        int(getattr(ob, "payload_summary", {}).get("score", 0))
        for ob in fired
        if getattr(ob, "topic", "") == "cost.ledger.recorded"
    )
    if cost_cents > 0:
        headers.append(("X-Enchanter-Cost-Cents", str(cost_cents)))
    return tuple(headers)


def _as_async_iter(value: Any) -> AsyncIterator[CanonicalChunk]:
    """Coerce a value that already is an async iterator into one.

    Some pipeline implementations might return an async-generator object
    directly (which is already an async iterator), or a wrapper that
    needs ``__aiter__()``.  This helper handles both.
    """
    if hasattr(value, "__aiter__"):
        return value.__aiter__()
    return value  # already an async iterator


# ---------------------------------------------------------------------------
# Top-level convenience.
# ---------------------------------------------------------------------------


async def serve_proxy(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    accept: frozenset[str] = frozenset({"anthropic", "openai", "gemini", "codex"}),
    conduct: bool = True,
    passthrough_auth: bool = False,
) -> None:
    """Start a :class:`ProxyServer` and serve forever.

    Clean-shutdown on :class:`KeyboardInterrupt` — the server is closed
    before the exception propagates back to the caller (which is
    typically :func:`asyncio.run` in the CLI).
    """
    server = ProxyServer(
        host=host,
        port=port,
        accept=accept,
        conduct=conduct,
        passthrough_auth=passthrough_auth,
    )
    bound_host, bound_port = await server.start()
    logger.info("enchanter proxy listening on %s:%d", bound_host, bound_port)
    try:
        await server.serve_forever()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await server.close()


__all__ = [
    "ProxyServer",
    "serve_proxy",
    "PER_MESSAGE_BODY_MAX_BYTES",
]
