"""Integration tests for enchanter.proxy.server — real TCP, mocked upstream.

Spins up a :class:`enchanter.proxy.server.ProxyServer` on ``127.0.0.1:0`` (OS-
assigned port) per test, sends real HTTP/1.1 requests via
``asyncio.open_connection``, and asserts on the wire-level response.  The
underlying LiteLLM ``acompletion`` is patched so no network traffic leaves
the process.

The test file imports the pipeline module lazily inside the module body so
that if Sibling D's pipeline isn't on disk yet the suite skips cleanly
rather than failing on collection.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# Skip the entire module gracefully if Sibling D's pipeline didn't land.
pipeline = pytest.importorskip("enchanter.proxy.pipeline")
streaming = pytest.importorskip("enchanter.proxy.streaming")

from enchanter.proxy import upstream  # noqa: E402
from enchanter.proxy.server import ProxyServer  # noqa: E402


# ---------------------------------------------------------------------------
# LiteLLM-shaped fakes (lifted from tests/proxy/test_upstream_litellm.py).
# ---------------------------------------------------------------------------


def _make_completion(
    text: str = "hello back",
    *,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _make_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    usage=None,
):
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice], usage=usage)


class _AsyncStream:
    """Trivial async iterator over a list of fabricated chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _basic_stream_chunks():
    return [
        _make_chunk(content="hel"),
        _make_chunk(content="lo"),
        _make_chunk(finish_reason="stop"),
    ]


# ---------------------------------------------------------------------------
# Raw-HTTP helpers.
# ---------------------------------------------------------------------------


async def _send_raw(
    host: str,
    port: int,
    request: bytes,
) -> tuple[int, dict[str, str], bytes]:
    """Send a raw HTTP request, return (status, headers, body).

    Reads the entire connection until EOF — works for both Content-Length-
    delimited and connection-close-delimited bodies (SSE).
    """
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(request)
        await writer.drain()

        status_line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
        parts = status_line.split(" ", 2)
        status = int(parts[1]) if len(parts) >= 2 else 0

        headers: dict[str, str] = {}
        while True:
            line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
            if not line:
                break
            k, _, v = line.partition(":")
            headers[k.strip().lower()] = v.strip()

        # Read until EOF — simplest for both framing styles.  All test
        # responses are small.
        body = await reader.read()
        return status, headers, body
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


async def _post(
    host: str,
    port: int,
    path: str,
    body: bytes,
    *,
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> tuple[int, dict[str, str], bytes]:
    header_lines = [
        f"POST {path} HTTP/1.1",
        f"Host: {host}:{port}",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
        "Connection: close",
    ]
    for k, v in extra_headers:
        header_lines.append(f"{k}: {v}")
    raw = ("\r\n".join(header_lines) + "\r\n\r\n").encode("latin-1") + body
    return await _send_raw(host, port, raw)


# ---------------------------------------------------------------------------
# Fixture: a started ProxyServer the test owns.
# ---------------------------------------------------------------------------


class _ServerHandle:
    def __init__(self, server: ProxyServer, host: str, port: int, task: asyncio.Task):
        self.server = server
        self.host = host
        self.port = port
        self.task = task

    async def aclose(self) -> None:
        await self.server.close()
        self.task.cancel()
        try:
            await self.task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


async def _start_server(**kwargs: Any) -> _ServerHandle:
    server = ProxyServer(host="127.0.0.1", port=0, **kwargs)
    host, port = await server.start()
    task = asyncio.create_task(server.serve_forever())
    # Yield once so serve_forever actually starts accepting.
    await asyncio.sleep(0)
    return _ServerHandle(server, host, port, task)


# ---------------------------------------------------------------------------
# Tests — non-streaming routes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_messages_round_trips_to_200():
    h = await _start_server()
    try:
        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")
        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(return_value=_make_completion(text="hello back")),
        ):
            status, headers, resp_body = await _post(
                h.host, h.port, "/v1/messages", body
            )
        assert status == 200
        assert "application/json" in headers.get("content-type", "")
        obj = json.loads(resp_body)
        assert obj["type"] == "message"
        assert obj["role"] == "assistant"
        assert obj["content"][0]["type"] == "text"
        assert obj["content"][0]["text"] == "hello back"
        # Bus header is always present, even when zero events fired.
        assert "x-enchanter-bus-events" in headers
    finally:
        await h.aclose()


@pytest.mark.asyncio
async def test_openai_chat_completions_round_trips_to_200():
    h = await _start_server()
    try:
        body = json.dumps(
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")
        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(return_value=_make_completion(text="hello openai")),
        ):
            status, headers, resp_body = await _post(
                h.host, h.port, "/v1/chat/completions", body
            )
        assert status == 200
        assert "application/json" in headers.get("content-type", "")
        obj = json.loads(resp_body)
        assert obj["object"] == "chat.completion"
        assert obj["choices"][0]["message"]["content"] == "hello openai"
    finally:
        await h.aclose()


@pytest.mark.asyncio
async def test_gemini_generate_content_round_trips_to_200():
    h = await _start_server()
    try:
        body = json.dumps(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "hi"}]},
                ],
            }
        ).encode("utf-8")
        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(return_value=_make_completion(text="hello gemini")),
        ):
            status, headers, resp_body = await _post(
                h.host,
                h.port,
                "/v1beta/models/gemini-1.5-flash:generateContent",
                body,
            )
        assert status == 200
        obj = json.loads(resp_body)
        cand = obj["candidates"][0]
        assert cand["content"]["parts"][0]["text"] == "hello gemini"
        assert cand["finishReason"] == "STOP"
    finally:
        await h.aclose()


# ---------------------------------------------------------------------------
# Tests — routing and accept filtering.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unmatched_path_returns_404():
    h = await _start_server()
    try:
        status, _headers, _body = await _post(
            h.host, h.port, "/totally/unknown/path", b"{}"
        )
        assert status == 404
    finally:
        await h.aclose()


@pytest.mark.asyncio
async def test_accept_filter_hides_disabled_family_as_404():
    """Anthropic family removed from accept → /v1/messages reads as 404.

    The intent is: the proxy must not leak that the endpoint exists but is
    disabled.  Returning 403 would be a fingerprint; 404 is correct.
    """
    h = await _start_server(accept=frozenset({"openai", "gemini"}))
    try:
        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")
        status, _headers, _body = await _post(h.host, h.port, "/v1/messages", body)
        assert status == 404
    finally:
        await h.aclose()


# ---------------------------------------------------------------------------
# Tests — error envelopes per family.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_malformed_json_returns_400_with_anthropic_envelope():
    h = await _start_server()
    try:
        status, headers, resp_body = await _post(
            h.host, h.port, "/v1/messages", b"not json at all"
        )
        assert status == 400
        assert "application/json" in headers.get("content-type", "")
        obj = json.loads(resp_body)
        assert obj["type"] == "error"
        assert obj["error"]["type"] == "invalid_request_error"
        assert isinstance(obj["error"]["message"], str)
    finally:
        await h.aclose()


# ---------------------------------------------------------------------------
# Test — veto path (destructive-op-gate fires on "rm -rf /").
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_destructive_prompt_returns_451_with_veto_header():
    h = await _start_server()
    try:
        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "user",
                        "content": "please run rm -rf / on the build host",
                    }
                ],
            }
        ).encode("utf-8")
        # Upstream should never be called when the gate vetoes; patch
        # acompletion defensively so a regression here doesn't reach out.
        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(return_value=_make_completion(text="should not get here")),
        ) as mocked:
            status, headers, resp_body = await _post(
                h.host, h.port, "/v1/messages", body
            )
        assert status == 451
        assert mocked.await_count == 0
        assert "x-enchanter-veto" in headers
        # Veto plugin should be one of the trust-gate engines that
        # subscribe to mcp.tool.call.requested (the exact engine depends
        # on which pattern fires first; both are valid signals).
        assert headers["x-enchanter-veto"] in {
            "destructive-op-gate",
            "cve-pattern-gate",
        }
        obj = json.loads(resp_body)
        assert obj["type"] == "policy_veto"
        assert obj["plugin"] in {"destructive-op-gate", "cve-pattern-gate"}
    finally:
        await h.aclose()


# ---------------------------------------------------------------------------
# Tests — streaming routes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_streaming_emits_sse_event_sequence():
    h = await _start_server()
    try:
        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")
        # Each test gets a fresh fake stream — acompletion is awaited once.
        async def _fake_acompletion(**_kwargs):
            return _AsyncStream(_basic_stream_chunks())

        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(side_effect=_fake_acompletion),
        ):
            status, headers, resp_body = await _post(
                h.host, h.port, "/v1/messages", body
            )
        assert status == 200
        assert "text/event-stream" in headers.get("content-type", "")
        text = resp_body.decode("utf-8", errors="replace")
        # Anthropic event names appear on the wire.
        assert "event: message_start" in text
        assert "event: content_block_delta" in text
        assert "event: message_stop" in text
    finally:
        await h.aclose()


@pytest.mark.asyncio
async def test_openai_streaming_emits_done_sentinel():
    h = await _start_server()
    try:
        body = json.dumps(
            {
                "model": "gpt-4o-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")

        async def _fake_acompletion(**_kwargs):
            return _AsyncStream(_basic_stream_chunks())

        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(side_effect=_fake_acompletion),
        ):
            status, headers, resp_body = await _post(
                h.host, h.port, "/v1/chat/completions", body
            )
        assert status == 200
        assert "text/event-stream" in headers.get("content-type", "")
        text = resp_body.decode("utf-8", errors="replace")
        assert "data: [DONE]\n\n" in text
    finally:
        await h.aclose()


@pytest.mark.asyncio
async def test_gemini_streaming_omits_done_sentinel():
    h = await _start_server()
    try:
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            }
        ).encode("utf-8")

        async def _fake_acompletion(**_kwargs):
            return _AsyncStream(_basic_stream_chunks())

        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(side_effect=_fake_acompletion),
        ):
            status, headers, resp_body = await _post(
                h.host,
                h.port,
                "/v1beta/models/gemini-1.5-flash:streamGenerateContent",
                body,
            )
        assert status == 200
        assert "text/event-stream" in headers.get("content-type", "")
        text = resp_body.decode("utf-8", errors="replace")
        # Gemini explicitly does not send [DONE].
        assert "[DONE]" not in text
        # But we did see at least one SSE-shaped frame.
        assert "data:" in text
    finally:
        await h.aclose()


# ---------------------------------------------------------------------------
# Test — ?conduct=off disables conduct injection on a single request.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conduct_off_query_param_skips_conduct_injection():
    """Verify conduct=off via query string suppresses the conduct XML.

    The server-wide flag is ``conduct=True``; the request URL flips it
    off for one request.  We assert via the upstream-side messages that
    no ``<conduct>`` XML envelope reached the system prompt.
    """
    h = await _start_server(conduct=True)
    try:
        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")
        with patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(return_value=_make_completion(text="ok")),
        ) as mocked:
            status, _headers, _body = await _post(
                h.host, h.port, "/v1/messages?conduct=off", body
            )
        assert status == 200
        assert mocked.await_count == 1
        kwargs = mocked.await_args.kwargs
        # Reduce all upstream messages to a single corpus string and confirm
        # the conduct XML didn't slip in.
        corpus_parts: list[str] = []
        for m in kwargs["messages"]:
            content = m.get("content")
            if isinstance(content, str):
                corpus_parts.append(content)
        corpus = "\n".join(corpus_parts)
        assert "<conduct>" not in corpus
    finally:
        await h.aclose()
