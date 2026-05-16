"""Integration tests for robit.proxy.server — real TCP, mocked upstream.

Spins up a :class:`robit.proxy.server.ProxyServer` on ``127.0.0.1:0`` (OS-
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
pipeline = pytest.importorskip("robit.proxy.pipeline")
streaming = pytest.importorskip("robit.proxy.streaming")

from robit.proxy import upstream  # noqa: E402
from robit.proxy.server import ProxyServer  # noqa: E402


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


# ---------------------------------------------------------------------------
# Tests — pass-through auth (Wave 16.1).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passthrough_auth_true_captures_x_api_key_into_metadata():
    """With passthrough_auth=True, the inbound x-api-key reaches the
    LiteLLM call as the ``api_key`` kwarg, instead of forcing the
    operator to set ANTHROPIC_API_KEY in the env."""
    h = await _start_server(passthrough_auth=True)
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
                h.host,
                h.port,
                "/v1/messages",
                body,
                extra_headers=(("x-api-key", "sk-ant-host-agent-key"),),
            )
        assert status == 200
        assert mocked.await_count == 1
        kwargs = mocked.await_args.kwargs
        assert kwargs["api_key"] == "sk-ant-host-agent-key"
    finally:
        await h.aclose()


@pytest.mark.asyncio
async def test_passthrough_auth_false_default_does_not_capture_auth():
    """With passthrough_auth disabled (default), the inbound x-api-key
    is NOT forwarded as an api_key kwarg — LiteLLM falls back to env
    vars, preserving the legacy contract."""
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
            new=AsyncMock(return_value=_make_completion(text="ok")),
        ) as mocked:
            status, _headers, _body = await _post(
                h.host,
                h.port,
                "/v1/messages",
                body,
                extra_headers=(("x-api-key", "sk-ant-should-be-ignored"),),
            )
        assert status == 200
        kwargs = mocked.await_args.kwargs
        assert "api_key" not in kwargs
        # And the sentinel never reaches the LiteLLM metadata bag either.
        meta = kwargs.get("metadata", {})
        assert "_enchanter_passthrough_auth" not in meta


    finally:
        await h.aclose()


@pytest.mark.asyncio
async def test_passthrough_auth_openai_bearer_captured():
    """OpenAI family — Authorization: Bearer ... reaches LiteLLM."""
    h = await _start_server(passthrough_auth=True)
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
            new=AsyncMock(return_value=_make_completion(text="ok")),
        ) as mocked:
            status, _headers, _body = await _post(
                h.host,
                h.port,
                "/v1/chat/completions",
                body,
                extra_headers=(("Authorization", "Bearer sk-openai-host"),),
            )
        assert status == 200
        kwargs = mocked.await_args.kwargs
        assert kwargs["api_key"] == "sk-openai-host"
    finally:
        await h.aclose()


# ---------------------------------------------------------------------------
# Tests — JWT-shape detection in _extract_inbound_auth (Wave 17.2).
# ---------------------------------------------------------------------------


def test_extract_inbound_auth_codex_detects_jwt_shape():
    """Codex /v1/responses + Bearer eyJ… → kind='chatgpt-jwt' with account_id."""
    from robit.proxy.server import _extract_inbound_auth

    headers = {
        "authorization": "Bearer eyJhbGciOi.payload-body.sig-tail",
        "chatgpt-account-id": "acct_123",
    }
    auth = _extract_inbound_auth(headers, "codex")
    assert auth is not None
    assert auth["kind"] == "chatgpt-jwt"
    assert auth["value"] == "eyJhbGciOi.payload-body.sig-tail"
    assert auth["account_id"] == "acct_123"


def test_extract_inbound_auth_codex_jwt_missing_account_id_is_none():
    """Missing ChatGPT-Account-ID is captured as None, not synthesised."""
    from robit.proxy.server import _extract_inbound_auth

    headers = {"authorization": "Bearer eyJabc.def.ghi"}
    auth = _extract_inbound_auth(headers, "codex")
    assert auth is not None
    assert auth["kind"] == "chatgpt-jwt"
    assert auth["account_id"] is None


def test_extract_inbound_auth_codex_non_jwt_bearer_is_openai_bearer():
    """A non-JWT Bearer (sk-…) on /v1/responses stays as openai-bearer."""
    from robit.proxy.server import _extract_inbound_auth

    headers = {"authorization": "Bearer sk-proj-abc-1234"}
    auth = _extract_inbound_auth(headers, "codex")
    assert auth is not None
    assert auth["kind"] == "openai-bearer"
    assert auth["value"] == "sk-proj-abc-1234"
    assert "account_id" not in auth


def test_extract_inbound_auth_openai_family_also_detects_jwt():
    """JWT-shaped tokens on the OpenAI family route to chatgpt-jwt too.

    Wave 17.2: some host agents may route Codex through the OpenAI adapter
    by accident — we honour the JWT shape regardless of family.
    """
    from robit.proxy.server import _extract_inbound_auth

    headers = {
        "authorization": "Bearer eyJraw.body.tail",
        "chatgpt-account-id": "acct_oai",
    }
    auth = _extract_inbound_auth(headers, "openai")
    assert auth is not None
    assert auth["kind"] == "chatgpt-jwt"
    assert auth["account_id"] == "acct_oai"


def test_looks_like_jwt_shape_matcher():
    """The JWT regex matches the canonical three-segment compact form."""
    from robit.proxy.server import _looks_like_jwt

    # Three base64url segments, eyJ prefix → match.
    assert _looks_like_jwt("eyJabc.def-_AB.gh_-iZ") is True
    # Missing one segment → no match.
    assert _looks_like_jwt("eyJabc.def") is False
    # Wrong prefix → no match.
    assert _looks_like_jwt("sk-abc.def.ghi") is False
    # Empty → no match.
    assert _looks_like_jwt("") is False
    # JWT-ish but with a forbidden char (=).
    assert _looks_like_jwt("eyJabc.de=f.ghi") is False


# ---------------------------------------------------------------------------
# Test — end-to-end Codex /v1/responses + JWT routes through
# _call_chatgpt_internal, NOT LiteLLM.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chatgpt_jwt_request_routes_through_chatgpt_internal_path():
    """POST /v1/responses with Bearer JWT + ChatGPT-Account-ID:
    upstream.call_upstream is invoked, _call_chatgpt_internal handles it,
    LiteLLM is never called.
    """
    from robit.proxy.canonical import (
        CanonicalResponse,
        CanonicalUsage,
        TextPart,
    )

    fake_canonical = CanonicalResponse(
        model="gpt-5-codex",
        content=(TextPart(text="hello chatgpt"),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=4, output_tokens=2),
    )

    h = await _start_server(passthrough_auth=True)
    try:
        body = json.dumps(
            {
                "model": "gpt-5-codex",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            }
        ).encode("utf-8")

        async def _fake_internal(req, auth):
            # Assert the auth blob carried through.
            assert auth["kind"] == "chatgpt-jwt"
            assert auth["value"].startswith("eyJ")
            assert auth["account_id"] == "acct_e2e"
            return fake_canonical

        with patch.object(
            upstream, "_call_chatgpt_internal", new=AsyncMock(side_effect=_fake_internal)
        ) as mocked_internal, patch.object(
            upstream.litellm, "acompletion", new=AsyncMock()
        ) as mocked_litellm:
            status, headers, resp_body = await _post(
                h.host,
                h.port,
                "/v1/responses",
                body,
                extra_headers=(
                    ("Authorization", "Bearer eyJh.payload.sig"),
                    ("ChatGPT-Account-ID", "acct_e2e"),
                ),
            )

        assert status == 200
        assert mocked_internal.await_count == 1
        assert mocked_litellm.await_count == 0
        obj = json.loads(resp_body)
        # CodexAdapter.render_response shape: output[0].content[0].text
        assert obj["output"][0]["content"][0]["text"] == "hello chatgpt"
    finally:
        await h.aclose()


@pytest.mark.asyncio
async def test_chatgpt_jwt_request_without_passthrough_does_not_route():
    """With passthrough_auth=False, the JWT is NOT captured; the request
    falls through to LiteLLM (which will 404 in production, but in the
    test we mock acompletion and just assert the routing decision)."""
    h = await _start_server(passthrough_auth=False)
    try:
        body = json.dumps(
            {
                "model": "gpt-5-codex",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            }
        ).encode("utf-8")
        with patch.object(
            upstream, "_call_chatgpt_internal", new=AsyncMock()
        ) as mocked_internal, patch.object(
            upstream.litellm,
            "acompletion",
            new=AsyncMock(return_value=_make_completion(text="lite")),
        ) as mocked_litellm:
            status, _headers, _resp_body = await _post(
                h.host,
                h.port,
                "/v1/responses",
                body,
                extra_headers=(
                    ("Authorization", "Bearer eyJh.payload.sig"),
                    ("ChatGPT-Account-ID", "acct_e2e"),
                ),
            )
        assert status == 200
        # Routing decision: passthrough off → LiteLLM, not the internal path.
        assert mocked_internal.await_count == 0
        assert mocked_litellm.await_count == 1
    finally:
        await h.aclose()
