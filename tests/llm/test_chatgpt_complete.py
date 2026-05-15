"""Wave 16.3 — focused tests for ChatGptClient.complete().

Mocks ``urllib.request.urlopen`` so no real network traffic ever leaves the
test process. We verify:

* The right URL + headers (Authorization, ChatGPT-Account-ID, User-Agent).
* The response body is parsed via ``_codex_responses.parse_responses_completion``.
* HTTP 401 triggers exactly one refresh + retry.
* A persistent 401 raises with a clear "re-run codex login" message.
* ``req.stream=True`` raises NotImplementedError (v1 limitation).
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import urllib.error
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from enchanter.llm import ChatGptClient, CompletionRequest, Message
from enchanter.llm._chatgpt_auth import ChatGptToken


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _fresh_token(**overrides) -> ChatGptToken:
    base = {
        "access_token": "access-jwt",
        "refresh_token": "refresh-jwt",
        "id_token": None,
        "expires_at": time.time() + 3600,
        "chatgpt_account_id": "acct-xyz",
    }
    base.update(overrides)
    return ChatGptToken(**base)


def _success_body(text: str = "hello") -> dict:
    return {
        "id": "resp_abc",
        "object": "response",
        "model": "gpt-5-codex",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }


@dataclass
class _FakeUrlopenResponse:
    """Stand-in for the context manager returned by ``urllib.request.urlopen``."""

    body: bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.body


def _make_urlopen_capture():
    """Build a urlopen mock that captures the Request it was called with."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["url"] = req.get_full_url()
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        return _FakeUrlopenResponse(json.dumps(_success_body("ok")).encode("utf-8"))

    return fake_urlopen, captured


def _new_client_with(token: ChatGptToken) -> ChatGptClient:
    """Construct a ChatGptClient cleanly (no cache file, no env)."""
    with patch(
        "enchanter.llm.chatgpt_client.load_cached_token", return_value=None
    ):
        return ChatGptClient(token=token)


# ---------------------------------------------------------------------------
# Cases.
# ---------------------------------------------------------------------------


def test_complete_posts_to_chatgpt_internal_endpoint(monkeypatch):
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    client = _new_client_with(_fresh_token())
    fake_urlopen, captured = _make_urlopen_capture()

    req = CompletionRequest(
        model="gpt-5-codex",
        messages=[Message(role="user", content="hi")],
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resp = asyncio.run(client.complete(req))

    assert captured["url"] == (
        "https://chatgpt.com/backend-api/codex/responses"
    )
    # urllib lower-cases header names on Request.header_items() — match
    # case-insensitively.
    norm = {k.lower(): v for k, v in captured["headers"].items()}
    assert norm["authorization"] == "Bearer access-jwt"
    assert "User-Agent".lower() in norm
    assert "codex-responses" in norm["user-agent"]
    assert resp.text == "ok"
    assert resp.model == "gpt-5-codex"


def test_complete_includes_chatgpt_account_id_header(monkeypatch):
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    client = _new_client_with(_fresh_token(chatgpt_account_id="acct-xyz"))
    fake_urlopen, captured = _make_urlopen_capture()

    req = CompletionRequest(
        model="gpt-5-codex",
        messages=[Message(role="user", content="hi")],
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        asyncio.run(client.complete(req))

    norm = {k.lower(): v for k, v in captured["headers"].items()}
    assert norm["chatgpt-account-id"] == "acct-xyz"


def test_complete_parses_response_body_via_shared_helper(monkeypatch):
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    client = _new_client_with(_fresh_token())

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeUrlopenResponse(
            json.dumps(_success_body("multi-token reply")).encode("utf-8")
        )

    req = CompletionRequest(
        model="gpt-5-codex",
        messages=[Message(role="user", content="ping")],
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resp = asyncio.run(client.complete(req))

    assert resp.text == "multi-token reply"
    assert resp.input_tokens == 3
    assert resp.output_tokens == 2
    assert resp.stop_reason == "completed"


def test_complete_on_401_refreshes_token_and_retries(monkeypatch):
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    client = _new_client_with(_fresh_token(access_token="old-jwt"))

    calls: list[str] = []

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        norm = {k.lower(): v for k, v in dict(req.header_items()).items()}
        bearer = norm.get("authorization", "")
        calls.append(bearer)
        if bearer == "Bearer old-jwt":
            raise urllib.error.HTTPError(
                req.get_full_url(), 401, "Unauthorized", None, io.BytesIO(b"")
            )
        return _FakeUrlopenResponse(
            json.dumps(_success_body("after-refresh")).encode("utf-8")
        )

    refreshed = _fresh_token(access_token="new-jwt")
    refresh_calls = {"n": 0}

    async def fake_refresh(token, **kwargs):  # noqa: ARG001
        # First call (the proactive pre-flight refresh) is a no-op: the
        # cached token still has 1h of life. The second call (post-401)
        # actually swaps in the new JWT.
        refresh_calls["n"] += 1
        if refresh_calls["n"] == 1:
            return token
        return refreshed

    req = CompletionRequest(
        model="gpt-5-codex",
        messages=[Message(role="user", content="hi")],
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch(
        "enchanter.llm.chatgpt_client.refresh_if_needed", side_effect=fake_refresh
    ):
        resp = asyncio.run(client.complete(req))

    # Two requests: the initial one (old-jwt, 401) and the retry (new-jwt, 200).
    assert calls == ["Bearer old-jwt", "Bearer new-jwt"]
    assert resp.text == "after-refresh"
    assert client.token.access_token == "new-jwt"


def test_complete_on_persistent_401_raises_with_clear_message(monkeypatch):
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    client = _new_client_with(_fresh_token())

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(
            req.get_full_url(), 401, "Unauthorized", None, io.BytesIO(b"")
        )

    async def fake_refresh(token, **kwargs):  # noqa: ARG001
        return token  # pretend refresh succeeded but token still invalid

    req = CompletionRequest(
        model="gpt-5-codex",
        messages=[Message(role="user", content="hi")],
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch(
        "enchanter.llm.chatgpt_client.refresh_if_needed", side_effect=fake_refresh
    ):
        with pytest.raises(urllib.error.HTTPError) as exc:
            asyncio.run(client.complete(req))

    assert exc.value.code == 401
    msg = str(exc.value)
    assert "codex login" in msg


def test_complete_with_stream_true_raises_not_implemented(monkeypatch):
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    client = _new_client_with(_fresh_token())

    req = CompletionRequest(
        model="gpt-5-codex",
        messages=[Message(role="user", content="hi")],
    )
    # CompletionRequest is frozen and has no stream field — simulate the
    # forward-compat extension by patching getattr to claim stream=True.
    fake_req = MagicMock(wraps=req)
    fake_req.stream = True
    fake_req.model = req.model
    fake_req.messages = req.messages
    fake_req.system = req.system

    with pytest.raises(NotImplementedError) as exc:
        asyncio.run(client.complete(fake_req))

    assert "streaming" in str(exc.value).lower()
