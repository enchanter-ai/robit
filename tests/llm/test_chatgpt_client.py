"""Tests for ChatGptClient — auth resolution + stubbed complete().

Wave 16.2 v2: the upstream call lives in Wave 16.3. We assert the honest
stub here so that no caller accidentally ships a silently-degraded client.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import patch

import pytest

from robit.llm import (
    ChatGptClient,
    CompletionRequest,
    ConfigurationError,
    Message,
)
from robit.llm._chatgpt_auth import ChatGptToken


def _fresh(**overrides) -> ChatGptToken:
    base = {
        "access_token": "access-tok",
        "refresh_token": "refresh-tok",
        "id_token": None,
        "expires_at": time.time() + 3600,
        "chatgpt_account_id": "acct-abc",
    }
    base.update(overrides)
    return ChatGptToken(**base)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_init_raises_when_no_credentials_anywhere(monkeypatch) -> None:
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    with patch("robit.llm.chatgpt_client.load_cached_token", return_value=None):
        with pytest.raises(ConfigurationError):
            ChatGptClient()


def test_init_resolves_from_explicit_arg(monkeypatch) -> None:
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    token = _fresh()
    with patch("robit.llm.chatgpt_client.load_cached_token", return_value=None):
        client = ChatGptClient(token=token)
    assert client.token is token
    assert client.auth_mode == "chatgpt-subscription"


def test_init_resolves_from_env_var_bare_token(monkeypatch) -> None:
    monkeypatch.setenv("CHATGPT_SESSION_TOKEN", "bare-access-token-string")
    with patch("robit.llm.chatgpt_client.load_cached_token", return_value=None):
        client = ChatGptClient()
    assert client.token.access_token == "bare-access-token-string"
    assert client.token.refresh_token is None


def test_init_resolves_from_env_var_json_blob(monkeypatch) -> None:
    blob = json.dumps(
        {
            "access_token": "env-access",
            "refresh_token": "env-refresh",
            "id_token": None,
            "expires_at": time.time() + 1800,
            "chatgpt_account_id": "acct-env",
        }
    )
    monkeypatch.setenv("CHATGPT_SESSION_TOKEN", blob)
    with patch("robit.llm.chatgpt_client.load_cached_token", return_value=None):
        client = ChatGptClient()
    assert client.token.access_token == "env-access"
    assert client.token.refresh_token == "env-refresh"
    assert client.token.chatgpt_account_id == "acct-env"


def test_init_resolves_from_cache_file(monkeypatch) -> None:
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    cached = _fresh(access_token="from-cache")
    with patch(
        "robit.llm.chatgpt_client.load_cached_token", return_value=cached
    ):
        client = ChatGptClient()
    assert client.token.access_token == "from-cache"


# ---------------------------------------------------------------------------
# has_valid_token
# ---------------------------------------------------------------------------


def test_has_valid_token_true_for_fresh(monkeypatch) -> None:
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    with patch("robit.llm.chatgpt_client.load_cached_token", return_value=None):
        client = ChatGptClient(token=_fresh())
    assert client.has_valid_token() is True


def test_has_valid_token_false_for_expired(monkeypatch) -> None:
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    expired = _fresh(expires_at=time.time() - 60)
    with patch("robit.llm.chatgpt_client.load_cached_token", return_value=None):
        client = ChatGptClient(token=expired)
    assert client.has_valid_token() is False


# ---------------------------------------------------------------------------
# complete() — Wave 16.3 wired the Responses API upstream call.
# ---------------------------------------------------------------------------


def test_complete_no_longer_raises_not_implemented(monkeypatch) -> None:
    """Regression guard: ChatGptClient.complete() must not be a stub.

    Heavier behaviour tests live in tests/llm/test_chatgpt_complete.py — this
    test only verifies the stub-removal commitment.
    """
    monkeypatch.delenv("CHATGPT_SESSION_TOKEN", raising=False)
    with patch("robit.llm.chatgpt_client.load_cached_token", return_value=None):
        client = ChatGptClient(token=_fresh())

    fake_body = {
        "id": "resp_abc",
        "object": "response",
        "model": "gpt-5",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    req = CompletionRequest(
        model="gpt-5",
        messages=[Message(role="user", content="hi")],
    )
    with patch(
        "robit.llm.chatgpt_client._post_responses", return_value=fake_body
    ):
        resp = asyncio.run(client.complete(req))
    assert resp.text == "hi"


# ---------------------------------------------------------------------------
# Live PKCE flow — skipped (would open a browser and hit real OpenAI)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Live PKCE flow requires a real browser + real OpenAI auth. "
    "Wave 16.3 will add a contract test against a recorded flow."
)
def test_run_pkce_flow_live() -> None:  # pragma: no cover
    raise AssertionError("Not executed.")
