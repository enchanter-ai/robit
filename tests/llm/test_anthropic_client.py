"""Tests for AnthropicClient — no real API calls.

Four mandatory cases:
  1. Construct without API key raises ValueError (when env is also absent).
  2. complete() calls the SDK with correctly translated args.
  3. SDK response is translated back to CompletionResponse correctly.
  4. Tool-use response (tool_calls in SDK response) is captured.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robit.llm import AnthropicClient, CompletionRequest, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sdk_text_response(
    text: str = "hello",
    model: str = "claude-sonnet-4-6",
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Fabricate a minimal object that looks like an Anthropic SDK response."""
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        content=[block],
        model=model,
        stop_reason=stop_reason,
        usage=usage,
    )


def _make_sdk_tool_response(tool_id: str, name: str, inp: dict) -> MagicMock:
    """Fabricate a response that contains a tool_use block alongside text."""
    text_block = SimpleNamespace(type="text", text="I'll call a tool.")
    tool_block = SimpleNamespace(type="tool_use", id=tool_id, name=name, input=inp)
    usage = SimpleNamespace(input_tokens=20, output_tokens=8)
    return SimpleNamespace(
        content=[text_block, tool_block],
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        usage=usage,
    )


def _simple_req(system: str | None = None) -> CompletionRequest:
    return CompletionRequest(
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="Say hi")],
        system=system,
        max_tokens=256,
    )


# ---------------------------------------------------------------------------
# Shared fixture: patch anthropic.AsyncAnthropic so no network call is made.
# ---------------------------------------------------------------------------

def _make_async_anthropic_mock(sdk_response) -> tuple[MagicMock, MagicMock]:
    """Return (mock_class, mock_instance) with messages.create scripted."""
    mock_instance = MagicMock()
    mock_instance.messages.create = AsyncMock(return_value=sdk_response)
    mock_class = MagicMock(return_value=mock_instance)
    return mock_class, mock_instance


# ---------------------------------------------------------------------------
# Test 1 — construction without API key raises ValueError
# ---------------------------------------------------------------------------

def test_construct_without_api_key_raises(monkeypatch):
    # AnthropicClient now also accepts CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_AUTH_TOKEN
    # for subscription auth — clear all three so the no-creds path actually fires.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    # We still need anthropic importable so the deferred import succeeds.
    fake_anthropic = MagicMock()
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        with pytest.raises(ValueError, match="No Anthropic credentials"):
            AnthropicClient()


def test_construct_with_env_var_succeeds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-from-env")
    fake_anthropic = MagicMock()
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        client = AnthropicClient()
        # Just check it constructs without raising.
        assert client is not None


# ---------------------------------------------------------------------------
# Test 2 — complete() calls the SDK with correctly translated args
# ---------------------------------------------------------------------------

async def test_complete_calls_sdk_with_correct_args(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    sdk_response = _make_sdk_text_response()
    mock_class, mock_instance = _make_async_anthropic_mock(sdk_response)
    fake_anthropic = MagicMock()
    fake_anthropic.AsyncAnthropic = mock_class

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        client = AnthropicClient()
        req = CompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Say hi")],
            system="You are helpful.",
            max_tokens=512,
            temperature=0.7,
            stop_sequences=("</answer>",),
        )
        await client.complete(req)

    call_kwargs = mock_instance.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 512
    assert call_kwargs["system"] == "You are helpful."
    assert call_kwargs["temperature"] == pytest.approx(0.7)
    assert call_kwargs["stop_sequences"] == ["</answer>"]
    assert call_kwargs["messages"] == [{"role": "user", "content": "Say hi"}]


# ---------------------------------------------------------------------------
# Test 3 — SDK response is translated back to CompletionResponse correctly
# ---------------------------------------------------------------------------

async def test_sdk_response_translates_to_completion_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    sdk_resp = _make_sdk_text_response(
        text="Hello there!",
        model="claude-haiku-4-6",
        stop_reason="end_turn",
        input_tokens=7,
        output_tokens=3,
    )
    mock_class, _ = _make_async_anthropic_mock(sdk_resp)
    fake_anthropic = MagicMock()
    fake_anthropic.AsyncAnthropic = mock_class

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        client = AnthropicClient()
        response = await client.complete(_simple_req())

    assert response.text == "Hello there!"
    assert response.model == "claude-haiku-4-6"
    assert response.stop_reason == "end_turn"
    assert response.input_tokens == 7
    assert response.output_tokens == 3
    assert response.tool_calls == []


# ---------------------------------------------------------------------------
# Test 4 — tool-use response captures tool_calls in CompletionResponse
# ---------------------------------------------------------------------------

async def test_tool_use_response_captured_in_tool_calls(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    sdk_resp = _make_sdk_tool_response(
        tool_id="call_abc123",
        name="read_file",
        inp={"path": "/etc/hosts"},
    )
    mock_class, _ = _make_async_anthropic_mock(sdk_resp)
    fake_anthropic = MagicMock()
    fake_anthropic.AsyncAnthropic = mock_class

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        client = AnthropicClient()
        response = await client.complete(_simple_req())

    assert response.stop_reason == "tool_use"
    assert response.text == "I'll call a tool."
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc["id"] == "call_abc123"
    assert tc["name"] == "read_file"
    assert tc["input"] == {"path": "/etc/hosts"}


# ---------------------------------------------------------------------------
# Test 5 — system=None is omitted from SDK kwargs
# ---------------------------------------------------------------------------

async def test_no_system_prompt_omits_system_kwarg(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    sdk_resp = _make_sdk_text_response()
    mock_class, mock_instance = _make_async_anthropic_mock(sdk_resp)
    fake_anthropic = MagicMock()
    fake_anthropic.AsyncAnthropic = mock_class

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        client = AnthropicClient()
        await client.complete(_simple_req(system=None))

    call_kwargs = mock_instance.messages.create.call_args.kwargs
    assert "system" not in call_kwargs
