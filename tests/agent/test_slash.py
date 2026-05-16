"""Tests for robit.agent.slash — built-ins + dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from robit.agent.conversation import Conversation
from robit.agent.slash import (
    SlashContext,
    SlashExit,
    builtin_registry,
    dispatch_slash,
)
from robit.agent.tools import EchoTool, ToolRegistry


def _ctx(tmp_path: Path | None = None) -> SlashContext:
    tools = ToolRegistry()
    tools.register(EchoTool())
    return SlashContext(
        conversation=Conversation.new(model="m"),
        tool_registry=tools,
        audit_dir=tmp_path or Path("."),
    )


@pytest.mark.asyncio
async def test_help_lists_builtins(tmp_path):
    reg = builtin_registry()
    out = await dispatch_slash("/help", reg, _ctx(tmp_path))
    for name in ("/help", "/clear", "/exit", "/model", "/cost"):
        assert name in out
    # And lists the echo tool.
    assert "echo" in out


@pytest.mark.asyncio
async def test_clear_resets_messages_but_keeps_session_id(tmp_path):
    reg = builtin_registry()
    ctx = _ctx(tmp_path)
    ctx.conversation = ctx.conversation.append_user("hello")
    original_id = ctx.conversation.session_id

    out = await dispatch_slash("/clear", reg, ctx)
    assert "cleared" in out.lower()
    assert ctx.conversation.session_id == original_id
    assert ctx.conversation.messages == ()


@pytest.mark.asyncio
async def test_exit_raises_sentinel(tmp_path):
    reg = builtin_registry()
    with pytest.raises(SlashExit):
        await dispatch_slash("/exit", reg, _ctx(tmp_path))


@pytest.mark.asyncio
async def test_unknown_slash_returns_clear_message(tmp_path):
    reg = builtin_registry()
    out = await dispatch_slash("/nope", reg, _ctx(tmp_path))
    assert "Unknown" in out or "not found" in out.lower()


@pytest.mark.asyncio
async def test_model_swap(tmp_path):
    reg = builtin_registry()
    ctx = _ctx(tmp_path)
    assert ctx.conversation.model == "m"
    out = await dispatch_slash("/model new-model-1", reg, ctx)
    assert "new-model-1" in out
    assert ctx.conversation.model == "new-model-1"


@pytest.mark.asyncio
async def test_model_no_arg_reports_current(tmp_path):
    reg = builtin_registry()
    ctx = _ctx(tmp_path)
    out = await dispatch_slash("/model", reg, ctx)
    assert "m" in out
    assert ctx.conversation.model == "m"


@pytest.mark.asyncio
async def test_cost_is_placeholder(tmp_path):
    reg = builtin_registry()
    ctx = _ctx(tmp_path)
    out = await dispatch_slash("/cost", reg, ctx)
    assert "Wave 15.2" in out or "placeholder" in out.lower()
    assert ctx.conversation.session_id in out
