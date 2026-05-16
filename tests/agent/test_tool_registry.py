"""Tests for robit.agent.tools — registry semantics + echo dummy."""

from __future__ import annotations

from pathlib import Path

import pytest

from robit.agent.tools import EchoTool, Tool, ToolContext, ToolRegistry, ToolResult


def test_register_get_contains():
    r = ToolRegistry()
    t = EchoTool()
    r.register(t)
    assert "echo" in r
    assert r.get("echo") is t


def test_duplicate_registration_raises():
    r = ToolRegistry()
    r.register(EchoTool())
    with pytest.raises(ValueError, match="already registered"):
        r.register(EchoTool())


def test_get_missing_tool_raises():
    r = ToolRegistry()
    with pytest.raises(KeyError):
        r.get("nope")


def test_listing_shape_is_llm_compatible():
    r = ToolRegistry()
    r.register(EchoTool())
    listing = r.listing()
    assert len(listing) == 1
    entry = listing[0]
    assert set(entry.keys()) == {"name", "description", "input_schema"}
    assert entry["name"] == "echo"
    assert isinstance(entry["input_schema"], dict)
    assert "text" in entry["input_schema"]["properties"]


def test_non_protocol_object_rejected():
    r = ToolRegistry()
    with pytest.raises(TypeError):
        r.register("not a tool")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_echo_tool_round_trips_text():
    t = EchoTool()
    ctx = ToolContext(cwd=Path("."), session_id="abc")
    res = await t.execute({"text": "hello world"}, ctx)
    assert isinstance(res, ToolResult)
    assert res.content == "hello world"
    assert res.is_error is False


@pytest.mark.asyncio
async def test_echo_tool_rejects_non_string():
    t = EchoTool()
    ctx = ToolContext(cwd=Path("."), session_id="abc")
    res = await t.execute({"text": 123}, ctx)
    assert res.is_error is True
