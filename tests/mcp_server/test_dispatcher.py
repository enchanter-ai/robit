"""Unit tests for the MCP server dispatcher."""

from __future__ import annotations

import json

import pytest

from robit.mcp_server.dispatcher import Dispatcher, PROTOCOL_VERSION
from robit.mcp_server.tools import Tool, ToolRegistry, register_default_tools


def _make_dispatcher() -> Dispatcher:
    reg = ToolRegistry()
    register_default_tools(reg)
    return Dispatcher(reg)


def _decode(raw: str) -> dict:
    return json.loads(raw)


@pytest.mark.asyncio
async def test_initialize_returns_capabilities() -> None:
    d = _make_dispatcher()
    raw_req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    resp = await d.handle_raw(raw_req)
    assert resp is not None
    obj = _decode(resp)
    assert obj["id"] == 1
    assert obj["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert "tools" in obj["result"]["capabilities"]
    assert obj["result"]["serverInfo"]["name"] == "enchanter-mcp-server"


@pytest.mark.asyncio
async def test_tools_list_returns_registered_tools() -> None:
    d = _make_dispatcher()
    raw = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    resp = await d.handle_raw(raw)
    assert resp is not None
    obj = _decode(resp)
    names = {t["name"] for t in obj["result"]["tools"]}
    assert "robit.scan_secrets" in names
    assert "robit.check_destructive_op" in names
    # Each tool exposes an inputSchema
    for t in obj["result"]["tools"]:
        assert isinstance(t["inputSchema"], dict)
        assert t["inputSchema"]["type"] == "object"


@pytest.mark.asyncio
async def test_method_not_found_returns_error_code() -> None:
    d = _make_dispatcher()
    raw = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "no_such_method"})
    resp = await d.handle_raw(raw)
    obj = _decode(resp)  # type: ignore[arg-type]
    assert obj["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_parse_error_returns_null_id() -> None:
    d = _make_dispatcher()
    resp = await d.handle_raw("not json at all")
    obj = _decode(resp)  # type: ignore[arg-type]
    assert obj["id"] is None
    assert obj["error"]["code"] == -32700


@pytest.mark.asyncio
async def test_invalid_request_shape_returns_parse_error() -> None:
    # Missing "jsonrpc" field
    d = _make_dispatcher()
    raw = json.dumps({"id": 4, "method": "ping"})
    resp = await d.handle_raw(raw)
    obj = _decode(resp)  # type: ignore[arg-type]
    assert obj["error"]["code"] == -32700


@pytest.mark.asyncio
async def test_notifications_initialized_is_silent() -> None:
    d = _make_dispatcher()
    raw = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    resp = await d.handle_raw(raw)
    assert resp is None  # notifications never produce responses


@pytest.mark.asyncio
async def test_tools_call_invalid_params_no_name() -> None:
    d = _make_dispatcher()
    raw = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {}})
    resp = await d.handle_raw(raw)
    obj = _decode(resp)  # type: ignore[arg-type]
    assert obj["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_tools_call_unknown_tool() -> None:
    d = _make_dispatcher()
    raw = json.dumps({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "nope.bogus", "arguments": {}},
    })
    resp = await d.handle_raw(raw)
    obj = _decode(resp)  # type: ignore[arg-type]
    assert obj["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_ping_returns_empty_object() -> None:
    d = _make_dispatcher()
    raw = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"})
    resp = await d.handle_raw(raw)
    obj = _decode(resp)  # type: ignore[arg-type]
    assert obj["result"] == {}


@pytest.mark.asyncio
async def test_custom_tool_dispatched() -> None:
    reg = ToolRegistry()

    async def echo(args: dict) -> dict:
        return {"echo": args.get("v")}

    reg.register(Tool(
        name="echo",
        description="echo back",
        input_schema={"type": "object", "properties": {"v": {"type": "string"}}},
        handler=echo,
    ))
    d = Dispatcher(reg)
    raw = json.dumps({
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "echo", "arguments": {"v": "hi"}},
    })
    resp = await d.handle_raw(raw)
    obj = _decode(resp)  # type: ignore[arg-type]
    # MCP tools/call result envelope
    assert obj["result"]["isError"] is False
    inner = json.loads(obj["result"]["content"][0]["text"])
    assert inner == {"echo": "hi"}
