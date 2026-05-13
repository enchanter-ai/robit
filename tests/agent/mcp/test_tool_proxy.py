"""Tests for enchanter.agent.mcp.tool_proxy.MCPToolProxy."""

from __future__ import annotations

from pathlib import Path

import pytest

from enchanter.agent.mcp.client import MCPCallError, MCPClient
from enchanter.agent.mcp.config import MCPServerConfig
from enchanter.agent.mcp.tool_proxy import MCPToolProxy
from enchanter.agent.tools._types import ToolContext


class StubClient:
    """Minimal stand-in for MCPClient.

    We don't go through the JSON-RPC layer for proxy tests — we only care
    that the proxy invokes the right remote name with the right args.
    """

    def __init__(self, tools: list[dict] | None = None, result: dict | Exception | None = None) -> None:
        self.config = MCPServerConfig(name="srv", command="x", args=(), env_allowlist=())
        self._tools = tools if tools is not None else []
        self._result = result
        self._connected = True
        self._closed = False
        self.calls: list[tuple[str, dict]] = []

    async def connect(self) -> dict:
        return {}

    async def list_tools(self) -> list[dict]:
        return self._tools

    async def call_tool(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        if isinstance(self._result, Exception):
            raise self._result
        assert self._result is not None
        return self._result


def _ctx() -> ToolContext:
    return ToolContext(cwd=Path("."), session_id="t", max_output_bytes=4096, timeout_s=5.0)


async def test_from_server_builds_namespaced_proxies() -> None:
    tools = [
        {"name": "read", "description": "read file", "inputSchema": {"type": "object"}},
        {"name": "write", "description": "write file", "inputSchema": {"type": "object"}},
    ]
    client = StubClient(tools=tools)
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    assert [p.name for p in proxies] == ["srv.read", "srv.write"]
    # All MCP proxies require approval — see tool_proxy module docstring.
    assert all(p.requires_approval for p in proxies)
    # Each carries its remote schema through to the registry listing.
    assert all(p.input_schema == {"type": "object"} for p in proxies)


async def test_execute_calls_remote_tool_with_unprefixed_name() -> None:
    tools = [{"name": "read", "description": "", "inputSchema": {}}]
    client = StubClient(
        tools=tools,
        result={"content": [{"type": "text", "text": "hello"}], "isError": False},
    )
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    proxy = proxies[0]
    result = await proxy.execute({"path": "x"}, _ctx())
    assert result.is_error is False
    assert result.content == "hello"
    # The remote name is "read", not "srv.read".
    assert client.calls == [("read", {"path": "x"})]


async def test_execute_with_text_only_content() -> None:
    client = StubClient(
        tools=[{"name": "t", "description": "", "inputSchema": {}}],
        result={
            "content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ],
            "isError": False,
        },
    )
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    result = await proxies[0].execute({}, _ctx())
    assert result.content == "line1\nline2"
    assert result.is_error is False


async def test_execute_with_mixed_content_falls_back_to_json() -> None:
    client = StubClient(
        tools=[{"name": "t", "description": "", "inputSchema": {}}],
        result={
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image", "data": "base64...", "mimeType": "image/png"},
            ],
            "isError": False,
        },
    )
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    result = await proxies[0].execute({}, _ctx())
    # Non-text → JSON-stringify the whole content list.
    assert result.content.startswith("[")
    assert "image" in result.content
    assert "text" in result.content


async def test_execute_with_is_error_true() -> None:
    client = StubClient(
        tools=[{"name": "t", "description": "", "inputSchema": {}}],
        result={"content": [{"type": "text", "text": "boom"}], "isError": True},
    )
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    result = await proxies[0].execute({}, _ctx())
    assert result.is_error is True
    assert result.content == "boom"


async def test_execute_translates_mcp_call_error_into_tool_result() -> None:
    client = StubClient(
        tools=[{"name": "t", "description": "", "inputSchema": {}}],
        result=MCPCallError("tools/call", -32602, "bad params"),
    )
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    result = await proxies[0].execute({}, _ctx())
    assert result.is_error is True
    assert "bad params" in result.content
    assert "-32602" in result.content


async def test_skips_malformed_tool_entries() -> None:
    tools = [
        {"name": "ok", "description": "", "inputSchema": {}},
        {"description": "no name field"},   # malformed — skipped
        "not even a dict",                  # malformed — skipped
        {"name": "", "description": ""},    # malformed — empty name
    ]
    client = StubClient(tools=tools)
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    assert [p.name for p in proxies] == ["srv.ok"]


async def test_missing_input_schema_defaults_to_permissive_object() -> None:
    client = StubClient(
        tools=[{"name": "t", "description": "no schema"}],
    )
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    p = proxies[0]
    assert p.input_schema["type"] == "object"
    assert p.input_schema.get("additionalProperties") is True


async def test_proxy_conforms_to_tool_protocol() -> None:
    """MCPToolProxy must satisfy the Tool runtime protocol so ToolRegistry.register accepts it."""
    from enchanter.agent.tools import Tool, ToolRegistry

    client = StubClient(
        tools=[{"name": "t", "description": "d", "inputSchema": {}}],
        result={"content": [{"type": "text", "text": "x"}], "isError": False},
    )
    proxies = await MCPToolProxy.from_server(client)  # type: ignore[arg-type]
    assert isinstance(proxies[0], Tool)
    reg = ToolRegistry()
    reg.register(proxies[0])
    assert "srv.t" in reg
