"""Tests for robit.agent.mcp.client.MCPClient.

We mock :class:`robit.transport.stdio.StdioTransport` so no real
subprocess is spawned. The fake transport queues incoming messages from
"the server" and records anything the client sends.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from robit.agent.mcp.client import MCPCallError, MCPClient
from robit.agent.mcp.config import MCPServerConfig


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeTransport:
    """Stand-in for :class:`StdioTransport` driven by an asyncio queue."""

    def __init__(self) -> None:
        self._incoming: asyncio.Queue[dict | None] = asyncio.Queue()
        self.sent: list[dict] = []
        self.started = False
        self.closed = False
        # Optional hook: response_for(req) returns a dict to enqueue, or
        # None to enqueue nothing. The default echoes back a generic result.
        self.responder = self._default_responder
        # If set, send() will raise this exception instead of recording.
        self.send_error: Exception | None = None

    async def start(self, descriptor) -> None:  # noqa: ANN001
        self.started = True

    async def send(self, message: dict) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(message)
        if message.get("id") is not None:
            reply = self.responder(message)
            if reply is not None:
                await self._incoming.put(reply)

    async def receive(self) -> dict | None:
        return await self._incoming.get()

    async def close(self) -> None:
        self.closed = True
        # Wake any pending receive() with EOF.
        await self._incoming.put(None)

    # ---- helpers for tests ---------------------------------------------

    def push(self, message: dict | None) -> None:
        """Synchronously enqueue a message (for tests that drive the loop)."""
        self._incoming.put_nowait(message)

    def _default_responder(self, req: dict) -> dict | None:
        method = req.get("method")
        rid = req.get("id")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "fake-mcp", "version": "0.0"},
                },
            }
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "echo input",
                            "inputSchema": {"type": "object"},
                        },
                        {
                            "name": "add",
                            "description": "add two numbers",
                            "inputSchema": {"type": "object"},
                        },
                    ],
                },
            }
        if method == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                },
            }
        # Unknown — return method-not-found error.
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(transport: FakeTransport | None = None, **kwargs: Any) -> MCPClient:
    cfg = MCPServerConfig(
        name="test",
        command="dummy",
        args=(),
        env_allowlist=(),
    )
    t = transport if transport is not None else FakeTransport()
    return MCPClient(cfg, transport=t, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_runs_handshake() -> None:
    transport = FakeTransport()
    client = _make_client(transport)
    info = await client.connect()
    assert "serverInfo" in info
    assert info["serverInfo"]["name"] == "fake-mcp"

    # Must have sent initialize + notifications/initialized.
    methods = [m.get("method") for m in transport.sent]
    assert methods[0] == "initialize"
    assert "notifications/initialized" in methods
    # The notification carries no id (per JSON-RPC).
    note = next(m for m in transport.sent if m.get("method") == "notifications/initialized")
    assert "id" not in note

    await client.close()


@pytest.mark.asyncio
async def test_list_tools_returns_catalog() -> None:
    transport = FakeTransport()
    client = _make_client(transport)
    await client.connect()
    tools = await client.list_tools()
    assert [t["name"] for t in tools] == ["echo", "add"]
    await client.close()


@pytest.mark.asyncio
async def test_list_tools_is_cached() -> None:
    transport = FakeTransport()
    client = _make_client(transport)
    await client.connect()
    await client.list_tools()
    sent_before = sum(1 for m in transport.sent if m.get("method") == "tools/list")
    await client.list_tools()
    sent_after = sum(1 for m in transport.sent if m.get("method") == "tools/list")
    assert sent_before == 1
    assert sent_after == 1
    await client.close()


@pytest.mark.asyncio
async def test_call_tool_success() -> None:
    transport = FakeTransport()
    client = _make_client(transport)
    await client.connect()
    result = await client.call_tool("echo", {"text": "hi"})
    assert result["isError"] is False
    assert result["content"][0]["text"] == "ok"
    await client.close()


@pytest.mark.asyncio
async def test_call_tool_jsonrpc_error_raises() -> None:
    transport = FakeTransport()

    def responder(req: dict) -> dict | None:
        if req.get("method") == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": req["id"],
                "error": {"code": -32602, "message": "bad params"},
            }
        return transport._default_responder(req)

    transport.responder = responder
    client = _make_client(transport)
    await client.connect()
    with pytest.raises(MCPCallError) as exc:
        await client.call_tool("echo", {})
    assert exc.value.code == -32602
    assert "bad params" in exc.value.message
    assert exc.value.method == "tools/call"
    await client.close()


@pytest.mark.asyncio
async def test_call_tool_timeout_raises_mcp_call_error() -> None:
    transport = FakeTransport()

    # tools/call request never gets a response.
    def responder(req: dict) -> dict | None:
        if req.get("method") == "tools/call":
            return None
        return transport._default_responder(req)

    transport.responder = responder
    client = _make_client(transport, timeout_s=0.05)
    await client.connect()
    with pytest.raises(MCPCallError) as exc:
        await client.call_tool("echo", {})
    assert exc.value.code == -1
    assert "timeout" in exc.value.message.lower()
    await client.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_closes_transport() -> None:
    transport = FakeTransport()
    client = _make_client(transport)
    await client.connect()
    await client.close()
    assert transport.closed is True
    # Second close is a no-op (no exception).
    await client.close()


@pytest.mark.asyncio
async def test_call_after_close_raises() -> None:
    transport = FakeTransport()
    client = _make_client(transport)
    await client.connect()
    await client.close()
    with pytest.raises(MCPCallError):
        await client.call_tool("echo", {})


@pytest.mark.asyncio
async def test_connect_is_idempotent() -> None:
    transport = FakeTransport()
    client = _make_client(transport)
    info1 = await client.connect()
    info2 = await client.connect()
    assert info1 is info2
    # Only one initialize request actually went out.
    init_count = sum(1 for m in transport.sent if m.get("method") == "initialize")
    assert init_count == 1
    await client.close()
