"""In-process stdio integration test for the MCP server.

Builds two asyncio StreamReader/Writer pairs wired through asyncio.Queue
fakes — no subprocess. The server consumes from one side, the test
writes/reads on the other.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from robit.mcp_server.server import MCPServer
from robit.mcp_server.stdio import ServerStdioTransport


class _MemoryReader:
    """Just enough of asyncio.StreamReader for ServerStdioTransport."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._event = asyncio.Event()
        self._eof = False

    def write(self, data: bytes) -> None:
        self._buffer.extend(data)
        self._event.set()

    def write_line(self, line: str) -> None:
        self.write(line.encode("utf-8") + b"\n")

    def eof(self) -> None:
        self._eof = True
        self._event.set()

    async def read(self, n: int = -1) -> bytes:
        while not self._buffer and not self._eof:
            self._event.clear()
            await self._event.wait()
        if not self._buffer and self._eof:
            return b""
        size = len(self._buffer) if n < 0 else min(n, len(self._buffer))
        out = bytes(self._buffer[:size])
        del self._buffer[:size]
        return out

    def feed_data(self, data: bytes) -> None:
        # ServerStdioTransport calls this to push back unread bytes.
        self._buffer[:0] = data
        self._event.set()


class _MemoryWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._buffer = bytearray()
        self.event = asyncio.Event()

    def write(self, data: bytes) -> None:
        self._buffer.extend(data)
        # Split out complete lines as soon as they arrive
        while b"\n" in self._buffer:
            idx = self._buffer.index(b"\n")
            line = bytes(self._buffer[:idx]).decode("utf-8")
            del self._buffer[: idx + 1]
            self.lines.append(line)
            self.event.set()

    async def drain(self) -> None:
        return

    def close(self) -> None:
        return

    async def wait_closed(self) -> None:
        return


async def _next_response(writer: _MemoryWriter, timeout: float = 2.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    while not writer.lines:
        writer.event.clear()
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError("no response received")
        await asyncio.wait_for(writer.event.wait(), timeout=remaining)
    return json.loads(writer.lines.pop(0))


@pytest.mark.asyncio
async def test_stdio_initialize_list_call_round_trip() -> None:
    reader = _MemoryReader()
    writer = _MemoryWriter()
    server = MCPServer()
    transport = ServerStdioTransport(reader, writer, server.handle_raw)  # type: ignore[arg-type]

    serve_task = asyncio.create_task(transport.serve())

    try:
        # 1) initialize
        reader.write_line(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        }))
        init = await _next_response(writer)
        assert init["id"] == 1
        assert "serverInfo" in init["result"]

        # 2) notifications/initialized — no response
        reader.write_line(json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        }))

        # 3) tools/list
        reader.write_line(json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        }))
        listing = await _next_response(writer)
        names = {t["name"] for t in listing["result"]["tools"]}
        assert "robit.scan_secrets" in names

        # 4) tools/call → scan_secrets
        reader.write_line(json.dumps({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "robit.scan_secrets",
                "arguments": {"text": "no secrets here"},
            },
        }))
        scan = await _next_response(writer)
        body = json.loads(scan["result"]["content"][0]["text"])
        assert body == {"matched_patterns": [], "matched": False}

        # 5) tools/call → check_destructive_op (vetoed)
        reader.write_line(json.dumps({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {
                "name": "robit.check_destructive_op",
                "arguments": {"tool": "rm", "args": ["-rf", "/"]},
            },
        }))
        veto = await _next_response(writer)
        body = json.loads(veto["result"]["content"][0]["text"])
        assert body["vetoed"] is True

    finally:
        reader.eof()
        await asyncio.wait_for(serve_task, timeout=2.0)
