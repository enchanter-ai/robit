"""HTTP integration test: bind on a free port, send real TCP requests."""

from __future__ import annotations

import asyncio
import json

import pytest

from robit.mcp_server.http import ServerHttpTransport
from robit.mcp_server.server import MCPServer


async def _post(host: str, port: int, path: str, body: bytes) -> tuple[int, dict, bytes]:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            "\r\n"
        ).encode("latin-1") + body
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

        length_s = headers.get("content-length")
        if length_s is None:
            response_body = await reader.read()
        else:
            response_body = await reader.readexactly(int(length_s))
        return status, headers, response_body
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.asyncio
async def test_http_initialize_list_call_round_trip() -> None:
    server = MCPServer()
    transport = ServerHttpTransport(server.handle_raw, path="/mcp")
    host, port = await transport.start("127.0.0.1", 0)

    serve_task = asyncio.create_task(transport.serve_forever())

    try:
        # initialize
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        }).encode("utf-8")
        status, headers, resp_body = await _post(host, port, "/mcp", body)
        assert status == 200
        assert "application/json" in headers.get("content-type", "")
        obj = json.loads(resp_body)
        assert obj["id"] == 1
        assert "serverInfo" in obj["result"]

        # tools/list
        body = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).encode("utf-8")
        status, _h, resp_body = await _post(host, port, "/mcp", body)
        assert status == 200
        obj = json.loads(resp_body)
        names = {t["name"] for t in obj["result"]["tools"]}
        assert "robit.scan_secrets" in names

        # tools/call
        body = json.dumps({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "robit.scan_secrets",
                "arguments": {"text": "hello world"},
            },
        }).encode("utf-8")
        status, _h, resp_body = await _post(host, port, "/mcp", body)
        assert status == 200
        obj = json.loads(resp_body)
        inner = json.loads(obj["result"]["content"][0]["text"])
        assert inner["matched"] is False

        # Unknown path → 404
        status, _h, _b = await _post(host, port, "/wrong", b"{}")
        assert status == 404

    finally:
        await transport.close()
        serve_task.cancel()
        try:
            await serve_task
        except (asyncio.CancelledError, Exception):
            pass
