"""MCPServer: glue between the dispatcher and the chosen transport."""

from __future__ import annotations

import asyncio
import logging

from .dispatcher import Dispatcher
from .http import ServerHttpTransport
from .stdio import ServerStdioTransport, attach_to_sys_streams
from .tools import ToolRegistry, register_default_tools

logger = logging.getLogger(__name__)


class MCPServer:
    """Bundle a dispatcher + tool registry; runs over an injected transport."""

    def __init__(self, tools: ToolRegistry | None = None) -> None:
        self.tools = tools or ToolRegistry()
        if len(self.tools) == 0:
            register_default_tools(self.tools)
        self.dispatcher = Dispatcher(self.tools)

    async def handle_raw(self, raw: str) -> str | None:
        return await self.dispatcher.handle_raw(raw)


async def serve_stdio(
    server: MCPServer | None = None,
    *,
    reader: asyncio.StreamReader | None = None,
    writer: asyncio.StreamWriter | None = None,
) -> None:
    """Run the MCP server over stdio.

    If ``reader``/``writer`` are not provided, wires ``sys.stdin`` /
    ``sys.stdout`` automatically. Returns when the input stream EOFs.
    """
    srv = server or MCPServer()
    if reader is None or writer is None:
        reader, writer = await attach_to_sys_streams()
    transport = ServerStdioTransport(reader, writer, srv.handle_raw)
    await transport.serve()


async def serve_http(
    server: MCPServer | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    path: str = "/mcp",
) -> None:
    """Run the MCP server over Streamable-HTTP at ``host:port``."""
    srv = server or MCPServer()
    transport = ServerHttpTransport(srv.handle_raw, path=path)
    bound_host, bound_port = await transport.start(host, port)
    logger.info("MCP server listening on http://%s:%d%s", bound_host, bound_port, path)
    try:
        await transport.serve_forever()
    finally:
        await transport.close()
