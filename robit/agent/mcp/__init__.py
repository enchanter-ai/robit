"""robit.agent.mcp — bridge MCP servers into the agent tool registry.

Wave 15.3 / Agent J. Wires the existing MCP client transport
(``robit.transport.stdio.StdioTransport``) up as a source of tools the
agent loop can call. Each configured MCP server runs as its own subprocess
and its remote tools are surfaced through :class:`MCPToolProxy` so they look
identical to a local :class:`~robit.agent.tools.Tool` to the agent loop.

Public surface
--------------
- :class:`MCPClient`       — thin async wrapper around ``StdioTransport``.
- :class:`MCPCallError`    — JSON-RPC error or timeout from a server.
- :class:`MCPToolProxy`    — local ``Tool`` that proxies to a remote MCP tool.
- :class:`MCPServerConfig` — dataclass describing one configured server.
- :func:`load_mcp_config`  — load ``~/.enchanter/mcp.json``.
- :func:`register_mcp_tools` — connect each server and register its tools.

Wave 15.4 will wire :func:`register_mcp_tools` into the CLI after
``default_registry()`` is constructed and store the returned clients so the
session-end hook can close them cleanly.
"""

from __future__ import annotations

import logging

from .client import MCPCallError, MCPClient
from .config import MCPServerConfig, load_mcp_config
from .tool_proxy import MCPToolProxy

logger = logging.getLogger(__name__)


async def register_mcp_tools(
    registry,
    servers: list[MCPServerConfig] | None = None,
) -> list[MCPClient]:
    """Connect to each configured MCP server and register its tools.

    Parameters
    ----------
    registry:
        A :class:`~robit.agent.tools.ToolRegistry` instance. New tools are
        registered in place; the registry is unchanged on partial failure
        (each server is independent — one server's failure does not block
        others).
    servers:
        Optional explicit list of :class:`MCPServerConfig`. When ``None``,
        :func:`load_mcp_config` is called to read the default config file.

    Returns
    -------
    list[MCPClient]
        One connected client per server that came up successfully. The caller
        owns these and must call :meth:`MCPClient.close` on each at session
        end (typically wired into the CLI's shutdown hook).
    """
    if servers is None:
        servers = load_mcp_config()

    clients: list[MCPClient] = []
    for cfg in servers:
        client = MCPClient(cfg)
        try:
            await client.connect()
            proxies = await MCPToolProxy.from_server(client)
            for proxy in proxies:
                try:
                    registry.register(proxy)
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "mcp: failed to register %r from server %r: %s",
                        proxy.name, cfg.name, exc,
                    )
            clients.append(client)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "mcp: server %r failed to come up: %s", cfg.name, exc,
            )
            # Best-effort cleanup; do not raise — other servers can still work.
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass

    return clients


__all__ = [
    "MCPClient",
    "MCPCallError",
    "MCPToolProxy",
    "MCPServerConfig",
    "load_mcp_config",
    "register_mcp_tools",
]
