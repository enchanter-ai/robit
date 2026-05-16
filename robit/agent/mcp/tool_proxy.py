"""robit.agent.mcp.tool_proxy — wrap a remote MCP tool as a local Tool.

Each :class:`MCPToolProxy` makes a single remote MCP tool look like a local
:class:`~robit.agent.tools.Tool` to the agent loop. Names are namespaced
``"<server>.<remote>"`` so multiple servers exposing a tool with the same
remote name (e.g. ``read_file``) don't collide in the registry.

Approval policy
---------------
``requires_approval = True`` for every MCP tool. The agent can't introspect
what a remote tool actually does — it could shell out, write files, hit
external services, or all three. Treat each call as remote code execution
and require explicit user approval per call. Tightening or per-tool
overrides is a future-wave concern (a user-side allowlist, signed manifests,
etc.); the safe default ships today.

Result conversion
-----------------
MCP returns ``{"content": [{"type": "text", "text": "..."}, ...], "isError": bool}``.

* If every item is ``{"type": "text"}``, we concatenate the ``text`` fields
  with newlines.
* Otherwise (mixed content with images, embedded resources, etc.), we fall
  back to ``json.dumps`` of the whole content list so the LLM at least sees
  the structured payload rather than losing it.
* ``isError`` from the server maps straight to ``ToolResult.is_error``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..tools._types import ToolResult
from .client import MCPCallError, MCPClient

logger = logging.getLogger(__name__)


class MCPToolProxy:
    """Local-Tool-protocol-compliant proxy for one remote MCP tool."""

    requires_approval: bool = True

    def __init__(
        self,
        client: MCPClient,
        remote_tool: dict,
    ) -> None:
        """Build a proxy from a single entry of a ``tools/list`` response.

        Parameters
        ----------
        client:
            The :class:`MCPClient` that owns the connection to the server.
        remote_tool:
            One element of the ``tools/list`` ``tools`` array. Must have
            ``name``; ``description`` and ``inputSchema`` are best-effort.
        """
        self._client = client
        remote_name = remote_tool.get("name")
        if not isinstance(remote_name, str) or not remote_name:
            raise ValueError(f"MCP tool missing 'name': {remote_tool!r}")
        self._remote_name = remote_name
        server_name = client.config.name
        self.name = f"{server_name}.{remote_name}"
        description = remote_tool.get("description") or f"MCP tool {self.name}"
        self.description = description if isinstance(description, str) else str(description)
        schema = remote_tool.get("inputSchema") or remote_tool.get("input_schema")
        # Fall back to a permissive object schema so the LLM still gets a
        # well-formed tool definition.
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}, "additionalProperties": True}
        self.input_schema: dict = schema

    @classmethod
    async def from_server(cls, client: "MCPClient") -> list["MCPToolProxy"]:
        """Connect (if needed) and build one proxy per remote tool."""
        if not client._connected and not client._closed:  # noqa: SLF001
            await client.connect()
        tools = await client.list_tools()
        proxies: list[MCPToolProxy] = []
        for entry in tools:
            if not isinstance(entry, dict):
                logger.warning(
                    "mcp: server %r returned non-object tool entry %r",
                    client.config.name, entry,
                )
                continue
            try:
                proxies.append(cls(client, entry))
            except ValueError as exc:
                logger.warning(
                    "mcp: server %r: skipping malformed tool entry: %s",
                    client.config.name, exc,
                )
        return proxies

    async def execute(self, args: dict, ctx):  # noqa: ANN001 — ToolContext type
        """Call the remote tool and convert the result into a ToolResult."""
        try:
            mcp_result = await self._client.call_tool(self._remote_name, args)
        except MCPCallError as exc:
            return ToolResult(
                content=f"MCP call {self.name!r} failed: {exc.message} (code={exc.code})",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content=f"MCP call {self.name!r} raised {type(exc).__name__}: {exc}",
                is_error=True,
            )

        content = mcp_result.get("content", [])
        is_error = bool(mcp_result.get("isError", False))
        text = _content_to_text(content)
        return ToolResult(content=text, is_error=is_error)


def _content_to_text(content: Any) -> str:
    """Collapse MCP content array to a single string.

    * All-text content → joined ``\\n`` of the ``text`` fields.
    * Mixed content    → ``json.dumps`` of the whole list.
    * Anything else    → ``json.dumps`` of the raw value.
    """
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    if not content:
        return ""
    all_text = all(
        isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        for item in content
    )
    if all_text:
        return "\n".join(item["text"] for item in content)
    return json.dumps(content, ensure_ascii=False)


__all__ = ["MCPToolProxy"]
