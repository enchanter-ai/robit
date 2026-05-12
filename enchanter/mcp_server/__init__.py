"""enchanter.mcp_server — MCP server exposing engine adapters as tools.

Counterpart to enchanter.transport (which spawns *client* connections to
external MCP servers). This package implements the *server* side: it reads
JSON-RPC from stdio or Streamable-HTTP, routes to registered tools, and
returns results.
"""

from .server import MCPServer, serve_stdio, serve_http
from .tools import Tool, ToolRegistry, register_default_tools

__all__ = [
    "MCPServer",
    "serve_stdio",
    "serve_http",
    "Tool",
    "ToolRegistry",
    "register_default_tools",
]
