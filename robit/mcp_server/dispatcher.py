"""JSON-RPC dispatcher: maps method names to handler coroutines.

Owns the MCP handshake (initialize, tools/list, tools/call, ping) plus the
error-mapping policy from Python exceptions to JSON-RPC error responses.
"""

from __future__ import annotations

import logging
from typing import Any

from robit.core import SecurityVetoError
from robit.protocol.jsonrpc import (
    ErrorCode,
    ErrorObject,
    JsonRpcParseError,
    Notification,
    Request,
    Response,
    decode,
)

from .errors import InvalidParamsError, MethodNotFoundError, ServerError, ToolNotFoundError
from .tools import ToolRegistry, to_mcp_call_result

logger = logging.getLogger(__name__)

from robit import __version__ as _ENCHANTER_VERSION

SERVER_INFO = {
    "name": "enchanter-mcp-server",
    "version": _ENCHANTER_VERSION,
}
PROTOCOL_VERSION = "2025-06-18"


class Dispatcher:
    """Stateless (besides the tool registry) JSON-RPC method dispatcher."""

    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    async def handle_raw(self, raw: str) -> str | None:
        """Decode + dispatch a single raw JSON string.

        Returns a serialised JSON-RPC response string, or None for notifications
        (which never produce a reply).
        """
        try:
            msg = decode(raw)
        except JsonRpcParseError as exc:
            # No request id available; per JSON-RPC spec, id is null.
            return _encode(
                Response(
                    id=None,
                    error=ErrorObject(
                        code=int(ErrorCode.PARSE_ERROR),
                        message=str(exc),
                    ),
                )
            )

        if isinstance(msg, Notification):
            # MCP notifications (e.g. notifications/initialized) — accept silently.
            await self._handle_notification(msg)
            return None

        if isinstance(msg, Request):
            response = await self.handle_request(msg)
            return response

        # Responses arriving on a server input — not expected; ignore.
        return None

    async def _handle_notification(self, note: Notification) -> None:
        if note.method == "notifications/initialized":
            return
        logger.debug("mcp_server: ignoring notification %s", note.method)

    async def handle_request(self, req: Request) -> str:
        """Dispatch a single Request and return the serialised Response JSON."""
        try:
            result = await self._dispatch(req.method, req.params)
            return _encode(Response(id=req.id, result=result))
        except SecurityVetoError as exc:
            return _encode(
                Response(
                    id=req.id,
                    error=ErrorObject(
                        code=int(ErrorCode.SECURITY_VETO),
                        message=str(exc),
                    ),
                )
            )
        except ServerError as exc:
            return _encode(
                Response(
                    id=req.id,
                    error=ErrorObject(
                        code=exc.code,
                        message=str(exc),
                        data=exc.data,
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("mcp_server: unhandled exception in %s", req.method)
            return _encode(
                Response(
                    id=req.id,
                    error=ErrorObject(
                        code=int(ErrorCode.INTERNAL_ERROR),
                        message=f"internal error: {exc}",
                    ),
                )
            )

    async def _dispatch(self, method: str, params: Any) -> Any:
        if method == "initialize":
            return await self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return await self._tools_list(params)
        if method == "tools/call":
            return await self._tools_call(params)
        raise MethodNotFoundError(f"method not found: {method}")

    async def _initialize(self, params: Any) -> dict[str, Any]:
        # Echo back capabilities. We don't negotiate the protocol version
        # strictly — return ours and let the client decide.
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": SERVER_INFO,
        }

    async def _tools_list(self, params: Any) -> dict[str, Any]:
        return {"tools": self.tools.listing()}

    async def _tools_call(self, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise InvalidParamsError("tools/call: params must be an object")
        name = params.get("name")
        if not isinstance(name, str):
            raise InvalidParamsError("tools/call: 'name' (string) is required")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise InvalidParamsError("tools/call: 'arguments' must be an object")

        try:
            tool = self.tools.get(name)
        except ToolNotFoundError:
            raise

        result = await tool.handler(arguments)
        return to_mcp_call_result(result)


def _encode(resp: Response) -> str:
    """Encode a Response without going through encode() (which forbids newlines).

    The dispatcher's caller is responsible for adding any framing newlines;
    here we just JSON-serialise the response payload.
    """
    import json
    return json.dumps(resp.to_dict(), ensure_ascii=False, separators=(",", ":"))
