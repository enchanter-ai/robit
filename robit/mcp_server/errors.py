"""Server-internal exception types for the MCP server.

These map to JSON-RPC error codes at the dispatcher boundary.
"""

from __future__ import annotations

from robit.protocol.jsonrpc import ErrorCode


class ServerError(Exception):
    """Base class for MCP server-side errors with a JSON-RPC mapping."""

    code: int = int(ErrorCode.INTERNAL_ERROR)

    def __init__(self, message: str, *, data: object = None) -> None:
        super().__init__(message)
        self.data = data


class MethodNotFoundError(ServerError):
    code = int(ErrorCode.METHOD_NOT_FOUND)


class InvalidParamsError(ServerError):
    code = int(ErrorCode.INVALID_PARAMS)


class ToolNotFoundError(ServerError):
    code = int(ErrorCode.METHOD_NOT_FOUND)


class ToolExecutionError(ServerError):
    code = int(ErrorCode.INTERNAL_ERROR)
