"""enchanter/protocol/jsonrpc.py — JSON-RPC 2.0 wire types and codec.

Port of `client/enchanter/src/protocol/jsonrpc.ts`.
Stdlib only: json, dataclasses, typing, enum.

Counter (same as TS comment): a typed RPC framework (gRPC, tRPC) would give
compile-time safety, but the MCP spec mandates JSON-RPC 2.0 wire format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Union


# ---------------------------------------------------------------------------
# Standard + enchanter custom error codes
# ---------------------------------------------------------------------------

class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes + enchanter custom range (-32099..-32000)."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # enchanter custom (-32099..-32000 reserved range)
    SECURITY_VETO = -32099
    VENDOR_UNAVAILABLE = -32098
    SAMPLING_BOUND_EXCEEDED = -32097
    TOOL_NAME_COLLISION = -32096
    BUDGET_FLOOR_REFUSAL = -32095


# ---------------------------------------------------------------------------
# Wire types (frozen dataclasses)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorObject:
    """JSON-RPC 2.0 error object embedded in a Response."""

    code: int
    message: str
    data: Any = None  # optional; any JSON-serialisable value

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


@dataclass(frozen=True)
class Request:
    """JSON-RPC 2.0 request (has id, expects response)."""

    id: Union[int, str]
    method: str
    params: Any = None  # optional; object or array per spec

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": self.method,
        }
        if self.params is not None:
            d["params"] = self.params
        return d


@dataclass(frozen=True)
class Response:
    """JSON-RPC 2.0 response (success or error)."""

    id: Union[int, str, None]
    result: Any = None       # mutually exclusive with error
    error: ErrorObject | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d


@dataclass(frozen=True)
class Notification:
    """JSON-RPC 2.0 notification (no id, no response expected)."""

    method: str
    params: Any = None  # optional

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"jsonrpc": "2.0", "method": self.method}
        if self.params is not None:
            d["params"] = self.params
        return d


# Union of all wire message types
JsonRpcMessage = Union[Request, Response, Notification]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class JsonRpcParseError(ValueError):
    """Raised when the raw input cannot be parsed or fails shape validation."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class EmbeddedNewlineError(ValueError):
    """Raised when the serialised message contains an embedded newline.

    MCP stdio transport MUST NOT have embedded newlines (framing relies on
    newline delimiters). Defence-in-depth check mirrors the TS implementation.
    """

    def __init__(self) -> None:
        super().__init__(
            "JSON-RPC message contains embedded newline (MCP spec MUST NOT)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_object(v: Any) -> bool:
    """True if v is a plain dict (not list, not None)."""
    return isinstance(v, dict)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode(raw: str) -> JsonRpcMessage:
    """Parse a raw JSON string into a typed JsonRpcMessage.

    Raises:
        JsonRpcParseError: if the JSON is malformed or the shape is invalid.
    """
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JsonRpcParseError(f"invalid JSON: {exc}", raw) from exc

    if not _is_object(parsed) or parsed.get("jsonrpc") != "2.0":
        raise JsonRpcParseError('missing or invalid jsonrpc:"2.0" field', raw)

    # Validate method field when present
    if "method" in parsed and not isinstance(parsed["method"], str):
        raise JsonRpcParseError(
            f"method must be string, got {type(parsed['method']).__name__}", raw
        )

    # Validate id field when present.
    # Note: bool is a subclass of int in Python, but JSON booleans are not
    # valid JSON-RPC ids — exclude them explicitly (mirrors the TS check:
    # `typeof id !== 'number' && typeof id !== 'string'`, where typeof true
    # is 'boolean', not 'number').
    if "id" in parsed:
        id_val = parsed["id"]
        if id_val is not None and (isinstance(id_val, bool) or not isinstance(id_val, (int, str))):
            raise JsonRpcParseError(
                f"id must be number | string | null, got {type(id_val).__name__}", raw
            )

    # Validate error object when present
    if "error" in parsed:
        err = parsed["error"]
        if (
            not _is_object(err)
            or not isinstance(err.get("code"), int)
            or not isinstance(err.get("message"), str)
        ):
            raise JsonRpcParseError(
                "error must have integer code and string message", raw
            )

    # Classify into the three message types
    has_id = "id" in parsed
    has_method = "method" in parsed
    has_result = "result" in parsed
    has_error = "error" in parsed

    if has_method and not has_id:
        # Notification
        return Notification(
            method=parsed["method"],
            params=parsed.get("params"),
        )

    if has_method and has_id:
        # Request
        return Request(
            id=parsed["id"],
            method=parsed["method"],
            params=parsed.get("params"),
        )

    if has_id and (has_result or has_error):
        # Response
        error_obj: ErrorObject | None = None
        if has_error:
            e = parsed["error"]
            error_obj = ErrorObject(
                code=e["code"],
                message=e["message"],
                data=e.get("data"),
            )
        return Response(
            id=parsed["id"],
            result=parsed.get("result"),
            error=error_obj,
        )

    # Ambiguous / unrecognised shape — treat as InvalidRequest
    raise JsonRpcParseError(
        "message shape not recognised as request, response, or notification", raw
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode(msg: JsonRpcMessage) -> str:
    """Serialise a JsonRpcMessage to a JSON string (no trailing newline).

    Raises:
        EmbeddedNewlineError: if the serialised JSON contains a newline
            (defence-in-depth; json.dumps with no indent never produces one
            unless the payload data itself contained a newline character).
    """
    as_dict = msg.to_dict()
    serialised = json.dumps(as_dict, ensure_ascii=False, separators=(",", ":"))
    if "\n" in serialised:
        raise EmbeddedNewlineError()
    return serialised
