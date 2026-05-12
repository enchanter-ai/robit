"""enchanter.protocol — JSON-RPC 2.0 protocol layer.

Public surface:
    Types:       Request, Response, Notification, ErrorObject, JsonRpcMessage
    Codec:       encode, decode
    Error codes: ErrorCode (IntEnum)
    Exceptions:  JsonRpcParseError, EmbeddedNewlineError, JsonRpcResponseError
    Correlation: PendingRequests, next_request_id, _RequestIdGenerator
"""

from .jsonrpc import (
    ErrorCode,
    ErrorObject,
    Request,
    Response,
    Notification,
    JsonRpcMessage,
    JsonRpcParseError,
    EmbeddedNewlineError,
    encode,
    decode,
)
from .correlation import (
    PendingRequests,
    JsonRpcResponseError,
    next_request_id,
    _RequestIdGenerator,
)

__all__ = [
    # types
    "ErrorCode",
    "ErrorObject",
    "Request",
    "Response",
    "Notification",
    "JsonRpcMessage",
    # codec
    "encode",
    "decode",
    # exceptions
    "JsonRpcParseError",
    "EmbeddedNewlineError",
    "JsonRpcResponseError",
    # correlation
    "PendingRequests",
    "next_request_id",
    "_RequestIdGenerator",
]
