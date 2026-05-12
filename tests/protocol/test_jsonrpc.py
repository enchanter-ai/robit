"""Tests for enchanter.protocol — JSON-RPC 2.0 codec and correlation layer.

Coverage:
  [1]  encode Request → correct JSON shape
  [2]  encode Response (success) → correct JSON shape
  [3]  encode Response (error) → correct JSON shape
  [4]  encode Notification → no 'id' field
  [5]  decode valid Request → typed result
  [6]  decode valid Notification → typed result
  [7]  decode valid Response (success) → typed result
  [8]  decode malformed JSON → JsonRpcParseError with ErrorCode.PARSE_ERROR context
  [9]  decode invalid request shape (bad method type) → JsonRpcParseError
  [10] decode invalid request shape (bad id type) → JsonRpcParseError
  [11] register + resolve → future receives result
  [12] register + reject (ErrorObject) → future raises JsonRpcResponseError
  [13] register + reject (Exception) → future raises that exception
  [14] register + timeout → future raises asyncio.TimeoutError
  [15] multiple concurrent pending requests are independent
  [16] resolve unknown id → returns False (no crash)
  [17] reject_all → all pending futures are rejected
  [18] EmbeddedNewlineError raised on encode with embedded newline in payload
"""

from __future__ import annotations

import asyncio
import json
import pytest

from enchanter.protocol.jsonrpc import (
    ErrorCode,
    ErrorObject,
    Request,
    Response,
    Notification,
    JsonRpcParseError,
    EmbeddedNewlineError,
    encode,
    decode,
)
from enchanter.protocol.correlation import (
    PendingRequests,
    JsonRpcResponseError,
    _RequestIdGenerator,
    next_request_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(raw: str) -> dict:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# [1] encode Request
# ---------------------------------------------------------------------------

def test_encode_request_shape() -> None:
    req = Request(id=1, method="tools/list", params={"cursor": None})
    wire = _json(encode(req))
    assert wire["jsonrpc"] == "2.0"
    assert wire["id"] == 1
    assert wire["method"] == "tools/list"
    assert wire["params"] == {"cursor": None}
    assert "result" not in wire
    assert "error" not in wire


def test_encode_request_no_params_field_when_none() -> None:
    req = Request(id="abc", method="ping")
    wire = _json(encode(req))
    assert "params" not in wire


# ---------------------------------------------------------------------------
# [2] encode Response (success)
# ---------------------------------------------------------------------------

def test_encode_response_success_shape() -> None:
    resp = Response(id=42, result={"tools": []})
    wire = _json(encode(resp))
    assert wire["jsonrpc"] == "2.0"
    assert wire["id"] == 42
    assert wire["result"] == {"tools": []}
    assert "error" not in wire


# ---------------------------------------------------------------------------
# [3] encode Response (error)
# ---------------------------------------------------------------------------

def test_encode_response_error_shape() -> None:
    err = ErrorObject(code=ErrorCode.METHOD_NOT_FOUND, message="no such method", data={"hint": "check spelling"})
    resp = Response(id=7, error=err)
    wire = _json(encode(resp))
    assert wire["jsonrpc"] == "2.0"
    assert wire["id"] == 7
    assert "result" not in wire
    error_dict = wire["error"]
    assert error_dict["code"] == -32601
    assert error_dict["message"] == "no such method"
    assert error_dict["data"] == {"hint": "check spelling"}


def test_encode_error_object_no_data_field_when_none() -> None:
    err = ErrorObject(code=ErrorCode.INTERNAL_ERROR, message="oops")
    resp = Response(id=1, error=err)
    wire = _json(encode(resp))
    assert "data" not in wire["error"]


# ---------------------------------------------------------------------------
# [4] encode Notification — no 'id' field
# ---------------------------------------------------------------------------

def test_encode_notification_no_id() -> None:
    notif = Notification(method="notifications/initialized")
    wire = _json(encode(notif))
    assert wire["jsonrpc"] == "2.0"
    assert wire["method"] == "notifications/initialized"
    assert "id" not in wire
    assert "params" not in wire


def test_encode_notification_with_params() -> None:
    notif = Notification(method="notifications/progress", params={"progress": 50})
    wire = _json(encode(notif))
    assert wire["params"] == {"progress": 50}


# ---------------------------------------------------------------------------
# [5] decode valid Request
# ---------------------------------------------------------------------------

def test_decode_valid_request() -> None:
    raw = '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fs_read"}}'
    msg = decode(raw)
    assert isinstance(msg, Request)
    assert msg.id == 3
    assert msg.method == "tools/call"
    assert msg.params == {"name": "fs_read"}


def test_decode_request_string_id() -> None:
    raw = '{"jsonrpc":"2.0","id":"req-1","method":"ping"}'
    msg = decode(raw)
    assert isinstance(msg, Request)
    assert msg.id == "req-1"


# ---------------------------------------------------------------------------
# [6] decode valid Notification
# ---------------------------------------------------------------------------

def test_decode_notification() -> None:
    raw = '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    msg = decode(raw)
    assert isinstance(msg, Notification)
    assert msg.method == "notifications/initialized"
    assert msg.params is None


# ---------------------------------------------------------------------------
# [7] decode valid Response (success)
# ---------------------------------------------------------------------------

def test_decode_response_success() -> None:
    raw = '{"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"test","version":"1.0"}}}'
    msg = decode(raw)
    assert isinstance(msg, Response)
    assert msg.id == 1
    assert msg.result == {"serverInfo": {"name": "test", "version": "1.0"}}
    assert msg.error is None


def test_decode_response_error() -> None:
    raw = '{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"method not found"}}'
    msg = decode(raw)
    assert isinstance(msg, Response)
    assert msg.error is not None
    assert msg.error.code == -32601
    assert msg.error.message == "method not found"


# ---------------------------------------------------------------------------
# [8] decode malformed JSON → JsonRpcParseError
# ---------------------------------------------------------------------------

def test_decode_malformed_json_raises_parse_error() -> None:
    with pytest.raises(JsonRpcParseError) as exc_info:
        decode("{not valid json}")
    assert "invalid JSON" in str(exc_info.value)
    assert exc_info.value.raw == "{not valid json}"


def test_decode_missing_jsonrpc_field_raises_parse_error() -> None:
    with pytest.raises(JsonRpcParseError) as exc_info:
        decode('{"id":1,"method":"ping"}')
    assert "jsonrpc" in str(exc_info.value)


def test_decode_wrong_jsonrpc_version_raises_parse_error() -> None:
    with pytest.raises(JsonRpcParseError):
        decode('{"jsonrpc":"1.0","id":1,"method":"ping"}')


# ---------------------------------------------------------------------------
# [9] decode invalid request shape (bad method type)
# ---------------------------------------------------------------------------

def test_decode_bad_method_type_raises_parse_error() -> None:
    raw = '{"jsonrpc":"2.0","id":1,"method":42}'
    with pytest.raises(JsonRpcParseError) as exc_info:
        decode(raw)
    assert "method" in str(exc_info.value)


# ---------------------------------------------------------------------------
# [10] decode invalid request shape (bad id type)
# ---------------------------------------------------------------------------

def test_decode_bad_id_type_raises_parse_error() -> None:
    raw = '{"jsonrpc":"2.0","id":true,"method":"ping"}'
    with pytest.raises(JsonRpcParseError) as exc_info:
        decode(raw)
    assert "id" in str(exc_info.value)


def test_decode_null_id_is_valid() -> None:
    # id: null is explicitly allowed by JSON-RPC spec for error responses
    raw = '{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"parse error"}}'
    msg = decode(raw)
    assert isinstance(msg, Response)
    assert msg.id is None


# ---------------------------------------------------------------------------
# [11] register + resolve → future receives result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_register_then_resolve() -> None:
    pending = PendingRequests()
    fut = pending.register(1)
    assert not fut.done()
    pending.resolve(1, {"status": "ok"})
    result = await fut
    assert result == {"status": "ok"}
    assert 1 not in pending


# ---------------------------------------------------------------------------
# [12] register + reject (ErrorObject) → future raises JsonRpcResponseError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_register_then_reject_error_object() -> None:
    pending = PendingRequests()
    fut = pending.register(2)
    err = ErrorObject(code=ErrorCode.INTERNAL_ERROR, message="something broke")
    pending.reject(2, err)
    with pytest.raises(JsonRpcResponseError) as exc_info:
        await fut
    assert exc_info.value.error.code == ErrorCode.INTERNAL_ERROR
    assert 2 not in pending


# ---------------------------------------------------------------------------
# [13] register + reject (Exception) → future raises that exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_register_then_reject_exception() -> None:
    pending = PendingRequests()
    fut = pending.register("req-x")
    pending.reject("req-x", RuntimeError("transport closed"))
    with pytest.raises(RuntimeError, match="transport closed"):
        await fut


# ---------------------------------------------------------------------------
# [14] register + timeout → future raises asyncio.TimeoutError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_timeout_cancels_future() -> None:
    pending = PendingRequests()
    fut = pending.register(99)
    # Fire the timeout task with a very short delay
    timeout_task = asyncio.create_task(pending.timeout(99, after_seconds=0.01))
    with pytest.raises(asyncio.TimeoutError):
        await fut
    await timeout_task  # ensure cleanup
    assert 99 not in pending


@pytest.mark.asyncio
async def test_pending_timeout_no_effect_when_already_resolved() -> None:
    """timeout() is a no-op if the request was already resolved."""
    pending = PendingRequests()
    fut = pending.register(100)
    pending.resolve(100, "done")
    # Timeout fires after request already resolved → should not raise
    await pending.timeout(100, after_seconds=0.01)
    result = await fut
    assert result == "done"


# ---------------------------------------------------------------------------
# [15] multiple concurrent pending requests are independent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_concurrent_pending_requests_independent() -> None:
    pending = PendingRequests()
    fut_a = pending.register(10)
    fut_b = pending.register(20)
    fut_c = pending.register(30)

    # Resolve in a different order than registration
    pending.resolve(20, "result-b")
    pending.resolve(30, "result-c")
    pending.resolve(10, "result-a")

    results = await asyncio.gather(fut_a, fut_b, fut_c)
    assert results == ["result-a", "result-b", "result-c"]
    assert len(pending) == 0


# ---------------------------------------------------------------------------
# [16] resolve unknown id → returns False (no crash)
# ---------------------------------------------------------------------------

def test_resolve_unknown_id_returns_false() -> None:
    pending = PendingRequests()
    assert pending.resolve(999, "whatever") is False


def test_reject_unknown_id_returns_false() -> None:
    pending = PendingRequests()
    assert pending.reject(999, RuntimeError("x")) is False


# ---------------------------------------------------------------------------
# [17] reject_all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reject_all_clears_all_pending() -> None:
    pending = PendingRequests()
    fut_a = pending.register(1)
    fut_b = pending.register(2)
    count = pending.reject_all(RuntimeError("shutdown"))
    assert count == 2
    assert len(pending) == 0
    with pytest.raises(RuntimeError):
        await fut_a
    with pytest.raises(RuntimeError):
        await fut_b


# ---------------------------------------------------------------------------
# [18] EmbeddedNewlineError — defence-in-depth check
# ---------------------------------------------------------------------------

def test_newline_in_string_value_is_safely_escaped() -> None:
    # json.dumps always escapes \n → \\n, so a newline inside a string value
    # is never a literal newline in the serialised output.  The codec must NOT
    # raise; the EmbeddedNewlineError defence-in-depth only fires if a literal
    # newline byte somehow appears in the final JSON (e.g. buggy custom encoder
    # or indent=N mode — neither is used here).  This mirrors the TS note:
    # "JSON.stringify never produces \n unless the input contained one AND we
    # set indent (we don't)."
    resp = Response(id=1, result={"text": "line1\nline2"})
    serialised = encode(resp)
    # The literal \n is escaped to \\n in the output — no actual newline byte.
    assert "\n" not in serialised
    assert r"\n" in serialised  # the escaped form is present


def test_embedded_newline_error_is_raised_when_output_contains_literal_newline() -> None:
    # Monkeypatch json.dumps to return a string with a literal newline to
    # exercise the defence-in-depth guard.
    import enchanter.protocol.jsonrpc as jrpc_mod
    import json as _json_mod
    original = _json_mod.dumps

    def _patched_dumps(obj, **kwargs):  # type: ignore[override]
        return "{\n}"  # literal newline — should be caught

    jrpc_mod.json.dumps = _patched_dumps  # type: ignore[attr-defined]
    try:
        with pytest.raises(EmbeddedNewlineError):
            encode(Notification(method="test"))
    finally:
        jrpc_mod.json.dumps = original  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ID generator
# ---------------------------------------------------------------------------

def test_request_id_generator_is_monotonic() -> None:
    gen = _RequestIdGenerator(start=1)
    ids = [gen.next_id() for _ in range(10)]
    assert ids == list(range(1, 11))


def test_request_id_generator_independent_instances() -> None:
    gen_a = _RequestIdGenerator(start=1)
    gen_b = _RequestIdGenerator(start=100)
    assert gen_a.next_id() == 1
    assert gen_b.next_id() == 100
    assert gen_a.next_id() == 2


def test_next_request_id_returns_int() -> None:
    id1 = next_request_id()
    id2 = next_request_id()
    assert isinstance(id1, int)
    assert id2 > id1


# ---------------------------------------------------------------------------
# ErrorCode enum values
# ---------------------------------------------------------------------------

def test_error_code_standard_values() -> None:
    assert ErrorCode.PARSE_ERROR == -32700
    assert ErrorCode.INVALID_REQUEST == -32600
    assert ErrorCode.METHOD_NOT_FOUND == -32601
    assert ErrorCode.INVALID_PARAMS == -32602
    assert ErrorCode.INTERNAL_ERROR == -32603


def test_error_code_custom_values() -> None:
    assert ErrorCode.SECURITY_VETO == -32099
    assert ErrorCode.VENDOR_UNAVAILABLE == -32098
    assert ErrorCode.SAMPLING_BOUND_EXCEEDED == -32097
    assert ErrorCode.TOOL_NAME_COLLISION == -32096
    assert ErrorCode.BUDGET_FLOOR_REFUSAL == -32095
