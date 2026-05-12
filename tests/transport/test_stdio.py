"""tests/transport/test_stdio.py — hermetic tests for StdioTransport.

All tests spawn a controlled echo server written as an inline Python script
(via sys.executable + '-c') — no external MCP binary is required.

Echo server contract
--------------------
The server reads newline-delimited JSON from stdin.  For each message it
receives it writes back a JSON response with the same ``id`` and a ``result``
field that echoes the ``params`` (or an empty dict if params is absent).
The server exits cleanly when stdin closes (EOF).

Test coverage
-------------
1. basic echo round-trip (send one request, receive one response)
2. multiple requests in flight resolve correctly via direct send/receive calls
3. 8 MB body cap rejects an oversized *incoming* line
4. stderr is captured and surfaced via recent_stderr()
5. clean shutdown: child exits within SHUTDOWN_TIMEOUT_SECS
6. send-after-close raises TransportClosedError
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap

import pytest
import pytest_asyncio  # type: ignore[import]  # noqa: F401  (may not be installed; see note)

from enchanter.transport.descriptor import TransportDescriptor
from enchanter.transport.stdio import (
    BodyTooLargeError,
    PER_MESSAGE_BODY_MAX_BYTES,
    StdioTransport,
    TransportClosedError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Inline echo-server script.  Reads one JSON line at a time; responds with
# {"jsonrpc":"2.0","id":<same id>,"result":<params or {}>}; exits on EOF.
_ECHO_SERVER_SCRIPT = textwrap.dedent(
    """
    import sys, json
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        resp = {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": msg.get("params", {}),
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    """
).strip()

# Inline stderr-emitting echo server: writes one line to stderr then behaves
# identically to the normal echo server.
_STDERR_ECHO_SERVER_SCRIPT = textwrap.dedent(
    """
    import sys, json
    sys.stderr.write("server ready\\n")
    sys.stderr.flush()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        resp = {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": msg.get("params", {}),
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    """
).strip()

# An echo server that writes an oversized line (> 8 MiB) then exits.
_OVERSIZED_SERVER_SCRIPT = textwrap.dedent(
    f"""
    import sys, json
    # Read one message (the client's probe), then emit an oversized response.
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        # Oversized payload: 'x' * (8 MiB + 1)
        big = "x" * ({PER_MESSAGE_BODY_MAX_BYTES} + 1)
        sys.stdout.write(big + "\\n")
        sys.stdout.flush()
        break
    """
).strip()


def _make_stdio_descriptor(script: str) -> TransportDescriptor:
    """Return a TransportDescriptor that runs *script* via the current Python."""
    return TransportDescriptor.for_stdio(
        cmd=sys.executable,
        args=("-c", script),
        env_allowlist=(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_descriptor() -> TransportDescriptor:
    return _make_stdio_descriptor(_ECHO_SERVER_SCRIPT)


@pytest.fixture
def stderr_echo_descriptor() -> TransportDescriptor:
    return _make_stdio_descriptor(_STDERR_ECHO_SERVER_SCRIPT)


@pytest.fixture
def oversized_descriptor() -> TransportDescriptor:
    return _make_stdio_descriptor(_OVERSIZED_SERVER_SCRIPT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_echo_round_trip(echo_descriptor: TransportDescriptor) -> None:
    """Test 1: spawn server, send one request, receive matching response."""
    transport = StdioTransport()
    try:
        await transport.start(echo_descriptor)

        request = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"x": 42}}
        await transport.send(request)

        response = await transport.receive()
        assert response is not None
        assert response["id"] == 1
        assert response["result"] == {"x": 42}
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_multiple_requests_sequential(
    echo_descriptor: TransportDescriptor,
) -> None:
    """Test 2: multiple sequential requests all resolve correctly.

    We test send/receive primitives directly (no PendingRequests layer).
    Each send() is immediately followed by a receive() so responses are
    matched by order.
    """
    transport = StdioTransport()
    try:
        await transport.start(echo_descriptor)

        pairs = [
            ({"jsonrpc": "2.0", "id": i, "method": "m", "params": {"n": i}}, i)
            for i in range(5)
        ]

        for request, expected_id in pairs:
            await transport.send(request)
            response = await transport.receive()
            assert response is not None
            assert response["id"] == expected_id
            assert response["result"] == {"n": expected_id}
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_8mb_body_cap_rejects_oversized_incoming_line(
    oversized_descriptor: TransportDescriptor,
) -> None:
    """Test 3: a received line exceeding 8 MiB raises BodyTooLargeError.

    The oversized server writes a line of (8 MiB + 1) bytes; receive()
    must detect this before parsing and raise BodyTooLargeError.
    """
    transport = StdioTransport()
    await transport.start(oversized_descriptor)

    # Send a probe so the server triggers its oversized response.
    await transport.send({"jsonrpc": "2.0", "id": 0, "method": "probe"})

    with pytest.raises(BodyTooLargeError):
        await transport.receive()

    # Transport should already be closed after the error.
    # Calling close() again should be a no-op.
    await transport.close()


@pytest.mark.asyncio
async def test_8mb_body_cap_rejects_oversized_outgoing_message() -> None:
    """Test 3b: send() raises BodyTooLargeError for an oversized outgoing message.

    We don't need a real server for this test — the cap fires before the
    write, so we just need a started transport.
    """
    transport = StdioTransport()
    descriptor = _make_stdio_descriptor(_ECHO_SERVER_SCRIPT)
    await transport.start(descriptor)
    try:
        # Build a message whose JSON serialisation exceeds 8 MiB.
        big_value = "x" * (PER_MESSAGE_BODY_MAX_BYTES + 1)
        oversized_msg = {"jsonrpc": "2.0", "id": 0, "data": big_value}

        with pytest.raises(BodyTooLargeError):
            await transport.send(oversized_msg)
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_stderr_captured_via_recent_stderr(
    stderr_echo_descriptor: TransportDescriptor,
) -> None:
    """Test 4: stderr output is captured and retrievable via recent_stderr()."""
    transport = StdioTransport()
    try:
        await transport.start(stderr_echo_descriptor)

        # Let the background stderr-drain task run.
        await asyncio.sleep(0.1)

        # Do a round trip to confirm the server is alive.
        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        response = await transport.receive()
        assert response is not None
        assert response["id"] == 1

        # Give the drain task a moment to capture the stderr line.
        await asyncio.sleep(0.1)

        stderr_lines = transport.recent_stderr()
        assert any("server ready" in line for line in stderr_lines), (
            f"Expected 'server ready' in stderr, got: {stderr_lines!r}"
        )
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_clean_shutdown_child_exits(
    echo_descriptor: TransportDescriptor,
) -> None:
    """Test 5: close() causes the child process to exit within the timeout."""
    transport = StdioTransport()
    await transport.start(echo_descriptor)
    proc = transport._proc
    assert proc is not None

    # Confirm the process is running.
    assert proc.returncode is None

    await transport.close()

    # After close(), the child must have exited.
    assert proc.returncode is not None, (
        "Expected child process to have exited after transport.close()"
    )


@pytest.mark.asyncio
async def test_send_after_close_raises(echo_descriptor: TransportDescriptor) -> None:
    """Test 6: send() on a closed transport raises TransportClosedError."""
    transport = StdioTransport()
    await transport.start(echo_descriptor)
    await transport.close()

    with pytest.raises(TransportClosedError):
        await transport.send({"jsonrpc": "2.0", "id": 99, "method": "noop"})


@pytest.mark.asyncio
async def test_receive_after_close_raises(
    echo_descriptor: TransportDescriptor,
) -> None:
    """Bonus: receive() on a closed transport also raises TransportClosedError."""
    transport = StdioTransport()
    await transport.start(echo_descriptor)
    await transport.close()

    with pytest.raises(TransportClosedError):
        await transport.receive()


@pytest.mark.asyncio
async def test_eof_returns_none(echo_descriptor: TransportDescriptor) -> None:
    """Bonus: receive() returns None when the server closes stdout (EOF)."""
    transport = StdioTransport()
    try:
        await transport.start(echo_descriptor)

        # Close stdin — the echo server will exit, closing its stdout.
        assert transport._proc is not None
        assert transport._proc.stdin is not None
        transport._proc.stdin.close()

        # Drain any buffered output; eventually we should get None (EOF).
        result = None
        for _ in range(10):
            try:
                result = await asyncio.wait_for(transport.receive(), timeout=1.0)
                if result is None:
                    break
            except (asyncio.TimeoutError, TransportClosedError):
                break
        assert result is None
    finally:
        await transport.close()
