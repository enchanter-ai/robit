"""enchanter.agent.mcp.client — async MCP client wrapper.

A thin protocol layer on top of
:class:`enchanter.transport.stdio.StdioTransport`. The transport handles
subprocess lifecycle and newline-framed JSON-RPC. This module handles:

* The MCP **initialize** handshake (request + ``notifications/initialized``).
* Response correlation by id (background reader task with futures).
* ``tools/list`` and ``tools/call`` convenience methods.
* A per-call timeout, surfaced as :class:`MCPCallError`.
* Mapping JSON-RPC error responses to :class:`MCPCallError`.

One :class:`MCPClient` per configured server. The agent owns the lifecycle:
:meth:`connect` on first use, :meth:`close` at session end.

Wire-protocol contract
----------------------
The methods we call on the server, in the order they typically appear:

* ``initialize``                 — request, returns server capabilities.
* ``notifications/initialized``  — fire-and-forget notification.
* ``tools/list``                 — request, returns ``{"tools": [...]}``.
* ``tools/call``                 — request with ``{"name", "arguments"}``,
  returns ``{"content": [...], "isError": false}``.

All exchanges use JSON-RPC 2.0 envelopes carried as newline-delimited UTF-8
lines on the subprocess's stdin/stdout, per the MCP spec.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from enchanter.protocol.jsonrpc import (
    ErrorObject,
    Notification,
    Request,
    Response,
    decode,
)
from enchanter.transport.descriptor import TransportDescriptor
from enchanter.transport.stdio import StdioTransport

from .config import MCPServerConfig

logger = logging.getLogger(__name__)

#: Default protocol version we advertise on initialize. Servers are allowed
#: to negotiate down; we accept whatever they return.
PROTOCOL_VERSION = "2025-06-18"

#: Default per-call timeout (seconds). Applied to initialize, tools/list,
#: and tools/call. Overridable per-call via the *timeout* parameter.
DEFAULT_TIMEOUT_S: float = 30.0

#: How we identify ourselves to the server on initialize.
CLIENT_INFO = {
    "name": "enchanter-agent",
    "version": "0.5",
}


class MCPCallError(Exception):
    """An MCP method call failed.

    Raised on:

    * JSON-RPC error response (``error.code``, ``error.message`` preserved).
    * Per-call timeout (``code=-1``, ``message`` describes the timeout).
    * Transport closed mid-call (``code=-2``).

    We chose to wrap timeouts in :class:`MCPCallError` (rather than letting
    :class:`asyncio.TimeoutError` bubble) so callers have a single failure
    type to handle, and so the error carries the offending method name.
    """

    def __init__(self, method: str, code: int, message: str) -> None:
        super().__init__(f"MCP {method!r} failed (code={code}): {message}")
        self.method = method
        self.code = code
        self.message = message


class MCPClient:
    """Async client for one MCP server.

    Not thread-safe; use from a single event-loop thread, same as the
    underlying :class:`StdioTransport`.
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: StdioTransport | None = None,
    ) -> None:
        self.config = config
        self._timeout_s = timeout_s
        self._transport: StdioTransport = transport if transport is not None else StdioTransport()
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int | str, asyncio.Future[Response]] = {}
        self._next_id: int = 1
        self._connected: bool = False
        self._closed: bool = False
        self._tools_cache: list[dict] | None = None
        self._server_info: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> dict:
        """Spawn the server, run the MCP initialize handshake.

        Returns the ``initialize`` result dict (server capabilities + info).
        Idempotent: calling ``connect`` twice returns the cached result.
        """
        if self._connected:
            assert self._server_info is not None
            return self._server_info
        if self._closed:
            raise RuntimeError("MCPClient.connect on a closed client")

        descriptor = TransportDescriptor.for_stdio(
            cmd=self.config.command,
            args=self.config.args,
            env_allowlist=self.config.env_allowlist,
        )
        await self._transport.start(descriptor)

        # Start the background reader so responses correlate to pending
        # futures.
        self._reader_task = asyncio.get_event_loop().create_task(
            self._read_loop(), name=f"mcp-reader[{self.config.name}]",
        )

        # initialize handshake.
        init_params = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        }
        result = await self._request("initialize", init_params)

        # Fire-and-forget notifications/initialized.
        await self._notify("notifications/initialized", {})

        self._server_info = result if isinstance(result, dict) else {"result": result}
        self._connected = True
        return self._server_info

    async def close(self) -> None:
        """Close the transport and cancel pending requests.

        Safe to call repeatedly. Pending requests resolve with a
        :class:`MCPCallError` describing the closure.
        """
        if self._closed:
            return
        self._closed = True

        # Cancel the reader so it doesn't fight with close().
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

        # Fail any pending requests so awaiters wake up.
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(
                    MCPCallError(
                        method="<pending>",
                        code=-2,
                        message="MCPClient closed while request was in flight",
                    )
                )
        self._pending.clear()

        try:
            await self._transport.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("mcp: transport close raised: %r", exc)

    # ------------------------------------------------------------------
    # MCP methods
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict]:
        """Fetch the server's tool catalog. Cached after first success."""
        if self._tools_cache is not None:
            return self._tools_cache
        result = await self._request("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise MCPCallError(
                "tools/list", -1,
                f"server returned malformed tools/list result: {result!r}",
            )
        self._tools_cache = tools
        return tools

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        """Invoke a remote tool. Returns the raw MCP ``tools/call`` result."""
        result = await self._request(
            "tools/call",
            {"name": tool_name, "arguments": args},
        )
        if not isinstance(result, dict):
            raise MCPCallError(
                "tools/call", -1,
                f"server returned non-object tools/call result: {result!r}",
            )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    async def _request(self, method: str, params: dict, *, timeout: float | None = None) -> Any:
        """Send a Request, await its Response, return the result payload."""
        if self._closed:
            raise MCPCallError(method, -2, "MCPClient is closed")
        req_id = self._alloc_id()
        req = Request(id=req_id, method=method, params=params)
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[Response] = loop.create_future()
        self._pending[req_id] = fut
        try:
            await self._transport.send(req.to_dict())
            try:
                response = await asyncio.wait_for(
                    fut,
                    timeout=timeout if timeout is not None else self._timeout_s,
                )
            except asyncio.TimeoutError as exc:
                raise MCPCallError(method, -1, f"timeout after {self._timeout_s}s") from exc
        finally:
            self._pending.pop(req_id, None)

        if response.error is not None:
            err: ErrorObject = response.error
            raise MCPCallError(method, err.code, err.message)
        return response.result

    async def _notify(self, method: str, params: dict) -> None:
        note = Notification(method=method, params=params)
        await self._transport.send(note.to_dict())

    async def _read_loop(self) -> None:
        """Background task: read incoming messages, route by id."""
        try:
            while True:
                try:
                    raw = await self._transport.receive()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("mcp reader: receive raised: %r", exc)
                    break
                if raw is None:
                    break  # clean EOF
                try:
                    # raw is already a parsed dict from StdioTransport.receive
                    # — re-frame through our JSON-RPC decoder to get the
                    # typed Response.
                    import json as _json
                    msg = decode(_json.dumps(raw))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("mcp reader: undecodable message %r: %r", raw, exc)
                    continue

                if isinstance(msg, Response):
                    fut = self._pending.get(msg.id)
                    if fut is None or fut.done():
                        logger.debug(
                            "mcp reader: response for unknown/finished id %r", msg.id,
                        )
                        continue
                    fut.set_result(msg)
                elif isinstance(msg, Notification):
                    # Server-initiated notifications (tools/listChanged, log
                    # messages, etc). We don't subscribe to anything in 15.3
                    # — just trace and drop.
                    logger.debug(
                        "mcp reader: notification %s ignored", msg.method,
                    )
                else:
                    # Server-initiated request (sampling, etc) — not
                    # supported in 15.3.
                    logger.debug(
                        "mcp reader: unsolicited request %r ignored", msg,
                    )
        except asyncio.CancelledError:
            raise
        finally:
            # On exit, fail anything still pending so callers wake up.
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(
                        MCPCallError(
                            method="<pending>",
                            code=-2,
                            message="MCP reader loop ended; transport closed",
                        )
                    )


__all__ = ["MCPClient", "MCPCallError", "PROTOCOL_VERSION", "DEFAULT_TIMEOUT_S"]
