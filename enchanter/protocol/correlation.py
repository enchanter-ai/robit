"""enchanter/protocol/correlation.py — request/response correlation primitive.

Port of the `pending` map pattern in `client/enchanter/src/client/mcp-client.ts`.

PendingRequests is a dict[id, asyncio.Future] keyed by request id with
register / resolve / reject / timeout helpers. Thread-safe via asyncio
single-threaded event loop; the ID generator additionally uses a Lock
for safety when called from threaded contexts.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from typing import Any, Union

from .jsonrpc import ErrorObject


# ---------------------------------------------------------------------------
# ID generator — monotonic int counter, thread-safe
# ---------------------------------------------------------------------------

class _RequestIdGenerator:
    """Monotonic integer counter for JSON-RPC request IDs.

    Uses itertools.count() under a threading.Lock so it remains safe if
    called from multiple threads (e.g. test runners with thread-pool
    executors). Within a normal asyncio session, the lock is uncontested.
    """

    def __init__(self, start: int = 1) -> None:
        self._counter = itertools.count(start)
        self._lock = threading.Lock()

    def next_id(self) -> int:
        with self._lock:
            return next(self._counter)


# Module-level default generator — one per process, matches TS's per-client
# nextRequestId counter pattern. Individual McpClient instances should
# construct their own generator to keep ID spaces independent.
_default_generator = _RequestIdGenerator()


def next_request_id() -> int:
    """Return the next unique request ID from the module-level generator."""
    return _default_generator.next_id()


# ---------------------------------------------------------------------------
# PendingRequests
# ---------------------------------------------------------------------------

class PendingRequests:
    """Correlates outgoing requests to their pending asyncio Futures.

    Mirrors the `pending` Map in McpClient:
        Map<number | string, { resolve, reject }>

    Each request ID maps to an asyncio.Future[Any] that resolves with the
    decoded result payload or raises an exception (from reject / timeout).
    """

    def __init__(self) -> None:
        self._pending: dict[Union[int, str], asyncio.Future[Any]] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def register(self, request_id: Union[int, str]) -> "asyncio.Future[Any]":
        """Register a new pending request; return its Future.

        The Future must be awaited by the caller to obtain the result.

        Raises:
            KeyError: if request_id is already registered (duplicate ID bug).
        """
        if request_id in self._pending:
            raise KeyError(f"request id {request_id!r} is already pending")
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = fut
        return fut

    def resolve(self, request_id: Union[int, str], result: Any) -> bool:
        """Resolve the pending Future for request_id with result.

        Returns True if a pending entry was found and resolved, False if the
        ID was not registered (e.g. already resolved or timed out).
        """
        fut = self._pending.pop(request_id, None)
        if fut is None:
            return False
        if not fut.done():
            fut.set_result(result)
        return True

    def reject(self, request_id: Union[int, str], error: Union[ErrorObject, Exception]) -> bool:
        """Reject the pending Future for request_id with an exception.

        If error is an ErrorObject (JSON-RPC error), wraps it in a
        JsonRpcResponseError exception. If error is already an Exception,
        propagates it directly.

        Returns True if a pending entry was found and rejected, False otherwise.
        """
        fut = self._pending.pop(request_id, None)
        if fut is None:
            return False
        if not fut.done():
            if isinstance(error, ErrorObject):
                exc = JsonRpcResponseError(error)
            else:
                exc = error
            fut.set_exception(exc)
        return True

    async def timeout(self, request_id: Union[int, str], after_seconds: float) -> None:
        """Cancel the pending Future after after_seconds with asyncio.TimeoutError.

        Waits for the given duration then, if the request is still pending,
        removes it and sets a CancelledError on the future.

        This is a fire-and-forget coroutine — callers await it directly to
        set a one-shot deadline, or schedule it as a task.
        """
        await asyncio.sleep(after_seconds)
        fut = self._pending.pop(request_id, None)
        if fut is not None and not fut.done():
            fut.set_exception(asyncio.TimeoutError(
                f"request {request_id!r} timed out after {after_seconds}s"
            ))

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reject_all(self, error: Exception) -> int:
        """Reject all pending requests with error (e.g. transport closed).

        Returns the number of futures that were rejected.
        """
        count = 0
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(error)
                count += 1
        self._pending.clear()
        return count

    def pending_ids(self) -> list[Union[int, str]]:
        """Return a snapshot of currently registered (unresolved) IDs."""
        return list(self._pending.keys())

    def __len__(self) -> int:
        return len(self._pending)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._pending


# ---------------------------------------------------------------------------
# Exception type for JSON-RPC error responses
# ---------------------------------------------------------------------------

class JsonRpcResponseError(Exception):
    """Raised when a JSON-RPC response contains an error object."""

    def __init__(self, error: ErrorObject) -> None:
        super().__init__(f"[{error.code}] {error.message}")
        self.error = error
