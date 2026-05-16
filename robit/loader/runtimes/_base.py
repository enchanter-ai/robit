"""Shared error types for the runtime registry."""

from __future__ import annotations


class SidecarBaseError(Exception):
    """Base for all sidecar runtime errors. All sidecar errors are coerced into
    a veto-shaped PluginAck by SidecarAdapter.on_phase; callers that want to
    distinguish them can catch SidecarBaseError directly."""

    def __init__(self, message: str, *, stderr_tail: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


class SidecarCrashError(SidecarBaseError):
    """Raised when the sidecar subprocess exits unexpectedly mid-request."""


class SidecarTimeoutError(SidecarBaseError):
    """Raised when the sidecar fails to respond within the configured timeout."""


class SidecarProtocolError(SidecarBaseError):
    """Raised when the sidecar emits a malformed JSON-RPC response."""


class SidecarInitError(SidecarBaseError):
    """Raised when the sidecar's initialize handshake fails or returns invalid data."""
