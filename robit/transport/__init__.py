"""robit.transport — MCP transport layer.

Provides transport abstractions for communicating with MCP servers.
Implemented: stdio (newline-delimited JSON-RPC) and streamable-HTTP (POST + SSE GET).
"""

from robit.transport.descriptor import TransportDescriptor
from robit.transport.http import (
    BodyTooLargeError,
    PER_MESSAGE_BODY_MAX_BYTES,
    StreamableHttpMaxRetriesError,
    StreamableHttpResumeError,
    StreamableHttpTransport,
)
from robit.transport.stdio import StdioTransport
from robit.transport.tls_pin import (
    InMemoryTlsPinStore,
    PersistentTlsPinStore,
    TlsPinEntry,
    TlsPinMismatchError,
    TlsPinStore,
    TlsPinUnknownError,
    compute_cert_fingerprint,
    verify_tls_pin,
)

__all__ = [
    # descriptor
    "TransportDescriptor",
    # shared constants / errors
    "BodyTooLargeError",
    "PER_MESSAGE_BODY_MAX_BYTES",
    # stdio
    "StdioTransport",
    # streamable-http
    "StreamableHttpTransport",
    "StreamableHttpMaxRetriesError",
    "StreamableHttpResumeError",
    # tls pin
    "TlsPinStore",
    "InMemoryTlsPinStore",
    "PersistentTlsPinStore",
    "TlsPinEntry",
    "TlsPinMismatchError",
    "TlsPinUnknownError",
    "compute_cert_fingerprint",
    "verify_tls_pin",
]
