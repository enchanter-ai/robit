"""enchanter/transport/descriptor.py — port of transport-descriptor.ts v0.4.

TransportDescriptor is the typed carrier that threads launch-time inputs
(cmd, args, env_allowlist, binary_digest) through the runtime so they can
contribute to the trust-pin digest without being recomputed at each gate.

Two shapes:
  - stdio: cmd + args + env_allowlist + (best-effort) binary_digest
  - http:  url only

`env_allowlist` carries env-var NAMES, never values.  Values rotate
legitimately; names changing IS security-relevant.

The dataclass is frozen (immutable after construction) to prevent accidental
mutation as it propagates through the orchestration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TransportDescriptor:
    """Carrier for transport-launch-time inputs.

    Fields
    ------
    kind:
        ``"stdio"`` for subprocess-based MCP servers,
        ``"http"``  for Streamable-HTTP MCP servers.
    cmd:
        Executable to spawn (stdio only).  May be a bare name (resolved via
        PATH at runtime) or an absolute path.  ``None`` for http descriptors.
    args:
        Positional arguments passed after *cmd* (stdio only).
    env_allowlist:
        Environment-variable *names* whose values are forwarded to the child
        process.  An empty tuple means the child inherits nothing beyond the
        minimal environment needed to find its executable.
    url:
        Endpoint URL (http only).  ``None`` for stdio descriptors.
    binary_digest:
        Best-effort SHA-256 hex digest of the executable binary (stdio only).
        ``None`` when the digest could not be computed (file too large, missing,
        or unreadable) or when the descriptor was built with
        ``skip_binary_digest=True``.  A missing digest shrinks the trust-pin
        coverage by one field but does not raise.
    """

    kind: Literal["stdio", "http"]
    cmd: str | None = None
    args: tuple[str, ...] = ()
    env_allowlist: tuple[str, ...] = ()
    url: str | None = None
    binary_digest: str | None = None

    # ------------------------------------------------------------------
    # Convenience constructors (mirrors describeStdio / describeHttp in TS)
    # ------------------------------------------------------------------

    @classmethod
    def for_stdio(
        cls,
        cmd: str,
        args: tuple[str, ...] = (),
        env_allowlist: tuple[str, ...] = (),
        binary_digest: str | None = None,
    ) -> "TransportDescriptor":
        """Build a stdio descriptor without computing a binary digest.

        If you need the digest, call the async helper
        ``robit.transport.descriptor.describe_stdio`` instead.
        """
        return cls(
            kind="stdio",
            cmd=cmd,
            args=args,
            env_allowlist=env_allowlist,
            binary_digest=binary_digest,
        )

    @classmethod
    def for_http(cls, url: str) -> "TransportDescriptor":
        """Build an http descriptor."""
        return cls(kind="http", url=url)
