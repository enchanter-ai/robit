"""enchanter/registry/namespace.py — NamespaceRegistry.

Prevents FM-1 tool-name collisions across MCP servers and detects FM-10
MCPoison schema mutations via SHA-256 digest pinning.

Two internal maps mirror the TypeScript implementation:
  byQualified : "server_id.tool_name" → (ToolDescriptor, digest)
  byBare      : bare_name             → set[server_id]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from robit.registry.digest import compute_schema_digest
from robit.registry.errors import SchemaDigestMismatchError, ToolNameCollisionError
from robit.registry.types import ToolDescriptor

if TYPE_CHECKING:
    pass


@dataclass
class _Entry:
    descriptor: ToolDescriptor
    digest: str


class NamespaceRegistry:
    """Thread-safety note: this implementation is not thread-safe.
    Callers that register/unregister from multiple threads must synchronise
    externally — matching the single-threaded TS original.
    """

    def __init__(self) -> None:
        # qualified "server_id.tool_name" → _Entry
        self._by_qualified: dict[str, _Entry] = {}
        # bare_name → set of server_ids that expose it
        self._by_bare: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, server_id: str, tools: list[ToolDescriptor]) -> None:
        """Register *tools* for *server_id*.

        For each tool:
        - Compute the SHA-256 digest of the combined descriptor schema
          (description, input_schema, output_schema).
        - If the qualified name already exists with a *different* digest,
          raise ``SchemaDigestMismatchError`` (MCPoison defence).
        - If the same digest: idempotent, no-op.
        - Otherwise: store and update the bare-name index.
        """
        for tool in tools:
            qualified = self._qualify(server_id, tool.name)
            schema = self._descriptor_to_schema(tool)
            digest = compute_schema_digest(schema)

            existing = self._by_qualified.get(qualified)
            if existing is not None:
                if existing.digest != digest:
                    raise SchemaDigestMismatchError(
                        server_id, tool.name, existing.digest, digest
                    )
                # same digest → idempotent
                continue

            self._by_qualified[qualified] = _Entry(descriptor=tool, digest=digest)

            servers = self._by_bare.setdefault(tool.name, set())
            servers.add(server_id)

    def resolve(self, name: str) -> tuple[str, str]:
        """Resolve *name* to ``(server_id, tool_name)``.

        Resolution order (mirrors TS):
        1. Try *name* as a qualified ``server_id.tool_name`` — exact map hit.
        2. Fall back to bare-name lookup.
           - Not found → ``KeyError``.
           - Ambiguous (>1 server) → ``ToolNameCollisionError``.
           - Unique → return ``(server_id, bare_name)``.

        Note: a bare name that also happens to look like a qualified name
        (contains a dot) is tried as qualified first, consistent with the TS
        comment about tools like ``shell.exec``.
        """
        # 1. Try as qualified
        if name in self._by_qualified:
            entry = self._by_qualified[name]
            # Reconstruct server_id as everything before the first '.'
            # We stored it at registration; retrieve from the entry's descriptor.
            # Find the server_id by scanning bare map — more robust.
            server_id = self._server_id_for_qualified(name)
            return (server_id, entry.descriptor.name)

        # 2. Bare-name lookup
        servers = self._by_bare.get(name)
        if not servers:
            raise KeyError(f"tool not found: {name!r}")
        if len(servers) > 1:
            raise ToolNameCollisionError(name, sorted(servers))
        (server_id,) = servers
        return (server_id, name)

    def unregister(self, server_id: str) -> None:
        """Drop all tools registered for *server_id*.

        After this call, bare names that were ambiguous solely because of
        *server_id*'s tools become resolvable again.
        """
        prefix = f"{server_id}."
        to_remove = [q for q in self._by_qualified if q.startswith(prefix)]
        for qualified in to_remove:
            entry = self._by_qualified.pop(qualified)
            bare = entry.descriptor.name
            servers = self._by_bare.get(bare)
            if servers is not None:
                servers.discard(server_id)
                if not servers:
                    del self._by_bare[bare]

    def tools_for(self, server_id: str) -> list[ToolDescriptor]:
        """Return all ``ToolDescriptor`` objects registered for *server_id*."""
        prefix = f"{server_id}."
        return [
            entry.descriptor
            for q, entry in self._by_qualified.items()
            if q.startswith(prefix)
        ]

    def all_qualified_names(self) -> list[str]:
        """Return every registered qualified name (``server_id.tool_name``)."""
        return list(self._by_qualified.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _qualify(server_id: str, bare_name: str) -> str:
        return f"{server_id}.{bare_name}"

    def _server_id_for_qualified(self, qualified: str) -> str:
        """Recover the server_id stored at registration by scanning bare map."""
        entry = self._by_qualified[qualified]
        bare = entry.descriptor.name
        servers = self._by_bare.get(bare, set())
        # The qualified key starts with server_id + "."
        for sid in servers:
            if qualified == self._qualify(sid, bare):
                return sid
        # Fallback: parse from the key (safe because bare_name is in entry)
        return qualified[: len(qualified) - len(bare) - 1]

    @staticmethod
    def _descriptor_to_schema(tool: ToolDescriptor) -> dict:
        """Build the schema dict that is digested — mirrors TS ToolSchema."""
        schema: dict = {
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }
        if tool.output_schema is not None:
            schema["outputSchema"] = tool.output_schema
        return schema
