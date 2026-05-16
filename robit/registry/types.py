"""enchanter/registry/types.py — shared data types for the namespace registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolDescriptor:
    """Describes a single tool exposed by an MCP server."""

    name: str
    description: str
    input_schema: dict
    output_schema: dict | None = field(default=None)
