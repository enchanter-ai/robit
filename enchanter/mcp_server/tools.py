"""Tool registry and default tool wrappers around engine adapters.

Each Tool wraps an engine adapter's ``on_phase`` interface by constructing a
synthetic EnchantedEvent in the phase the engine expects, dispatching it
directly, and translating the resulting PluginAck into a tool result dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from enchanter.core import create_request_context
from enchanter.core.bus import build_event

from .errors import InvalidParamsError, ToolExecutionError, ToolNotFoundError


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_listing(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFoundError(f"unknown tool: {name}")
        return self._tools[name]

    def listing(self) -> list[dict[str, Any]]:
        return [t.to_listing() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Engine adapter wrappers
# ---------------------------------------------------------------------------


async def scan_secrets_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    """Wrap secret_mask.adapter: scan a text input for secret patterns."""
    text = arguments.get("text")
    if not isinstance(text, str):
        raise InvalidParamsError("scan_secrets: 'text' (string) is required")

    from enchanter.engines.secret_mask.adapter import adapter

    ctx = create_request_context()
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="post-response",
        topic="mcp.tool.result.received",
        source="mcp-server",
        budget_tier=ctx.budget_tier,
        payload={"result": text},
    )

    try:
        ack = await adapter.on_phase(event, ctx)
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"scan_secrets failed: {exc}") from exc

    matched: list[str] = []
    for derived in ack.derived_events:
        if derived.topic == "secret-mask.matched":
            payload = dict(derived.payload or {})
            ids = payload.get("matched_patterns") or []
            if isinstance(ids, list):
                matched.extend(str(x) for x in ids)

    return {
        "matched_patterns": matched,
        "matched": bool(matched),
    }


async def check_destructive_op_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    """Wrap destructive_op_gate.adapter: report whether a command would veto."""
    tool = arguments.get("tool")
    args = arguments.get("args")
    if not isinstance(tool, str):
        raise InvalidParamsError("check_destructive_op: 'tool' (string) is required")
    if args is None:
        args = []
    if not (isinstance(args, list) and all(isinstance(a, str) for a in args)):
        raise InvalidParamsError("check_destructive_op: 'args' must be a list of strings")

    from enchanter.engines.destructive_op_gate.adapter import adapter

    ctx = create_request_context()
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="trust-gate",
        topic="mcp.tool.call.requested",
        source="mcp-server",
        budget_tier=ctx.budget_tier,
        payload={"tool": tool, "args": args},
    )

    try:
        ack = await adapter.on_phase(event, ctx)
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"check_destructive_op failed: {exc}") from exc

    vetoed = ack.status == "veto"
    pattern_id: str | None = None
    pattern_name: str | None = None
    if ack.derived_events:
        d_payload = dict(ack.derived_events[0].payload or {})
        pid = d_payload.get("pattern_id")
        pname = d_payload.get("pattern_name")
        if isinstance(pid, str):
            pattern_id = pid
        if isinstance(pname, str):
            pattern_name = pname

    return {
        "vetoed": vetoed,
        "degraded": bool(ack.degraded),
        "reason": ack.reason,
        "pattern_id": pattern_id,
        "pattern_name": pattern_name,
    }


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


_SCAN_SECRETS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Text corpus to scan for secret patterns.",
        },
    },
    "required": ["text"],
    "additionalProperties": False,
}

_CHECK_DESTRUCTIVE_OP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "description": "Tool name being invoked (e.g. 'shell', 'git').",
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Argument vector passed to the tool.",
        },
    },
    "required": ["tool"],
    "additionalProperties": False,
}


def register_default_tools(registry: ToolRegistry) -> None:
    """Register the default set of engine-wrapping tools.

    Deep research is intentionally NOT registered by default — it requires
    an injected LlmClient + TierRouter pair. Callers wanting that tool should
    construct their own Tool and pass it to ``registry.register()``.
    """
    registry.register(
        Tool(
            name="enchanter.scan_secrets",
            description=(
                "Scan a text corpus for secret patterns "
                "(API keys, bearer tokens, etc.). Returns matched pattern IDs."
            ),
            input_schema=_SCAN_SECRETS_SCHEMA,
            handler=scan_secrets_handler,
        )
    )
    registry.register(
        Tool(
            name="enchanter.check_destructive_op",
            description=(
                "Evaluate a shell-like command against the destructive-op gate "
                "and report whether it would be vetoed."
            ),
            input_schema=_CHECK_DESTRUCTIVE_OP_SCHEMA,
            handler=check_destructive_op_handler,
        )
    )


# JSON serialisation helper for tools/call result payload
def to_mcp_call_result(value: Any) -> dict[str, Any]:
    """Wrap a tool's return value into MCP tools/call result shape.

    MCP tools/call result envelope: {"content": [{"type": "text", "text": ...}]}
    We serialise the handler dict as JSON text content.
    """
    text = json.dumps(value, ensure_ascii=False)
    return {
        "content": [{"type": "text", "text": text}],
        "isError": False,
    }
