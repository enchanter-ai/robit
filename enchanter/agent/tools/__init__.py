"""enchanter.agent.tools — Tool Protocol + registry + the dummy echo tool.

Wave 15.0 ships ONE tool — :class:`EchoTool` — so the loop has something to
wire end-to-end. Real tools (file_read, file_write, bash, ...) land in
Wave 15.1 against this contract.

The contract:

    class Tool(Protocol):
        name: str
        description: str
        input_schema: dict           # JSON schema, dispatched to the LLM
        requires_approval: bool      # True → ApprovalRequested event before execute

        async def execute(self, args: dict, ctx: ToolContext) -> ToolResult: ...

Every tool implementation conforms to the Protocol above. The :class:`ToolRegistry`
holds a name → instance map and exposes :meth:`ToolRegistry.listing` to render
LLM-compatible tool definitions for the request.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ._types import ToolCall, ToolContext, ToolResult


@runtime_checkable
class Tool(Protocol):
    """Contract every tool implementation must satisfy.

    Wave 15.1 tool authors: subclass nothing, just implement the four
    attributes + ``execute``. The :func:`isinstance(t, Tool)` check at
    registration time is a runtime structural guard.
    """

    name: str
    description: str
    input_schema: dict
    requires_approval: bool

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    """Name → :class:`Tool` map used by the agent loop.

    Registration is one-shot per name; re-registering the same name raises
    :class:`ValueError` so a typo or accidental duplicate import is caught
    loudly rather than silently shadowing the original.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError(
                f"object does not conform to Tool protocol: {tool!r}"
            )
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"no such tool: {name!r}") from exc

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def listing(self) -> list[dict]:
        """Return LLM-compatible tool definitions for the current request.

        Shape matches :class:`enchanter.proxy.canonical.Tool` fields so the
        loop can pass these straight through to a ``CanonicalRequest.tools``
        tuple after a thin dataclass conversion.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]


# ---------------------------------------------------------------------------
# Dummy echo tool — end-to-end smoke for the loop.
# ---------------------------------------------------------------------------


class EchoTool:
    """Returns the ``text`` arg unchanged. Auto-approved.

    Exists so Wave 15.0 can demonstrate a full loop turn (LLM → tool call →
    tool result → LLM) without any side effects. Wave 15.1 will replace this
    with real tools.
    """

    name: str = "echo"
    description: str = "Echo back the provided text. Useful for round-trip smoke."
    input_schema: dict = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to echo back verbatim.",
            }
        },
        "required": ["text"],
    }
    requires_approval: bool = False

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        text = args.get("text", "")
        if not isinstance(text, str):
            return ToolResult(
                content=f"echo: expected string 'text' arg, got {type(text).__name__}",
                is_error=True,
            )
        return ToolResult(content=text, is_error=False)


def default_registry(*, include_echo: bool = False) -> "ToolRegistry":
    """Build a registry with all production tools registered.

    Wave 15.1 tools:
      - file_read, file_write, file_edit (atomic + line-diff output)
      - glob, grep (mtime-sorted, skip-dirs, context-aware)
      - bash (destructive-op-gate vetoes BEFORE execution)
      - web_fetch (HTTPS-only, SSRF-guarded, HTML-to-text)

    Pass include_echo=True to also register the round-trip smoke tool.
    """
    from .bash import BashTool
    from .file_edit import FileEditTool
    from .file_read import FileReadTool
    from .file_write import FileWriteTool
    from .glob import GlobTool
    from .grep import GrepTool
    from .web_fetch import WebFetchTool

    reg = ToolRegistry()
    reg.register(FileReadTool())
    reg.register(FileWriteTool())
    reg.register(FileEditTool())
    reg.register(GlobTool())
    reg.register(GrepTool())
    reg.register(BashTool())
    reg.register(WebFetchTool())
    if include_echo:
        reg.register(EchoTool())
    return reg


__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolContext",
    "ToolResult",
    "ToolCall",
    "EchoTool",
    "default_registry",
]
