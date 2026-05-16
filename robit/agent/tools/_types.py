"""robit.agent.tools._types — tool contract dataclasses.

Foundation contracts for Wave 15.0. Wave 15.1+ tool implementations conform
to :class:`Tool` (defined in ``robit.agent.tools.__init__``); the supporting
context + result dataclasses live here so they import cleanly without pulling in
the registry.

Design rules:

* ``ToolContext`` and ``ToolResult`` are ``frozen=True`` — tool code receives
  them as read-only handles and cannot mutate them by accident.
* ``ToolContext.cwd`` is a :class:`pathlib.Path` so tool implementations get
  unambiguous platform-correct join semantics.
* ``ToolResult.content`` is the string the LLM sees. The agent loop is
  responsible for truncating beyond ``max_output_bytes``; tools may pre-truncate
  too, but they MUST set the truncation marker themselves if they do.
* ``side_effects`` is a tuple of short human-readable strings the UI may render
  separately ("wrote 42 lines to foo.py"). It never reaches the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ToolContext:
    """Read-only execution context handed to every :meth:`Tool.execute` call.

    Attributes
    ----------
    cwd:
        Working directory tools should resolve relative paths against. The
        agent loop normalises this once at session start; tool implementations
        SHOULD NOT call :func:`os.chdir` to drift away from it.
    session_id:
        Current session UUID hex. Useful for tools that emit audit records
        keyed by session.
    max_output_bytes:
        Soft cap on ``ToolResult.content`` length. The loop truncates beyond
        this point and appends a ``...[truncated]`` marker before feeding the
        result back to the LLM.
    timeout_s:
        Wall-clock budget for the tool's :meth:`execute` coroutine. The loop
        wraps :meth:`execute` in :func:`asyncio.wait_for`; on timeout the
        tool is cancelled and a :class:`ToolResult` with ``is_error=True``
        is fed back to the LLM.
    """

    cwd: Path
    session_id: str
    max_output_bytes: int = 64 * 1024
    timeout_s: float = 30.0


@dataclass(frozen=True)
class ToolResult:
    """The outcome of a single tool execution.

    Attributes
    ----------
    content:
        Stringified result body. Becomes the ``content`` field of the
        ``tool_result`` part the agent sends back to the LLM.
    is_error:
        When True, the loop sets ``is_error=True`` on the canonical
        :class:`~robit.proxy.canonical.ToolResultPart` so the LLM can
        distinguish success from failure.
    side_effects:
        Display-only summary strings. The loop forwards these to the UI as
        ``ToolCallExecuted.side_effects``; they never leave the agent process
        and never reach the LLM.
    """

    content: str
    is_error: bool = False
    side_effects: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolCall:
    """A pending tool invocation extracted from an assistant turn.

    The loop builds one of these per ``tool_use`` content part the LLM
    emitted. Wave 15.2 will surface this dataclass through the approval UI
    when ``requires_approval=True``.
    """

    id: str            # Matches the LLM's ToolUsePart.id; used to wire the result back.
    name: str
    args: dict
    requires_approval: bool


__all__ = ["ToolContext", "ToolResult", "ToolCall"]
