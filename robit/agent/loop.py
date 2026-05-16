"""robit.agent.loop — the user-prompt → LLM → tool loop.

Wave 15.0 ships the contracts + a fully-working mock LLM path. Wave 15.1+ will
swap the mock for real :func:`robit.proxy.pipeline.run` calls (the seam is
the :attr:`AgentLoop.dispatch_fn` attribute — tests override it directly).

The loop drives one user turn through the canonical request shape:

  1. Append the user message to the conversation.
  2. Call ``dispatch_fn(canonical_request) -> PipelineResult | VetoResult``.
  3. On veto: yield :class:`VetoFired` + :class:`TurnComplete`; the turn ends
     without a tool execution.
  4. On success: yield :class:`AssistantTextDelta` for every text part, append
     the assistant message, and for each ``tool_use`` part:
       a. yield :class:`ToolCallProposed`
       b. if ``requires_approval`` and no approval handler is wired,
          yield :class:`ApprovalRequested` and STOP (Wave 15.2G renders this)
       c. otherwise run the tool, yield :class:`ToolCallExecuted`, append
          the tool result, and loop back to step 2.
  5. When the assistant returns text-only (no tool_use), yield
     :class:`TurnComplete` and exit.

A hard cap (``MAX_ITERATIONS``) on per-turn LLM calls prevents infinite loops
where the model keeps requesting the same tool.

Events are tiny frozen dataclasses. The Textual app subscribes to the
async iterator and renders each event to the RichLog widget.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Union

from robit.proxy.canonical import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    Message as CanonicalMessage,
    TextPart,
    Tool as CanonicalToolDef,
    ToolResultPart,
    ToolUsePart,
)
from robit.proxy.pipeline import (
    PipelineResult,
    VetoResult,
    run as pipeline_run,
)

from .conversation import Conversation
from .session import SessionWriter
from .tools import Tool, ToolCall, ToolContext, ToolRegistry, ToolResult


_log = logging.getLogger(__name__)


# Hard cap to break runaway tool-call loops.
MAX_ITERATIONS: int = 20


# ---------------------------------------------------------------------------
# AgentEvent — what the loop yields.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistantThinking:
    """Emitted once per LLM call, right before the request flies."""

    iteration: int


@dataclass(frozen=True)
class AssistantTextDelta:
    """One chunk of assistant text. For Wave 15.0 (non-streaming mock) this
    fires once per text part with the whole text; Wave 15.2 will switch to
    true streaming and fire many small deltas instead."""

    text: str


@dataclass(frozen=True)
class ToolCallProposed:
    """The LLM emitted a ``tool_use`` part. The loop has not yet executed."""

    tool_name: str
    args: dict
    requires_approval: bool
    tool_use_id: str


@dataclass(frozen=True)
class ApprovalRequested:
    """A ``requires_approval=True`` tool is pending. The UI must call
    :meth:`AgentLoop.approve` or :meth:`AgentLoop.reject` to continue.

    Wave 15.0 surfaces this event when no approval handler is wired; in
    Wave 15.2G the Textual app supplies a real approval handler and this
    event becomes the trigger for the diff/approval pane."""

    tool_name: str
    args: dict
    tool_use_id: str


@dataclass(frozen=True)
class ToolCallExecuted:
    """A tool just ran. ``result`` is the LLM-visible content; ``side_effects``
    is the display-only summary the UI may render in a chip."""

    tool_name: str
    result: str
    is_error: bool
    side_effects: tuple[str, ...]
    tool_use_id: str


@dataclass(frozen=True)
class VetoFired:
    """The proxy pipeline vetoed this request. No tool execution happened."""

    plugin: str
    reason: str
    phase: str
    pattern_id: str | None


@dataclass(frozen=True)
class TurnComplete:
    """The turn finished cleanly. ``usage`` aggregates tokens across every
    LLM call inside the turn."""

    usage: CanonicalUsage
    iterations: int
    stop_reason: str | None = None


AgentEvent = Union[
    AssistantThinking,
    AssistantTextDelta,
    ToolCallProposed,
    ApprovalRequested,
    ToolCallExecuted,
    VetoFired,
    TurnComplete,
]


# ---------------------------------------------------------------------------
# Dispatch type alias — pluggable for tests.
# ---------------------------------------------------------------------------

DispatchFn = Callable[
    [CanonicalRequest], Awaitable[Union[PipelineResult, VetoResult]]
]


async def _real_pipeline_dispatch(
    req: CanonicalRequest,
) -> Union[PipelineResult, VetoResult]:
    """Default dispatch: route through the full proxy pipeline."""
    return await pipeline_run(req)


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


@dataclass
class AgentLoop:
    """Drives one user-prompt → LLM → tool-execute → loop-until-done cycle.

    Each LLM call goes through :func:`robit.proxy.pipeline.run` by
    default. Tests override :attr:`dispatch_fn` to inject a deterministic
    mock so no real network is touched.

    Attributes
    ----------
    conversation:
        Mutable reference to the active conversation. The loop replaces it
        with a fresh instance per append.
    tool_registry:
        Active tools the LLM may invoke.
    session_writer:
        Optional JSONL writer. When set, every appended message is also
        persisted to disk.
    dispatch_fn:
        Coroutine that takes a :class:`CanonicalRequest` and returns a
        :class:`PipelineResult` or :class:`VetoResult`. Tests swap this for
        a mock; production uses :func:`_real_pipeline_dispatch`.
    cwd:
        Working directory handed to tools via :class:`ToolContext`.
    approval_pending:
        Internal queue of (tool_use_id, future) for tools that requested
        approval. Wave 15.2 wires the Textual UI to resolve these futures.
    """

    conversation: Conversation
    tool_registry: ToolRegistry
    session_writer: SessionWriter | None = None
    dispatch_fn: DispatchFn = field(default=_real_pipeline_dispatch)
    cwd: Path = field(default_factory=Path.cwd)
    auto_approve: bool = field(default=False)

    # Internal — set by approval handler hooks
    _approval_resolver: dict[str, asyncio.Future] = field(default_factory=dict)

    # ----- public approval API ---------------------------------------------

    def approve(self, tool_use_id: str) -> None:
        fut = self._approval_resolver.pop(tool_use_id, None)
        if fut is not None and not fut.done():
            fut.set_result(True)

    def reject(self, tool_use_id: str) -> None:
        fut = self._approval_resolver.pop(tool_use_id, None)
        if fut is not None and not fut.done():
            fut.set_result(False)

    # ----- turn driver -----------------------------------------------------

    async def run_turn(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Drive a single user turn to completion (or veto, or cap).

        Generator: yields :class:`AgentEvent` instances as the turn unfolds.
        Caller consumes via ``async for ev in loop.run_turn(...)``.
        """
        # Append the user message (and persist).
        self.conversation = self.conversation.append_user(user_input)
        if self.session_writer is not None:
            self.session_writer.write_message(self.conversation.messages[-1])

        total_in = 0
        total_out = 0
        iterations = 0
        stop_reason: str | None = None

        while iterations < MAX_ITERATIONS:
            iterations += 1
            yield AssistantThinking(iteration=iterations)

            req = self._build_request()
            result = await self.dispatch_fn(req)

            if isinstance(result, VetoResult):
                yield VetoFired(
                    plugin=result.plugin,
                    reason=result.reason,
                    phase=result.phase,
                    pattern_id=result.pattern_id,
                )
                stop_reason = "veto"
                break

            resp: CanonicalResponse = result.response
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens

            # Append the assistant turn.
            self.conversation = self.conversation.append_assistant(resp.content)
            if self.session_writer is not None:
                self.session_writer.write_message(self.conversation.messages[-1])

            # Stream out the text parts to the UI.
            for part in resp.content:
                if isinstance(part, TextPart):
                    if part.text:
                        yield AssistantTextDelta(text=part.text)

            # Collect tool_use parts to execute.
            tool_uses = [p for p in resp.content if isinstance(p, ToolUsePart)]
            stop_reason = resp.stop_reason

            if not tool_uses:
                # No tool requested — the model has answered, turn is done.
                break

            # Execute each tool_use in order; their results become the
            # next user message.
            for use in tool_uses:
                tool = self.tool_registry.get(use.name) if use.name in self.tool_registry else None
                if tool is None:
                    err = f"unknown tool: {use.name}"
                    yield ToolCallExecuted(
                        tool_name=use.name,
                        result=err,
                        is_error=True,
                        side_effects=(),
                        tool_use_id=use.id,
                    )
                    self.conversation = self.conversation.append_tool_result(
                        use.id, err, is_error=True
                    )
                    if self.session_writer is not None:
                        self.session_writer.write_message(self.conversation.messages[-1])
                    continue

                requires_approval = bool(getattr(tool, "requires_approval", False))
                yield ToolCallProposed(
                    tool_name=tool.name,
                    args=use.input,
                    requires_approval=requires_approval,
                    tool_use_id=use.id,
                )

                if requires_approval and not self.auto_approve:
                    fut: asyncio.Future = asyncio.get_event_loop().create_future()
                    self._approval_resolver[use.id] = fut
                    yield ApprovalRequested(
                        tool_name=tool.name,
                        args=use.input,
                        tool_use_id=use.id,
                    )
                    approved = await fut
                    if not approved:
                        msg = f"user rejected tool call: {tool.name}"
                        yield ToolCallExecuted(
                            tool_name=tool.name,
                            result=msg,
                            is_error=True,
                            side_effects=(),
                            tool_use_id=use.id,
                        )
                        self.conversation = self.conversation.append_tool_result(
                            use.id, msg, is_error=True
                        )
                        if self.session_writer is not None:
                            self.session_writer.write_message(self.conversation.messages[-1])
                        continue

                # Execute under timeout + output cap.
                result_obj = await self._execute_tool(tool, use.input, use.id)
                yield ToolCallExecuted(
                    tool_name=tool.name,
                    result=result_obj.content,
                    is_error=result_obj.is_error,
                    side_effects=result_obj.side_effects,
                    tool_use_id=use.id,
                )
                self.conversation = self.conversation.append_tool_result(
                    use.id, result_obj.content, is_error=result_obj.is_error
                )
                if self.session_writer is not None:
                    self.session_writer.write_message(self.conversation.messages[-1])

            # Loop back: the LLM gets another shot now that it has tool
            # results in the conversation.
            continue

        if self.session_writer is not None:
            self.session_writer.write_usage(total_in, total_out)

        yield TurnComplete(
            usage=CanonicalUsage(input_tokens=total_in, output_tokens=total_out),
            iterations=iterations,
            stop_reason=stop_reason,
        )

    # ----- helpers ---------------------------------------------------------

    def _build_request(self) -> CanonicalRequest:
        """Render the current conversation as a :class:`CanonicalRequest`.

        Tool definitions come from the registry; the system prompt is the
        conversation's stored value. The proxy pipeline takes it from here
        — conduct injection, trust-gate, dispatch all happen inside ``run``.
        """
        tools = tuple(
            CanonicalToolDef(
                name=defn["name"],
                description=defn["description"],
                input_schema=defn["input_schema"],
            )
            for defn in self.tool_registry.listing()
        )
        return CanonicalRequest(
            model=self.conversation.model,
            messages=self.conversation.messages,
            system=self.conversation.system_prompt,
            tools=tools,
            tool_choice="auto" if tools else None,
        )

    async def _execute_tool(
        self, tool: Tool, args: dict, tool_use_id: str
    ) -> ToolResult:
        ctx = ToolContext(
            cwd=self.cwd,
            session_id=self.conversation.session_id,
            max_output_bytes=64 * 1024,
            timeout_s=30.0,
        )
        try:
            res = await asyncio.wait_for(tool.execute(args, ctx), timeout=ctx.timeout_s)
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"tool '{tool.name}' timed out after {ctx.timeout_s}s",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            _log.exception("tool %r raised", tool.name)
            return ToolResult(
                content=f"tool '{tool.name}' raised {type(exc).__name__}: {exc}",
                is_error=True,
            )

        # Enforce the output cap.
        if len(res.content.encode("utf-8")) > ctx.max_output_bytes:
            truncated = res.content.encode("utf-8")[: ctx.max_output_bytes].decode(
                "utf-8", errors="ignore"
            )
            return replace(
                res,
                content=truncated + "\n...[truncated]",
            )
        return res


__all__ = [
    "AgentLoop",
    "AgentEvent",
    "AssistantThinking",
    "AssistantTextDelta",
    "ToolCallProposed",
    "ApprovalRequested",
    "ToolCallExecuted",
    "VetoFired",
    "TurnComplete",
    "MAX_ITERATIONS",
    "DispatchFn",
]
