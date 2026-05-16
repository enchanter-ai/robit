"""robit.agent.subagents.dispatch — the ``subagent`` Tool.

The :class:`SubagentTool` is the bridge from the main :class:`AgentLoop` to
a recursively-spawned inner :class:`AgentLoop`. The main loop sees only a
normal tool result: a string (optionally JSON-shaped) summarizing what the
subagent did. The subagent's own LLM rounds, intermediate tool calls, and
session log live entirely inside the inner loop and never leak into the
main conversation.

Design rules
------------

1. **Fresh conversation.** The inner loop gets its own ``Conversation``
   with its own ``session_id`` and the role's ``system_prompt``. The main
   conversation is not threaded through; the caller must supply enough
   ``task`` + ``context_summary`` for the subagent to act alone.

2. **Filtered tools.** Only tools in ``role.allowed_tools`` are registered
   in the inner registry. ``None`` means "inherit everything from the
   parent registry". The ``subagent`` tool itself is NEVER registered into
   the inner registry unless explicitly listed — see recursion guard below.

3. **Turn cap.** The inner loop's :data:`robit.agent.loop.MAX_ITERATIONS`
   default is replaced for this invocation by ``role.max_turns`` via the
   ``max_iterations_override`` knob (we monkey-patch the inner loop's
   attribute on a per-instance basis without touching the module constant).

4. **Result extraction.** The final ``AssistantTextDelta`` events are
   concatenated to form the subagent's summary. When ``role.summary_schema``
   is set we try ``json.loads`` on the concatenated text; on failure we
   return the raw text and surface a side-effect warning so the main agent
   knows the structured contract wasn't honored.

5. **Recursion guard.** :data:`MAX_SUBAGENT_DEPTH` caps nesting. The
   :class:`SubagentTool` tracks its own active-depth counter (per-instance
   ``_depth``) so a subagent that *does* have ``subagent`` in its
   ``allowed_tools`` cannot exceed the cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from robit.proxy.canonical import TextPart

from ..conversation import Conversation
from ..tools import Tool, ToolRegistry
from ..tools._types import ToolContext, ToolResult
from .registry import SubagentRegistry, SubagentRole


# ---------------------------------------------------------------------------
# Recursion bound.
# ---------------------------------------------------------------------------

# Maximum subagent call nesting depth.
#
# Depth 1 = main agent → subagent (the normal case).
# Depth 2 = subagent → subagent (allowed only if the role's allowed_tools
#           explicitly includes "subagent"; rare).
# Depth 3+ = refused. The dispatch tool returns a ToolResult error so the
#           inner subagent's LLM sees the refusal and can adapt.
#
# This mirrors the inference-substrate "no depth-2 recursion" rule but is
# one level more permissive — a single nested level is occasionally useful
# (e.g. deep-research delegating a find-references search), but the engine
# never lets the stack grow unbounded.
MAX_SUBAGENT_DEPTH: int = 2


# ---------------------------------------------------------------------------
# Inner-loop factory contract.
# ---------------------------------------------------------------------------

# Tests inject this so no real LLM is hit. The factory receives the inner
# Conversation + the inner ToolRegistry and returns an object with an
# async ``run_turn(task: str)`` method yielding AgentEvents (duck-typed
# against AgentLoop).
LoopFactory = Callable[..., Any]


# ---------------------------------------------------------------------------
# SubagentTool
# ---------------------------------------------------------------------------


class SubagentTool:
    """Dispatch a focused subagent for a bounded task.

    Construction
    ------------
    The tool takes a :class:`SubagentRegistry` (role catalog), a
    ``parent_loop_factory`` (so tests can inject a mock inner loop), and an
    optional ``parent_tool_registry`` from which the per-role filtered
    registry is built.

    A factory is preferred over a direct ``AgentLoop`` import so the call
    site can decide whether the inner loop should share the parent's
    dispatch_fn (production) or use a deterministic mock (tests). The
    factory's signature is:

        factory(*, conversation, tool_registry, cwd, session_writer=None,
                max_iterations) -> AgentLoop-like

    The inner loop must expose ``run_turn(text) -> AsyncIterator[AgentEvent]``
    matching :class:`robit.agent.loop.AgentLoop`.
    """

    name: str = "subagent"
    description: str = (
        "Dispatch a focused subagent for a bounded task. Subagents have an "
        "isolated conversation and a constrained tool set. Use this when the "
        "main task has a self-contained sub-task that benefits from a clean "
        "context (e.g., research, find references, review a diff). Returns "
        "the subagent's final summary."
    )
    requires_approval: bool = False

    def __init__(
        self,
        registry: SubagentRegistry,
        parent_loop_factory: LoopFactory,
        *,
        parent_tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self._registry = registry
        self._loop_factory = parent_loop_factory
        self._parent_tools = parent_tool_registry
        # Active-depth counter. Bumped on entry, decremented on exit.
        # A subagent that calls ``subagent`` again shares this instance.
        self._depth: int = 0

    # ----- input schema (computed from registry contents) ------------------

    @property
    def input_schema(self) -> dict:
        roles = self._registry.names()
        role_doc = (
            "Which subagent role to use. Available: "
            + ", ".join(roles)
            if roles
            else "Which subagent role to use."
        )
        schema: dict = {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": role_doc,
                },
                "task": {
                    "type": "string",
                    "description": (
                        "The task description for the subagent. Include "
                        "enough context for it to act alone — the subagent "
                        "does NOT see the main conversation."
                    ),
                },
                "context_summary": {
                    "type": "string",
                    "description": (
                        "Optional 1-2 paragraph summary of relevant main-"
                        "conversation context. Do NOT include the full "
                        "conversation."
                    ),
                },
            },
            "required": ["role", "task"],
            "additionalProperties": False,
        }
        if roles:
            schema["properties"]["role"]["enum"] = list(roles)
        return schema

    # ----- execute ---------------------------------------------------------

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        # 1. Validate role name.
        role_name = args.get("role")
        if not isinstance(role_name, str) or not role_name:
            return ToolResult(
                content="subagent: 'role' is required and must be a non-empty string",
                is_error=True,
            )
        try:
            role = self._registry.get(role_name)
        except KeyError:
            available = ", ".join(self._registry.names()) or "(none registered)"
            return ToolResult(
                content=(
                    f"subagent: unknown role {role_name!r}. "
                    f"Available roles: {available}."
                ),
                is_error=True,
            )

        # 2. Validate task.
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            return ToolResult(
                content="subagent: 'task' is required and must be a non-empty string",
                is_error=True,
            )

        # 3. Recursion guard.
        if self._depth >= MAX_SUBAGENT_DEPTH:
            return ToolResult(
                content=(
                    f"subagent: recursion depth {self._depth} would exceed "
                    f"MAX_SUBAGENT_DEPTH={MAX_SUBAGENT_DEPTH}. "
                    "Finish the task in the current subagent instead of "
                    "delegating further."
                ),
                is_error=True,
                side_effects=(
                    f"subagent role={role_name} refused (recursion depth cap)",
                ),
            )

        # 4. Build the inner Conversation.
        ctx_summary = args.get("context_summary")
        user_message = task
        if isinstance(ctx_summary, str) and ctx_summary.strip():
            user_message = (
                "Context from the main conversation:\n"
                f"{ctx_summary.strip()}\n\n"
                "Your task:\n"
                f"{task}"
            )

        inner_conv = Conversation.new(
            model=_pick_model_for_role(role),
            system_prompt=role.system_prompt,
        )

        # 5. Build the filtered tool registry.
        inner_tools = self._build_filtered_registry(role)

        # 6. Spawn the inner loop via the factory.
        inner_loop = self._loop_factory(
            conversation=inner_conv,
            tool_registry=inner_tools,
            cwd=ctx.cwd,
            max_iterations=role.max_turns,
        )

        # 7. Drive the inner loop, collecting assistant text + tool counts.
        self._depth += 1
        try:
            final_text, turns_used, tools_used, hit_cap = await _drive_inner_loop(
                inner_loop, user_message, role.max_turns
            )
        finally:
            self._depth -= 1

        # 8. Parse structured output if a schema is set.
        warning: Optional[str] = None
        if role.summary_schema is not None:
            parsed, parse_err = _try_parse_json(final_text)
            if parsed is None:
                warning = (
                    f"subagent role={role.name} returned non-JSON output "
                    f"({parse_err}); falling back to raw text"
                )
                content = final_text
            else:
                # Re-emit canonical JSON so the main agent sees consistent
                # formatting regardless of whether the subagent emitted
                # pretty-printed or compact JSON.
                content = json.dumps(parsed, ensure_ascii=False, indent=2)
        else:
            content = final_text

        # 9. Build side-effects.
        side: list[str] = [
            f"subagent role={role.name} ran {turns_used} turn(s), "
            f"used {tools_used} tool(s)"
        ]
        if hit_cap:
            side.append(
                f"subagent hit max_turns={role.max_turns} before completion; "
                "result may be partial"
            )
        if warning:
            side.append(warning)

        return ToolResult(
            content=content or f"(subagent role={role.name} produced no output)",
            is_error=False,
            side_effects=tuple(side),
        )

    # ----- internals -------------------------------------------------------

    def _build_filtered_registry(self, role: SubagentRole) -> ToolRegistry:
        """Return a fresh registry with the role's allowed tools.

        If ``role.allowed_tools is None``, mirror the parent registry as-is
        (minus the ``subagent`` tool itself unless explicitly named — we
        never silently grant recursion).
        """
        out = ToolRegistry()
        if self._parent_tools is None:
            # Nothing to filter from; the subagent gets an empty registry.
            # Roles with required tools will surface that gap inside their
            # own LLM turn ("I asked for grep but it isn't available").
            return out

        wanted: Optional[set[str]] = (
            set(role.allowed_tools) if role.allowed_tools is not None else None
        )
        for name in self._parent_tools.names():
            if wanted is not None and name not in wanted:
                continue
            if name == self.name and (
                wanted is None or self.name not in wanted
            ):
                # Never auto-grant the subagent tool to a subagent — would
                # break the recursion bound's intent.
                continue
            out.register(self._parent_tools.get(name))
        return out


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _pick_model_for_role(role: SubagentRole) -> str:
    """Pick a model id for the spawned inner conversation.

    Wave 15.3 ships with a single placeholder ("mock-model" in tests, the
    parent's model in production via factory override). Wixie's tier-
    sizing module would normally route deep-research → Sonnet,
    find-references → Haiku, review-diff → Sonnet. We delegate that choice
    to the factory rather than hard-coding here so the same code path
    works for tests AND production.
    """
    return "subagent-model"


async def _drive_inner_loop(
    inner_loop: Any,
    user_message: str,
    max_turns: int,
) -> tuple[str, int, int, bool]:
    """Run the inner loop and harvest (final_text, turns, tools_used, hit_cap).

    We consume the AgentEvent stream and accumulate:

    * concatenated assistant text from :class:`AssistantTextDelta`,
      cleared each iteration so only the LAST assistant turn survives —
      that's the subagent's "final answer" by convention,
    * iteration count from the :class:`TurnComplete` event (or by counting
      :class:`AssistantThinking` if the stream ends early),
    * tool execution count from :class:`ToolCallExecuted`,
    * whether the cap was hit (iterations == max_turns AND a tool was the
      last action, suggesting the loop was truncated mid-task).
    """
    # Lazy import to avoid module-load cycle (loop.py imports tools, which
    # may import subagents/__init__.py during future wiring).
    from ..loop import (
        AssistantTextDelta,
        AssistantThinking,
        ToolCallExecuted,
        TurnComplete,
    )

    final_text_parts: list[str] = []
    current_text_parts: list[str] = []
    thinking_count = 0
    tool_count = 0
    iterations_reported = 0
    stop_reason: Optional[str] = None

    async for ev in inner_loop.run_turn(user_message):
        if isinstance(ev, AssistantThinking):
            # New LLM round: flush the previous round's text into final_text
            # (it'll be overwritten by the next round if there is one) and
            # start collecting fresh.
            if current_text_parts:
                final_text_parts = current_text_parts
                current_text_parts = []
            thinking_count += 1
        elif isinstance(ev, AssistantTextDelta):
            current_text_parts.append(ev.text)
        elif isinstance(ev, ToolCallExecuted):
            tool_count += 1
        elif isinstance(ev, TurnComplete):
            iterations_reported = ev.iterations
            stop_reason = ev.stop_reason
            if current_text_parts:
                final_text_parts = current_text_parts
                current_text_parts = []

    # Fall back to whatever's still buffered if the stream ended without a
    # TurnComplete (shouldn't happen but defensive).
    if current_text_parts and not final_text_parts:
        final_text_parts = current_text_parts

    turns_used = iterations_reported or thinking_count
    hit_cap = turns_used >= max_turns and stop_reason != "end_turn"
    return ("".join(final_text_parts).strip(), turns_used, tool_count, hit_cap)


def _try_parse_json(text: str) -> tuple[Optional[Any], Optional[str]]:
    """Try to parse ``text`` as JSON, tolerating common LLM wrapper noise.

    Returns ``(parsed, None)`` on success or ``(None, error_message)`` on
    failure. Strips markdown code fences and leading/trailing prose if
    the JSON object is recoverable.
    """
    s = text.strip()
    if not s:
        return None, "empty output"

    # Strip a single ```json ... ``` or ``` ... ``` fence if present.
    if s.startswith("```"):
        # Drop opening fence line.
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        # Drop closing fence.
        if s.rstrip().endswith("```"):
            s = s.rstrip()[: -3].rstrip()

    # First attempt: parse as-is.
    try:
        return json.loads(s), None
    except json.JSONDecodeError as exc:
        first_err = str(exc)

    # Second attempt: find the outermost JSON object by brace matching.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1]), None
        except json.JSONDecodeError as exc:
            return None, f"{first_err}; brace-scan also failed: {exc}"

    return None, first_err


__all__ = [
    "SubagentTool",
    "MAX_SUBAGENT_DEPTH",
    "LoopFactory",
]
