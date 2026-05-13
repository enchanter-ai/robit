"""enchanter.agent.plan — plan data structures + LLM-driven planner.

Plan mode is the "write a plan, execute later" workflow:

  1. ``/plan <goal>`` drives the LLM through a planning-only prompt that asks
     for a JSON plan (no tool calls).
  2. The returned :class:`Plan` is rendered as a checklist for the user to
     review, edit, or cancel.
  3. ``/execute`` (Wave 15.3+) walks the plan one step at a time, surfacing
     each ``tool_call`` to the existing approval flow.

The planner reuses :func:`enchanter.proxy.pipeline.run` so conduct injection
and the trust-gate still apply.  The only difference is the system prompt
addendum (``PLAN_PROMPT_TEMPLATE``) which tells the model NOT to emit
``tool_use`` blocks — produce a JSON plan instead.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable, Literal, Union

from enchanter.proxy.canonical import (
    CanonicalRequest,
    Message as CanonicalMessage,
    TextPart,
)
from enchanter.proxy.pipeline import (
    PipelineResult,
    VetoResult,
    run as pipeline_run,
)

from .conversation import Conversation
from .tools import ToolRegistry


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class PlanParseError(ValueError):
    """Raised when the LLM's planning response cannot be parsed into a Plan.

    The error message includes a short excerpt of the offending output so
    operators can diagnose drift without re-running the call.
    """


# ---------------------------------------------------------------------------
# Plan data classes (frozen, immutable).
# ---------------------------------------------------------------------------


PlanStepStatus = Literal["pending", "running", "done", "skipped", "failed"]
PlanStatus = Literal["draft", "executing", "completed", "cancelled"]


@dataclass(frozen=True)
class PlanStep:
    """One atomic action inside a :class:`Plan`.

    ``tool_name`` of ``None`` denotes a "think/discuss" step — no tool will
    be invoked at ``/execute`` time; the LLM (or user) just considers the
    step and marks it done.

    Steps are 1-indexed because that's what users see in the rendered
    checklist (``[ ] 1. Read auth.py``) and what they type into ``/edit``.
    """

    index: int
    description: str
    tool_name: str | None
    tool_args: dict | None
    status: PlanStepStatus = "pending"
    result: str | None = None


@dataclass(frozen=True)
class Plan:
    """A reviewable, editable list of steps the agent will execute later.

    Frozen + functional: every mutation (``with_step_status``,
    ``with_replaced_step``) returns a fresh instance.  The slash-command
    layer swaps its stashed reference on each mutation — the same pattern
    :class:`Conversation` uses.
    """

    goal: str
    steps: tuple[PlanStep, ...]
    created_ts: float
    status: PlanStatus = "draft"

    # ----- mutators -------------------------------------------------------

    def with_step_status(
        self, index: int, status: PlanStepStatus, result: str | None = None
    ) -> "Plan":
        """Return a new Plan with step ``index`` mutated to ``status``.

        Raises :class:`IndexError` if ``index`` is out of range so callers
        don't silently no-op on a typo.
        """
        new_steps: list[PlanStep] = []
        found = False
        for s in self.steps:
            if s.index == index:
                new_steps.append(replace(s, status=status, result=result))
                found = True
            else:
                new_steps.append(s)
        if not found:
            raise IndexError(f"no step with index {index}")
        return replace(self, steps=tuple(new_steps))

    def with_replaced_step(self, index: int, new_step: PlanStep) -> "Plan":
        """Replace step ``index`` wholesale (preserves index field)."""
        new_steps: list[PlanStep] = []
        found = False
        for s in self.steps:
            if s.index == index:
                new_steps.append(replace(new_step, index=index))
                found = True
            else:
                new_steps.append(s)
        if not found:
            raise IndexError(f"no step with index {index}")
        return replace(self, steps=tuple(new_steps))

    def with_status(self, status: PlanStatus) -> "Plan":
        return replace(self, status=status)

    # ----- queries --------------------------------------------------------

    def is_complete(self) -> bool:
        """All steps must be either ``done`` or ``skipped`` (never pending,
        running, or failed) for the plan to count as complete."""
        if not self.steps:
            return True
        return all(s.status in ("done", "skipped") for s in self.steps)

    def render_checklist(self) -> str:
        """ASCII checklist for slash-command output."""
        lines = [f"Plan for: {self.goal}"]
        if not self.steps:
            lines.append("  (no steps)")
        else:
            for s in self.steps:
                box = {
                    "pending": "[ ]",
                    "running": "[~]",
                    "done": "[x]",
                    "skipped": "[-]",
                    "failed": "[!]",
                }.get(s.status, "[?]")
                tool_hint = f"  ({s.tool_name})" if s.tool_name else ""
                lines.append(
                    f"  {box} {s.index}. {s.description}{tool_hint}"
                )
        lines.append("")
        lines.append(
            "Type /execute to run, /edit <n> <new description> to modify, "
            "/cancel to abort."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planner — LLM-driven plan generation.
# ---------------------------------------------------------------------------


PlanDispatchFn = Callable[
    [CanonicalRequest], Awaitable[Union[PipelineResult, VetoResult]]
]


async def _real_dispatch(
    req: CanonicalRequest,
) -> Union[PipelineResult, VetoResult]:
    return await pipeline_run(req)


@dataclass
class Planner:
    """Drive the LLM in planning mode.

    The LLM call goes through the proxy pipeline like any other turn —
    conduct injection + trust-gate still apply.  The only difference is the
    appended planning instructions, which tell the model to produce JSON
    instead of emitting tool_use blocks.

    Tests inject ``dispatch_fn`` so no real network is touched.
    """

    tool_registry: ToolRegistry
    dispatch_fn: PlanDispatchFn = field(default=_real_dispatch)

    # Public so Wave 15.4 can quote it in user-facing docs.
    PLAN_PROMPT_TEMPLATE: str = (
        "You are in PLANNING MODE. Do NOT call any tools. Instead, produce a "
        "JSON object with this shape and nothing else:\n"
        "\n"
        "{\n"
        '  "steps": [\n'
        "    {\n"
        '      "description": "human-readable summary",\n'
        '      "tool": "file_read" | "file_write" | "file_edit" | "bash" | '
        '"glob" | "grep" | "web_fetch" | null,\n'
        '      "args": {}\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Each step is one atomic action. Use null for the tool when a step "
        "is a thinking step (e.g., \"Review the design notes\"). Prefer many "
        "small steps over a few large ones — the user will review each one. "
        "Return ONLY the JSON object (no markdown fences, no commentary)."
    )

    async def plan(self, goal: str, conversation: Conversation) -> Plan:
        """Send the goal + planning instructions to the LLM and parse the
        returned plan.

        Tolerates JSON wrapped in markdown code fences and leading prose; on
        outright malformed output, falls back to numbered-list parsing
        before raising :class:`PlanParseError`.
        """
        # Build a planning-mode system prompt by appending the template to
        # whatever the conversation already carries.  We do NOT mutate the
        # caller's conversation — the goal text is appended as a user turn
        # in the request only, leaving conversation history intact.
        base_system = conversation.system_prompt or ""
        sep = "\n\n" if base_system else ""
        planning_system = f"{base_system}{sep}{self.PLAN_PROMPT_TEMPLATE}"

        user_msg = CanonicalMessage(
            role="user",
            content=(
                TextPart(text=f"Plan the following goal:\n\n{goal}"),
            ),
        )

        req = CanonicalRequest(
            model=conversation.model,
            messages=conversation.messages + (user_msg,),
            system=planning_system,
            tools=(),  # CRITICAL: no tools in planning mode.
            tool_choice=None,
        )

        result = await self.dispatch_fn(req)
        if isinstance(result, VetoResult):
            raise PlanParseError(
                f"planning was vetoed by {result.plugin}: {result.reason}"
            )

        text = _extract_text(result)
        steps = _parse_plan_text(text)
        return Plan(
            goal=goal,
            steps=tuple(steps),
            created_ts=time.time(),
            status="draft",
        )


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def _extract_text(result: PipelineResult) -> str:
    """Concatenate all TextPart content from the response."""
    chunks: list[str] = []
    for part in result.response.content:
        if isinstance(part, TextPart):
            chunks.append(part.text)
    return "".join(chunks).strip()


_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def _parse_plan_text(text: str) -> list[PlanStep]:
    """Parse an LLM planning response into a list of :class:`PlanStep`.

    Strategy (in order):
      1. Try to find a fenced JSON block (```json ... ``` or ``` ... ```).
      2. Try to find a bare top-level JSON object anywhere in the text.
      3. Fall back to numbered-list parsing (``1. step``).
      4. Otherwise raise :class:`PlanParseError` with an excerpt.

    An empty ``steps`` list is a valid plan (returns ``[]``), not an error.
    """
    if not text:
        # Empty response = empty plan, per the contract.
        return []

    obj = _try_extract_json_object(text)
    if obj is not None:
        return _steps_from_json_object(obj)

    # Numbered-list fallback.
    fallback = _parse_numbered_list(text)
    if fallback:
        return fallback

    raise PlanParseError(
        f"could not parse plan from LLM output (excerpt: {text[:200]!r})"
    )


def _try_extract_json_object(text: str) -> dict | None:
    """Locate the first JSON object in ``text``, tolerating fences + prose."""
    # 1. Fenced block.
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 2. Bare top-level object: walk to first '{' and try increasingly long
    # slices until one parses.  Cheap because the LLM rarely emits anything
    # before the JSON.
    start = text.find("{")
    if start == -1:
        return None
    # Use a balanced-brace scan to bound the slice.
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
                break
    return None


def _steps_from_json_object(obj: dict) -> list[PlanStep]:
    raw_steps = obj.get("steps")
    if raw_steps is None:
        raise PlanParseError(
            "JSON plan missing 'steps' key"
        )
    if not isinstance(raw_steps, list):
        raise PlanParseError(
            f"JSON plan 'steps' must be a list, got {type(raw_steps).__name__}"
        )
    out: list[PlanStep] = []
    for i, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            raise PlanParseError(
                f"step {i}: expected object, got {type(raw).__name__}"
            )
        desc = raw.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise PlanParseError(
                f"step {i}: missing or empty 'description'"
            )
        tool = raw.get("tool")
        if tool is not None and not isinstance(tool, str):
            raise PlanParseError(
                f"step {i}: 'tool' must be a string or null"
            )
        args = raw.get("args")
        if args is not None and not isinstance(args, dict):
            raise PlanParseError(
                f"step {i}: 'args' must be an object or null"
            )
        out.append(
            PlanStep(
                index=i,
                description=desc.strip(),
                tool_name=tool,
                tool_args=args if args is not None else ({} if tool else None),
            )
        )
    return out


_NUMBERED_RE = re.compile(r"^\s*(\d+)[\.\)]\s+(.+?)\s*$")


def _parse_numbered_list(text: str) -> list[PlanStep]:
    """Fallback for LLM output like ``1. Do X\n2. Do Y`` (no JSON)."""
    out: list[PlanStep] = []
    seen_index = 0
    for line in text.splitlines():
        m = _NUMBERED_RE.match(line)
        if not m:
            continue
        seen_index += 1
        out.append(
            PlanStep(
                index=seen_index,
                description=m.group(2).strip(),
                tool_name=None,
                tool_args=None,
            )
        )
    return out


__all__ = [
    "Plan",
    "PlanStep",
    "PlanStatus",
    "PlanStepStatus",
    "PlanParseError",
    "Planner",
    "PlanDispatchFn",
]
