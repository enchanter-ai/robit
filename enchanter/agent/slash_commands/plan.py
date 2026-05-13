"""enchanter.agent.slash_commands.plan — /plan, /edit, /cancel, /execute.

Wave 15.3 / Agent K — plan mode.

The user types ``/plan refactor the auth module``; the agent drives an LLM
planning call that returns a structured plan; the plan is rendered as a
checklist; the user reviews / edits / cancels; finally ``/execute`` walks
the steps.

The :class:`SlashContext` does not (yet) expose a writable scratch field, so
plan state is parked in a module-level dict keyed by ``session_id``.  This
is a v1 shortcut — Wave 15.4+ should refactor to a proper ``ctx.scratch``
attribute so concurrent sessions don't share global state.  Today's REPL is
single-session, so the simplification is safe.

``/execute`` cannot directly drive the agent loop from inside a slash
command — slash commands run *outside* the LLM turn-driver.  The v1 design
therefore renders the plan as a natural-language prompt the user can submit
as their next turn (or that the loop's outer caller can dispatch).  Wave
15.4+ will wire a true step-by-step executor that hands each step to the
loop with the existing approval flow.
"""

from __future__ import annotations

from dataclasses import replace

from ..plan import (
    Plan,
    PlanParseError,
    PlanStep,
    Planner,
)
from ..slash import SlashContext


# ---------------------------------------------------------------------------
# Module-level scratch — keyed by session_id.
# ---------------------------------------------------------------------------


_PLAN_SCRATCH: dict[str, Plan] = {}


def _get_plan(ctx: SlashContext) -> Plan | None:
    return _PLAN_SCRATCH.get(ctx.conversation.session_id)


def _set_plan(ctx: SlashContext, plan: Plan | None) -> None:
    sid = ctx.conversation.session_id
    if plan is None:
        _PLAN_SCRATCH.pop(sid, None)
    else:
        _PLAN_SCRATCH[sid] = plan


def _reset_scratch_for_tests() -> None:
    """Test helper — wipe the module-level scratch."""
    _PLAN_SCRATCH.clear()


# ---------------------------------------------------------------------------
# Planner factory — overridable by tests.
# ---------------------------------------------------------------------------


def _default_planner_factory(ctx: SlashContext) -> Planner:
    """Build a real Planner using the proxy pipeline.

    Tests monkey-patch this module attribute to inject a mock Planner so no
    real network is touched.
    """
    return Planner(tool_registry=ctx.tool_registry)


# This indirection lets tests replace the factory without monkey-patching
# the Planner class itself.
_planner_factory = _default_planner_factory


# ---------------------------------------------------------------------------
# /plan
# ---------------------------------------------------------------------------


class PlanCommand:
    """``/plan <goal>`` — enter planning mode and produce a draft plan."""

    name = "/plan"
    description = (
        "Plan a goal without executing tools. Argument: the goal to plan."
    )

    async def execute(self, args: str, ctx: SlashContext) -> str:
        goal = args.strip()
        if not goal:
            return (
                "Usage: /plan <goal>\n"
                "Example: /plan refactor the auth module to use env-var "
                "credentials"
            )

        planner = _planner_factory(ctx)
        try:
            plan = await planner.plan(goal, ctx.conversation)
        except PlanParseError as exc:
            return f"Plan generation failed: {exc}"

        _set_plan(ctx, plan)
        return plan.render_checklist()


# ---------------------------------------------------------------------------
# /edit
# ---------------------------------------------------------------------------


class EditStepCommand:
    """``/edit <n> <new description>`` — replace a step's description."""

    name = "/edit"
    description = (
        "Edit a step in the current plan. Usage: /edit <step_index> "
        "<new description>"
    )

    async def execute(self, args: str, ctx: SlashContext) -> str:
        plan = _get_plan(ctx)
        if plan is None:
            return "No active plan. Run /plan <goal> first."

        head, _, rest = args.strip().partition(" ")
        if not head or not rest.strip():
            return "Usage: /edit <step_index> <new description>"

        try:
            idx = int(head)
        except ValueError:
            return f"Step index must be an integer, got {head!r}."

        old_step = next((s for s in plan.steps if s.index == idx), None)
        if old_step is None:
            return (
                f"No step {idx} in the current plan "
                f"(plan has {len(plan.steps)} steps)."
            )

        new_step = replace(old_step, description=rest.strip())
        try:
            plan = plan.with_replaced_step(idx, new_step)
        except IndexError as exc:
            return f"Edit failed: {exc}"

        _set_plan(ctx, plan)
        return plan.render_checklist()


# ---------------------------------------------------------------------------
# /cancel
# ---------------------------------------------------------------------------


class CancelPlanCommand:
    """``/cancel`` — discard the current plan."""

    name = "/cancel"
    description = "Cancel the current plan."

    async def execute(self, args: str, ctx: SlashContext) -> str:
        plan = _get_plan(ctx)
        if plan is None:
            return "No active plan to cancel."
        _set_plan(ctx, None)
        return f"Plan cancelled: {plan.goal}"


# ---------------------------------------------------------------------------
# /execute
# ---------------------------------------------------------------------------


class ExecutePlanCommand:
    """``/execute`` — emit a prompt that drives the LLM through the plan.

    v1 limitation: slash commands cannot directly drive the agent loop
    (slash commands run *outside* a turn).  This implementation therefore
    renders the plan as a natural-language prompt the user can submit as
    their next turn — the LLM then proposes tool_use calls per step, each
    routed through the existing approval flow in :mod:`enchanter.agent.loop`.

    Wave 15.4+ should replace this with a true step-by-step executor that
    yields :class:`enchanter.agent.loop.AgentEvent` instances directly.
    """

    name = "/execute"
    description = "Execute the current plan step by step."

    async def execute(self, args: str, ctx: SlashContext) -> str:
        plan = _get_plan(ctx)
        if plan is None:
            return "No active plan. Run /plan <goal> first."
        if not plan.steps:
            _set_plan(ctx, None)
            return f"Plan has no steps; nothing to execute. (goal: {plan.goal})"

        # Mark plan as executing for the consuming layer.
        plan = plan.with_status("executing")
        _set_plan(ctx, plan)

        lines = [
            f"Executing plan for: {plan.goal}",
            "",
            "Submit the following as your next message so the LLM walks the "
            "plan step by step. Each tool call will go through the usual "
            "approval flow.",
            "",
            "----- PLAN PROMPT -----",
            f"Please execute the following plan for: {plan.goal}",
            "",
        ]
        for s in plan.steps:
            tool_hint = (
                f" (using tool: {s.tool_name})" if s.tool_name else " (think)"
            )
            lines.append(f"  Step {s.index}. {s.description}{tool_hint}")
        lines.append("")
        lines.append(
            "Work through the steps in order. After each step, briefly "
            "summarise what you did before moving to the next."
        )
        lines.append("----- END PLAN PROMPT -----")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def all_plan_commands() -> list[object]:
    """Return all plan-mode slash commands as a list.

    Typed as ``list[object]`` rather than ``list[SlashCommand]`` to avoid a
    runtime Protocol check at import time — the registry already validates
    the structural contract at ``register()`` time.
    """
    return [
        PlanCommand(),
        EditStepCommand(),
        CancelPlanCommand(),
        ExecutePlanCommand(),
    ]


__all__ = [
    "PlanCommand",
    "EditStepCommand",
    "CancelPlanCommand",
    "ExecutePlanCommand",
    "all_plan_commands",
]
