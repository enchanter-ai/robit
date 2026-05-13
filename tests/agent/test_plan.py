"""Tests for enchanter.agent.plan — data structures + Planner."""

from __future__ import annotations

import pytest

from enchanter.agent.conversation import Conversation
from enchanter.agent.plan import (
    Plan,
    PlanParseError,
    PlanStep,
    Planner,
    _parse_plan_text,
)
from enchanter.agent.tools import ToolRegistry
from enchanter.proxy.canonical import (
    CanonicalResponse,
    CanonicalUsage,
    TextPart,
)
from enchanter.proxy.pipeline import PipelineResult, VetoResult


def _make_plan(steps=()) -> Plan:
    return Plan(goal="g", steps=tuple(steps), created_ts=0.0)


def _step(idx: int, desc: str = "x", tool: str | None = None) -> PlanStep:
    return PlanStep(
        index=idx,
        description=desc,
        tool_name=tool,
        tool_args={} if tool else None,
    )


def _resp(text: str) -> PipelineResult:
    return PipelineResult(
        response=CanonicalResponse(
            model="m",
            content=(TextPart(text=text),),
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=1, output_tokens=1),
        ),
        fired=(),
    )


# ---------------------------------------------------------------------------
# Plan dataclass.
# ---------------------------------------------------------------------------


def test_with_step_status_updates_immutably():
    p = _make_plan([_step(1), _step(2), _step(3)])
    p2 = p.with_step_status(2, "done", result="ok")
    assert p2 is not p
    assert p.steps[1].status == "pending"
    assert p2.steps[1].status == "done"
    assert p2.steps[1].result == "ok"
    # Untouched steps remain identical objects (immutability).
    assert p2.steps[0].status == "pending"
    assert p2.steps[2].status == "pending"


def test_with_step_status_unknown_index_raises():
    p = _make_plan([_step(1)])
    with pytest.raises(IndexError):
        p.with_step_status(99, "done")


def test_with_replaced_step_replaces_by_index():
    p = _make_plan([_step(1, "old"), _step(2, "keep")])
    new = PlanStep(
        index=42,  # Should be coerced back to 1 by with_replaced_step.
        description="new",
        tool_name="bash",
        tool_args={"cmd": "ls"},
    )
    p2 = p.with_replaced_step(1, new)
    assert p2.steps[0].index == 1
    assert p2.steps[0].description == "new"
    assert p2.steps[0].tool_name == "bash"
    # Step 2 untouched.
    assert p2.steps[1].description == "keep"


def test_with_replaced_step_unknown_index_raises():
    p = _make_plan([_step(1)])
    with pytest.raises(IndexError):
        p.with_replaced_step(99, _step(99))


def test_is_complete():
    # Empty plan → complete.
    assert _make_plan([]).is_complete() is True

    p = _make_plan([_step(1), _step(2)])
    assert p.is_complete() is False

    p2 = p.with_step_status(1, "done").with_step_status(2, "skipped")
    assert p2.is_complete() is True

    # Failed counts as NOT complete.
    p3 = p.with_step_status(1, "done").with_step_status(2, "failed")
    assert p3.is_complete() is False


# ---------------------------------------------------------------------------
# Planner.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_parses_valid_json():
    async def dispatch(req):
        return _resp(
            '{"steps": ['
            '{"description": "Read auth.py", "tool": "file_read", '
            '"args": {"path": "auth.py"}},'
            '{"description": "Discuss", "tool": null, "args": {}}'
            "]}"
        )

    planner = Planner(tool_registry=ToolRegistry(), dispatch_fn=dispatch)
    conv = Conversation.new(model="m")
    plan = await planner.plan("refactor auth", conv)

    assert plan.goal == "refactor auth"
    assert len(plan.steps) == 2
    assert plan.steps[0].index == 1
    assert plan.steps[0].tool_name == "file_read"
    assert plan.steps[0].tool_args == {"path": "auth.py"}
    assert plan.steps[1].tool_name is None
    assert plan.steps[1].index == 2


@pytest.mark.asyncio
async def test_planner_parses_fenced_json():
    async def dispatch(req):
        return _resp(
            "Sure, here's the plan:\n\n"
            "```json\n"
            '{"steps": [{"description": "Step A", "tool": null, "args": {}}]}\n'
            "```\n"
            "Let me know if anything looks off."
        )

    planner = Planner(tool_registry=ToolRegistry(), dispatch_fn=dispatch)
    conv = Conversation.new(model="m")
    plan = await planner.plan("g", conv)
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "Step A"


@pytest.mark.asyncio
async def test_planner_falls_back_to_numbered_list():
    async def dispatch(req):
        return _resp(
            "I think the plan is:\n"
            "1. Read auth.py\n"
            "2. Patch credentials\n"
            "3. Write tests\n"
        )

    planner = Planner(tool_registry=ToolRegistry(), dispatch_fn=dispatch)
    conv = Conversation.new(model="m")
    plan = await planner.plan("g", conv)
    assert len(plan.steps) == 3
    assert plan.steps[0].description == "Read auth.py"
    assert plan.steps[2].description == "Write tests"
    # Numbered-list fallback yields thinking steps (no tool).
    assert all(s.tool_name is None for s in plan.steps)


@pytest.mark.asyncio
async def test_planner_malformed_raises_plan_parse_error():
    async def dispatch(req):
        return _resp("not a plan, just prose without numbers or json")

    planner = Planner(tool_registry=ToolRegistry(), dispatch_fn=dispatch)
    conv = Conversation.new(model="m")
    with pytest.raises(PlanParseError) as exc_info:
        await planner.plan("g", conv)
    # Error message includes an excerpt for diagnosis.
    assert "could not parse plan" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_planner_empty_steps_returns_empty_plan():
    async def dispatch(req):
        return _resp('{"steps": []}')

    planner = Planner(tool_registry=ToolRegistry(), dispatch_fn=dispatch)
    conv = Conversation.new(model="m")
    plan = await planner.plan("g", conv)
    assert plan.steps == ()
    assert plan.is_complete() is True


@pytest.mark.asyncio
async def test_planner_veto_raises():
    async def dispatch(req):
        return VetoResult(
            phase="pre",
            plugin="trust-gate",
            reason="dangerous",
        )

    planner = Planner(tool_registry=ToolRegistry(), dispatch_fn=dispatch)
    conv = Conversation.new(model="m")
    with pytest.raises(PlanParseError) as exc_info:
        await planner.plan("g", conv)
    assert "trust-gate" in str(exc_info.value)


@pytest.mark.asyncio
async def test_planner_does_not_send_tools_to_llm():
    """In planning mode the LLM must never see tool definitions — it would
    just produce tool_use blocks instead of a JSON plan."""
    seen = {"tools": None}

    async def dispatch(req):
        seen["tools"] = req.tools
        return _resp('{"steps": []}')

    reg = ToolRegistry()
    # Even with tools registered, the request to the LLM has tools=().
    from enchanter.agent.tools import EchoTool

    reg.register(EchoTool())
    planner = Planner(tool_registry=reg, dispatch_fn=dispatch)
    conv = Conversation.new(model="m")
    await planner.plan("g", conv)
    assert seen["tools"] == ()


def test_parse_plan_text_empty_returns_empty():
    """An empty LLM response is a valid empty plan, not an error."""
    assert _parse_plan_text("") == []


def test_parse_plan_text_rejects_non_list_steps():
    with pytest.raises(PlanParseError):
        _parse_plan_text('{"steps": "not a list"}')


def test_parse_plan_text_rejects_step_without_description():
    with pytest.raises(PlanParseError):
        _parse_plan_text('{"steps": [{"tool": null, "args": {}}]}')


def test_plan_prompt_template_forbids_tool_calls():
    """The template is load-bearing — Wave 15.4 docs quote it verbatim, and
    the contract is that the LLM never emits tool_use in planning mode."""
    template = Planner.PLAN_PROMPT_TEMPLATE
    assert "PLANNING MODE" in template
    assert "Do NOT call any tools" in template
    # Must reference the JSON shape so the LLM has something concrete to
    # emit.
    assert '"steps"' in template
    assert '"description"' in template
