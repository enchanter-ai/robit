"""Tests for robit.agent.slash_commands.plan — /plan, /edit, /cancel, /execute."""

from __future__ import annotations

from pathlib import Path

import pytest

from robit.agent.conversation import Conversation
from robit.agent.slash import SlashContext
from robit.agent.slash_commands import plan as plan_slash
from robit.agent.slash_commands.plan import (
    CancelPlanCommand,
    EditStepCommand,
    ExecutePlanCommand,
    PlanCommand,
    _reset_scratch_for_tests,
)
from robit.agent.tools import EchoTool, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_scratch():
    _reset_scratch_for_tests()
    yield
    _reset_scratch_for_tests()


def _ctx(tmp_path: Path) -> SlashContext:
    tools = ToolRegistry()
    tools.register(EchoTool())
    return SlashContext(
        conversation=Conversation.new(model="m"),
        tool_registry=tools,
        audit_dir=tmp_path,
    )


def _install_fake_planner(steps_json: str):
    """Replace the planner factory with one that returns a fixed plan."""

    from robit.agent.plan import Planner
    from robit.proxy.canonical import (
        CanonicalResponse,
        CanonicalUsage,
        TextPart,
    )
    from robit.proxy.pipeline import PipelineResult

    async def dispatch(req):
        return PipelineResult(
            response=CanonicalResponse(
                model="m",
                content=(TextPart(text=steps_json),),
                stop_reason="end_turn",
                usage=CanonicalUsage(input_tokens=1, output_tokens=1),
            ),
            fired=(),
        )

    def factory(ctx):
        return Planner(tool_registry=ctx.tool_registry, dispatch_fn=dispatch)

    plan_slash._planner_factory = factory


# ---------------------------------------------------------------------------
# /plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_command_renders_checklist_and_stashes(tmp_path):
    _install_fake_planner(
        '{"steps": ['
        '{"description": "Read auth.py", "tool": "file_read", "args": {}},'
        '{"description": "Discuss findings", "tool": null, "args": {}}'
        "]}"
    )
    ctx = _ctx(tmp_path)
    cmd = PlanCommand()
    out = await cmd.execute("refactor the auth module", ctx)

    assert "Plan for: refactor the auth module" in out
    assert "1. Read auth.py" in out
    assert "2. Discuss findings" in out
    assert "[ ]" in out
    assert "/execute" in out
    # Plan is stashed under the session id.
    assert plan_slash._get_plan(ctx) is not None


@pytest.mark.asyncio
async def test_plan_command_without_args_prints_usage(tmp_path):
    ctx = _ctx(tmp_path)
    cmd = PlanCommand()
    out = await cmd.execute("", ctx)
    assert "Usage:" in out
    assert plan_slash._get_plan(ctx) is None


# ---------------------------------------------------------------------------
# /edit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_replaces_step_description(tmp_path):
    _install_fake_planner(
        '{"steps": ['
        '{"description": "old A", "tool": null, "args": {}},'
        '{"description": "old B", "tool": null, "args": {}}'
        "]}"
    )
    ctx = _ctx(tmp_path)
    await PlanCommand().execute("g", ctx)

    out = await EditStepCommand().execute("2 new description here", ctx)
    assert "new description here" in out
    assert "old B" not in out
    # Step 1 untouched.
    assert "old A" in out


@pytest.mark.asyncio
async def test_edit_out_of_range_returns_clear_error(tmp_path):
    _install_fake_planner(
        '{"steps": [{"description": "only one", "tool": null, "args": {}}]}'
    )
    ctx = _ctx(tmp_path)
    await PlanCommand().execute("g", ctx)

    out = await EditStepCommand().execute("99 foo", ctx)
    assert "No step 99" in out


@pytest.mark.asyncio
async def test_edit_without_plan_errors(tmp_path):
    ctx = _ctx(tmp_path)
    out = await EditStepCommand().execute("1 foo", ctx)
    assert "No active plan" in out


@pytest.mark.asyncio
async def test_edit_bad_index_returns_clear_error(tmp_path):
    _install_fake_planner(
        '{"steps": [{"description": "only one", "tool": null, "args": {}}]}'
    )
    ctx = _ctx(tmp_path)
    await PlanCommand().execute("g", ctx)
    out = await EditStepCommand().execute("abc foo", ctx)
    assert "integer" in out


@pytest.mark.asyncio
async def test_edit_missing_description_returns_usage(tmp_path):
    _install_fake_planner(
        '{"steps": [{"description": "x", "tool": null, "args": {}}]}'
    )
    ctx = _ctx(tmp_path)
    await PlanCommand().execute("g", ctx)
    out = await EditStepCommand().execute("1", ctx)
    assert "Usage:" in out


# ---------------------------------------------------------------------------
# /cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_clears_plan(tmp_path):
    _install_fake_planner(
        '{"steps": [{"description": "x", "tool": null, "args": {}}]}'
    )
    ctx = _ctx(tmp_path)
    await PlanCommand().execute("g", ctx)
    assert plan_slash._get_plan(ctx) is not None
    out = await CancelPlanCommand().execute("", ctx)
    assert "cancelled" in out.lower()
    assert plan_slash._get_plan(ctx) is None


@pytest.mark.asyncio
async def test_cancel_without_plan_is_safe(tmp_path):
    ctx = _ctx(tmp_path)
    out = await CancelPlanCommand().execute("", ctx)
    assert "No active plan" in out


# ---------------------------------------------------------------------------
# /execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_without_plan_errors(tmp_path):
    ctx = _ctx(tmp_path)
    out = await ExecutePlanCommand().execute("", ctx)
    assert "No active plan" in out


@pytest.mark.asyncio
async def test_execute_renders_prompt_for_llm(tmp_path):
    _install_fake_planner(
        '{"steps": ['
        '{"description": "Read auth.py", "tool": "file_read", "args": {}},'
        '{"description": "Patch creds", "tool": "file_edit", "args": {}}'
        "]}"
    )
    ctx = _ctx(tmp_path)
    await PlanCommand().execute("refactor auth", ctx)
    out = await ExecutePlanCommand().execute("", ctx)

    assert "Executing plan for: refactor auth" in out
    assert "PLAN PROMPT" in out
    assert "Step 1. Read auth.py" in out
    assert "Step 2. Patch creds" in out
    # Tool hints surface so the LLM knows what to use.
    assert "file_read" in out
    assert "file_edit" in out
    # Plan status updated to executing.
    plan = plan_slash._get_plan(ctx)
    assert plan is not None
    assert plan.status == "executing"


@pytest.mark.asyncio
async def test_execute_empty_plan_clears_state(tmp_path):
    _install_fake_planner('{"steps": []}')
    ctx = _ctx(tmp_path)
    await PlanCommand().execute("g", ctx)
    out = await ExecutePlanCommand().execute("", ctx)
    assert "no steps" in out.lower()
    assert plan_slash._get_plan(ctx) is None


# ---------------------------------------------------------------------------
# all_plan_commands.
# ---------------------------------------------------------------------------


def test_all_plan_commands_returns_four():
    from robit.agent.slash_commands import all_plan_commands

    cmds = all_plan_commands()
    names = {c.name for c in cmds}
    assert names == {"/plan", "/edit", "/cancel", "/execute"}
    for c in cmds:
        # SlashCommand Protocol attributes.
        assert isinstance(c.name, str) and c.name.startswith("/")
        assert isinstance(c.description, str) and c.description
