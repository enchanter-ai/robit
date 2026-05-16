"""Tests for robit.agent.loop — turn driver, events, veto, iteration cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from robit.agent.conversation import Conversation
from robit.agent.loop import (
    AgentLoop,
    AssistantTextDelta,
    AssistantThinking,
    MAX_ITERATIONS,
    ToolCallExecuted,
    ToolCallProposed,
    TurnComplete,
    VetoFired,
)
from robit.agent.tools import EchoTool, ToolRegistry
from robit.proxy.canonical import (
    CanonicalResponse,
    CanonicalUsage,
    TextPart,
    ToolUsePart,
)
from robit.proxy.pipeline import PipelineResult, VetoResult


def _make_loop(dispatch_fn, cwd=None) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(EchoTool())
    conv = Conversation.new(model="mock-model")
    loop = AgentLoop(
        conversation=conv,
        tool_registry=tools,
        cwd=cwd or Path("."),
    )
    loop.dispatch_fn = dispatch_fn  # type: ignore[assignment]
    return loop


def _resp(content, stop="end_turn"):
    return PipelineResult(
        response=CanonicalResponse(
            model="mock-model",
            content=content,
            stop_reason=stop,
            usage=CanonicalUsage(input_tokens=1, output_tokens=1),
        ),
        fired=(),
    )


@pytest.mark.asyncio
async def test_text_only_turn():
    async def dispatch(req):
        return _resp((TextPart(text="hello"),), stop="end_turn")

    loop = _make_loop(dispatch)
    events = [ev async for ev in loop.run_turn("hi")]
    types = [type(e).__name__ for e in events]
    assert "AssistantThinking" in types
    assert "AssistantTextDelta" in types
    assert "TurnComplete" in types
    deltas = [e for e in events if isinstance(e, AssistantTextDelta)]
    assert deltas[0].text == "hello"


@pytest.mark.asyncio
async def test_turn_with_tool_call():
    calls = {"n": 0}

    async def dispatch(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(
                (
                    TextPart(text="Sure."),
                    ToolUsePart(id="tu_1", name="echo", input={"text": "yo"}),
                ),
                stop="tool_use",
            )
        return _resp((TextPart(text="Done."),), stop="end_turn")

    loop = _make_loop(dispatch)
    events = [ev async for ev in loop.run_turn("echo yo")]
    proposed = [e for e in events if isinstance(e, ToolCallProposed)]
    executed = [e for e in events if isinstance(e, ToolCallExecuted)]
    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(proposed) == 1
    assert proposed[0].tool_name == "echo"
    assert len(executed) == 1
    assert executed[0].result == "yo"
    assert executed[0].is_error is False
    assert len(completes) == 1
    assert calls["n"] == 2  # one to propose tool, one to wrap up


@pytest.mark.asyncio
async def test_veto_emits_event_no_tool_exec():
    async def dispatch(req):
        return VetoResult(
            phase="trust-gate",
            plugin="destructive-op-gate",
            reason="destructive-op-gate:rm-rf",
            pattern_id="rm-rf",
        )

    loop = _make_loop(dispatch)
    events = [ev async for ev in loop.run_turn("rm -rf /")]
    vetoes = [e for e in events if isinstance(e, VetoFired)]
    executed = [e for e in events if isinstance(e, ToolCallExecuted)]
    assert len(vetoes) == 1
    assert vetoes[0].plugin == "destructive-op-gate"
    assert executed == []


@pytest.mark.asyncio
async def test_iteration_cap_prevents_infinite_loop():
    """If the LLM keeps emitting tool_use indefinitely, we stop at MAX_ITERATIONS."""

    async def dispatch(req):
        # Always emit a tool_use → tool result → next iteration; keep going.
        return _resp(
            (ToolUsePart(id="tu_infinite", name="echo", input={"text": "loop"}),),
            stop="tool_use",
        )

    loop = _make_loop(dispatch)
    events = [ev async for ev in loop.run_turn("loop forever")]
    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(completes) == 1
    assert completes[0].iterations == MAX_ITERATIONS


@pytest.mark.asyncio
async def test_unknown_tool_name_is_reported_as_error():
    calls = {"n": 0}

    async def dispatch(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(
                (ToolUsePart(id="tu_1", name="does_not_exist", input={}),),
                stop="tool_use",
            )
        return _resp((TextPart(text="done"),), stop="end_turn")

    loop = _make_loop(dispatch)
    events = [ev async for ev in loop.run_turn("call ghost tool")]
    executed = [e for e in events if isinstance(e, ToolCallExecuted)]
    assert len(executed) == 1
    assert executed[0].is_error is True
    assert "unknown tool" in executed[0].result


@pytest.mark.asyncio
async def test_tool_output_cap_truncates():
    """A tool returning > max_output_bytes should be truncated."""

    class BigTool:
        name = "big"
        description = "emits a huge string"
        input_schema: dict = {"type": "object", "properties": {}}
        requires_approval = False

        async def execute(self, args, ctx):
            from robit.agent.tools import ToolResult
            return ToolResult(content="A" * (ctx.max_output_bytes + 100))

    calls = {"n": 0}

    async def dispatch(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(
                (ToolUsePart(id="tu_big", name="big", input={}),),
                stop="tool_use",
            )
        return _resp((TextPart(text="done"),), stop="end_turn")

    loop = _make_loop(dispatch)
    loop.tool_registry.register(BigTool())
    events = [ev async for ev in loop.run_turn("big")]
    executed = [e for e in events if isinstance(e, ToolCallExecuted)]
    assert len(executed) == 1
    assert "[truncated]" in executed[0].result
