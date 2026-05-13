"""Tests for enchanter.agent.subagents.dispatch — SubagentTool behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enchanter.agent.loop import (
    AssistantTextDelta,
    AssistantThinking,
    ToolCallExecuted,
    TurnComplete,
)
from enchanter.agent.subagents.dispatch import (
    MAX_SUBAGENT_DEPTH,
    SubagentTool,
)
from enchanter.agent.subagents.registry import SubagentRegistry, SubagentRole
from enchanter.agent.subagents.roles import default_roles
from enchanter.agent.tools import ToolRegistry
from enchanter.agent.tools._types import ToolContext
from enchanter.proxy.canonical import CanonicalUsage


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session")


class MockInnerLoop:
    """Stand-in for AgentLoop with a scripted event stream."""

    def __init__(self, events_factory):
        # events_factory: () -> list[AgentEvent]  (rebuilt per run_turn)
        self._events_factory = events_factory
        self.last_input: str | None = None
        self.run_count: int = 0

    async def run_turn(self, user_input: str):
        self.last_input = user_input
        self.run_count += 1
        for ev in self._events_factory():
            yield ev


def _factory_from_events(events):
    def factory(*, conversation, tool_registry, cwd, max_iterations, **kw):
        return MockInnerLoop(lambda: list(events))

    return factory


def _populated_registry() -> SubagentRegistry:
    reg = SubagentRegistry()
    for role in default_roles():
        reg.register(role)
    return reg


def _parent_tool_registry() -> ToolRegistry:
    """A parent registry holding stub tools matching role allowed_tools."""

    class _Stub:
        def __init__(self, name):
            self.name = name
            self.description = f"stub {name}"
            self.input_schema = {"type": "object", "properties": {}}
            self.requires_approval = False

        async def execute(self, args, ctx):
            from enchanter.agent.tools import ToolResult

            return ToolResult(content=f"{self.name} ran", is_error=False)

    reg = ToolRegistry()
    for name in ("file_read", "file_write", "glob", "grep", "bash", "web_fetch"):
        reg.register(_Stub(name))
    return reg


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_tool_name_and_schema_shape():
    reg = _populated_registry()
    tool = SubagentTool(reg, parent_loop_factory=_factory_from_events([]))
    assert tool.name == "subagent"
    assert tool.requires_approval is False
    schema = tool.input_schema
    assert schema["type"] == "object"
    assert "role" in schema["properties"]
    assert "task" in schema["properties"]
    assert "context_summary" in schema["properties"]
    assert schema["required"] == ["role", "task"]
    assert schema["additionalProperties"] is False
    # role enum lists known roles
    assert "deep-research" in schema["properties"]["role"]["enum"]


@pytest.mark.asyncio
async def test_unknown_role_returns_error(tmp_path):
    reg = _populated_registry()
    tool = SubagentTool(reg, parent_loop_factory=_factory_from_events([]))
    res = await tool.execute(
        {"role": "no-such-role", "task": "do something"}, _ctx(tmp_path)
    )
    assert res.is_error is True
    assert "unknown role" in res.content
    assert "deep-research" in res.content  # lists available


@pytest.mark.asyncio
async def test_missing_task_returns_error(tmp_path):
    reg = _populated_registry()
    tool = SubagentTool(reg, parent_loop_factory=_factory_from_events([]))
    res = await tool.execute({"role": "find-references"}, _ctx(tmp_path))
    assert res.is_error is True
    assert "task" in res.content


@pytest.mark.asyncio
async def test_missing_role_returns_error(tmp_path):
    reg = _populated_registry()
    tool = SubagentTool(reg, parent_loop_factory=_factory_from_events([]))
    res = await tool.execute({"task": "x"}, _ctx(tmp_path))
    assert res.is_error is True
    assert "role" in res.content


@pytest.mark.asyncio
async def test_unstructured_role_returns_final_text(tmp_path):
    # A role without summary_schema → raw text passthrough.
    reg = SubagentRegistry()
    reg.register(
        SubagentRole(
            name="plain",
            description="plain role for testing" * 5,
            system_prompt="you are a plain subagent for testing" * 5,
            allowed_tools=("file_read",),
            max_turns=3,
            summary_schema=None,
        )
    )

    events = [
        AssistantThinking(iteration=1),
        AssistantTextDelta(text="Hello from "),
        AssistantTextDelta(text="the subagent."),
        TurnComplete(
            usage=CanonicalUsage(input_tokens=10, output_tokens=5),
            iterations=1,
            stop_reason="end_turn",
        ),
    ]
    tool = SubagentTool(
        reg,
        parent_loop_factory=_factory_from_events(events),
        parent_tool_registry=_parent_tool_registry(),
    )
    res = await tool.execute(
        {"role": "plain", "task": "say hello"}, _ctx(tmp_path)
    )
    assert res.is_error is False
    assert res.content == "Hello from the subagent."
    # side_effects describes turns + tools
    assert any("role=plain" in s for s in res.side_effects)
    assert any("1 turn" in s for s in res.side_effects)


@pytest.mark.asyncio
async def test_structured_role_parses_json(tmp_path):
    reg = SubagentRegistry()
    reg.register(
        SubagentRole(
            name="structured",
            description="structured role for testing" * 3,
            system_prompt="output json only and nothing else, exact schema" * 3,
            allowed_tools=("file_read",),
            max_turns=3,
            summary_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        )
    )

    json_text = '{"answer": "42"}'
    events = [
        AssistantThinking(iteration=1),
        AssistantTextDelta(text=json_text),
        TurnComplete(
            usage=CanonicalUsage(input_tokens=5, output_tokens=5),
            iterations=1,
            stop_reason="end_turn",
        ),
    ]
    tool = SubagentTool(
        reg,
        parent_loop_factory=_factory_from_events(events),
        parent_tool_registry=_parent_tool_registry(),
    )
    res = await tool.execute(
        {"role": "structured", "task": "answer"}, _ctx(tmp_path)
    )
    assert res.is_error is False
    parsed = json.loads(res.content)
    assert parsed == {"answer": "42"}
    # No JSON-warning side effect on the happy path.
    assert not any("non-JSON" in s for s in res.side_effects)


@pytest.mark.asyncio
async def test_structured_role_falls_back_on_malformed_json(tmp_path):
    reg = SubagentRegistry()
    reg.register(
        SubagentRole(
            name="structured",
            description="structured role for testing" * 3,
            system_prompt="output json only" * 30,
            allowed_tools=None,
            max_turns=3,
            summary_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        )
    )

    events = [
        AssistantThinking(iteration=1),
        AssistantTextDelta(text="this is not JSON, it's just prose"),
        TurnComplete(
            usage=CanonicalUsage(input_tokens=5, output_tokens=5),
            iterations=1,
            stop_reason="end_turn",
        ),
    ]
    tool = SubagentTool(
        reg, parent_loop_factory=_factory_from_events(events)
    )
    res = await tool.execute(
        {"role": "structured", "task": "answer"}, _ctx(tmp_path)
    )
    assert res.is_error is False
    # Raw text returned.
    assert "not JSON" in res.content
    # Warning side-effect emitted.
    assert any("non-JSON" in s for s in res.side_effects)


@pytest.mark.asyncio
async def test_structured_role_strips_code_fences(tmp_path):
    """Common LLM behavior: wrap JSON in ```json ... ``` — we should recover."""
    reg = SubagentRegistry()
    reg.register(
        SubagentRole(
            name="structured",
            description="structured role" * 3,
            system_prompt="output json" * 30,
            allowed_tools=None,
            summary_schema={"type": "object"},
        )
    )

    fenced = '```json\n{"a": 1, "b": [1, 2]}\n```'
    events = [
        AssistantThinking(iteration=1),
        AssistantTextDelta(text=fenced),
        TurnComplete(
            usage=CanonicalUsage(input_tokens=1, output_tokens=1),
            iterations=1,
            stop_reason="end_turn",
        ),
    ]
    tool = SubagentTool(
        reg, parent_loop_factory=_factory_from_events(events)
    )
    res = await tool.execute(
        {"role": "structured", "task": "x"}, _ctx(tmp_path)
    )
    assert res.is_error is False
    parsed = json.loads(res.content)
    assert parsed == {"a": 1, "b": [1, 2]}


@pytest.mark.asyncio
async def test_max_turns_truncation_emits_partial_warning(tmp_path):
    """Inner loop that never completes hits max_turns; we surface a warning."""
    reg = SubagentRegistry()
    reg.register(
        SubagentRole(
            name="capped",
            description="capped role" * 3,
            system_prompt="you will hit the cap" * 20,
            allowed_tools=None,
            max_turns=3,
            summary_schema=None,
        )
    )

    # Simulate a loop that ran the cap, ended with tool_use stop_reason
    # (no clean end_turn).
    events = [
        AssistantThinking(iteration=1),
        AssistantTextDelta(text="step 1"),
        ToolCallExecuted(
            tool_name="grep",
            result="some output",
            is_error=False,
            side_effects=(),
            tool_use_id="t1",
        ),
        AssistantThinking(iteration=2),
        AssistantTextDelta(text="step 2"),
        ToolCallExecuted(
            tool_name="grep",
            result="more",
            is_error=False,
            side_effects=(),
            tool_use_id="t2",
        ),
        AssistantThinking(iteration=3),
        AssistantTextDelta(text="step 3 (cut off)"),
        TurnComplete(
            usage=CanonicalUsage(input_tokens=10, output_tokens=10),
            iterations=3,
            stop_reason="tool_use",  # not end_turn → truncated
        ),
    ]
    tool = SubagentTool(
        reg, parent_loop_factory=_factory_from_events(events)
    )
    res = await tool.execute(
        {"role": "capped", "task": "go"}, _ctx(tmp_path)
    )
    assert res.is_error is False
    assert "step 3" in res.content
    assert any("max_turns=3" in s for s in res.side_effects)
    assert any("partial" in s for s in res.side_effects)
    # Tool count is 2.
    assert any("2 tool(s)" in s for s in res.side_effects)


@pytest.mark.asyncio
async def test_side_effects_describe_role_turns_and_tools(tmp_path):
    reg = _populated_registry()
    events = [
        AssistantThinking(iteration=1),
        AssistantTextDelta(text='{"symbol":"x","references":[]}'),
        TurnComplete(
            usage=CanonicalUsage(input_tokens=1, output_tokens=1),
            iterations=1,
            stop_reason="end_turn",
        ),
    ]
    tool = SubagentTool(
        reg,
        parent_loop_factory=_factory_from_events(events),
        parent_tool_registry=_parent_tool_registry(),
    )
    res = await tool.execute(
        {"role": "find-references", "task": "find x"}, _ctx(tmp_path)
    )
    assert res.is_error is False
    side = " | ".join(res.side_effects)
    assert "role=find-references" in side
    assert "1 turn(s)" in side
    assert "0 tool(s)" in side


@pytest.mark.asyncio
async def test_recursion_depth_cap_refuses(tmp_path):
    """Set the depth high enough that the next call must refuse."""
    reg = _populated_registry()
    events = [
        AssistantThinking(iteration=1),
        AssistantTextDelta(text="hi"),
        TurnComplete(
            usage=CanonicalUsage(input_tokens=1, output_tokens=1),
            iterations=1,
            stop_reason="end_turn",
        ),
    ]
    tool = SubagentTool(
        reg,
        parent_loop_factory=_factory_from_events(events),
        parent_tool_registry=_parent_tool_registry(),
    )
    # Simulate being already at the depth cap.
    tool._depth = MAX_SUBAGENT_DEPTH
    res = await tool.execute(
        {"role": "find-references", "task": "x"}, _ctx(tmp_path)
    )
    assert res.is_error is True
    assert "recursion" in res.content.lower()
    assert any("recursion depth cap" in s for s in res.side_effects)


@pytest.mark.asyncio
async def test_context_summary_is_prepended_to_task(tmp_path):
    reg = _populated_registry()
    captured = {}

    class _Spy(MockInnerLoop):
        async def run_turn(self, user_input):
            captured["input"] = user_input
            yield AssistantThinking(iteration=1)
            yield AssistantTextDelta(text='{"symbol":"y","references":[]}')
            yield TurnComplete(
                usage=CanonicalUsage(input_tokens=1, output_tokens=1),
                iterations=1,
                stop_reason="end_turn",
            )

    def factory(*, conversation, tool_registry, cwd, max_iterations, **kw):
        return _Spy(lambda: [])

    tool = SubagentTool(
        reg,
        parent_loop_factory=factory,
        parent_tool_registry=_parent_tool_registry(),
    )
    res = await tool.execute(
        {
            "role": "find-references",
            "task": "find y",
            "context_summary": "we are refactoring module M",
        },
        _ctx(tmp_path),
    )
    assert res.is_error is False
    assert "refactoring module M" in captured["input"]
    assert "find y" in captured["input"]


@pytest.mark.asyncio
async def test_filtered_registry_excludes_disallowed_tools(tmp_path):
    """The inner loop receives only role.allowed_tools from the parent."""
    reg = _populated_registry()
    captured = {}

    def factory(*, conversation, tool_registry, cwd, max_iterations, **kw):
        captured["tools"] = tool_registry.names()
        captured["max_iter"] = max_iterations
        return MockInnerLoop(
            lambda: [
                AssistantThinking(iteration=1),
                AssistantTextDelta(text='{"symbol":"q","references":[]}'),
                TurnComplete(
                    usage=CanonicalUsage(input_tokens=1, output_tokens=1),
                    iterations=1,
                    stop_reason="end_turn",
                ),
            ]
        )

    tool = SubagentTool(
        reg,
        parent_loop_factory=factory,
        parent_tool_registry=_parent_tool_registry(),
    )
    await tool.execute(
        {"role": "find-references", "task": "q"}, _ctx(tmp_path)
    )
    # find-references allows only glob, grep, file_read.
    assert set(captured["tools"]) == {"glob", "grep", "file_read"}
    # And the max_iterations matches the role.
    assert captured["max_iter"] == 5
    # Subagent tool is never inherited.
    assert "subagent" not in captured["tools"]


@pytest.mark.asyncio
async def test_empty_final_text_returns_placeholder(tmp_path):
    """Subagent that produces no text gets a placeholder, not empty string."""
    reg = SubagentRegistry()
    reg.register(
        SubagentRole(
            name="silent",
            description="silent role" * 5,
            system_prompt="say nothing" * 30,
            allowed_tools=None,
            summary_schema=None,
        )
    )
    events = [
        AssistantThinking(iteration=1),
        TurnComplete(
            usage=CanonicalUsage(input_tokens=1, output_tokens=0),
            iterations=1,
            stop_reason="end_turn",
        ),
    ]
    tool = SubagentTool(
        reg, parent_loop_factory=_factory_from_events(events)
    )
    res = await tool.execute(
        {"role": "silent", "task": "x"}, _ctx(tmp_path)
    )
    assert res.is_error is False
    assert "no output" in res.content
