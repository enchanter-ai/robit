"""End-to-end tests for the destructive-op-gate engine.

Verifies that a real tool call event triggers veto on dangerous patterns
and passes through on benign ones, with the orchestrator surfacing the
veto as SecurityVetoError so dispatch never fires.
"""

from __future__ import annotations

import pytest

from robit.core import (
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    PluginAck,
    SecurityVetoError,
    create_request_context,
)
from robit.core.bus import build_event
from robit.core.context import RequestContext
from robit.engines.destructive_op_gate import adapter as gate


async def _publish_tool_call(bus: InProcessBus, ctx, tool: str, args: list[str]) -> None:
    """Publish an mcp.tool.call.requested event for the trust-gate scanner to see."""
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="trust-gate",
        topic="mcp.tool.call.requested",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={"tool": tool, "args": args},
    )
    await bus.publish(event.topic, event)


@pytest.mark.parametrize(
    "tool, args, expected_pattern",
    [
        ("git", ["push", "--force", "origin", "main"], "w5-force-push"),
        ("git", ["reset", "--hard", "HEAD~3"], "w5-reset-hard"),
        ("git", ["branch", "-D", "stale-feature"], "w5-branch-delete-force"),
        ("rm", ["-rf", "/tmp/build"], "w5-rm-rf"),
        ("git", ["push", "--force-with-lease", "origin", "main"], "w5-force-push-with-lease-protected"),
    ],
)
async def test_dangerous_tool_call_is_vetoed(tool, args, expected_pattern):
    bus = InProcessBus()
    registry = {gate.name: gate}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> str:
        raise AssertionError("dispatch must not run when trust-gate vetoes")

    ctx = create_request_context()
    await _publish_tool_call(bus, ctx, tool, args)

    with pytest.raises(SecurityVetoError) as excinfo:
        await orch.run(ctx, dispatch)

    assert excinfo.value.plugin == "destructive-op-gate"
    assert excinfo.value.phase == "trust-gate"
    assert expected_pattern in excinfo.value.reason


@pytest.mark.parametrize(
    "tool, args",
    [
        ("git", ["status"]),
        ("git", ["log", "--oneline", "-n", "10"]),
        ("ls", ["-la"]),
        ("cat", ["README.md"]),
        ("npm", ["test"]),
    ],
)
async def test_benign_tool_call_passes(tool, args):
    bus = InProcessBus()
    registry = {gate.name: gate}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    dispatched = False

    async def dispatch(ctx: RequestContext) -> str:
        nonlocal dispatched
        dispatched = True
        return f"ran-{tool}"

    ctx = create_request_context()
    await _publish_tool_call(bus, ctx, tool, args)

    result = await orch.run(ctx, dispatch)
    assert dispatched is True
    assert result == f"ran-{tool}"


async def test_plain_git_push_is_advisory_not_veto():
    """Plain `git push` (no --force) is W5 advisory: ack with degraded=True, no veto."""
    bus = InProcessBus()
    registry = {gate.name: gate}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> str:
        return "pushed"

    ctx = create_request_context()
    await _publish_tool_call(bus, ctx, "git", ["push", "origin", "main"])

    # No SecurityVetoError — plain push is advisory.
    result = await orch.run(ctx, dispatch)
    assert result == "pushed"
    # ... but degraded findings should NOT be recorded for required plugins
    # that returned ack (even with degraded=True). The degraded_findings list
    # is only for advisory plugins. Required plugins use the veto channel
    # for hard stops; degraded is informational. Verify no degraded entry
    # is added for this required plugin.
    assert all(f.plugin != gate.name for f in ctx.degraded_findings)


async def test_force_push_in_string_arg_is_vetoed():
    """Tool args sometimes arrive as a single string, not a list. Both must trigger the regex."""
    bus = InProcessBus()
    registry = {gate.name: gate}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> str:
        raise AssertionError("dispatch must not run")

    ctx = create_request_context()
    # args as a single string instead of a list
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="trust-gate",
        topic="mcp.tool.call.requested",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={"tool": "git", "args": "push --force origin main"},
    )
    await bus.publish(event.topic, event)

    with pytest.raises(SecurityVetoError) as excinfo:
        await orch.run(ctx, dispatch)
    assert "w5-force-push" in excinfo.value.reason


async def test_derived_veto_event_is_published():
    """When a veto fires, the derived destructive-op-gate.veto event must hit the bus."""
    bus = InProcessBus()
    registry = {gate.name: gate}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> str:
        raise AssertionError("dispatch must not run")

    ctx = create_request_context()
    await _publish_tool_call(bus, ctx, "rm", ["-rf", "/important/data"])

    with pytest.raises(SecurityVetoError):
        await orch.run(ctx, dispatch)

    # The derived event should have been published before the veto propagated up.
    veto_events = [e for e in bus.tap(ctx.correlation_id) if e.topic == "destructive-op-gate.veto"]
    assert len(veto_events) == 1
    assert veto_events[0].payload["pattern_id"] == "w5-rm-rf"
