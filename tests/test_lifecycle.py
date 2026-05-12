"""Phase 0 validation — dispatch a fake event through all 7 phases with a
no-op plugin and assert correct ordering + ACKs.

Success criteria:
- All 7 phases are visited in order
- Plugin receives one event per phase (no dedup leak)
- Dispatch handler is called exactly once, at phase 'dispatch'
- All ACKs are collected with status='ack'
- No PhaseTimeoutError, no SecurityVetoError
- The bus tap shows 7 lifecycle.<phase> events for this correlation_id
"""

from __future__ import annotations

import pytest

from enchanter.core import (
    InProcessBus,
    LIFECYCLE_PHASES,
    Orchestrator,
    OrchestratorConfig,
    PluginAck,
    PluginAdapter,
    SecurityVetoError,
    create_request_context,
)
from enchanter.core.plugin import PluginTopics
from enchanter.core.events import EnchantedEvent
from enchanter.core.context import LifecyclePhase, RequestContext


class NoOpPlugin:
    """Participates in every phase, always ACKs. Records the events it sees."""

    name = "noop"
    phases = tuple(LIFECYCLE_PHASES)
    required = True
    topics = PluginTopics(subscribes=(), emits=())
    budget_tier = "always"

    def __init__(self) -> None:
        self.seen_phases: list[LifecyclePhase] = []

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        self.seen_phases.append(event.phase)
        return PluginAck(status="ack")


class VetoPlugin:
    """A required plugin that vetoes on the trust-gate phase."""

    name = "veto"
    phases = ("trust-gate",)
    required = True
    topics = PluginTopics(subscribes=(), emits=())
    budget_tier = "always"

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        return PluginAck(status="veto", reason="test veto")


class AdvisoryFailPlugin:
    """An advisory plugin that errors on post-response. Should not fail the run."""

    name = "advisory"
    phases = ("post-response",)
    required = False
    topics = PluginTopics(subscribes=(), emits=())
    budget_tier = "always"

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        raise RuntimeError("simulated advisory failure")


async def test_happy_path_all_phases_visited():
    plugin = NoOpPlugin()
    bus = InProcessBus()
    registry = {plugin.name: plugin}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    dispatched = False

    async def dispatch(ctx: RequestContext) -> str:
        nonlocal dispatched
        dispatched = True
        assert ctx.phase == "dispatch"
        return "dispatched-ok"

    ctx = create_request_context()
    result = await orch.run(ctx, dispatch)

    # Plugin saw every phase, in order, exactly once.
    assert plugin.seen_phases == list(LIFECYCLE_PHASES)

    # Dispatch fired and returned.
    assert dispatched is True
    assert result == "dispatched-ok"

    # Bus has 7 lifecycle.<phase> events for this correlation_id.
    tapped = bus.tap(ctx.correlation_id)
    phases_in_buffer = [e.phase for e in tapped if e.topic.startswith("lifecycle.")]
    assert phases_in_buffer == list(LIFECYCLE_PHASES)


async def test_required_plugin_veto_short_circuits():
    plugin = VetoPlugin()
    bus = InProcessBus()
    registry = {plugin.name: plugin}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> None:
        raise AssertionError("dispatch should not be called when trust-gate vetoes")

    ctx = create_request_context()
    with pytest.raises(SecurityVetoError) as excinfo:
        await orch.run(ctx, dispatch)
    assert excinfo.value.plugin == "veto"
    assert excinfo.value.phase == "trust-gate"


async def test_advisory_failure_does_not_block_run():
    advisory = AdvisoryFailPlugin()
    noop = NoOpPlugin()
    bus = InProcessBus()
    registry = {advisory.name: advisory, noop.name: noop}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    dispatched = False

    async def dispatch(ctx: RequestContext) -> str:
        nonlocal dispatched
        dispatched = True
        return "ok"

    ctx = create_request_context()
    result = await orch.run(ctx, dispatch)

    assert dispatched is True
    assert result == "ok"
    # Advisory failure should have been recorded as a degraded finding.
    advisory_findings = [f for f in ctx.degraded_findings if f.plugin == "advisory"]
    assert len(advisory_findings) == 1
    assert "simulated advisory failure" in advisory_findings[0].reason


async def test_no_plugins_runs_cleanly():
    bus = InProcessBus()
    registry: dict[str, PluginAdapter] = {}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> str:
        return "no-plugins-ok"

    ctx = create_request_context()
    result = await orch.run(ctx, dispatch)
    assert result == "no-plugins-ok"
