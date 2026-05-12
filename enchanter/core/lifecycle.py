"""7-phase orchestrator lifecycle — port of `src/orchestration/lifecycle.ts`.

ADR-001: hybrid coordination (orchestrator + bus) chose 25 in the decision
matrix, beating pure-orchestrator and pure-bus tied at 22. Per-tier subsystem
activity (phase_5): advisory plugins fail-open with degraded=True; required
plugins fail-closed on missing ACK.

Phase 0 port omits the trust-gate hook and control-channel approval (Tier 2;
also the control channel is the inspector-facing surface we explicitly dropped).
Both can be wired in Phase 2 as dedicated middleware plugins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .bus import Bus, build_event
from .context import (
    DEFAULT_PHASE_TIMEOUTS_MS,
    LIFECYCLE_PHASES,
    DegradedFinding,
    LifecyclePhase,
    RequestContext,
)
from .events import EnchantedEvent, PluginAck
from .plugin import PluginAdapter, PluginRegistry


class SecurityVetoError(Exception):
    def __init__(self, plugin: str, phase: LifecyclePhase, reason: str) -> None:
        super().__init__(f"security veto from plugin {plugin} at phase {phase}: {reason}")
        self.plugin = plugin
        self.phase = phase
        self.reason = reason


class PhaseTimeoutError(Exception):
    def __init__(self, phase: LifecyclePhase, missing: tuple[str, ...]) -> None:
        super().__init__(
            f"phase {phase} timed out without ACK from required plugins: {', '.join(missing)}"
        )
        self.phase = phase
        self.missing = missing


# Dispatch handler — the only callback permitted to touch external transport.
DispatchHandler = Callable[[RequestContext], Awaitable[object]]


@dataclass
class OrchestratorConfig:
    registry: PluginRegistry
    bus: Bus
    timeouts: dict[LifecyclePhase, int] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_TIMEOUTS_MS)
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


class Orchestrator:
    """Runs the 7-phase lifecycle. The dispatch handler is the only callback
    permitted to talk to an external MCP server."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self._registry = config.registry
        self._bus = config.bus
        self._timeouts = config.timeouts
        self._wire_subscriptions()

    async def run(self, ctx: RequestContext, dispatch: DispatchHandler) -> object:
        dispatch_result: object = None

        for phase in LIFECYCLE_PHASES:
            ctx.phase = phase
            phase_event = self._build_phase_event(ctx, phase)
            await self._bus.publish(phase_event.topic, phase_event)

            subscribers = self._subscribers_for_phase(phase)
            required = tuple(p.name for p in subscribers if p.required)
            advisory = tuple(p.name for p in subscribers if not p.required)
            all_names = required + advisory

            if all_names:
                acks = await self._bus.acks.wait_for_acks(
                    ctx.correlation_id,
                    phase,
                    all_names,
                    self._timeouts[phase],
                )

                # Required plugins must ack; missing ack = phase timeout = fail closed.
                missing_required = tuple(p for p in required if p not in acks)
                if missing_required:
                    raise PhaseTimeoutError(phase, missing_required)

                # Veto check — any required plugin returning veto fails closed.
                for p in required:
                    a = acks.get(p)
                    if a is not None and a.status == "veto":
                        raise SecurityVetoError(p, phase, a.reason or "veto")

                # Advisory plugins fail open — record degraded findings on missing/error.
                for p in advisory:
                    a = acks.get(p)
                    if a is None:
                        ctx.degraded_findings = [
                            *ctx.degraded_findings,
                            DegradedFinding(plugin=p, reason="no-ack-within-timeout"),
                        ]
                    elif a.status == "error" or a.degraded:
                        ctx.degraded_findings = [
                            *ctx.degraded_findings,
                            DegradedFinding(plugin=p, reason=a.reason or "degraded"),
                        ]

            # Dispatch is the only phase that calls external transport.
            if phase == "dispatch":
                dispatch_result = await dispatch(ctx)

        return dispatch_result

    def _wire_subscriptions(self) -> None:
        for plugin in self._registry.values():
            self._wire_plugin(plugin)

    def _wire_plugin(self, plugin: PluginAdapter) -> None:
        async def handler(event: EnchantedEvent) -> None:
            if event.phase not in plugin.phases:
                return
            # Dedup: if this plugin already acked for (correlation_id, phase),
            # skip — multiple subscribed topics in the same phase would otherwise
            # fire the handler more than once.
            if self._bus.acks.has(event.correlation_id, event.phase, plugin.name):
                return
            try:
                ack = await plugin.on_phase(event, self._context_from_event(event))
                self._bus.acks.ack(event.correlation_id, event.phase, plugin.name, ack)
                # Publish each derived event the plugin returned.
                for de in ack.derived_events:
                    await self._bus.publish(de.topic, de)
            except Exception as e:
                err_ack = PluginAck(
                    status="error",
                    reason=str(e),
                    degraded=not plugin.required,
                )
                self._bus.acks.ack(event.correlation_id, event.phase, plugin.name, err_ack)

        # Subscribe to plugin's declared domain topics PLUS the lifecycle.<phase>
        # event for every phase the plugin participates in. Dedup via set so a
        # plugin that declares both a domain topic and the lifecycle topic does
        # not get double-wired.
        topics: set[str] = set(plugin.topics.subscribes) | {
            f"lifecycle.{p}" for p in plugin.phases
        }
        for topic in topics:
            self._bus.subscribe(topic, handler)

    def _subscribers_for_phase(self, phase: LifecyclePhase) -> list[PluginAdapter]:
        return [p for p in self._registry.values() if phase in p.phases]

    def _build_phase_event(self, ctx: RequestContext, phase: LifecyclePhase) -> EnchantedEvent:
        return build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase=phase,
            topic=f"lifecycle.{phase}",
            source="orchestrator",
            budget_tier=ctx.budget_tier,
            payload={
                "sampling_depth": ctx.sampling_depth,
                "deadline_ms": ctx.deadline_ms,
                "elapsed_ms": _now_ms() - ctx.started_ms,
            },
        )

    def _context_from_event(self, event: EnchantedEvent) -> RequestContext:
        # Plugins receive a read-only view; the orchestrator is the single mutator.
        # v0.1 reconstructs minimally; full context propagation is a v0.2 follow-up
        # via a shared per-correlation-id context store.
        return RequestContext(
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            phase=event.phase,
            budget_tier=event.budget_tier,
            sampling_depth=0,
            deadline_ms=30_000,
            started_ms=event.ts,
            degraded_findings=[],
        )
