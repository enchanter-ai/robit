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

import asyncio
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


def _is_concurrent_safe(plugin: PluginAdapter) -> bool:
    """Read the engine's concurrent-dispatch opt-in.

    Defaults to False (serial-only) when the attribute is absent — backwards
    compatible with engines and test fakes that pre-date Wave 13.3.
    """
    return bool(getattr(plugin, "concurrent_safe", False))


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
            # Publish to the bus so non-plugin observers (recorders, taps,
            # domain-topic subscribers) still see the phase event. Plugins
            # themselves are NOT subscribed to lifecycle.<phase> topics
            # (see _wire_plugin) — the orchestrator drives them directly
            # below via the two-bucket dispatch.
            await self._bus.publish(phase_event.topic, phase_event)

            subscribers = self._subscribers_for_phase(phase)
            await self._dispatch_phase(phase, phase_event, subscribers)

            required = tuple(p.name for p in subscribers if p.required)
            advisory = tuple(p.name for p in subscribers if not p.required)
            all_names = required + advisory

            if all_names:
                # acks have already been recorded synchronously by
                # _dispatch_phase; wait_for_acks short-circuits on present
                # entries, so this is effectively a snapshot read.
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
                # Iteration order is the required tuple's order (subscribers
                # are listed in registry order); within the concurrent bucket
                # _dispatch_phase has already enforced lexicographic ordering
                # of result processing so attribution is reproducible.
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

    async def _dispatch_phase(
        self,
        phase: LifecyclePhase,
        event: EnchantedEvent,
        subscribers: list[PluginAdapter],
    ) -> None:
        """Two-bucket phase dispatch (Wave 13.3).

        Bucket A — ``concurrent_safe`` engines — runs in parallel via
        ``asyncio.gather``. ACK processing for the bucket is then performed in
        deterministic alphabetical order of plugin name so dedup / veto
        attribution does not depend on gather completion order.

        Bucket B — serial engines (default) — runs one-by-one, in registry
        order, AFTER the concurrent bucket completes. This ordering means
        serial engines may observe any derived events the concurrent bucket
        already published to the bus.

        On exception:
          - required=True → record a veto-status ACK with
            ``reason="plugin-exception"``; the orchestrator's veto-check at
            run() will raise SecurityVetoError.
          - required=False → record an error/degraded ACK; the orchestrator
            records a DegradedFinding (advisory fail-open).
        """
        if not subscribers:
            return

        concurrent = [p for p in subscribers if _is_concurrent_safe(p)]
        serial = [p for p in subscribers if not _is_concurrent_safe(p)]

        # Dedup BEFORE invocation: if a plugin's domain-topic handler (wired
        # by _wire_plugin) already fired for this (correlation_id, phase),
        # an ACK is already recorded — skip re-invocation to preserve the
        # historical "on_phase runs at most once per phase" contract.
        def _already_acked(p: PluginAdapter) -> bool:
            return self._bus.acks.has(event.correlation_id, event.phase, p.name)

        concurrent = [p for p in concurrent if not _already_acked(p)]
        # serial is filtered lazily inside the loop so a domain handler that
        # fires between concurrent and serial dispatch is still respected.

        # ── Bucket A: concurrent, parallel dispatch ───────────────────────
        if concurrent:
            results = await asyncio.gather(
                *(self._invoke_plugin(p, event) for p in concurrent),
                return_exceptions=True,
            )
            # Process in deterministic alphabetical order. _invoke_plugin
            # never raises (it converts exceptions into ACK objects), so the
            # return_exceptions=True is defence-in-depth; we still handle
            # bare exceptions here for safety.
            sorted_results = sorted(
                zip(concurrent, results),
                key=lambda t: t[0].name,
            )
            for plugin, result in sorted_results:
                ack: PluginAck
                if isinstance(result, BaseException):
                    ack = PluginAck(
                        status="veto" if plugin.required else "error",
                        reason="plugin-exception",
                        degraded=not plugin.required,
                    )
                else:
                    ack = result  # type: ignore[assignment]
                # Late-dedup: a derived event published mid-gather could
                # theoretically have re-entered this plugin via a domain
                # handler; in that case the bus handler recorded its own ACK
                # first. Skip overwrite.
                if self._bus.acks.has(event.correlation_id, event.phase, plugin.name):
                    continue
                self._bus.acks.ack(
                    event.correlation_id, event.phase, plugin.name, ack
                )
                for de in ack.derived_events:
                    await self._bus.publish(de.topic, de)

        # ── Bucket B: serial dispatch ─────────────────────────────────────
        for plugin in serial:
            if _already_acked(plugin):
                continue
            ack = await self._invoke_plugin(plugin, event)
            # Re-check after await in case a derived event from this plugin's
            # own emit re-entered via a domain handler (should not happen
            # because the handler runs synchronously inside the same publish,
            # but defence-in-depth).
            if self._bus.acks.has(event.correlation_id, event.phase, plugin.name):
                continue
            self._bus.acks.ack(
                event.correlation_id, event.phase, plugin.name, ack
            )
            for de in ack.derived_events:
                await self._bus.publish(de.topic, de)

    async def _invoke_plugin(
        self, plugin: PluginAdapter, event: EnchantedEvent
    ) -> PluginAck:
        """Invoke a single plugin's on_phase, coercing exceptions into ACKs.

        Mirrors the historical bus-handler error contract: required → veto
        on exception (so the orchestrator fails closed); advisory → error
        with degraded=True (fail open + degraded finding).
        """
        try:
            return await plugin.on_phase(
                event, self._context_from_event(event)
            )
        except Exception as e:
            return PluginAck(
                status="veto" if plugin.required else "error",
                reason="plugin-exception" if plugin.required else str(e),
                degraded=not plugin.required,
            )

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

        # Subscribe to plugin's declared DOMAIN topics only. As of Wave 13.3,
        # ``lifecycle.<phase>`` is driven directly by Orchestrator._dispatch_phase
        # (two-bucket: concurrent_safe in parallel, then serial) so we must NOT
        # also subscribe the plugin's handler to those topics on the bus — doing
        # so would let the bus race the orchestrator and break the deterministic
        # ordering / veto-attribution contract. The bus still publishes
        # ``lifecycle.<phase>`` events for non-plugin observers (recorders,
        # taps, domain-topic chains).
        for topic in plugin.topics.subscribes:
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
