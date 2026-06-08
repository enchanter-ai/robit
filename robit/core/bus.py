"""In-process pub-sub bus — port of `src/bus/pubsub.ts`.

Ring-buffer event store + topic subscription + ACK tracker. ADR-001
chose in-process over external (NATS, Redis Streams) at v1 because we
have no scale evidence yet; revisit at 100k requests/day.

Phase 0 port: faithful asyncio translation. The TS uses Promise chains
with manual waiter lists; Python uses asyncio.Event per ack-key. Same
semantics, idiomatic Python.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable, Mapping, MutableMapping

from .context import LifecyclePhase
from .events import EnchantedEvent, EventHandler, PluginAck


RING_BUFFER_DEFAULT_SIZE = 10_000

# G5 — maximum number of derived re-publish hops. An event whose hop_count
# would exceed this is dropped (and the drop recorded) rather than re-published,
# bounding any runaway derived-event cycle to a finite depth.
MAX_DERIVED_HOPS = 8


@dataclass(frozen=True)
class DroppedEvent:
    """Record of an event the bus refused to re-publish.

    ``reason`` is a stable machine code: ``"hop-cap"`` for a depth-guard drop.
    """

    topic: str
    source: str
    hop_count: int
    reason: str


@dataclass(frozen=True)
class HandlerFailure:
    """Record of a subscriber that raised during dispatch (G6).

    The bus stays crash-isolated — the exception never propagates out of
    ``publish`` — but the failure is captured here so it is observable instead
    of silently swallowed.
    """

    topic: str
    source: str
    error: str


def _topic_matches(pattern: str, topic: str) -> bool:
    """Match a topic pattern.

    Supports four forms:
      - `*`          — matches any topic
      - `foo.*`      — prefix wildcard; matches any topic starting with "foo."
      - `*.foo`      — suffix wildcard; matches any topic ending with ".foo"
      - exact match  — pattern equals topic verbatim

    Without suffix-wildcard support, subscriptions like `*.veto` (collect all
    veto events regardless of which engine fired them) silently never match.
    """
    if pattern == "*":
        return True
    if pattern == topic:
        return True
    if pattern.endswith(".*"):
        prefix = pattern[:-1]  # keeps the dot
        return topic.startswith(prefix)
    if pattern.startswith("*."):
        suffix = pattern[1:]  # keeps the dot
        return topic.endswith(suffix)
    return False


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class AckTracker:
    """Tracks plugin ACKs keyed by (correlation_id, phase, plugin)."""

    _acks: MutableMapping[str, PluginAck]
    _events: MutableMapping[str, asyncio.Event]

    @classmethod
    def new(cls) -> "AckTracker":
        return cls(_acks={}, _events={})

    @staticmethod
    def _key(correlation_id: str, phase: LifecyclePhase, plugin: str) -> str:
        return f"{correlation_id}::{phase}::{plugin}"

    def ack(
        self,
        correlation_id: str,
        phase: LifecyclePhase,
        plugin: str,
        result: PluginAck,
    ) -> None:
        key = self._key(correlation_id, phase, plugin)
        self._acks[key] = result
        ev = self._events.get(key)
        if ev is not None:
            ev.set()

    def has(self, correlation_id: str, phase: LifecyclePhase, plugin: str) -> bool:
        return self._key(correlation_id, phase, plugin) in self._acks

    async def wait_for_acks(
        self,
        correlation_id: str,
        phase: LifecyclePhase,
        plugins: tuple[str, ...],
        timeout_ms: int,
    ) -> dict[str, PluginAck]:
        """Wait until all named plugins ack or timeout fires.

        Returns a map of plugin → ack. Plugins that didn't ack before the
        deadline are absent from the map; the caller distinguishes required
        vs. advisory and applies fail-closed vs. fail-open policy.
        """
        result: dict[str, PluginAck] = {}
        pending: list[str] = []
        for p in plugins:
            key = self._key(correlation_id, phase, p)
            existing = self._acks.get(key)
            if existing is not None:
                result[p] = existing
            else:
                pending.append(p)

        if not pending:
            return result

        # Create an asyncio.Event per pending plugin if not already present.
        events: list[asyncio.Event] = []
        for p in pending:
            key = self._key(correlation_id, phase, p)
            ev = self._events.get(key)
            if ev is None:
                ev = asyncio.Event()
                self._events[key] = ev
            events.append(ev)

        # Wait for all events with a single deadline.
        timeout_s = timeout_ms / 1000.0
        try:
            await asyncio.wait_for(
                asyncio.gather(*(ev.wait() for ev in events)),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            pass  # fall through; collect whatever did ack

        for p in pending:
            key = self._key(correlation_id, phase, p)
            a = self._acks.get(key)
            if a is not None:
                result[p] = a

        return result


class Bus:
    """Abstract bus interface. See InProcessBus for the v0 implementation."""

    acks: AckTracker

    async def publish(self, topic: str, event: EnchantedEvent) -> None:
        raise NotImplementedError

    def subscribe(self, topic: str, handler: EventHandler) -> "Subscription":
        raise NotImplementedError

    def tap(self, correlation_id: str | None = None) -> list[EnchantedEvent]:
        raise NotImplementedError


@dataclass
class Subscription:
    topic: str
    handler: EventHandler
    _bus: "InProcessBus"

    def unsubscribe(self) -> None:
        handlers = self._bus._subscriptions.get(self.topic)
        if handlers is not None:
            handlers.discard(self.handler)
            if not handlers:
                self._bus._subscriptions.pop(self.topic, None)


class InProcessBus(Bus):
    """In-process bus with bounded ring-buffer event store."""

    def __init__(
        self,
        buffer_max: int = RING_BUFFER_DEFAULT_SIZE,
        *,
        on_handler_error: Callable[[HandlerFailure], None] | None = None,
        on_event_dropped: Callable[[DroppedEvent], None] | None = None,
    ) -> None:
        self._subscriptions: dict[str, set[EventHandler]] = {}
        self._buffer: list[EnchantedEvent] = []
        self._buffer_max = buffer_max
        self.acks = AckTracker.new()
        # G6 — observability sinks. Handler crashes are still isolated (never
        # propagate out of publish) but are now recorded here and optionally
        # forwarded to a caller-supplied callback instead of silently dropped.
        self.handler_failures: list[HandlerFailure] = []
        self._on_handler_error = on_handler_error
        # G5 — record of events dropped by the hop-count guard.
        self.dropped_events: list[DroppedEvent] = []
        self._on_event_dropped = on_event_dropped

    async def publish(self, topic: str, event: EnchantedEvent) -> None:
        # Stamp id + ts + topic if the caller passed an incomplete event.
        # In Python we accept fully-formed events; the orchestrator builds them.
        # Ring buffer (drop oldest).
        self._buffer.append(event)
        if len(self._buffer) > self._buffer_max:
            self._buffer.pop(0)

        # Dispatch to matching subscriptions.
        matched: list[EventHandler] = []
        for pattern, handlers in self._subscriptions.items():
            if _topic_matches(pattern, topic):
                matched.extend(handlers)

        # Run handlers concurrently; collect derived events to re-publish.
        derived: list[EnchantedEvent] = []

        async def _run(h: EventHandler) -> None:
            try:
                out = await h(event)
                if out:
                    derived.extend(out)
            except Exception as exc:  # noqa: BLE001 — crash isolation is the contract
                # G6 — subscriber failures are isolated by design (the bus does
                # not crash and the exception does not propagate out of publish),
                # but they are no longer silently swallowed: record + notify.
                failure = HandlerFailure(
                    topic=topic, source=event.source, error=repr(exc)
                )
                self.handler_failures.append(failure)
                if self._on_handler_error is not None:
                    try:
                        self._on_handler_error(failure)
                    except Exception:
                        # The error sink itself must never break the bus.
                        pass

        if matched:
            await asyncio.gather(*(_run(h) for h in matched))

        for e in derived:
            # G5 — derived events inherit parent.hop_count + 1. Drop (and record)
            # any event that would exceed the depth guard, bounding runaway
            # derived-event cycles to a finite depth instead of recursing forever.
            next_hop = event.hop_count + 1
            if next_hop > MAX_DERIVED_HOPS:
                dropped = DroppedEvent(
                    topic=e.topic,
                    source=e.source,
                    hop_count=next_hop,
                    reason="hop-cap",
                )
                self.dropped_events.append(dropped)
                if self._on_event_dropped is not None:
                    try:
                        self._on_event_dropped(dropped)
                    except Exception:
                        pass
                continue
            await self.publish(e.topic, replace(e, hop_count=next_hop))

    def subscribe(self, topic: str, handler: EventHandler) -> Subscription:
        handlers = self._subscriptions.setdefault(topic, set())
        handlers.add(handler)
        return Subscription(topic=topic, handler=handler, _bus=self)

    def tap(self, correlation_id: str | None = None) -> list[EnchantedEvent]:
        if correlation_id is None:
            return list(self._buffer)
        return [e for e in self._buffer if e.correlation_id == correlation_id]


def new_event_id() -> str:
    return str(uuid.uuid4())


def build_event(
    *,
    correlation_id: str,
    session_id: str,
    phase: LifecyclePhase,
    topic: str,
    source: str,
    budget_tier: str,
    payload: Mapping[str, object] | None = None,
    hop_count: int = 0,
) -> EnchantedEvent:
    """Helper to construct a fully-formed EnchantedEvent with id + ts stamped.

    ``hop_count`` defaults to 0 (a root event). The bus inherits and increments
    it for derived re-publishes; callers rarely set it explicitly.
    """
    return EnchantedEvent(
        id=new_event_id(),
        correlation_id=correlation_id,
        session_id=session_id,
        phase=phase,
        topic=topic,
        source=source,
        budget_tier=budget_tier,  # type: ignore[arg-type]
        ts=_now_ms(),
        payload=payload or {},
        hop_count=hop_count,
    )
