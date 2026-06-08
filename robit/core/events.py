"""Bus event types — port of `src/bus/event-types.ts`.

EnchantedEvent is the cross-plugin lingua franca for the in-process bus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Mapping

from .context import BudgetTier, LifecyclePhase
from .verdict import Verdict


PluginAckStatus = Literal["ack", "veto", "error"]


# G3 — current wire-contract version. Bump when the EnchantedEvent / PluginAck
# shape changes in a way decoders must distinguish. Decoders treat a missing
# schema_version as 1 (see ``decode_*`` helpers below).
CURRENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EnchantedEvent:
    id: str
    correlation_id: str
    session_id: str
    phase: LifecyclePhase
    topic: str
    source: str  # plugin name or 'orchestrator'
    budget_tier: BudgetTier
    ts: int  # ms since epoch
    payload: Mapping[str, object] = field(default_factory=dict)
    # G3 — contract version. Default-valued so every existing constructor
    # still works; decoders treat a missing field as 1.
    schema_version: int = CURRENT_SCHEMA_VERSION
    # G5 — cycle/depth guard. Number of re-publish hops this event has taken
    # as a derived event. Root events start at 0; InProcessBus.publish inherits
    # ``parent.hop_count + 1`` for each derived re-publish and drops events that
    # exceed MAX_DERIVED_HOPS.
    hop_count: int = 0


@dataclass
class PluginAck:
    status: PluginAckStatus
    reason: str | None = None
    derived_events: list[EnchantedEvent] = field(default_factory=list)
    degraded: bool = False
    # G1 — structured veto. When ``status == "veto"`` an engine may attach a
    # Verdict so downstream consumers read pattern_id / pattern_name directly
    # instead of string-slicing ``reason``. Optional + default None for
    # backwards-compat with every existing constructor.
    verdict: Verdict | None = None
    # G3 — contract version (see EnchantedEvent.schema_version).
    schema_version: int = CURRENT_SCHEMA_VERSION


# A subscriber to one or more topics. May return derived events to be
# re-published, or None.
EventHandler = Callable[[EnchantedEvent], Awaitable[list[EnchantedEvent] | None]]
