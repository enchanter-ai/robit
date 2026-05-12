"""Bus event types — port of `src/bus/event-types.ts`.

EnchantedEvent is the cross-plugin lingua franca for the in-process bus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Mapping

from .context import BudgetTier, LifecyclePhase


PluginAckStatus = Literal["ack", "veto", "error"]


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


@dataclass
class PluginAck:
    status: PluginAckStatus
    reason: str | None = None
    derived_events: list[EnchantedEvent] = field(default_factory=list)
    degraded: bool = False


# A subscriber to one or more topics. May return derived events to be
# re-published, or None.
EventHandler = Callable[[EnchantedEvent], Awaitable[list[EnchantedEvent] | None]]
