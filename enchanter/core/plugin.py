"""Plugin contract — port of `src/plugins/plugin-contract.ts`.

Every plugin adapter implements PluginAdapter. ADR-001 fail-open
(advisory plugins) vs. fail-closed (required plugins) policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Literal, Mapping, Protocol, runtime_checkable

from .context import LifecyclePhase, RequestContext
from .events import EnchantedEvent, PluginAck


BudgetTierGate = Literal["always", "med-or-higher", "high-only"]


@dataclass(frozen=True)
class PluginTopics:
    subscribes: tuple[str, ...]
    emits: tuple[str, ...]


@runtime_checkable
class PluginAdapter(Protocol):
    """The single interface every plugin implements."""

    name: str
    phases: tuple[LifecyclePhase, ...]
    required: bool  # True → fail-closed on missing ACK; False → fail-open with degraded=True
    topics: PluginTopics
    budget_tier: BudgetTierGate
    # Wave 13.3 — optional opt-in for concurrent dispatch. Default False
    # (serial-only). Engines that mutate shared in-process state MUST leave
    # this False. The lifecycle dispatcher reads this via
    # ``getattr(plugin, "concurrent_safe", False)`` so adapters that don't
    # set the attribute remain serial — full backwards compatibility.
    concurrent_safe: bool

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        """Called by the orchestrator at each subscribed phase. Must return within phase timeout."""
        ...


PluginRegistry = Mapping[str, PluginAdapter]
