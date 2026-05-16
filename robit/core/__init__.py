"""robit.core — lifecycle, bus, plugin protocol, request context."""

from .context import (
    BudgetTier,
    DEFAULT_PHASE_TIMEOUTS_MS,
    LIFECYCLE_PHASES,
    LifecyclePhase,
    PhaseTimeoutMap,
    RequestContext,
    create_request_context,
)
from .events import EnchantedEvent, EventHandler, PluginAck, PluginAckStatus
from .plugin import BudgetTierGate, PluginAdapter, PluginRegistry
from .bus import Bus, InProcessBus
from .lifecycle import (
    Orchestrator,
    OrchestratorConfig,
    PhaseTimeoutError,
    SecurityVetoError,
)

__all__ = [
    "BudgetTier",
    "BudgetTierGate",
    "Bus",
    "DEFAULT_PHASE_TIMEOUTS_MS",
    "EnchantedEvent",
    "EventHandler",
    "InProcessBus",
    "LIFECYCLE_PHASES",
    "LifecyclePhase",
    "Orchestrator",
    "OrchestratorConfig",
    "PhaseTimeoutError",
    "PhaseTimeoutMap",
    "PluginAck",
    "PluginAckStatus",
    "PluginAdapter",
    "PluginRegistry",
    "RequestContext",
    "SecurityVetoError",
    "create_request_context",
]
