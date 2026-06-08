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
from .verdict import Verdict, render_veto_http
from .plugin import BudgetTierGate, PluginAdapter, PluginRegistry
from .bus import (
    Bus,
    DroppedEvent,
    HandlerFailure,
    InProcessBus,
    MAX_DERIVED_HOPS,
)
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
    "DroppedEvent",
    "EnchantedEvent",
    "EventHandler",
    "HandlerFailure",
    "InProcessBus",
    "LIFECYCLE_PHASES",
    "LifecyclePhase",
    "MAX_DERIVED_HOPS",
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
    "Verdict",
    "create_request_context",
    "render_veto_http",
]
