"""robit.core — lifecycle, bus, plugin protocol, request context."""

from .context import (
    BudgetTier,
    DEFAULT_PHASE_TIMEOUTS_MS,
    DegradedFinding,
    EmitterScratch,
    LIFECYCLE_PHASES,
    LifecyclePhase,
    PhaseTimeoutMap,
    RequestContext,
    RequestScratchpad,
    ScratchCompatMapping,
    create_request_context,
)
from .events import EnchantedEvent, EventHandler, PluginAck, PluginAckStatus
from .verdict import Verdict, render_veto_http
from .topics import (
    TOPIC_REGISTRY,
    TopicKind,
    TopicOwner,
    TopicSpec,
    all_emitted_topics,
    get_topic,
    is_known_topic,
    is_lifecycle_subscription,
    is_wildcard,
)
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
    "DegradedFinding",
    "DroppedEvent",
    "EmitterScratch",
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
    "RequestScratchpad",
    "ScratchCompatMapping",
    "SecurityVetoError",
    "TOPIC_REGISTRY",
    "TopicKind",
    "TopicOwner",
    "TopicSpec",
    "Verdict",
    "all_emitted_topics",
    "create_request_context",
    "get_topic",
    "is_known_topic",
    "is_lifecycle_subscription",
    "is_wildcard",
    "render_veto_http",
]
