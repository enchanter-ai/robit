"""enchanter.agent — coding-agent CLI built on the enchanter proxy.

Public surface (Wave 15.0):

* :class:`AgentLoop`        — drives one user turn → LLM → tools → done.
* :class:`Conversation`     — append-only canonical-message ledger.
* :class:`Tool`             — Protocol every tool implementation conforms to.
* :class:`ToolRegistry`     — name → Tool map handed to the loop.
* :class:`SlashCommand`     — Protocol for in-REPL slash commands.

Wave 15.1+ tool implementations register themselves against the same
contracts. Don't change the shapes here without bumping the wave plan —
twelve downstream agents read these verbatim.
"""

from __future__ import annotations

from .conversation import Conversation
from .loop import (
    AgentEvent,
    AgentLoop,
    ApprovalRequested,
    AssistantTextDelta,
    AssistantThinking,
    MAX_ITERATIONS,
    ToolCallExecuted,
    ToolCallProposed,
    TurnComplete,
    VetoFired,
)
from .slash import (
    SlashCommand,
    SlashContext,
    SlashExit,
    SlashRegistry,
    builtin_registry,
    dispatch_slash,
)
from .tools import (
    EchoTool,
    Tool,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
)


__all__ = [
    # Core
    "AgentLoop",
    "Conversation",
    "MAX_ITERATIONS",
    # Tools
    "Tool",
    "ToolRegistry",
    "ToolContext",
    "ToolResult",
    "ToolCall",
    "EchoTool",
    # Slash
    "SlashCommand",
    "SlashContext",
    "SlashRegistry",
    "SlashExit",
    "builtin_registry",
    "dispatch_slash",
    # Events
    "AgentEvent",
    "AssistantThinking",
    "AssistantTextDelta",
    "ToolCallProposed",
    "ApprovalRequested",
    "ToolCallExecuted",
    "VetoFired",
    "TurnComplete",
]
