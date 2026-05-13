"""enchanter.agent.subagents — focused, context-isolated sub-conversations.

A *subagent* is a recursive :class:`~enchanter.agent.loop.AgentLoop` driven
by a constrained system prompt, a filtered tool subset, and a per-role
turn cap. The main agent dispatches one via the :class:`SubagentTool`;
the subagent runs its own conversation (own ``session_id``) and returns a
single string summary (optionally parsed as structured JSON) which the
main agent reads as a tool result.

The point is NOT parallelism — subagents run sequentially. The point is
CONTEXT ISOLATION: a bounded task (research, find-references, review)
gets a clean conversation that doesn't bloat the main agent's window.

Module layout
-------------

* :mod:`.registry` — :class:`SubagentRole` dataclass + :class:`SubagentRegistry`.
* :mod:`.roles` — three production-ready roles + :func:`default_roles`.
* :mod:`.dispatch` — :class:`SubagentTool` (the Tool the main agent calls).

Recursion guard
---------------

A subagent's tool set never includes ``subagent`` itself unless the role
explicitly lists it in ``allowed_tools``. The :class:`SubagentTool` also
counts nesting depth via a ToolContext-side flag check and refuses to
recurse beyond :data:`dispatch.MAX_SUBAGENT_DEPTH`.
"""

from __future__ import annotations

from .dispatch import MAX_SUBAGENT_DEPTH, SubagentTool
from .registry import SubagentRegistry, SubagentRole
from .roles import (
    DEEP_RESEARCH,
    FIND_REFERENCES,
    REVIEW_DIFF,
    default_roles,
)

__all__ = [
    "SubagentRole",
    "SubagentRegistry",
    "SubagentTool",
    "MAX_SUBAGENT_DEPTH",
    "DEEP_RESEARCH",
    "FIND_REFERENCES",
    "REVIEW_DIFF",
    "default_roles",
]
