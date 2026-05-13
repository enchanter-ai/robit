"""enchanter.agent.subagents.registry — role definitions + name → role map.

A :class:`SubagentRole` is a frozen description of a specialist: name,
description for tool-choice routing, system prompt, tool whitelist,
per-invocation turn cap, and optional JSON output schema. The
:class:`SubagentRegistry` is a thin name → role index with the same
"register-once, fail-loud-on-collision" semantics as the main
:class:`enchanter.agent.tools.ToolRegistry`.

Roles are *data*, not code. The runtime behavior lives entirely in
:class:`enchanter.agent.subagents.dispatch.SubagentTool` — the role
just configures the AgentLoop the dispatch tool spawns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SubagentRole:
    """Frozen description of a subagent specialist.

    Attributes
    ----------
    name:
        Short, hyphenated identifier. The main agent's tool args carry this
        string; the registry resolves it to the full role.
    description:
        One-paragraph natural-language description of what this subagent is
        for. Surfaces in :class:`SubagentTool.input_schema` so the LLM can
        pick the right role.
    system_prompt:
        System prompt the spawned :class:`AgentLoop` runs with. Must
        identify the role, state scope, name the allowed tools, and
        specify the output format if ``summary_schema`` is set.
    allowed_tools:
        Tuple of tool names the subagent may call. ``None`` means "all
        tools the main agent has" (rare — most roles want a filter).
    max_turns:
        Hard cap on internal LLM rounds. The dispatch tool truncates the
        loop at this many iterations and folds the final assistant text
        (or a "ran out of turns" notice) into the result.
    summary_schema:
        Optional JSON schema describing the structured output. When set,
        the dispatch tool will try to parse the subagent's final
        assistant-text as JSON matching this schema; on parse failure it
        falls back to the raw text with a warning side-effect.
    """

    name: str
    description: str
    system_prompt: str
    allowed_tools: Optional[tuple[str, ...]]
    max_turns: int = 10
    summary_schema: Optional[dict] = None


class SubagentRegistry:
    """Name → :class:`SubagentRole` map.

    Registration is one-shot per name; re-registering raises
    :class:`ValueError`. Lookup via :meth:`get` raises :class:`KeyError`
    on unknown names (the dispatch tool catches this and surfaces a
    structured error).
    """

    def __init__(self) -> None:
        self._roles: dict[str, SubagentRole] = {}

    def register(self, role: SubagentRole) -> None:
        if not isinstance(role, SubagentRole):
            raise TypeError(
                f"expected SubagentRole, got {type(role).__name__}"
            )
        if role.name in self._roles:
            raise ValueError(f"subagent role already registered: {role.name!r}")
        self._roles[role.name] = role

    def get(self, name: str) -> SubagentRole:
        try:
            return self._roles[name]
        except KeyError as exc:
            raise KeyError(f"no such subagent role: {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._roles))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._roles

    def __len__(self) -> int:
        return len(self._roles)


def default_registry() -> "SubagentRegistry":
    """Build a registry preloaded with the built-in roles."""
    from .roles import default_roles

    reg = SubagentRegistry()
    for role in default_roles():
        reg.register(role)
    return reg


__all__ = ["SubagentRole", "SubagentRegistry", "default_registry"]
