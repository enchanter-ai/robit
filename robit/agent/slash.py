"""robit.agent.slash — slash command Protocol + registry + built-ins.

Slash commands run synchronously inside the REPL — they never touch the LLM.
They mutate session-level state (clear conversation, change model, request
exit) or return informational strings the UI prints inline.

The contract:

    class SlashCommand(Protocol):
        name: str               # "/help"
        description: str
        async def execute(self, args: str, ctx: SlashContext) -> str: ...

Built-ins shipped in Wave 15.0: ``/help``, ``/clear``, ``/exit``, ``/model``,
``/cost`` (placeholder — Wave 15.2I will populate from the cost-ledger).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .conversation import Conversation
from .tools import ToolRegistry


# ---------------------------------------------------------------------------
# Exit sentinel
# ---------------------------------------------------------------------------


class SlashExit(Exception):
    """Raised by ``/exit`` to signal the REPL should terminate cleanly.

    The Textual app catches this and shuts down without printing a traceback.
    The one-shot CLI ignores it (the loop has already produced output).
    """


# ---------------------------------------------------------------------------
# Protocol + context
# ---------------------------------------------------------------------------


@runtime_checkable
class SlashCommand(Protocol):
    name: str
    description: str

    async def execute(self, args: str, ctx: "SlashContext") -> str: ...


@dataclass(frozen=False)
class SlashContext:
    """Mutable handle a slash command may use to update session state.

    Why mutable: ``/clear`` and ``/model`` need to swap the conversation
    reference. The Conversation itself stays immutable; SlashContext is the
    container that owns the current reference.

    Attributes
    ----------
    conversation:
        Current conversation. Slash commands MAY replace this reference
        (``/clear``, ``/model``) but must NOT mutate the existing instance.
    tool_registry:
        Read-only reference to the active tool registry, exposed so
        ``/help`` can list available tools.
    audit_dir:
        Filesystem path for session JSONL logs — handed in for tests so
        they can redirect away from the user's real home directory.
    """

    conversation: Conversation
    tool_registry: ToolRegistry
    audit_dir: Path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SlashRegistry:
    """Name (with leading slash) → :class:`SlashCommand` map.

    ``parse(raw)`` splits the leading token from the rest of the line. The
    REPL feeds it ``"/model claude-opus-4-7"`` and gets back
    ``(command, "claude-opus-4-7")``.
    """

    def __init__(self) -> None:
        self._cmds: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        if not cmd.name.startswith("/"):
            raise ValueError(f"slash command name must start with '/': {cmd.name!r}")
        if cmd.name in self._cmds:
            raise ValueError(f"slash command already registered: {cmd.name!r}")
        self._cmds[cmd.name] = cmd

    def get(self, name: str) -> SlashCommand | None:
        return self._cmds.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._cmds

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._cmds))

    def all(self) -> tuple[SlashCommand, ...]:
        return tuple(self._cmds.values())

    def parse(self, raw: str) -> tuple[str, str]:
        """Split ``raw`` into (command, args). Leading whitespace tolerated."""
        stripped = raw.strip()
        if not stripped.startswith("/"):
            raise ValueError(f"not a slash command: {raw!r}")
        head, _, rest = stripped.partition(" ")
        return head, rest.strip()


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------


class HelpCommand:
    name = "/help"
    description = "Show available slash commands and registered tools."

    async def execute(self, args: str, ctx: SlashContext) -> str:
        # Note: ctx imports avoid a circular dep on the registry — we walk
        # the *current* slash registry through a module-level singleton
        # the app constructs and the REPL hands in via the dispatch.
        lines = ["Available slash commands:"]
        # The dispatch hands us the registry via ctx-bound closure in app.py;
        # for now we ship a static built-in roster + tool listing here.
        for name, desc in _BUILTIN_HELP_TABLE:
            lines.append(f"  {name:<10} {desc}")
        lines.append("")
        lines.append("Registered tools:")
        if not ctx.tool_registry.names():
            lines.append("  (none)")
        else:
            for tname in ctx.tool_registry.names():
                tool = ctx.tool_registry.get(tname)
                lines.append(f"  {tname:<10} {tool.description}")
        return "\n".join(lines)


class ClearCommand:
    name = "/clear"
    description = "Reset the conversation (keeps session_id and system prompt)."

    async def execute(self, args: str, ctx: SlashContext) -> str:
        ctx.conversation = ctx.conversation.cleared()
        return "Conversation cleared. Session id preserved."


class ExitCommand:
    name = "/exit"
    description = "Exit the REPL."

    async def execute(self, args: str, ctx: SlashContext) -> str:
        raise SlashExit("user requested exit")


class ModelCommand:
    name = "/model"
    description = "Switch the active model id for subsequent turns."

    async def execute(self, args: str, ctx: SlashContext) -> str:
        name = args.strip()
        if not name:
            return f"Current model: {ctx.conversation.model}"
        ctx.conversation = ctx.conversation.with_model(name)
        return f"Model switched to: {name}"


class CostCommand:
    name = "/cost"
    description = "Show session cost (placeholder — Wave 15.2I will populate)."

    async def execute(self, args: str, ctx: SlashContext) -> str:
        return (
            "Cost ticker not yet wired (Wave 15.2I).\n"
            f"Session id: {ctx.conversation.session_id}"
        )


# Help table kept in sync with built-ins by hand; tests assert this.
_BUILTIN_HELP_TABLE: tuple[tuple[str, str], ...] = (
    ("/help", "Show available slash commands and registered tools."),
    ("/clear", "Reset the conversation (keeps session_id and system prompt)."),
    ("/exit", "Exit the REPL."),
    ("/model", "Switch the active model id for subsequent turns."),
    ("/cost", "Show session cost (placeholder — Wave 15.2I will populate)."),
)


def builtin_registry() -> SlashRegistry:
    """Build a registry pre-loaded with the Wave 15.0 built-in commands."""
    reg = SlashRegistry()
    reg.register(HelpCommand())
    reg.register(ClearCommand())
    reg.register(ExitCommand())
    reg.register(ModelCommand())
    reg.register(CostCommand())
    return reg


async def dispatch_slash(
    raw: str, registry: SlashRegistry, ctx: SlashContext
) -> str:
    """Parse + dispatch a single slash command. Returns the command's output.

    Unknown commands return a clear "not found" string (rc-wise this is not
    an error — the REPL just prints the message). ``SlashExit`` from
    ``/exit`` propagates.
    """
    name, args = registry.parse(raw)
    cmd = registry.get(name)
    if cmd is None:
        return f"Unknown slash command: {name}. Type /help for the list."
    return await cmd.execute(args, ctx)


__all__ = [
    "SlashCommand",
    "SlashContext",
    "SlashRegistry",
    "SlashExit",
    "HelpCommand",
    "ClearCommand",
    "ExitCommand",
    "ModelCommand",
    "CostCommand",
    "builtin_registry",
    "dispatch_slash",
]
