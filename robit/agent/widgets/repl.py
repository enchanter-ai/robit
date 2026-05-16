"""robit.agent.widgets.repl — main REPL widget.

Composes:

* a top ``RichLog`` (``#output``) that auto-scrolls unless the user scrolled
  up — Textual's ``RichLog.auto_scroll`` already implements that contract
* a middle approval slot (``#approval-slot``) where the diff/approval modal
  mounts when a ``file_edit`` / ``file_write`` tool requests approval
* a bottom ``Input`` (``#prompt``) with ``>>> `` prefix + Up/Down history

The widget owns the event-rendering helpers ``append_*`` and the modal
``show_approval`` coroutine. It does NOT own the agent loop — the parent
``EnchanterApp`` drives ``loop.run_turn`` and forwards each event here.

Sibling integration points (frozen Wave 15.2 contracts):

* G's ``DiffView(diff_text)`` mounts under ``#approval-slot`` together with
  Accept/Reject buttons when a ``file_edit`` / ``file_write`` proposal lands.
  Fallback: a plain ``Static`` containing the raw arg dict.
* H's ``EnforcementChip(kind, label)`` mounts inline in the log on every
  ``VetoFired``. Fallback: red ``Static`` line.
* I's ``CostTicker`` lives in the footer, not here.

Imports of sibling widgets are guarded so the app boots even if the
sibling files haven't landed yet.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Button, Input, RichLog, Static

if TYPE_CHECKING:  # pragma: no cover
    from ..loop import (
        ApprovalRequested,
        ToolCallExecuted,
        ToolCallProposed,
        VetoFired,
    )


# Optional sibling imports — fall back to None if absent.
try:  # pragma: no cover - import branch
    from .diff import DiffView  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    DiffView = None  # type: ignore[assignment, misc]

try:  # pragma: no cover - import branch
    from .enforcement import EnforcementChip  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    EnforcementChip = None  # type: ignore[assignment, misc]


# Tools that benefit from G's diff renderer.
_DIFFABLE_TOOLS = frozenset({"file_edit", "file_write"})


class _HistoryInput(Input):
    """Input subclass with persistent up/down command history."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history: list[str] = []
        self._cursor: int | None = None  # None = bottom (new line)

    def push_history(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        # De-dupe consecutive identical entries.
        if self._history and self._history[-1] == line:
            self._cursor = None
            return
        self._history.append(line)
        self._cursor = None

    async def _on_key(self, event) -> None:  # type: ignore[override]
        # Textual fires "key" events; we intercept Up/Down only when the
        # cursor is at the corresponding edge so normal editing still works
        # for multi-line-ish entries. (Input is single-row but Up/Down would
        # otherwise be a no-op.)
        if event.key == "up":
            if not self._history:
                return
            if self._cursor is None:
                self._cursor = len(self._history) - 1
            elif self._cursor > 0:
                self._cursor -= 1
            self.value = self._history[self._cursor]
            event.stop()
        elif event.key == "down":
            if self._cursor is None:
                return
            if self._cursor < len(self._history) - 1:
                self._cursor += 1
                self.value = self._history[self._cursor]
            else:
                self._cursor = None
                self.value = ""
            event.stop()


class ReplWidget(Container):
    """Main REPL: log + approval slot + input.

    The parent app (``EnchanterApp``) handles submission and event
    rendering; this widget only owns the visual surface.
    """

    DEFAULT_CSS = """
    ReplWidget {
        layout: vertical;
        height: 1fr;
    }
    ReplWidget RichLog#output {
        height: 1fr;
        border: round $panel;
        scrollbar-gutter: stable;
    }
    ReplWidget Container#approval-slot {
        height: auto;
        max-height: 60%;
        background: $boost;
        display: none;
    }
    ReplWidget Container#approval-slot.visible {
        display: block;
    }
    ReplWidget Input#prompt {
        dock: bottom;
        height: 1;
        border: tall $accent;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="repl")

    def compose(self) -> ComposeResult:
        yield RichLog(id="output", wrap=True, highlight=True, markup=True)
        yield Container(id="approval-slot")
        yield _HistoryInput(placeholder=">>> Type a task or /help", id="prompt")

    # ----- public helpers used by the app -------------------------------

    def log(self) -> RichLog:
        return self.query_one("#output", RichLog)

    def input(self) -> _HistoryInput:
        return self.query_one("#prompt", _HistoryInput)

    def push_input_history(self, line: str) -> None:
        self.input().push_history(line)

    def clear_log(self) -> None:
        self.log().clear()

    # ----- event rendering ----------------------------------------------

    def append_user(self, text: str) -> None:
        self.log().write(f"[bold cyan]>>> {text}[/]")

    def append_thinking(self, iteration: int) -> None:
        self.log().write(f"[dim]thinking… (iter {iteration})[/]")

    def append_assistant(self, text: str) -> None:
        # Streamed delta: append without newline. RichLog renders each call
        # on its own line, so for Wave 15.0 (one-shot deltas) this is fine;
        # when true streaming lands we can switch to a buffered renderer.
        if text:
            self.log().write(text)

    def append_tool_call(self, event: "ToolCallProposed") -> None:
        approval = " [requires approval]" if event.requires_approval else ""
        self.log().write(
            f"[yellow]→ tool[/] {event.tool_name}({event.args}){approval}"
        )

    def append_tool_result(self, event: "ToolCallExecuted") -> None:
        kind = "[red]error[/]" if event.is_error else "[green]ok[/]"
        self.log().write(f"  {kind} {event.tool_name} → {event.result}")
        # If a diffable tool just ran and sibling G shipped, mount inline.
        if (
            not event.is_error
            and event.tool_name in _DIFFABLE_TOOLS
            and DiffView is not None
        ):
            try:
                self.log().write(DiffView(event.result))  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                # Diff renderer is best-effort; if it raises we just skip.
                pass
        for note in event.side_effects:
            self.log().write(f"    [dim]· {note}[/]")

    def append_enforcement(self, event: "VetoFired") -> None:
        label = f"{event.plugin}: {event.reason}"
        if EnforcementChip is not None:
            try:
                self.log().write(EnforcementChip(kind="veto", label=label))  # type: ignore[arg-type]
                return
            except Exception:  # noqa: BLE001
                pass
        self.log().write(
            f"[bold red]✖ veto[/] [{event.plugin}/{event.phase}] {event.reason}"
        )

    def append_turn_complete(self, iterations: int, usage) -> None:
        self.log().write(
            f"[dim]turn done — iters={iterations} "
            f"tokens={usage.input_tokens}+{usage.output_tokens}[/]"
        )

    def append_error(self, text: str) -> None:
        self.log().write(f"[bold red]{text}[/]")

    def append_info(self, text: str) -> None:
        self.log().write(text)

    # ----- approval modal -----------------------------------------------

    async def show_approval(
        self,
        event: "ApprovalRequested",
        on_accept: Callable[[str], Awaitable[None] | None],
        on_reject: Callable[[str], Awaitable[None] | None],
    ) -> None:
        """Mount an approval pane and wait for the user to decide.

        Parameters
        ----------
        event:
            The ``ApprovalRequested`` carrying tool_use_id + args.
        on_accept / on_reject:
            Callbacks the buttons fire; each receives ``event.tool_use_id``.
            They may be sync or async.
        """
        slot = self.query_one("#approval-slot", Container)
        # Wipe any previous approval pane.
        await slot.remove_children()
        slot.add_class("visible")

        # Pick the diff renderer if the tool is diffable and G shipped.
        diff_text = _summarise_args_for_approval(event.tool_name, event.args)
        if event.tool_name in _DIFFABLE_TOOLS and DiffView is not None:
            try:
                body = DiffView(diff_text)  # type: ignore[call-arg]
            except Exception:  # noqa: BLE001
                body = Static(diff_text)
        else:
            body = Static(diff_text)

        pane = Vertical(
            Static(
                f"[bold]Approval required[/] for [yellow]{event.tool_name}[/]",
                id="approval-title",
            ),
            body,
            Container(
                Button("Accept (a)", id="approve-btn", variant="success"),
                Button("Reject (r)", id="reject-btn", variant="error"),
                id="approval-buttons",
            ),
        )
        await slot.mount(pane)

        done: asyncio.Future[bool] = asyncio.get_event_loop().create_future()

        async def _handle(approved: bool) -> None:
            if not done.done():
                done.set_result(approved)

        # Attach button presses. We use Textual's message system at the
        # parent level normally, but for a self-contained modal it's
        # cleaner to wire a single watcher here.
        async def _on_button(message) -> None:  # pragma: no cover - UI glue
            btn = getattr(message, "button", None)
            if btn is None:
                return
            if btn.id == "approve-btn":
                await _handle(True)
            elif btn.id == "reject-btn":
                await _handle(False)

        # The parent forwards Button.Pressed to us via the watcher attribute.
        self._approval_handler = _on_button  # type: ignore[attr-defined]

        try:
            approved = await done
        finally:
            slot.remove_class("visible")
            await slot.remove_children()
            self._approval_handler = None  # type: ignore[attr-defined]

        if approved:
            result = on_accept(event.tool_use_id)
        else:
            result = on_reject(event.tool_use_id)
        if asyncio.iscoroutine(result):
            await result

    async def on_button_pressed(self, message) -> None:  # type: ignore[override]
        handler = getattr(self, "_approval_handler", None)
        if handler is not None:
            await handler(message)


def _summarise_args_for_approval(tool_name: str, args: dict) -> str:
    """Pretty arg summary shown in the approval pane.

    For diffable tools we surface the path + content/old/new triple as
    plain text so even without sibling G's ``DiffView`` the user can
    read what's about to change.
    """
    if tool_name in _DIFFABLE_TOOLS:
        path = args.get("path") or args.get("file_path") or "(unknown path)"
        if "old_string" in args and "new_string" in args:
            return (
                f"path: {path}\n\n"
                f"--- old ---\n{args.get('old_string', '')}\n"
                f"+++ new +++\n{args.get('new_string', '')}\n"
            )
        if "content" in args:
            return f"path: {path}\n\n--- content ---\n{args['content']}\n"
        return f"path: {path}\nargs: {args}"
    return f"args: {args}"


__all__ = ["ReplWidget"]
