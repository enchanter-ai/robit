"""robit.agent.app — Textual REPL shell.

Wave 15.2F polish: the bare-bones Wave 15.0 app is rebuilt around the new
widget package. The app owns the loop + slash registry; the
:class:`~robit.agent.widgets.repl.ReplWidget` owns the visual surface;
the :class:`~robit.agent.widgets.footer.FooterWidget` owns status +
cost ticker mount.

Imports are still deferred — Textual is heavy and we don't want
``import robit.agent`` to drag it in. Call :func:`launch` (or
``EnchanterApp(...).run()``) explicitly.

Event flow
----------

User submits text →
    starts with ``/`` → :func:`dispatch_slash` then log result
    otherwise        → spawn ``_run_turn`` task → forward each
                       :class:`AgentEvent` to ``self.repl`` /
                       ``self.footer``

Approval is handled inside :meth:`ReplWidget.show_approval` which mounts
sibling G's ``DiffView`` (when available) under ``#approval-slot`` and
resolves the loop's pending approval future.

Sibling widgets land via duck-typed imports — see
``enchanter/agent/widgets/__init__.py``. A missing sibling never crashes
the app; the widget falls back to plain-text rendering.

Bindings
--------

* ``ctrl+l``  — clear log
* ``escape``  — cancel in-flight turn (existing turn task gets cancelled)
* ``ctrl+c``  — quit
* ``up`` / ``down`` — input history (handled inside ``_HistoryInput``)
"""

from __future__ import annotations

import asyncio

from .loop import (
    AgentLoop,
    ApprovalRequested,
    AssistantTextDelta,
    AssistantThinking,
    ToolCallExecuted,
    ToolCallProposed,
    TurnComplete,
    VetoFired,
)
from .slash import (
    SlashContext,
    SlashExit,
    builtin_registry,
    dispatch_slash,
)


def _build_app_class():
    """Import Textual lazily and return the App subclass.

    Kept in a function so ``import robit.agent`` doesn't require Textual.
    The CLI entry point calls this only when the REPL is actually launched.
    """
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Input

    from .widgets.footer import FooterWidget
    from .widgets.repl import ReplWidget

    class EnchanterApp(App):  # type: ignore[misc]
        """The agent REPL.

        Construct with a pre-wired :class:`AgentLoop`; the app does NOT
        build its own — the CLI builds the loop so it can be reused in
        one-shot mode too.
        """

        BINDINGS = [
            ("ctrl+l", "clear", "Clear log"),
            ("escape", "cancel_turn", "Cancel turn"),
            ("ctrl+c", "quit", "Quit"),
        ]

        CSS = """
        Screen {
            background: $surface;
        }
        Header {
            background: $primary;
        }
        """

        def __init__(self, loop: AgentLoop, slash_ctx: SlashContext) -> None:
            super().__init__()
            self.loop = loop
            self.slash_ctx = slash_ctx
            self.slash_registry = builtin_registry()
            self._turn_task: asyncio.Task | None = None
            self.repl: ReplWidget | None = None
            self.footer: FooterWidget | None = None

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header(show_clock=False)
            self.repl = ReplWidget()
            yield self.repl
            self.footer = FooterWidget(
                model=self.loop.conversation.model,
                session_id=self.loop.conversation.session_id,
            )
            yield self.footer

        # ----- input submission ------------------------------------------

        async def on_input_submitted(self, event: Input.Submitted) -> None:  # type: ignore[override]
            text = event.value.strip()
            if not text or self.repl is None:
                return
            input_widget = self.repl.input()
            input_widget.value = ""
            self.repl.push_input_history(text)
            self.repl.append_user(text)

            if text.startswith("/"):
                await self._handle_slash(text)
                return

            self._turn_task = asyncio.create_task(self._run_turn(text))

        async def _handle_slash(self, text: str) -> None:
            assert self.repl is not None
            try:
                out = await dispatch_slash(text, self.slash_registry, self.slash_ctx)
            except SlashExit:
                await self.action_quit()
                return
            self.repl.append_info(out)
            # Slash may have swapped conversation (e.g. /clear, /model).
            self.loop.conversation = self.slash_ctx.conversation
            if self.footer is not None:
                self.footer.update_model(self.loop.conversation.model)

        # ----- turn driver -----------------------------------------------

        async def _run_turn(self, text: str) -> None:
            assert self.repl is not None
            try:
                async for ev in self.loop.run_turn(text):
                    await self._render_event(ev)
                # Mirror conversation reference back to slash ctx.
                self.slash_ctx.conversation = self.loop.conversation
            except asyncio.CancelledError:
                self.repl.append_info("[yellow]turn cancelled by user[/]")
                # Reject any pending approval futures so the loop unwinds.
                self._cancel_pending_approvals()
                raise
            except Exception as exc:  # noqa: BLE001
                self.repl.append_error(
                    f"turn errored: {type(exc).__name__}: {exc}"
                )

        async def _render_event(self, event) -> None:
            assert self.repl is not None
            if isinstance(event, AssistantThinking):
                self.repl.append_thinking(event.iteration)
            elif isinstance(event, AssistantTextDelta):
                self.repl.append_assistant(event.text)
            elif isinstance(event, ToolCallProposed):
                self.repl.append_tool_call(event)
            elif isinstance(event, ApprovalRequested):
                await self.repl.show_approval(
                    event,
                    on_accept=self.loop.approve,
                    on_reject=self.loop.reject,
                )
            elif isinstance(event, ToolCallExecuted):
                self.repl.append_tool_result(event)
            elif isinstance(event, VetoFired):
                self.repl.append_enforcement(event)
            elif isinstance(event, TurnComplete):
                self.repl.append_turn_complete(event.iterations, event.usage)
                if self.footer is not None:
                    self.footer.update_cost(event.usage)

        def _cancel_pending_approvals(self) -> None:
            """Reject any in-flight approval futures the loop is waiting on."""
            resolver = getattr(self.loop, "_approval_resolver", None)
            if not resolver:
                return
            for tool_use_id in list(resolver):
                self.loop.reject(tool_use_id)

        # ----- actions ----------------------------------------------------

        def action_clear(self) -> None:
            if self.repl is not None:
                self.repl.clear_log()

        async def action_cancel_turn(self) -> None:
            if self._turn_task is not None and not self._turn_task.done():
                self._turn_task.cancel()
                self._cancel_pending_approvals()

    return EnchanterApp


def launch(loop: AgentLoop, slash_ctx: SlashContext) -> None:
    """Boot the Textual app. Blocks until the user exits."""
    App = _build_app_class()
    App(loop, slash_ctx).run()


__all__ = ["launch", "_build_app_class"]
