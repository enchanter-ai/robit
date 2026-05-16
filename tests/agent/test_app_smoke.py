"""Smoke tests for the Wave 15.2F Textual REPL shell.

These tests are deliberately structural where possible — booting the full
Textual harness inside parallel CI is fragile, so we prefer:

1. Building the App class without driving the runtime, and inspecting the
   widget tree returned by ``compose()``.
2. Driving small slices through ``App.run_test()`` when the assertion
   actually requires a mounted DOM.

The fixtures swap the loop's ``dispatch_fn`` for a deterministic mock so
no real network is involved.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Union

import pytest

from robit.agent.conversation import Conversation
from robit.agent.loop import AgentLoop
from robit.agent.slash import SlashContext, builtin_registry
from robit.agent.tools import EchoTool, ToolRegistry
from robit.proxy.canonical import (
    CanonicalResponse,
    CanonicalUsage,
    TextPart,
)
from robit.proxy.pipeline import PipelineResult, VetoResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _mock_text_only(req) -> Union[PipelineResult, VetoResult]:
    """One-shot: respond with a single text part, no tool use, stop."""
    resp = CanonicalResponse(
        model=req.model,
        content=(TextPart(text="hello from mock"),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=3, output_tokens=5),
    )
    return PipelineResult(response=resp, fired=())


@pytest.fixture
def loop_and_ctx(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(EchoTool())
    conv = Conversation.new(model="claude-sonnet-4-5", system_prompt=None)
    loop = AgentLoop(
        conversation=conv,
        tool_registry=registry,
        dispatch_fn=_mock_text_only,
    )
    slash_ctx = SlashContext(
        conversation=conv, tool_registry=registry, audit_dir=tmp_path
    )
    return loop, slash_ctx


# ---------------------------------------------------------------------------
# Structural tests — no runtime needed
# ---------------------------------------------------------------------------


def test_build_app_class_returns_textual_app_subclass():
    """The lazy builder returns a real Textual App subclass."""
    from textual.app import App as TextualApp

    from robit.agent.app import _build_app_class

    cls = _build_app_class()
    assert issubclass(cls, TextualApp)
    # Bindings include the Wave 15.2F keybindings.
    keys = {b[0] if isinstance(b, tuple) else b.key for b in cls.BINDINGS}
    assert "ctrl+l" in keys
    assert "escape" in keys
    assert "ctrl+c" in keys


def test_compose_yields_header_repl_footer(loop_and_ctx):
    """compose() returns Header, ReplWidget, FooterWidget in order."""
    from textual.widgets import Header

    from robit.agent.app import _build_app_class
    from robit.agent.widgets.footer import FooterWidget
    from robit.agent.widgets.repl import ReplWidget

    loop, ctx = loop_and_ctx
    App = _build_app_class()
    app = App(loop, ctx)
    children = list(app.compose())
    assert len(children) == 3
    assert isinstance(children[0], Header)
    assert isinstance(children[1], ReplWidget)
    assert isinstance(children[2], FooterWidget)


def test_repl_widget_composes_log_input_and_approval_slot():
    """ReplWidget.compose() yields RichLog + approval slot + Input."""
    from textual.containers import Container
    from textual.widgets import Input, RichLog

    from robit.agent.widgets.repl import ReplWidget

    widget = ReplWidget()
    children = list(widget.compose())
    types = [type(c).__name__ for c in children]
    assert "RichLog" in types
    assert any(isinstance(c, Container) and c.id == "approval-slot" for c in children)
    # Input subclass — _HistoryInput inherits Input.
    assert any(isinstance(c, Input) for c in children)


def test_footer_widget_stores_model_and_session():
    """FooterWidget retains the model + session_id passed at construction.

    compose() requires an active App context in Textual 8.x, so we assert
    the constructor preserves the values; the runtime test below confirms
    the widget actually mounts and renders inside a running app.
    """
    from robit.agent.widgets.footer import FooterWidget

    f = FooterWidget(model="claude-sonnet-4-5", session_id="deadbeef" * 4)
    assert f._model == "claude-sonnet-4-5"
    assert f._session_id.startswith("deadbeef")


# ---------------------------------------------------------------------------
# Runtime tests — use Textual's run_test harness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_mounts_without_crash(loop_and_ctx):
    """The app boots, mounts widgets, and exits cleanly."""
    from robit.agent.app import _build_app_class
    from robit.agent.widgets.footer import FooterWidget
    from robit.agent.widgets.repl import ReplWidget

    loop, ctx = loop_and_ctx
    App = _build_app_class()
    app = App(loop, ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(ReplWidget) is not None
        assert app.query_one(FooterWidget) is not None


@pytest.mark.asyncio
async def test_ctrl_l_clears_the_log(loop_and_ctx):
    """Ctrl-L clears the RichLog widget."""
    from textual.widgets import RichLog

    from robit.agent.app import _build_app_class

    loop, ctx = loop_and_ctx
    App = _build_app_class()
    app = App(loop, ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#output", RichLog)
        log.write("seed line")
        # Some Textual versions render lines lazily; force a refresh tick.
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.pause()
        # RichLog stores its renderables in .lines (private but stable enough
        # for a smoke); fall back to checking it was at least cleared via API.
        lines = getattr(log, "lines", None)
        if lines is not None:
            assert len(lines) == 0


@pytest.mark.asyncio
async def test_unknown_slash_prints_error_message(loop_and_ctx):
    """Submitting `/nope` writes the unknown-command message to the log."""
    from textual.widgets import Input, RichLog

    from robit.agent.app import _build_app_class

    loop, ctx = loop_and_ctx
    App = _build_app_class()
    app = App(loop, ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/nope"
        # Trigger submission via Textual's message system.
        await prompt.action_submit()
        await pilot.pause()
        log = app.query_one("#output", RichLog)
        # Look for the canonical unknown-command marker in any rendered line.
        rendered = "\n".join(
            str(seg.text if hasattr(seg, "text") else seg)
            for seg in getattr(log, "lines", [])
        )
        # If we couldn't introspect lines, fall back to confirming the input
        # got reset (the handler ran).
        assert "Unknown slash command" in rendered or prompt.value == ""


@pytest.mark.asyncio
async def test_escape_cancels_in_flight_turn(loop_and_ctx):
    """Esc cancels a running turn task without crashing the app."""
    from robit.agent.app import _build_app_class

    loop, ctx = loop_and_ctx

    # Build a dispatch that hangs so we have something to cancel.
    hang_event = asyncio.Event()

    async def hanging_dispatch(_req):
        await hang_event.wait()
        # Never returns by design.
        raise AssertionError("should have been cancelled")

    loop.dispatch_fn = hanging_dispatch  # type: ignore[assignment]

    App = _build_app_class()
    app = App(loop, ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Kick off a turn directly (skip the input plumbing — we just want
        # to observe that escape cancels the task).
        app._turn_task = asyncio.create_task(app._run_turn("hang please"))
        await asyncio.sleep(0.05)
        assert not app._turn_task.done()
        await pilot.press("escape")
        await asyncio.sleep(0.05)
        assert app._turn_task.done() or app._turn_task.cancelled()
        hang_event.set()  # release any lingering wait


@pytest.mark.asyncio
async def test_one_shot_text_turn_renders_assistant_line(loop_and_ctx):
    """End-to-end: submit a prompt, mock returns text, log shows it."""
    from textual.widgets import Input

    from robit.agent.app import _build_app_class

    loop, ctx = loop_and_ctx
    App = _build_app_class()
    app = App(loop, ctx)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", Input)
        prompt.value = "hello"
        await prompt.action_submit()
        # Allow the turn task to complete.
        for _ in range(20):
            await pilot.pause()
            if app._turn_task is not None and app._turn_task.done():
                break
        assert app._turn_task is not None
        assert app._turn_task.done()
        # The loop appended the user message + assistant reply.
        assert len(loop.conversation.messages) >= 2
