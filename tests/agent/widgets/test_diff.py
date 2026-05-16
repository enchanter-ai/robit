"""Tests for ``robit.agent.widgets.diff``.

Covers:

* parsing + rendering a simple edit
* multi-hunk diffs
* language inference from the diff header
* missing-header tolerance
* :meth:`DiffView.new_file` synthesises an all-added diff
* empty body tolerance
* long-line wrap (does not raise)
* :class:`ApprovalPrompt` key bindings end-to-end via Textual's test harness
* :class:`ApprovalPrompt` structural shape (compose tree)
"""

from __future__ import annotations

import asyncio

import pytest
from rich.console import Console
from textual.app import App, ComposeResult
from textual.widgets import Button

from robit.agent.widgets.diff import ApprovalPrompt, DiffView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture(renderable) -> str:
    """Render a Rich renderable to an ANSI-stripped string for assertions."""
    console = Console(record=True, width=120, color_system=None, force_terminal=False)
    console.print(renderable)
    return console.export_text()


SIMPLE_DIFF = (
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ line 2 @@\n"
    " def greet(name):\n"
    "-    print('hello ' + name)\n"
    "+    print(f'hello {name}')\n"
    " \n"
    " greet('world')\n"
)

MULTI_HUNK_DIFF = (
    "--- a/bar.py\n"
    "+++ b/bar.py\n"
    "@@ line 1 @@\n"
    "-import os\n"
    "+import sys\n"
    "@@ line 42 @@\n"
    " def main():\n"
    "-    return 0\n"
    "+    return 1\n"
)


# ---------------------------------------------------------------------------
# DiffView — rendering
# ---------------------------------------------------------------------------


def test_simple_edit_renders_all_lines() -> None:
    """A standard one-add / one-remove diff renders every line we fed it."""
    view = DiffView(SIMPLE_DIFF)
    rendered = _capture(view.render())
    # Each original line should appear in the output (modulo any trailing
    # whitespace the console may trim — we strip before matching).
    for line in SIMPLE_DIFF.splitlines():
        if not line.strip():
            continue
        assert line.strip() in rendered, f"missing diff line: {line!r}"


def test_multi_hunk_renders_all_hunks() -> None:
    """Every ``@@`` header and both add/remove pairs survive rendering."""
    view = DiffView(MULTI_HUNK_DIFF)
    rendered = _capture(view.render())
    assert "@@ line 1 @@" in rendered
    assert "@@ line 42 @@" in rendered
    assert "import os" in rendered
    assert "import sys" in rendered
    assert "return 0" in rendered
    assert "return 1" in rendered


def test_language_inferred_from_py_header() -> None:
    """No explicit language hint → inference from ``+++ b/foo.py`` returns python."""
    view = DiffView(SIMPLE_DIFF)
    assert view.language == "python"


def test_language_explicit_hint_wins() -> None:
    """Explicit language overrides inference."""
    view = DiffView(SIMPLE_DIFF, language="rust")
    assert view.language == "rust"


def test_no_header_renders_gracefully() -> None:
    """A body-only diff (no ``---`` / ``+++`` lines) must still render."""
    body = "-old\n+new\n"
    view = DiffView(body)
    rendered = _capture(view.render())
    assert "old" in rendered
    assert "new" in rendered
    # Inference fails without a header → language is None.
    assert view.language is None


def test_new_file_synthesises_all_added() -> None:
    """``new_file(path, content)`` prefixes every body line with ``+``."""
    content = "line one\nline two\nline three\n"
    view = DiffView.new_file("hello.py", content)
    diff = view.diff_text
    assert "--- a/hello.py" in diff
    assert "+++ b/hello.py" in diff
    assert "+line one" in diff
    assert "+line two" in diff
    assert "+line three" in diff
    # No "-" body lines exist (the "---" header doesn't count).
    body_lines = [
        ln
        for ln in diff.splitlines()
        if ln.startswith(("+", "-")) and not ln.startswith(("+++ ", "--- "))
    ]
    assert all(ln.startswith("+") for ln in body_lines)
    # Language inferred from .py extension.
    assert view.language == "python"


def test_empty_diff_body_renders_without_exception() -> None:
    """A header-only or empty diff renders a placeholder and does not raise."""
    view_empty = DiffView("")
    rendered_empty = _capture(view_empty.render())
    assert "empty diff" in rendered_empty.lower()

    header_only = "--- a/x.txt\n+++ b/x.txt\n@@ line 1 @@\n"
    view_header = DiffView(header_only)
    rendered_header = _capture(view_header.render())
    # No crash; header lines appear.
    assert "x.txt" in rendered_header
    assert "@@" in rendered_header


def test_long_single_line_does_not_crash() -> None:
    """A pathologically long line must render (best-effort wrap, no crash)."""
    long_payload = "x" * 5000
    diff = (
        "--- a/long.txt\n"
        "+++ b/long.txt\n"
        "@@ line 1 @@\n"
        f"+{long_payload}\n"
    )
    view = DiffView(diff)
    rendered = _capture(view.render())
    # The payload appears somewhere in the output (wrapped or not).
    assert "x" * 50 in rendered  # at least a substantial chunk survives


def test_new_file_with_empty_content() -> None:
    """``new_file`` with an empty body emits header but no ``+`` body lines."""
    view = DiffView.new_file("blank.txt", "")
    diff = view.diff_text
    assert "--- a/blank.txt" in diff
    body = [
        ln
        for ln in diff.splitlines()
        if ln.startswith("+") and not ln.startswith("+++ ")
    ]
    assert body == []


def test_classify_handles_malformed_input() -> None:
    """Lines without the expected prefix render as plain text (no crash)."""
    diff = "this line has no prefix\nneither does this one\n"
    view = DiffView(diff)
    rendered = _capture(view.render())
    assert "this line has no prefix" in rendered
    assert "neither does this one" in rendered


def test_syntax_renderable_returns_none_without_language() -> None:
    """No inferred language → ``syntax_renderable`` returns ``None``."""
    view = DiffView("-x\n+y\n")  # no header → no language
    assert view.syntax_renderable() is None


def test_syntax_renderable_extracts_added_lines() -> None:
    """When a language was inferred, the syntax renderable holds only the
    added-line bodies (no ``+`` prefix)."""
    view = DiffView(SIMPLE_DIFF)
    syn = view.syntax_renderable()
    assert syn is not None
    # rich.Syntax stores the source under .code.
    assert "f'hello {name}'" in syn.code
    # The minus-line body must NOT be in the added-only view.
    assert "'hello ' + name" not in syn.code


# ---------------------------------------------------------------------------
# ApprovalPrompt — structural + behavioural
# ---------------------------------------------------------------------------


def test_approval_bindings_contain_y_n_escape() -> None:
    """Structural smoke test: BINDINGS expose y, n and escape."""
    keys = {b[0] for b in ApprovalPrompt.BINDINGS}
    assert "y" in keys
    assert "n" in keys
    assert "escape" in keys


def test_approval_compose_tree_has_diff_and_buttons() -> None:
    """Mount an ApprovalPrompt and verify it contains a DiffView and two Buttons."""

    diff_view = DiffView(SIMPLE_DIFF)

    class _Host(App):
        def compose(self) -> ComposeResult:
            yield ApprovalPrompt(
                tool_name="file_edit",
                args={"path": "foo.py"},
                diff_view=diff_view,
                tool_use_id="tu-1",
            )

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one(ApprovalPrompt)
            assert prompt.tool_name == "file_edit"
            assert prompt.tool_use_id == "tu-1"
            # DiffView is mounted as a child.
            assert app.query_one(DiffView) is diff_view
            buttons = list(app.query(Button))
            assert len(buttons) == 2
            ids = {b.id for b in buttons}
            assert ids == {"approval-approve", "approval-reject"}

    asyncio.run(_run())


def test_approval_y_posts_approved_message() -> None:
    """Pressing ``y`` resolves the prompt by posting ``Approved``."""

    received: list[tuple[str, str]] = []

    class _Host(App):
        def compose(self) -> ComposeResult:
            yield ApprovalPrompt(
                tool_name="file_edit",
                args={"path": "foo.py"},
                diff_view=None,
                tool_use_id="tu-approve",
            )

        def on_approval_prompt_approved(
            self, message: ApprovalPrompt.Approved
        ) -> None:
            received.append(("approved", message.tool_use_id))

        def on_approval_prompt_rejected(
            self, message: ApprovalPrompt.Rejected
        ) -> None:
            received.append(("rejected", message.tool_use_id))

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()

    asyncio.run(_run())

    assert ("approved", "tu-approve") in received
    assert not any(kind == "rejected" for kind, _ in received)


def test_approval_n_posts_rejected_message() -> None:
    """Pressing ``n`` resolves the prompt by posting ``Rejected``."""

    received: list[tuple[str, str]] = []

    class _Host(App):
        def compose(self) -> ComposeResult:
            yield ApprovalPrompt(
                tool_name="file_edit",
                args={"path": "foo.py"},
                diff_view=None,
                tool_use_id="tu-reject",
            )

        def on_approval_prompt_approved(
            self, message: ApprovalPrompt.Approved
        ) -> None:
            received.append(("approved", message.tool_use_id))

        def on_approval_prompt_rejected(
            self, message: ApprovalPrompt.Rejected
        ) -> None:
            received.append(("rejected", message.tool_use_id))

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()

    asyncio.run(_run())

    assert ("rejected", "tu-reject") in received
    assert not any(kind == "approved" for kind, _ in received)


def test_approval_escape_posts_rejected_message() -> None:
    """Pressing ``Esc`` is the safe default and posts ``Rejected``."""

    received: list[tuple[str, str]] = []

    class _Host(App):
        def compose(self) -> ComposeResult:
            yield ApprovalPrompt(
                tool_name="file_write",
                args={"path": "new.py"},
                diff_view=None,
                tool_use_id="tu-esc",
            )

        def on_approval_prompt_rejected(
            self, message: ApprovalPrompt.Rejected
        ) -> None:
            received.append(("rejected", message.tool_use_id))

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

    asyncio.run(_run())

    assert ("rejected", "tu-esc") in received


def test_approval_double_press_does_not_double_post() -> None:
    """A second key press after resolution is a no-op (no double message)."""

    received: list[str] = []

    class _Host(App):
        def compose(self) -> ComposeResult:
            yield ApprovalPrompt(
                tool_name="file_edit",
                args={},
                diff_view=None,
                tool_use_id="tu-double",
            )

        def on_approval_prompt_approved(
            self, message: ApprovalPrompt.Approved
        ) -> None:
            received.append("approved")

        def on_approval_prompt_rejected(
            self, message: ApprovalPrompt.Rejected
        ) -> None:
            received.append("rejected")

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("y")
            await pilot.press("n")  # should be ignored
            await pilot.pause()

    asyncio.run(_run())

    assert received == ["approved"]


if __name__ == "__main__":  # pragma: no cover - convenience runner
    pytest.main([__file__, "-v"])
