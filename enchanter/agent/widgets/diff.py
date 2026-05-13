"""enchanter.agent.widgets.diff — unified-diff renderer for the REPL.

Consumes the compact diff format emitted by
:mod:`enchanter.agent.tools.file_edit`:

    --- a/<relpath>
    +++ b/<relpath>
    @@ line <N> @@
     context
    -removed
    +added
     context

Also synthesises a "new file" diff from raw content (used for the output of
:mod:`enchanter.agent.tools.file_write`, which emits ``"wrote N bytes to X"``
rather than a diff).

Tolerance contract: real-world diff strings from the loop will sometimes be
empty, header-only, malformed-whitespace, or contain extremely long lines.
This renderer never raises on parse — it renders what it can and degrades to
plain text otherwise.

Color scheme (committed):

* hunk header (``@@``, ``---``, ``+++``): ``bold blue``
* context lines (leading space):          ``dim``
* removed lines (leading ``-``):          ``red strike``
* added lines (leading ``+``):            ``green``
* file path inside header:                ``cyan`` (inherited from bold blue)
"""

from __future__ import annotations

from pathlib import PurePosixPath

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text
from textual.widgets import Static

# Map file extensions to Pygments lexer names used by rich.Syntax. The
# inference is best-effort — unknown extensions fall back to plain text.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".jsonl": "json",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".xml": "xml",
}

# Max line length before we mark a line as "wrappable" rather than truncate
# it outright. rich.Text supports overflow="fold" so long lines wrap rather
# than crash the renderer.
_LONG_LINE_OVERFLOW = "fold"


def _infer_language_from_diff(diff_text: str) -> str | None:
    """Inspect the ``+++ b/<path>`` header (falling back to ``--- a/<path>``)
    and look up the extension in :data:`_EXT_TO_LANG`. Returns ``None`` if no
    header is present or the extension is unrecognised.
    """
    if not diff_text:
        return None
    for line in diff_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("+++ ") or stripped.startswith("--- "):
            # Format: "+++ b/path/to/file.py" or "--- a/path/to/file.py".
            # Tolerate missing "a/" / "b/" prefix.
            payload = stripped[4:].strip()
            if payload.startswith(("a/", "b/")):
                payload = payload[2:]
            if not payload or payload == "/dev/null":
                continue
            try:
                ext = PurePosixPath(payload).suffix.lower()
            except (ValueError, TypeError):
                continue
            lang = _EXT_TO_LANG.get(ext)
            if lang is not None:
                return lang
    return None


def _classify_line(line: str) -> str:
    """Return a Rich style string for a single diff line.

    Classification is by leading character, matching the file_edit emit
    format exactly. Lines that don't start with one of the expected prefixes
    are rendered as plain text (no style) so malformed input doesn't blow up.
    """
    if not line:
        return ""
    if line.startswith("@@") or line.startswith("--- ") or line.startswith("+++ "):
        return "bold blue"
    head = line[0]
    if head == "+":
        return "green"
    if head == "-":
        return "red strike"
    if head == " ":
        return "dim"
    return ""


class DiffView(Static):
    """Textual widget rendering a unified-diff string with color highlighting.

    Tolerates malformed input (empty body, missing header, very long lines)
    and never raises during render. Use :meth:`new_file` to render the body
    of a freshly-created file as an all-added synthetic diff.
    """

    DEFAULT_CSS = """
    DiffView {
        height: auto;
        width: 100%;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        diff_text: str,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> None:
        # Coerce non-string input defensively — callers may pass None or
        # other oddities from upstream tool results.
        if diff_text is None:
            diff_text = ""
        elif not isinstance(diff_text, str):
            diff_text = str(diff_text)
        self._diff_text = diff_text
        self._language = language if language is not None else _infer_language_from_diff(
            diff_text
        )
        super().__init__(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ ctors

    @classmethod
    def new_file(
        cls,
        path: str,
        content: str,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> "DiffView":
        """Build a synthetic all-added diff for a freshly-created file.

        Every line of ``content`` becomes a ``+``-prefixed line. The header
        names ``/dev/null`` as the source so consumers can distinguish a
        "new file" diff from an edit.
        """
        if path is None:
            path = ""
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        # Normalise CRLF / CR to LF so we don't render rogue \r glyphs.
        normalised = content.replace("\r\n", "\n").replace("\r", "\n")
        body_lines = normalised.split("\n") if normalised else []
        # Trim trailing empty line if the content ended with a newline — the
        # split otherwise inserts a phantom blank "+" line at the bottom.
        if body_lines and body_lines[-1] == "" and normalised.endswith("\n"):
            body_lines.pop()
        parts: list[str] = [
            f"--- a/{path}" if path else "--- a/",
            f"+++ b/{path}" if path else "+++ b/",
            "@@ line 1 @@",
        ]
        parts.extend(f"+{line}" for line in body_lines)
        diff_text = "\n".join(parts) + ("\n" if parts else "")
        return cls(diff_text, language=language, **kwargs)

    # ----------------------------------------------------------------- render

    def render(self) -> RenderableType:
        """Produce a color-coded renderable for the diff.

        The strategy: split the diff into "header" lines (rendered as styled
        :class:`rich.text.Text`) and "body" lines (also rendered as styled
        :class:`rich.text.Text` with per-line classification). We *don't*
        pipe the body through :class:`rich.syntax.Syntax` because Syntax
        lexers don't understand ``+``/``-`` prefixes — they'd report syntax
        errors on a perfectly-valid diff. The language hint is exposed via
        :attr:`language` for downstream consumers (e.g. side-by-side views).
        """
        if not self._diff_text:
            # Empty diff — render a single dim placeholder rather than nothing.
            return Text("(empty diff)", style="dim italic")

        lines: list[Text] = []
        for raw_line in self._diff_text.splitlines():
            style = _classify_line(raw_line)
            # overflow="fold" makes long lines wrap rather than crash the
            # terminal; no_wrap=False keeps Rich's wrapper enabled.
            text = Text(
                raw_line if raw_line else " ",
                style=style,
                overflow=_LONG_LINE_OVERFLOW,
                no_wrap=False,
            )
            lines.append(text)
        if not lines:
            return Text("(empty diff)", style="dim italic")
        return Group(*lines)

    # ----------------------------------------------------------------- helpers

    @property
    def language(self) -> str | None:
        """The inferred or supplied language hint (read-only)."""
        return self._language

    @property
    def diff_text(self) -> str:
        """The raw diff string this widget was constructed with."""
        return self._diff_text

    def syntax_renderable(self) -> Syntax | None:
        """Optional helper: return a :class:`rich.syntax.Syntax` for the
        *added-lines-only* body, with the inferred language. Returns
        ``None`` if no language was inferred or no `+` body lines exist.

        Used by sibling F's REPL when it wants a side-by-side preview of
        what the file would look like after the edit lands.
        """
        if self._language is None:
            return None
        added: list[str] = []
        for line in self._diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++ "):
                added.append(line[1:])
        if not added:
            return None
        return Syntax(
            "\n".join(added),
            self._language,
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )


# ---------------------------------------------------------------------------
# ApprovalPrompt — inline approval UI mounted by sibling F's REPL.
# ---------------------------------------------------------------------------
#
# This widget lives next to DiffView because the Wave 15.2G allowed-files
# list scopes us to ``widgets/diff.py``. A future refactor may split it into
# ``widgets/approval.py`` once the sibling sprint has landed.

from textual.app import ComposeResult  # noqa: E402  (kept near consumer)
from textual.containers import Container  # noqa: E402
from textual.message import Message  # noqa: E402
from textual.widgets import Button, Label  # noqa: E402


class ApprovalPrompt(Container):
    """Inline approval UI for a pending tool call.

    Mounts inline (not as a full-screen modal — that's heavier and grabs
    focus in a way that interrupts the REPL log). Sibling F's REPL mounts
    one of these whenever the loop yields an ``ApprovalRequested`` event,
    awaits an :class:`Approved` or :class:`Rejected` Textual message, and
    then calls ``loop.approve(tool_use_id)`` or ``loop.reject(tool_use_id)``.

    Key bindings: ``y`` / ``Y`` → approve; ``n`` / ``N`` → reject; ``Esc``
    → reject (the safe default).
    """

    DEFAULT_CSS = """
    ApprovalPrompt {
        layout: vertical;
        height: auto;
        border: round $warning;
        padding: 1 2;
        margin: 1 0;
    }
    ApprovalPrompt > #approval-header {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    ApprovalPrompt > #approval-buttons {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    ApprovalPrompt > #approval-buttons > Button {
        margin-right: 2;
    }
    ApprovalPrompt > #approval-footer {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("y", "approve", "Approve"),
        ("n", "reject", "Reject"),
        ("escape", "reject", "Reject"),
    ]

    class Approved(Message):
        """Posted when the user approves the pending tool call.

        :attr:`tool_use_id` matches the id from the originating
        :class:`enchanter.agent.loop.ApprovalRequested` event; the REPL
        forwards it to :meth:`AgentLoop.approve`.
        """

        def __init__(self, tool_use_id: str) -> None:
            self.tool_use_id = tool_use_id
            super().__init__()

    class Rejected(Message):
        """Posted when the user rejects (or Escapes) the pending tool call."""

        def __init__(self, tool_use_id: str) -> None:
            self.tool_use_id = tool_use_id
            super().__init__()

    def __init__(
        self,
        *,
        tool_name: str,
        args: dict,
        diff_view: DiffView | None,
        tool_use_id: str,
        **kwargs: object,
    ) -> None:
        self._tool_name = tool_name
        self._args = args if isinstance(args, dict) else {}
        self._diff_view = diff_view
        self._tool_use_id = tool_use_id
        # Track whether we've already resolved so a fast double-press of
        # Y then N (or vice versa) doesn't double-post.
        self._resolved = False
        super().__init__(**kwargs)  # type: ignore[arg-type]
        # Give the widget a stable id so sibling F can query it by selector.
        if not self.id:
            self.id = "approval-prompt"

    def compose(self) -> ComposeResult:
        # Header summarises what's being asked.
        path = ""
        if isinstance(self._args, dict):
            raw = self._args.get("path")
            if isinstance(raw, str):
                path = raw
        header_text = f"Approve {self._tool_name}"
        if path:
            header_text += f" on {path}"
        header_text += "?"
        yield Label(header_text, id="approval-header")

        # The diff preview (optional — some approvable tools have no diff).
        if self._diff_view is not None:
            yield self._diff_view

        with Container(id="approval-buttons"):
            yield Button("Approve", id="approval-approve", variant="success")
            yield Button("Reject", id="approval-reject", variant="error")

        yield Label(
            "Press Y to approve, N to reject (Esc rejects).",
            id="approval-footer",
        )

    # ------------------------------------------------------------------ actions

    async def action_approve(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        self.post_message(self.Approved(self._tool_use_id))

    async def action_reject(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        self.post_message(self.Rejected(self._tool_use_id))

    # Button clicks route through the same resolution path so mouse and
    # keyboard share one code path.
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-approve":
            await self.action_approve()
        elif event.button.id == "approval-reject":
            await self.action_reject()

    # ----------------------------------------------------------------- helpers

    @property
    def tool_use_id(self) -> str:
        """The pending tool-use id this prompt is resolving."""
        return self._tool_use_id

    @property
    def tool_name(self) -> str:
        """The name of the tool awaiting approval."""
        return self._tool_name


__all__ = ["DiffView", "ApprovalPrompt"]
