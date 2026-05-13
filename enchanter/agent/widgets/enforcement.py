"""Enforcement chips — inline status indicators for the agent REPL.

A *chip* is a single-line Textual widget that renders alongside conversation
output to surface an enforcement event: a proxy veto, a redaction, a conduct
injection, a fast-path audit notice, or an inline cost notice.

Design notes
------------

* Each chip is a :class:`textual.widgets.Static` subclass. Its
  :meth:`render` returns a :class:`rich.text.Text` so styling stays in the
  rich layer (no markup string parsing surprises).
* Chips render on **one line**, padded/truncated to **at most 80 chars**.
  The detail is dimmed and bracketed; if the full string would overflow,
  the detail is truncated with an ellipsis before the closing bracket.
* The :data:`EnforcementChip.KIND_COLORS` table is the single source of
  truth for the five chip kinds. Adding a kind here updates render and
  validation in one step.

Chip kinds
~~~~~~~~~~

==========  ============  ==========  =======
kind        background    fg          prefix
==========  ============  ==========  =======
veto        red           white       ``✘``
redaction   yellow        black       ``⚠``
conduct     (none)        blue        ``ℹ``
audit       (none)        gray        ``📋``
cost        (none)        green       ``$``
==========  ============  ==========  =======

The conduct / audit / cost kinds use a foreground colour only (no
background) so they sit quietly inline; veto and redaction take a
background to grab the eye.

Subclasses
----------

* :class:`VetoChip` — builds from a :class:`enchanter.agent.loop.VetoFired`.
* :class:`RedactionChip` — builds from a list of matched pattern IDs.
* :class:`ConductChip` — builds from a list of injected module names.

The generic :class:`EnforcementChip` is the right tool for audit / cost
chips and for ad-hoc enforcement notices that don't have a dedicated
event type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from enchanter.agent.loop import VetoFired


__all__ = [
    "EnforcementChip",
    "VetoChip",
    "RedactionChip",
    "ConductChip",
]


# Hard cap on rendered chip width so it never wraps inside a RichLog row.
_MAX_WIDTH = 80


class EnforcementChip(Static):
    """Inline single-line status chip for an enforcement event.

    Parameters
    ----------
    kind:
        One of the keys in :data:`KIND_COLORS`. Anything else raises
        :class:`ValueError` with the list of valid kinds.
    label:
        The primary text — what happened. Rendered after the kind prefix.
    detail:
        Optional secondary text — why, or which pattern. Rendered in a
        dimmer style inside ``[ ]`` brackets after the label. Omitted
        entirely when ``None``.

    Notes
    -----
    A chip's rendered width is capped at 80 chars. When the full string
    would overflow, only the *detail* is truncated (with an ellipsis
    before the closing bracket); the kind prefix and label are preserved
    verbatim because they carry the actionable signal.
    """

    # (background, foreground, prefix). A foreground of ``None`` means
    # "default terminal foreground" — used when the chip has a background.
    KIND_COLORS: dict[str, tuple[str | None, str | None, str]] = {
        "veto":      ("red",    "white", "✘"),  # ✘
        "redaction": ("yellow", "black", "⚠"),  # ⚠
        "conduct":   (None,     "blue",  "ℹ"),  # ℹ
        "audit":     (None,     "gray",  "\U0001f4cb"),  # 📋
        "cost":      (None,     "green", "$"),
    }

    def __init__(
        self,
        kind: str,
        label: str,
        *,
        detail: str | None = None,
    ) -> None:
        if kind not in self.KIND_COLORS:
            valid = ", ".join(sorted(self.KIND_COLORS))
            raise ValueError(
                f"unknown chip kind {kind!r}; valid kinds are: {valid}"
            )
        self.kind = kind
        self.label = label
        self.detail = detail
        # Static accepts a renderable at init time; pass an empty string
        # and rely on render() so the rich.Text is built fresh each frame.
        super().__init__("")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Text:  # type: ignore[override]
        """Build the chip's :class:`rich.text.Text` representation.

        The structure is::

            <prefix> <label>  [<detail>]

        Two spaces separate the label from the bracketed detail when the
        detail is present. The whole string is clamped to
        :data:`_MAX_WIDTH` chars by truncating the detail.
        """
        bg, fg, prefix = self.KIND_COLORS[self.kind]

        # Build the head: "<prefix> <label>" — the part we never truncate.
        head = f"{prefix} {self.label}"

        # Compose the full string (no styling yet) so we can measure it.
        if self.detail is not None:
            tail = f"  [{self.detail}]"
        else:
            tail = ""
        full = head + tail

        if len(full) > _MAX_WIDTH and self.detail is not None:
            # Trim the detail. Reserve room for "  [" + "...]" = 6 chars.
            budget = _MAX_WIDTH - len(head) - len("  [") - len("...]")
            if budget < 1:
                # Head alone already fills/exceeds budget — drop detail.
                tail = ""
                full = head
            else:
                trimmed = self.detail[:budget]
                tail = f"  [{trimmed}...]"
                full = head + tail
        elif len(full) > _MAX_WIDTH:
            # No detail, but head somehow overflows — hard-truncate.
            full = full[: _MAX_WIDTH - 3] + "..."

        # Now style.
        style_parts: list[str] = []
        if bg is not None:
            style_parts.append(f"on {bg}")
        if fg is not None:
            style_parts.append(fg)
        head_style = " ".join(style_parts) if style_parts else ""

        text = Text()
        # Re-derive head/tail from the (possibly truncated) ``full`` so
        # truncation lands in the right span.
        if tail:
            text.append(full[: len(head)], style=head_style)
            text.append(full[len(head):], style="dim")
        else:
            text.append(full, style=head_style)
        return text


class VetoChip(EnforcementChip):
    """Specialised chip for :class:`enchanter.agent.loop.VetoFired`.

    Use :meth:`from_event` to build directly from a fired event; the
    constructor stays available for tests or for chips assembled by hand.
    """

    @classmethod
    def from_event(cls, event: "VetoFired") -> "VetoChip":
        """Build a :class:`VetoChip` from a fired veto event.

        The label is ``"<plugin> blocked"`` and the detail is the
        ``reason`` — optionally suffixed with ``[<pattern_id>]`` when the
        event carries one.
        """
        label = f"{event.plugin} blocked"
        detail = event.reason
        if event.pattern_id:
            detail += f" [{event.pattern_id}]"
        return cls(kind="veto", label=label, detail=detail)


class RedactionChip(EnforcementChip):
    """Specialised chip for redaction events from the secret-mask plugin."""

    @classmethod
    def from_patterns(cls, matched_pattern_ids: list[str]) -> "RedactionChip":
        """Build a :class:`RedactionChip` from the list of matched IDs.

        An empty list still renders — the count becomes ``0 pattern(s)``
        and the detail is an empty bracket. This is intentional: the
        chip exists to communicate that the secret-mask plugin *ran*,
        even when no patterns matched.
        """
        label = f"secret-mask redacted {len(matched_pattern_ids)} pattern(s)"
        detail = ", ".join(matched_pattern_ids)
        return cls(kind="redaction", label=label, detail=detail)


class ConductChip(EnforcementChip):
    """Specialised chip for conduct-module injection notices."""

    @classmethod
    def from_modules(cls, module_names: list[str]) -> "ConductChip":
        """Build a :class:`ConductChip` from the list of injected modules.

        Mirrors :meth:`RedactionChip.from_patterns` — an empty list still
        renders so consumers can detect a no-op injection round.
        """
        label = f"conduct: {len(module_names)} module(s) injected"
        detail = ", ".join(module_names)
        return cls(kind="conduct", label=label, detail=detail)
