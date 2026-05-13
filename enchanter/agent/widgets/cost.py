"""enchanter.agent.widgets.cost — Wave 15.2/Agent I live cost ticker.

Footer widget showing per-turn and accumulated session spend in real time.
The agent CLI calls :meth:`CostTicker.add_turn` from every
:class:`enchanter.agent.loop.TurnComplete` event so the displayed numbers
track the proxy's own ``X-Enchanter-Cost-Cents`` accounting.

Pricing source
==============

Cents are computed by reusing the proxy's private helper
``enchanter.proxy.events.cost_ledger._compute_cents`` so the proxy and the
agent CLI cannot drift on what each model costs. We import the
underscore-prefixed name deliberately — touching ``cost_ledger.py`` to
re-export it as a public symbol is out of scope for this wave (the audit
report flagged duplicate pricing tables as a finding; consolidating the
pricing surface is a **Wave 16+ honey-spot consolidation candidate**).

Render format
=============

Single line, fits one footer row::

    last: $0.0012 (42→18 tok) | session: $0.034 (4128→1209 tok, claude-3-5-sonnet)

* Dollar amounts use 4-decimal precision below $1 and 2-decimal at/above $1.
* Token counts use thousands separators (``f"{n:,}"``).
* ``in→out`` arrows show direction (input tokens → output tokens).
* ``last_model`` truncated to 30 chars.
* Trailing model name only appears once a turn has been recorded.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

# Reuse the proxy's private cost helper so the agent CLI and the proxy
# stay in sync on pricing. See the module docstring — this underscore
# import is technical debt, tracked for Wave 16+ consolidation.
from enchanter.proxy.events.cost_ledger import _compute_cents

__all__ = ["CostTicker"]


_MODEL_TRUNCATE = 30


def _format_dollars(cents: int) -> str:
    """Format integer cents as a dollar string.

    Below $1.00 → 4-decimal precision (e.g. ``$0.0012``).
    At/above $1.00 → 2-decimal precision (e.g. ``$1.23``).
    """
    dollars = cents / 100.0
    if dollars < 1.0:
        return f"${dollars:.4f}"
    return f"${dollars:.2f}"


def _format_tokens(n: int) -> str:
    """Format a token count with thousands separators."""
    return f"{n:,}"


def _truncate_model(model: str) -> str:
    if len(model) <= _MODEL_TRUNCATE:
        return model
    # Reserve one char for the ellipsis so the visible string is exactly
    # _MODEL_TRUNCATE chars wide.
    return model[: _MODEL_TRUNCATE - 1] + "…"


class CostTicker(Static):
    """Footer widget showing per-turn and total cost for the session.

    Reads pricing via :func:`enchanter.proxy.events.cost_ledger._compute_cents`
    so the proxy's ``cost.ledger.recorded`` events and this widget agree on
    what each model costs. Unknown models fall back to the cost-ledger
    module's default rate (no crash, conservative-middle estimate).

    State
    -----
    turn_tokens_in / turn_tokens_out
        Tokens for the last turn only.
    turn_cents
        Cents for the last turn only.
    session_tokens_in / session_tokens_out
        Accumulated across the whole session.
    session_cents
        Accumulated cost across the whole session.
    last_model
        Most recent model used (``None`` until the first ``add_turn``).

    Rendering
    ---------
    Static, single-line, fits one footer row::

        last: $0.0012 (42→18 tok) | session: $0.034 (4128→1209 tok, claude-3-5-sonnet)
    """

    def __init__(
        self,
        initial_tokens_in: int = 0,
        initial_tokens_out: int = 0,
    ) -> None:
        super().__init__()
        # Last-turn counters — overwritten on every add_turn.
        self.turn_tokens_in: int = 0
        self.turn_tokens_out: int = 0
        self.turn_cents: int = 0
        # Session counters — accumulate from the supplied initial values.
        self.session_tokens_in: int = int(initial_tokens_in)
        self.session_tokens_out: int = int(initial_tokens_out)
        self.session_cents: int = 0
        # Last model — populated by the first add_turn.
        self.last_model: str | None = None

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_turn(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> None:
        """Record one ``TurnComplete`` event.

        Called by the agent CLI's app (sibling F's REPL) on each
        :class:`enchanter.agent.loop.TurnComplete`. The widget updates
        both last-turn and session counters and triggers a re-render
        via :meth:`textual.widgets.Static.update`.
        """
        in_tok = int(input_tokens)
        out_tok = int(output_tokens)

        cents = _compute_cents(in_tok, out_tok, model)

        # Last-turn counters reflect ONLY this turn.
        self.turn_tokens_in = in_tok
        self.turn_tokens_out = out_tok
        self.turn_cents = cents

        # Session counters accumulate.
        self.session_tokens_in += in_tok
        self.session_tokens_out += out_tok
        self.session_cents += cents

        # Always update last_model so a turn with a new model is reflected
        # immediately (handles the multi-model conversation case).
        self.last_model = model

        # Push the new render into the Textual widget. Tests that
        # instantiate CostTicker outside a running app can still call
        # render() directly and inspect the returned renderable.
        try:
            self.update(self.render())
        except Exception:
            # Textual's Static.update touches the DOM; harmless to skip
            # when the widget is not mounted (unit-test path).
            pass

    def reset(self) -> None:
        """Zero session counters. Last-turn fields and ``last_model`` stick.

        Wired to ``/clear`` so the user can drop conversation context
        without losing the most recent turn's display (helps confirm
        the reset actually fired).
        """
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        self.session_cents = 0
        try:
            self.update(self.render())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Text:
        """Build the footer line.

        Returns a :class:`rich.text.Text` so Textual can size and style
        it. The body is plain text — styling is intentionally deferred
        to sibling F's footer container so the widget composes cleanly.
        """
        turn_cost = _format_dollars(self.turn_cents)
        turn_in = _format_tokens(self.turn_tokens_in)
        turn_out = _format_tokens(self.turn_tokens_out)

        session_cost = _format_dollars(self.session_cents)
        session_in = _format_tokens(self.session_tokens_in)
        session_out = _format_tokens(self.session_tokens_out)

        last_part = f"last: {turn_cost} ({turn_in}→{turn_out} tok)"

        if self.last_model is not None:
            model_display = _truncate_model(self.last_model)
            session_part = (
                f"session: {session_cost} "
                f"({session_in}→{session_out} tok, {model_display})"
            )
        else:
            session_part = f"session: {session_cost} ({session_in}→{session_out} tok)"

        return Text(f"{last_part} | {session_part}")
