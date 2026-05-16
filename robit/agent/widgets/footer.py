"""robit.agent.widgets.footer — status footer widget.

Layout (single row, dim text):

    model: <model_id> | session: <short_id> | <cost_slot>

The cost slot is a mount point for sibling I's ``CostTicker``. If that
widget can't be imported, the footer falls back to a static ``cost: $0.00``
string and exposes :meth:`update_cost` so the app can refresh it from
:class:`TurnComplete.usage` events.

The footer is intentionally a ``Container`` (not a plain ``Static``) so it
can host the live cost-ticker child widget. The text portion lives in a
``Static`` child labelled ``#status``.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static


# Try sibling I's cost ticker. Fall back to a stub.
try:  # pragma: no cover - import branch
    from .cost import CostTicker  # type: ignore[attr-defined]

    _COST_TICKER_AVAILABLE = True
except Exception:  # noqa: BLE001
    CostTicker = None  # type: ignore[assignment, misc]
    _COST_TICKER_AVAILABLE = False


def _short_session(session_id: str) -> str:
    """First 8 hex chars — long enough to disambiguate in practice."""
    return session_id[:8] if session_id else "(none)"


class FooterWidget(Container):
    """Single-row status footer.

    Parameters
    ----------
    model:
        Current model id, e.g. ``claude-sonnet-4-5``.
    session_id:
        Hex session id; the footer shows the first 8 chars.
    """

    DEFAULT_CSS = """
    FooterWidget {
        dock: bottom;
        height: 1;
        background: $panel;
    }
    FooterWidget #status {
        width: auto;
        color: $text-muted;
    }
    FooterWidget #cost-slot {
        width: auto;
        color: $text-muted;
        margin-left: 2;
    }
    """

    def __init__(self, model: str, session_id: str) -> None:
        super().__init__(id="footer")
        self._model = model
        self._session_id = session_id
        self._cost_widget = None  # filled in on_mount if sibling I shipped

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(
                f"model: {self._model} | session: {_short_session(self._session_id)} |",
                id="status",
            )
            if _COST_TICKER_AVAILABLE and CostTicker is not None:
                ticker = CostTicker()  # type: ignore[call-arg]
                self._cost_widget = ticker
                yield ticker
            else:
                yield Static("cost: $0.00", id="cost-slot")

    # ----- public API -----------------------------------------------------

    def update_model(self, model: str) -> None:
        """Refresh the model label after a ``/model`` switch."""
        self._model = model
        try:
            self.query_one("#status", Static).update(
                f"model: {self._model} | session: {_short_session(self._session_id)} |"
            )
        except Exception:  # noqa: BLE001
            # Not yet mounted — the new value will be picked up at compose.
            pass

    def update_cost(self, usage) -> None:
        """Forward a :class:`CanonicalUsage` to the cost ticker if present.

        Sibling I's ``CostTicker.add_turn(input_tokens, output_tokens, model)``
        is the contract. We do not know which model produced the tokens at
        the widget level, so we forward the footer's current model.
        """
        if self._cost_widget is not None:
            add = getattr(self._cost_widget, "add_turn", None)
            if callable(add):
                try:
                    add(
                        getattr(usage, "input_tokens", 0),
                        getattr(usage, "output_tokens", 0),
                        self._model,
                    )
                    return
                except Exception:  # noqa: BLE001
                    pass
        # Fallback path: dumb token total in the stub slot.
        try:
            slot = self.query_one("#cost-slot", Static)
            total = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
            slot.update(f"tokens: {total}")
        except Exception:  # noqa: BLE001
            pass


__all__ = ["FooterWidget"]
