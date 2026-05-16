"""Tests for the Wave 15.2 / Agent I live cost ticker.

The widget is plain enough to instantiate without a running Textual app —
we exercise it by calling ``add_turn`` / ``reset`` and inspecting the
``rich.text.Text`` returned by ``render()``.
"""

from __future__ import annotations

import pytest

from robit.agent.widgets.cost import CostTicker
from robit.proxy.events.cost_ledger import _compute_cents


def _text(ticker: CostTicker) -> str:
    """Return the rendered plain string of the ticker."""
    return ticker.render().plain


def test_fresh_ticker_renders_zero_state() -> None:
    """A new ticker reports zero cost and zero tokens, no model yet."""
    ticker = CostTicker()
    rendered = _text(ticker)

    assert (
        rendered
        == "last: $0.0000 (0→0 tok) | session: $0.0000 (0→0 tok)"
    )


def test_add_turn_updates_last_and_session() -> None:
    """``add_turn`` populates last-turn fields and starts session totals."""
    ticker = CostTicker()
    ticker.add_turn(100, 50, "claude-3-5-sonnet-20241022")

    # Independently compute the expected cost via the proxy helper so we
    # don't drift from the canonical pricing source.
    expected_cents = _compute_cents(100, 50, "claude-3-5-sonnet-20241022")
    assert expected_cents > 0  # sanity — pricing table is live

    assert ticker.turn_tokens_in == 100
    assert ticker.turn_tokens_out == 50
    assert ticker.turn_cents == expected_cents
    assert ticker.session_tokens_in == 100
    assert ticker.session_tokens_out == 50
    assert ticker.session_cents == expected_cents
    assert ticker.last_model == "claude-3-5-sonnet-20241022"

    rendered = _text(ticker)
    assert "last:" in rendered
    assert "session:" in rendered
    # Non-zero cost surfaces — guards against an accidental zero state.
    assert "$0.0000" not in rendered
    assert "claude-3-5-sonnet-20241022" in rendered


def test_multiple_turns_accumulate_session_last_reflects_most_recent() -> None:
    """Session counters sum across turns; ``last_*`` only carries the latest."""
    ticker = CostTicker()
    ticker.add_turn(100, 50, "claude-3-5-sonnet-20241022")
    ticker.add_turn(200, 80, "claude-3-5-sonnet-20241022")

    # Last-turn fields show ONLY the second call.
    assert ticker.turn_tokens_in == 200
    assert ticker.turn_tokens_out == 80
    # Session sums both calls.
    assert ticker.session_tokens_in == 300
    assert ticker.session_tokens_out == 130

    # Sum check against the proxy helper.
    expected_session_cents = _compute_cents(
        100, 50, "claude-3-5-sonnet-20241022"
    ) + _compute_cents(200, 80, "claude-3-5-sonnet-20241022")
    assert ticker.session_cents == expected_session_cents


def test_different_model_updates_last_model() -> None:
    """A turn on a new model overwrites ``last_model`` immediately."""
    ticker = CostTicker()
    ticker.add_turn(100, 50, "claude-3-5-sonnet-20241022")
    ticker.add_turn(200, 80, "gpt-4o-mini")

    assert ticker.last_model == "gpt-4o-mini"
    assert "gpt-4o-mini" in _text(ticker)


def test_reset_zeros_session_but_keeps_last_model() -> None:
    """``reset`` clears session totals only; last-turn + model persist."""
    ticker = CostTicker()
    ticker.add_turn(100, 50, "claude-3-5-sonnet-20241022")
    ticker.add_turn(200, 80, "claude-3-5-sonnet-20241022")

    ticker.reset()

    assert ticker.session_tokens_in == 0
    assert ticker.session_tokens_out == 0
    assert ticker.session_cents == 0
    # Last-turn fields are deliberately untouched so the user can verify
    # the reset fired against a still-visible reference turn.
    assert ticker.turn_tokens_in == 200
    assert ticker.turn_tokens_out == 80
    assert ticker.last_model == "claude-3-5-sonnet-20241022"


def test_unknown_model_falls_back_without_crash() -> None:
    """Models missing from the pricing table use the default rate."""
    ticker = CostTicker()
    # Force a non-trivial usage so the default rate produces non-zero cents.
    ticker.add_turn(10_000, 5_000, "fictitious-model-9000")

    assert ticker.last_model == "fictitious-model-9000"
    # Default rate is conservative middle estimate — must be > 0.
    assert ticker.turn_cents > 0
    assert ticker.session_cents == ticker.turn_cents


def test_token_count_formatting_uses_thousands_separators() -> None:
    """Six-figure token counts render as ``1,234,567`` not ``1234567``."""
    ticker = CostTicker()
    ticker.add_turn(1_234_567, 0, "claude-3-5-sonnet-20241022")

    rendered = _text(ticker)
    assert "1,234,567" in rendered
    assert "1234567" not in rendered


def test_initial_token_seeds_only_seed_session() -> None:
    """``initial_tokens_in/out`` carry over from a prior session for the
    session counters but do not pollute the last-turn fields."""
    ticker = CostTicker(initial_tokens_in=500, initial_tokens_out=200)

    assert ticker.session_tokens_in == 500
    assert ticker.session_tokens_out == 200
    assert ticker.turn_tokens_in == 0
    assert ticker.turn_tokens_out == 0
    # No cost ascribed to the seed (we don't know which model produced it).
    assert ticker.session_cents == 0


def test_dollar_formatting_switches_precision_at_one_dollar() -> None:
    """< $1.00 uses 4dp, ≥ $1.00 uses 2dp."""
    ticker = CostTicker()
    # Tiny turn → sub-cent rounds up via _compute_cents but still < $1.
    ticker.add_turn(10, 5, "claude-3-5-haiku-20241022")
    assert "$0." in _text(ticker)

    # Big turn → ≥ $1.00. Use opus prices to cross the threshold quickly:
    # opus output is 7500 cents per million → 100k output tokens = 750c = $7.50.
    big = CostTicker()
    big.add_turn(0, 100_000, "claude-3-opus-20240229")
    rendered = _text(big)
    # 2dp format means there's no '$N.NNNN' pattern in session/last.
    # Cheaper to assert the rendered cost contains a 2dp dollar amount.
    assert "$7.50" in rendered


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
