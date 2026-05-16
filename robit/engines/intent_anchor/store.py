"""Per-session intent-anchor state store.

Holds:
  • the session anchor (intent string + token list + set_at ms)
  • the per-session HMM instance (incremental forward state)
  • the EMA posterior (exponential moving average of LCS ratios)

One IntentAnchorStore per IntentAnchor adapter instance.  Never shared.

EMA recurrence: posterior = α * new_ratio + (1 - α) * posterior
  α = 0.05  (matches the TS comment "α=0.05 or similar"; slow-tracking,
  one observation barely moves the posterior — conservative for advisory signals)
  initial posterior = 1.0  (assume on-task at session start)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple

from .hmm import HMM, HmmStep
from .lcs import lcs_ratio


# ---------------------------------------------------------------------------
# Tokeniser — mirrors djinn.adapter.ts / c1_lcs.py normalize()
# ---------------------------------------------------------------------------

import re as _re

_TOKEN_RE = _re.compile(r"\w+")
_STOPWORDS = frozenset(
    {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "be"}
)


def tokenize(text: str) -> list[str]:
    """Lower-case, alpha-only tokens with stopword removal."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# SessionAnchor
# ---------------------------------------------------------------------------

class SessionAnchor(NamedTuple):
    intent: str
    tokens: list[str]
    set_at: int  # ms since epoch


# ---------------------------------------------------------------------------
# EMA constants
# ---------------------------------------------------------------------------

_EMA_ALPHA: float = 0.05
_EMA_INITIAL: float = 1.0  # assume on-task at session start


# ---------------------------------------------------------------------------
# IntentAnchorStore
# ---------------------------------------------------------------------------

@dataclass
class IntentAnchorStore:
    """All mutable per-session state for the intent-anchor engine."""

    # Populated on first anchor phase
    _anchor: SessionAnchor | None = field(default=None, init=False, repr=False)
    # Per-session HMM (forward recursion)
    _hmm: HMM = field(default_factory=HMM, init=False, repr=False)
    # EMA posterior over LCS ratios
    _ema_posterior: float = field(default=_EMA_INITIAL, init=False, repr=False)

    # ------------------------------------------------------------------
    # Anchor management
    # ------------------------------------------------------------------

    @property
    def anchor(self) -> SessionAnchor | None:
        return self._anchor

    @property
    def has_anchor(self) -> bool:
        return self._anchor is not None

    def set_anchor(self, intent: str, ts_ms: int | None = None) -> SessionAnchor:
        """Set the session anchor.  Immutable once set (first call wins)."""
        if self._anchor is not None:
            return self._anchor
        ts = ts_ms if ts_ms is not None else int(time.time() * 1000)
        tokens = tokenize(intent)
        self._anchor = SessionAnchor(intent=intent, tokens=tokens, set_at=ts)
        return self._anchor

    # ------------------------------------------------------------------
    # Observation recording
    # ------------------------------------------------------------------

    def record_observation(self, prompt: str) -> tuple[float, HmmStep, float]:
        """Record a new prompt observation against the anchor.

        Returns (lcs_ratio_value, hmm_step, ema_posterior).
        Raises RuntimeError if called before an anchor is set.
        """
        if self._anchor is None:
            raise RuntimeError("record_observation called before set_anchor")

        current_tokens = tokenize(prompt)
        ratio = lcs_ratio(self._anchor.tokens, current_tokens)

        # Update HMM (forward step)
        hmm_step = self._hmm.update(ratio)

        # Update EMA
        self._ema_posterior = _EMA_ALPHA * ratio + (1.0 - _EMA_ALPHA) * self._ema_posterior

        return ratio, hmm_step, self._ema_posterior

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def ema_posterior(self) -> float:
        return self._ema_posterior

    @property
    def hmm(self) -> HMM:
        return self._hmm

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all state — equivalent to a fresh session."""
        self._anchor = None
        self._hmm = HMM()
        self._ema_posterior = _EMA_INITIAL
