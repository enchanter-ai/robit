"""TokenRunwayStore — per-instance rolling window + drift detection + runway forecast.

Two algorithms faithful to emu.adapter.ts:

A1 Markov Drift Detection
    Two named patterns, checked on every post-response observation:
    - read-loop:  3+ consecutive observations share the same tool_call_id.
    - edit-revert: ABAB pattern across the last 4 observations.
    Only one pattern fires per observation (read-loop takes priority).

A2 Linear Runway Forecasting
    Over the last FORECAST_WINDOW observations, compute mean and σ of
    total_tokens (input + output).  Project remaining_budget / mean as the
    point estimate; apply 95% CI via error propagation (1.96·σ/mean·R_hat).
    Returns None when fewer than 2 observations exist (cold start).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


# ── Constants (faithful to emu.adapter.ts author-judgment comments) ───────────

# Window cap: enough history across a long session, avoids unbounded growth.
WINDOW_CAP: int = 100

# Forecast window: last N observations; recent velocity matters more than start.
FORECAST_WINDOW: int = 10

# Default remaining budget matching emu README example C_max = 200 000.
DEFAULT_REMAINING_BUDGET: int = 200_000


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TokenObservation:
    ts: int                 # ms since epoch
    input_tokens: int
    output_tokens: int
    tool_call_id: str


@dataclass(frozen=True)
class RunwayForecast:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    mean_tokens_per_call: float
    sigma: float
    observation_count: int


# ── Store ──────────────────────────────────────────────────────────────────────

class TokenRunwayStore:
    """Mutable state for one TokenRunway engine instance.

    All mutation goes through ``record_observation``; read-only accessors
    return a snapshot so callers cannot mutate internal state.

    State is per-instance — never module-level — so tests construct a fresh
    store per test case without teardown.
    """

    def __init__(self, remaining_budget: int = DEFAULT_REMAINING_BUDGET) -> None:
        self._window: deque[TokenObservation] = deque(maxlen=WINDOW_CAP)
        self.remaining_budget: int = remaining_budget

    # ── Mutation ──────────────────────────────────────────────────────────────

    def record_observation(
        self,
        input_tokens: int,
        output_tokens: int,
        tool_call_id: str,
        ts: int,
    ) -> None:
        """Append one observation; eviction is handled by deque(maxlen=WINDOW_CAP)."""
        self._window.append(
            TokenObservation(
                ts=ts,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_call_id=tool_call_id,
            )
        )

    # ── Drift detection (A1) ──────────────────────────────────────────────────

    def detect_read_loop(self) -> bool:
        """True when the last 3 observations share the same tool_call_id.

        Threshold θ=3 matches README §A1 and TS source detectReadLoop().
        """
        if len(self._window) < 3:
            return False
        tail = list(self._window)[-3:]
        first_id = tail[0].tool_call_id
        return all(o.tool_call_id == first_id for o in tail)

    def detect_edit_revert(self) -> bool:
        """True when the last 4 observations form an ABAB pattern.

        Minimum 4 observations (2 full ABAB cycles) is the tightest window
        that unambiguously identifies the pattern — matches TS detectEditRevert().
        """
        if len(self._window) < 4:
            return False
        tail = list(self._window)[-4:]
        a = tail[0].tool_call_id
        b = tail[1].tool_call_id
        return (
            a != b
            and tail[2].tool_call_id == a
            and tail[3].tool_call_id == b
        )

    def drift_pattern(self) -> Optional[str]:
        """Return the name of the active drift pattern, or None.

        Priority: read-loop > edit-revert (single pattern per event per TS source).
        """
        if self.detect_read_loop():
            return "read-loop"
        if self.detect_edit_revert():
            return "edit-revert"
        return None

    # ── Runway forecast (A2) ──────────────────────────────────────────────────

    def compute_runway(self) -> Optional[RunwayForecast]:
        """A2 Linear Runway Forecast.

        Formula (README §A2):
            R_hat = remaining_budget / t̄_w
            95% CI: R_hat ± 1.96 · (σ_t / t̄_w) · R_hat   (error propagation on ratio)

        Returns None when < 2 observations exist (cold start — insufficient data
        for a meaningful mean or CI), matching TS computeRunway() guard.
        """
        # Take the last FORECAST_WINDOW observations.
        tail = list(self._window)[-FORECAST_WINDOW:]
        if len(tail) < 2:
            return None  # cold start

        totals = [o.input_tokens + o.output_tokens for o in tail]
        n = len(totals)
        mean = sum(totals) / n
        if mean == 0:
            return None  # avoid division by zero

        # Population variance (matches TS: totals.reduce / totals.length — no Bessel).
        variance = sum((t - mean) ** 2 for t in totals) / n
        sigma = math.sqrt(variance)

        point_estimate = self.remaining_budget / mean
        half_width = 1.96 * (sigma / mean) * point_estimate

        return RunwayForecast(
            point_estimate=point_estimate,
            ci_lower=max(0.0, point_estimate - half_width),
            ci_upper=point_estimate + half_width,
            mean_tokens_per_call=mean,
            sigma=sigma,
            observation_count=n,
        )

    # ── Read-only accessors ───────────────────────────────────────────────────

    def observation_count(self) -> int:
        return len(self._window)

    def observations(self) -> list[TokenObservation]:
        """Return a shallow copy of the current window."""
        return list(self._window)
