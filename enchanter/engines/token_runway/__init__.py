"""token-runway engine — port of emu.adapter.ts (phase_1.emu).

Advisory plugin at post-response + pre-dispatch.

  post-response: records token usage, detects A1 drift patterns
                 (read-loop, edit-revert); emits emu.drift.pattern.
  pre-dispatch:  computes A2 runway forecast; emits emu.runway.forecast.

Required: False (fail-open). Budget tier: med-or-higher.
"""

from .adapter import TokenRunway, adapter
from .store import (
    DEFAULT_REMAINING_BUDGET,
    FORECAST_WINDOW,
    WINDOW_CAP,
    RunwayForecast,
    TokenObservation,
    TokenRunwayStore,
)

__all__ = [
    "DEFAULT_REMAINING_BUDGET",
    "FORECAST_WINDOW",
    "RunwayForecast",
    "TokenObservation",
    "TokenRunway",
    "TokenRunwayStore",
    "WINDOW_CAP",
    "adapter",
]
