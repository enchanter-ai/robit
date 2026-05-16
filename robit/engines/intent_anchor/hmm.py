"""3-state HMM with forward-algorithm inference — pure Python, stdlib only.

Mirrors the TS IntentHmm (plugins/djinn/hmm.ts) faithfully:
  • States: ON_TASK, SIDEQUEST, LOST  (ordered; index 0, 1, 2)
  • Observation buckets: high / mid / low — derived from an LCS ratio
  • Inference: incremental forward algorithm (NOT Viterbi), so the per-turn
    posterior is a normalised distribution over states.  Most-likely state
    is argmax of that posterior.  This is faithful to the TS comment:
    "forward variant picked over Viterbi: only the latest-turn label +
    a confidence number is needed".

HMM class also exposes a `decode()` method that runs Viterbi over a
complete observation sequence (as required by the port spec), but the
adapter uses the incremental `update()` path.

Log-probabilities are used in Viterbi to avoid underflow.

Default model parameters are copied verbatim from hmm.ts DEFAULT_*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

HmmStateLabel = Literal["ON_TASK", "SIDEQUEST", "LOST"]
ObservationBucket = Literal["high", "mid", "low"]

STATES: tuple[HmmStateLabel, ...] = ("ON_TASK", "SIDEQUEST", "LOST")
_N = len(STATES)  # 3

# ---------------------------------------------------------------------------
# Default parameters — verbatim from hmm.ts
# ---------------------------------------------------------------------------

# Row-stochastic 3×3: DEFAULT_TRANSITIONS[from_state][to_state]
DEFAULT_TRANSITIONS: list[list[float]] = [
    [0.85, 0.149, 0.001],  # from ON_TASK
    [0.40, 0.55, 0.05],    # from SIDEQUEST
    [0.05, 0.15, 0.80],    # from LOST
]

# Emission probabilities: DEFAULT_EMISSIONS[state_idx][bucket]
# Ordered as (high, mid, low) to map to indices 0, 1, 2
DEFAULT_EMISSIONS: list[list[float]] = [
    [0.75, 0.20, 0.05],  # ON_TASK
    [0.15, 0.45, 0.40],  # SIDEQUEST
    [0.02, 0.18, 0.80],  # LOST
]

DEFAULT_PRIOR: list[float] = [0.90, 0.08, 0.02]

DEFAULT_HIGH_CUTOFF = 0.6
DEFAULT_MID_CUTOFF = 0.3

_OBS_TO_IDX: dict[ObservationBucket, int] = {"high": 0, "mid": 1, "low": 2}


# ---------------------------------------------------------------------------
# Snapshot (mirrors HmmStateSnapshot in hmm-store.ts)
# ---------------------------------------------------------------------------

HMM_STATE_VERSION = 1


@dataclass
class HmmStateSnapshot:
    """Serialisable forward state — config excluded (same rationale as TS)."""
    version: int
    posterior: list[float]   # length 3, sums to ~1
    initialized: bool


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class HmmStep:
    state: HmmStateLabel
    posterior: dict[HmmStateLabel, float]   # {"ON_TASK": p, "SIDEQUEST": p, "LOST": p}
    observation: ObservationBucket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucketize(similarity: float, high_cutoff: float, mid_cutoff: float) -> ObservationBucket:
    if similarity >= high_cutoff:
        return "high"
    if similarity >= mid_cutoff:
        return "mid"
    return "low"


def _normalize(v: list[float]) -> list[float]:
    total = sum(v)
    if total <= 0.0:
        return [1.0 / _N] * _N
    return [x / total for x in v]


def _argmax(v: list[float]) -> int:
    best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


def _step_to_hmm_step(
    alpha: list[float],
    obs: ObservationBucket,
) -> HmmStep:
    idx = _argmax(alpha)
    return HmmStep(
        state=STATES[idx],
        posterior={
            "ON_TASK": alpha[0],
            "SIDEQUEST": alpha[1],
            "LOST": alpha[2],
        },
        observation=obs,
    )


# ---------------------------------------------------------------------------
# HMM — incremental forward recursion + Viterbi decode
# ---------------------------------------------------------------------------

class HMM:
    """3-state HMM with configurable transition/emission tables.

    Supports two inference modes:
      decode(observations)  — Viterbi over a full sequence (pure math, no state mutation)
      update(similarity)    — incremental forward step (mutates internal alpha)

    Args:
        states:           sequence of state labels (default: STATES)
        transition_prob:  row-stochastic 3×3 list-of-lists
        emission_prob:    per-state emission list-of-lists, rows = states, cols = obs buckets
        prior:            initial state distribution (length = len(states))
        high_cutoff:      LCS ratio ≥ this → 'high' bucket
        mid_cutoff:       LCS ratio ≥ this → 'mid' bucket, else 'low'
    """

    def __init__(
        self,
        states: tuple[HmmStateLabel, ...] = STATES,
        transition_prob: list[list[float]] | None = None,
        emission_prob: list[list[float]] | None = None,
        prior: list[float] | None = None,
        high_cutoff: float = DEFAULT_HIGH_CUTOFF,
        mid_cutoff: float = DEFAULT_MID_CUTOFF,
    ) -> None:
        self._states = states
        self._n = len(states)
        self._A = transition_prob if transition_prob is not None else DEFAULT_TRANSITIONS
        self._B = emission_prob if emission_prob is not None else DEFAULT_EMISSIONS
        self._prior = prior if prior is not None else DEFAULT_PRIOR
        self._high_cutoff = high_cutoff
        self._mid_cutoff = mid_cutoff

        # Forward state
        self._alpha: list[float] = list(self._prior)
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to the prior — equivalent to a fresh session."""
        self._alpha = list(self._prior)
        self._initialized = False

    # ------------------------------------------------------------------
    # Incremental forward update (used by the adapter)
    # ------------------------------------------------------------------

    def current(self) -> HmmStep:
        """Current posterior without folding a new observation in."""
        return _step_to_hmm_step(self._alpha, "high")  # neutral placeholder

    def update(self, similarity: float) -> HmmStep:
        """Fold a new LCS similarity value in; return updated posterior + state."""
        obs = _bucketize(similarity, self._high_cutoff, self._mid_cutoff)
        obs_idx = _OBS_TO_IDX[obs]

        if not self._initialized:
            # First update: use prior as the predicted distribution.
            predicted = list(self._prior)
            self._initialized = True
        else:
            # Predict: alpha_t = alpha_{t-1} * A
            predicted = [0.0] * self._n
            for j in range(self._n):
                s = 0.0
                for i in range(self._n):
                    s += self._alpha[i] * self._A[i][j]
                predicted[j] = s

        # Update: multiply by emission, normalise
        updated = [predicted[j] * self._B[j][obs_idx] for j in range(self._n)]
        updated = _normalize(updated)
        self._alpha = updated

        return _step_to_hmm_step(self._alpha, obs)

    # ------------------------------------------------------------------
    # Viterbi decode — pure, no state mutation
    # ------------------------------------------------------------------

    def decode(self, observations: list[ObservationBucket]) -> list[HmmStateLabel]:
        """Viterbi decoder over a complete observation sequence.

        Uses log-probabilities to avoid underflow.
        Returns the most-likely state sequence (same length as *observations*).
        Empty input → empty list.
        """
        T = len(observations)
        if T == 0:
            return []

        # Log transition + emission matrices
        _NEG_INF = float("-inf")
        log_A = [[math.log(p) if p > 0 else _NEG_INF for p in row] for row in self._A]
        log_B = [[math.log(p) if p > 0 else _NEG_INF for p in row] for row in self._B]
        log_pi = [math.log(p) if p > 0 else _NEG_INF for p in self._prior]

        # Initialise
        obs_idx_0 = _OBS_TO_IDX[observations[0]]
        delta = [log_pi[i] + log_B[i][obs_idx_0] for i in range(self._n)]
        psi: list[list[int]] = [[0] * self._n]  # backpointer; unused at t=0

        for t in range(1, T):
            obs_idx = _OBS_TO_IDX[observations[t]]
            new_delta = [_NEG_INF] * self._n
            new_psi = [0] * self._n
            for j in range(self._n):
                best_val = _NEG_INF
                best_i = 0
                for i in range(self._n):
                    val = delta[i] + log_A[i][j]
                    if val > best_val:
                        best_val = val
                        best_i = i
                new_delta[j] = best_val + log_B[j][obs_idx]
                new_psi[j] = best_i
            delta = new_delta
            psi.append(new_psi)

        # Backtrack
        path: list[int] = [0] * T
        path[T - 1] = _argmax(delta)
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]

        return [STATES[s] for s in path]

    # ------------------------------------------------------------------
    # Serialisation (mirrors IntentHmm.serialize / fromSnapshot)
    # ------------------------------------------------------------------

    def serialize(self) -> HmmStateSnapshot:
        return HmmStateSnapshot(
            version=HMM_STATE_VERSION,
            posterior=list(self._alpha),
            initialized=self._initialized,
        )

    @classmethod
    def from_snapshot(
        cls,
        snap: HmmStateSnapshot,
        **kwargs: object,
    ) -> "HMM | None":
        """Re-hydrate from a snapshot.  Returns None on version mismatch."""
        if snap.version != HMM_STATE_VERSION:
            return None
        if len(snap.posterior) != _N:
            return None
        obj = cls(**kwargs)  # type: ignore[arg-type]
        obj._alpha = list(snap.posterior)
        obj._initialized = snap.initialized
        return obj
