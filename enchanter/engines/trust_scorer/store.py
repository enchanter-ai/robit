"""Beta-Bernoulli per-key trust store.

Encapsulates all mutable posterior state.  Each TrustStore instance is
independent — never share across engine instances.

Key shape: (server_id, tool_name)  — matching the TS crow adapter.
Prior: Beta(1, 1) — uniform, posterior mean = 0.5 at cold start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# The key is a (server_id, tool_name) pair.
_Key = Tuple[str, str]


@dataclass
class _Posterior:
    alpha: float
    beta: float


@dataclass
class TrustStore:
    """Per-key Beta-Bernoulli posterior store.

    Args:
        prior_alpha: α of the uniform prior (default 1 → Beta(1,1)).
        prior_beta:  β of the uniform prior (default 1 → Beta(1,1)).
    """

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    _posteriors: dict[_Key, _Posterior] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, key: _Key) -> _Posterior:
        if key not in self._posteriors:
            self._posteriors[key] = _Posterior(
                alpha=self.prior_alpha,
                beta=self.prior_beta,
            )
        return self._posteriors[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_success(self, key: _Key) -> None:
        """Increment α for *key* (Bernoulli success observation)."""
        p = self._get_or_create(key)
        p.alpha += 1.0

    def record_failure(self, key: _Key) -> None:
        """Increment β for *key* (Bernoulli failure observation)."""
        p = self._get_or_create(key)
        p.beta += 1.0

    def score(self, key: _Key) -> float:
        """Return the posterior mean α/(α+β) for *key*.

        Returns the prior mean when the key has never been observed.
        """
        p = self._get_or_create(key)
        return p.alpha / (p.alpha + p.beta)

    def observation_count(self, key: _Key) -> int:
        """Number of observations recorded for *key* (excludes prior counts)."""
        if key not in self._posteriors:
            return 0
        p = self._posteriors[key]
        # Each update_posterior call adds 1 to either alpha or beta.
        # Prior contributes (prior_alpha + prior_beta); subtract to get n.
        return int(round(p.alpha + p.beta - self.prior_alpha - self.prior_beta))

    def alpha_beta(self, key: _Key) -> tuple[float, float]:
        """Return the raw (alpha, beta) for *key* — useful in tests."""
        p = self._get_or_create(key)
        return p.alpha, p.beta

    def reset(self) -> None:
        """Clear all posteriors — used in test teardown / singleton resets."""
        self._posteriors.clear()
