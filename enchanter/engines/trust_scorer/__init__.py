"""trust-scorer — Beta-Bernoulli per-(server_id, tool_name) trust engine.

Port of the TS crow adapter (crow.adapter.ts).  Advisory plugin at
trust-gate; emits crow.trust.scored on every call and crow.review.ordered
when the posterior mean drops below 0.5 after ≥ 3 observations.

Prior: Beta(1, 1) — uniform, cold-start mean = 0.5.
"""

from .adapter import TrustScorer, adapter
from .store import TrustStore

__all__ = [
    "TrustScorer",
    "TrustStore",
    "adapter",
]
