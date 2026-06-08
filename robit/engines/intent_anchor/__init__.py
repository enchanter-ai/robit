"""intent-anchor — LCS drift detection + HMM forward labelling + EMA posterior.

Port of the TS djinn adapter (djinn.adapter.ts v0.3.1+).

Phases: anchor (set session intent), post-session (detect drift).
Advisory (required=False), fail-open.

Emits:
  intent-anchor.anchor.set    — when the first anchor is captured
  intent-anchor.drift.detected — when LCS ratio < 0.3 vs. the anchor
"""

from .adapter import IntentAnchor, LlmCall, adapter
from .store import IntentAnchorStore
from .lcs import lcs_length, lcs_ratio
from .hmm import HMM

__all__ = [
    "IntentAnchor",
    "IntentAnchorStore",
    "HMM",
    "LlmCall",
    "adapter",
    "lcs_length",
    "lcs_ratio",
]
