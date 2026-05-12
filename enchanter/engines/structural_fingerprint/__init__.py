"""structural-fingerprint — triple-axis schema fingerprinting engine.

Port of the TS naga adapter (naga.adapter.ts). Implements N1 shape-hash,
N2 TF token-signature (Jaccard drift), and N3 naming-convention fingerprinting
at trust-gate.  Fail-closed on N1/N3 structural drift; fail-open (degraded)
on N2-only drift.

Additional multi-algorithm path: StructuralFingerprintStore exposes a
corpus-level TF-IDF cosine similarity API for fingerprint distance queries.

Algorithms:
  tfidf.py      — tokenize, compute_tfidf, cosine_similarity (stdlib only)
  levenshtein.py — levenshtein, levenshtein_ratio (O(m*n) DP)
  store.py      — StructuralFingerprintStore (corpus + TF-IDF cache)
  adapter.py    — StructuralFingerprint engine (PluginAdapter)
"""

from .adapter import StructuralFingerprint, adapter, TripleAxisFingerprint
from .store import StructuralFingerprintStore
from .tfidf import tokenize, compute_tfidf, cosine_similarity
from .levenshtein import levenshtein, levenshtein_ratio

__all__ = [
    "StructuralFingerprint",
    "StructuralFingerprintStore",
    "TripleAxisFingerprint",
    "adapter",
    "tokenize",
    "compute_tfidf",
    "cosine_similarity",
    "levenshtein",
    "levenshtein_ratio",
]
