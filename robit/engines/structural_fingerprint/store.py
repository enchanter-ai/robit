"""Structural fingerprint store — per-instance state for N2 corpus and fingerprint cache.

Each StructuralFingerprintStore instance is fully isolated (no module-level
singletons), matching the test-isolation contract established by TrustStore.

The store holds:
  - A growing corpus of (qualified_name, token_list) documents used for IDF.
  - A cache of the last computed TF-IDF vectors (invalidated on new doc add).
  - N3 naming-convention entries keyed by qualified_name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tfidf import compute_tfidf, cosine_similarity, tokenize


@dataclass
class _Entry:
    qualified_name: str
    tokens: list[str]


@dataclass
class StructuralFingerprintStore:
    """Mutable corpus store for N2 TF-IDF fingerprinting.

    Typical call sequence:
        store.add_document(qualified_name, description_text)
        sim = store.similarity(qualified_name_a, qualified_name_b)
    """

    # Doc corpus: order-preserving insertion list.
    _docs: list[_Entry] = field(default_factory=list, init=False, repr=False)
    # Index for O(1) lookup by qualified_name.
    _index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    # Cached TF-IDF vectors; None means the cache is dirty.
    _tfidf_cache: list[dict[str, float]] | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Corpus mutation
    # ------------------------------------------------------------------

    def add_document(self, qualified_name: str, text: str) -> None:
        """Add or replace the document for *qualified_name*.

        If the name already exists the old entry is updated in-place so that
        the order of the corpus does not change (important for IDF stability
        across incremental additions).
        """
        tokens = tokenize(text)
        if qualified_name in self._index:
            idx = self._index[qualified_name]
            self._docs[idx] = _Entry(qualified_name=qualified_name, tokens=tokens)
        else:
            self._index[qualified_name] = len(self._docs)
            self._docs.append(_Entry(qualified_name=qualified_name, tokens=tokens))
        # Invalidate cache.
        self._tfidf_cache = None

    def has(self, qualified_name: str) -> bool:
        return qualified_name in self._index

    def document_count(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------
    # TF-IDF vectors
    # ------------------------------------------------------------------

    def _ensure_cache(self) -> list[dict[str, float]]:
        """Recompute TF-IDF vectors for the full corpus if dirty."""
        if self._tfidf_cache is None:
            corpus = [entry.tokens for entry in self._docs]
            self._tfidf_cache = compute_tfidf(corpus)
        return self._tfidf_cache

    def vector(self, qualified_name: str) -> dict[str, float]:
        """Return the TF-IDF weight vector for *qualified_name*.

        Raises KeyError if the document is not in the store.
        """
        if qualified_name not in self._index:
            raise KeyError(qualified_name)
        idx = self._index[qualified_name]
        return self._ensure_cache()[idx]

    # ------------------------------------------------------------------
    # Similarity queries
    # ------------------------------------------------------------------

    def similarity(self, name_a: str, name_b: str) -> float:
        """Cosine similarity between the N2 TF-IDF vectors of two documents.

        Returns 0.0 if either document is not in the store.
        """
        if name_a not in self._index or name_b not in self._index:
            return 0.0
        va = self.vector(name_a)
        vb = self.vector(name_b)
        return cosine_similarity(va, vb)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state — used in test teardown."""
        self._docs.clear()
        self._index.clear()
        self._tfidf_cache = None
