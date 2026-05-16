"""TF-IDF pure math module — no external deps, stdlib math only.

Tokenizer, TF-IDF computation, and cosine similarity for the structural
fingerprint engine (naga N2 axis port).

Stopword list mirrors naga.adapter.ts exactly so fingerprints are
cross-runtime comparable.
"""

from __future__ import annotations

import math
import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Stop-word list — exact mirror of the TS adapter's STOPWORDS set
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "should", "could", "may", "might", "can", "this", "that", "it", "its",
    "which", "who", "what", "when", "where", "how", "not", "no", "if", "as",
    "than", "then", "so", "up", "out", "about", "into", "through", "during",
    "each", "all", "any", "both", "few", "more", "most", "other", "such",
})

# Tokenise on any run of non-alphanumeric characters (matches `[\s\W]+` in TS).
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop stop-words and single-char tokens.

    Mirrors the TS tokeniser:
        .toLowerCase().split(/[\\s\\W]+/).filter(t => t.length > 1 && !STOPWORDS.has(t))
    """
    lowered = text.lower()
    raw_tokens = _TOKEN_RE.split(lowered)
    return [t for t in raw_tokens if len(t) > 1 and t not in _STOPWORDS]


def compute_tfidf(docs: Sequence[list[str]]) -> list[dict[str, float]]:
    """Compute TF-IDF weight vectors for a corpus.

    TF  = count(term, doc) / len(doc)          (raw relative frequency)
    IDF = log(1 + N / (1 + df(term)))           (smoothed, natural log)
    Score = TF * IDF

    Smoothing (+1 in numerator and denominator of IDF) prevents zero-division
    and softens the penalty for terms appearing in every document.

    Args:
        docs: list of token lists, one list per document.

    Returns:
        list of {term: tfidf_score} dicts, parallel to *docs*.
        Empty documents return an empty dict.
    """
    n = len(docs)
    if n == 0:
        return []

    # Build document-frequency map.
    df: dict[str, int] = {}
    for tokens in docs:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    result: list[dict[str, float]] = []
    for tokens in docs:
        if not tokens:
            result.append({})
            continue

        # Term frequency (relative).
        tf: dict[str, float] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0.0) + 1.0
        doc_len = len(tokens)
        for term in tf:
            tf[term] /= doc_len

        # TF-IDF.
        tfidf: dict[str, float] = {}
        for term, freq in tf.items():
            idf = math.log(1.0 + n / (1.0 + df.get(term, 0)))
            tfidf[term] = freq * idf

        result.append(tfidf)

    return result


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF weight vectors.

    Returns a value in [0.0, 1.0].
    Returns 1.0 when both vectors are empty (identical empty documents).
    Returns 0.0 when one vector is empty and the other is not.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    # Dot product — only iterate over the smaller map.
    dot = 0.0
    for term, weight in a.items():
        if term in b:
            dot += weight * b[term]

    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
