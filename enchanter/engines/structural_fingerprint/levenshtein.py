"""Levenshtein edit distance — classical O(m*n) DP, stdlib only.

Costs: insertion=1, deletion=1, substitution=1.
"""

from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between *a* and *b*.

    Algorithm: full O(m*n) DP matrix, all costs = 1.

    Edge cases:
        levenshtein("", "")   == 0
        levenshtein("", "abc") == 3
        levenshtein("abc", "") == 3
        levenshtein("a", "a")  == 0
    """
    m, n = len(a), len(b)

    # Early-exit optimisations.
    if a == b:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    # Use two rolling rows instead of the full matrix to keep space O(n).
    # prev[j] = edit distance between a[:i] and b[:j]
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]          # no cost — match
            else:
                curr[j] = 1 + min(
                    prev[j],       # deletion  (remove a[i-1])
                    curr[j - 1],   # insertion (insert b[j-1])
                    prev[j - 1],   # substitution
                )
        prev, curr = curr, prev

    return prev[n]


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalised similarity: 1 - dist / max(len(a), len(b)).

    Returns 1.0 for identical strings.
    Returns 1.0 when both strings are empty (by convention — max=0 guard).
    Returns a value in [0.0, 1.0].
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max_len
