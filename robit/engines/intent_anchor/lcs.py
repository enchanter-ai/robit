"""Pure LCS DP — no engine logic.

Implements Hunt-Szymanski O(m*n) Longest Common Subsequence using a
two-row rolling DP to keep space at O(n).  No numpy, stdlib only.

Public API
----------
lcs_length(a, b) -> int
    Raw LCS length of two token sequences.
lcs_ratio(a, b) -> float
    LCS length / max(len(a), len(b)), in [0, 1].
    Both-empty returns 1.0 (identical empty sequences).
    One-empty returns 0.0 (nothing in common).
"""

from __future__ import annotations


def lcs_length(a: list[str], b: list[str]) -> int:
    """Return the LCS length of *a* and *b* via rolling two-row DP.

    Space: O(n).  Time: O(m * n).
    """
    m = len(a)
    n = len(b)
    if m == 0 or n == 0:
        return 0

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        # swap buffers — reuse prev as next curr
        prev, curr = curr, prev
        # reset new curr
        for k in range(n + 1):
            curr[k] = 0

    return prev[n]


def lcs_ratio(a: list[str], b: list[str]) -> float:
    """Return LCS similarity ratio in [0, 1].

    Definition: lcs_length(a, b) / max(len(a), len(b)).
    Special cases:
      - Both empty  → 1.0  (identical empty sequences)
      - One empty   → 0.0  (nothing in common)
    """
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return lcs_length(a, b) / max(len(a), len(b))
