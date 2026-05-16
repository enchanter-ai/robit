"""Jaccard similarity on filename path tokens.

Pure function — no state, no I/O.  Port of the `jaccardSim` function in
`src/plugins/sylph.adapter.ts` § "W2 Boundary Segmentation".
"""

from __future__ import annotations

import re

# Split on path separators, dots, underscores, dashes — identical to the TS
# tokenizer: `p.split(/[/\\._-]+/).filter((t) => t.length > 0)`.
_TOKEN_RE = re.compile(r"[/\\._\-]+")


def _tokenize(path: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_RE.split(path) if t)


def jaccard_similarity(path_a: str, path_b: str) -> float:
    """Return Jaccard similarity between the token sets of two file paths.

    Tokens are produced by splitting on ``/``, ``\\``, ``.``, ``_``, ``-``.

    Returns 1.0 when both paths produce an empty token set (both are empty
    or consist entirely of separators) — faithful to the TS edge case
    ``if (ta.size === 0 && tb.size === 0) return 1``.
    """
    ta = _tokenize(path_a)
    tb = _tokenize(path_b)

    if not ta and not tb:
        return 1.0

    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union > 0 else 0.0
