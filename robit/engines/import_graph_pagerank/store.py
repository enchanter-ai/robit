"""store — per-instance cumulative import-graph state.

Accumulates import edges across multiple ``add_file`` calls, then computes
PageRank centrality on demand.

Public API
----------
ImportGraphStore()
    .add_file(path: str, source: str) -> None
        Parse imports from *source* and add edges ``path → imported_module``
        to the cumulative graph.  Relative imports (``__relative__``) are
        added as an edge target but treated as low-signal — they still appear
        in the graph so the topology is complete.
    .compute_centrality() -> dict[str, float]
        Run PageRank on the current graph.  Returns {} if the graph is empty.
    .top_n(k: int) -> list[tuple[str, float]]
        Return the top *k* nodes by PageRank score as ``(node, score)`` pairs,
        sorted descending.  Returns all nodes if k > len(graph).
    .graph -> dict[str, list[str]]
        Read-only view of the current cumulative graph.
    .reset() -> None
        Clear all accumulated state — test teardown helper.
"""

from __future__ import annotations

from .python_extractor import extract_imports
from .pagerank import pagerank


class ImportGraphStore:
    """Accumulates a file-level import graph and computes PageRank centrality.

    One instance per adapter instance — never shared across sessions.
    """

    def __init__(self) -> None:
        # file path → list of imported module names / file paths
        self._graph: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Graph accumulation
    # ------------------------------------------------------------------

    def add_file(self, path: str, source: str) -> None:
        """Extract imports from *source* and record edges for *path*.

        If *path* has been seen before, its edges are replaced (the latest
        source wins — matches the TS ``setSourceMap`` override semantics).
        """
        imports = extract_imports(source)
        self._graph[path] = imports

    # ------------------------------------------------------------------
    # Centrality
    # ------------------------------------------------------------------

    def compute_centrality(self) -> dict[str, float]:
        """Run PageRank on the current graph.  Returns {} when graph is empty."""
        if not self._graph:
            return {}
        return pagerank(self._graph)

    def top_n(self, k: int) -> list[tuple[str, float]]:
        """Return the top *k* nodes by PageRank score, descending.

        Parameters
        ----------
        k:
            Number of entries to return.  Clamped to the total number of
            distinct nodes — never raises IndexError.
        """
        scores = self.compute_centrality()
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------
    # Read-only graph access
    # ------------------------------------------------------------------

    @property
    def graph(self) -> dict[str, list[str]]:
        """Shallow copy of the current graph (read-only view)."""
        return dict(self._graph)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated state — useful in tests."""
        self._graph.clear()
