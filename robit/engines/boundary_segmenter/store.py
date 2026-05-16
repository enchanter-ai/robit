"""Per-instance mutable state for the W2 boundary segmenter.

Port of the module-level ``_clusters`` array plus ``recordEdit``,
``getOpenClusters``, and ``closeIdleClusters`` in
``src/plugins/sylph.adapter.ts``.

Key difference from the TS original: state is per-instance (on
``ClusterStore``) rather than module-level globals, so each engine instance
gets its own isolated state — required for per-request isolation in tests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .jaccard import jaccard_similarity

# [author judgment] 5-minute active-edit window for same-cluster grouping.
CLUSTER_WINDOW_MS: int = 5 * 60 * 1000
# [author judgment] 10-minute idle gap before a cluster is considered closed.
CLUSTER_IDLE_MS: int = 10 * 60 * 1000
# [author judgment] Jaccard similarity threshold for co-clustering.
JACCARD_THRESHOLD: float = 0.4


def _new_cluster_id() -> str:
    return f"boundary-cluster-{uuid.uuid4().hex[:8]}"


@dataclass
class Cluster:
    """A single edit cluster.

    Fields mirror the TS interface:
        id: str
        files: list[str]
        last_edit_ts: int   (ms since epoch)
        closed: bool
    """

    id: str
    files: list[str]
    last_edit_ts: int
    closed: bool = False


class ClusterStore:
    """Per-instance mutable state: a list of clusters.

    Methods mirror the TS free functions that operated on the module-level
    ``_clusters`` array.
    """

    def __init__(self) -> None:
        self._clusters: list[Cluster] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_edit(self, file_path: str, ts: int) -> None:
        """Assign *file_path* to the best matching open cluster or open a new one.

        Matching criteria (faithful to TS ``recordEdit``):
        - Cluster must be open (not closed).
        - ``ts - cluster.last_edit_ts <= CLUSTER_WINDOW_MS`` (5-min window).
        - ``max(jaccard(f, file_path) for f in cluster.files) >= JACCARD_THRESHOLD``.

        Among eligible clusters the one with the highest max-Jaccard score wins.
        Ties are broken by iteration order (first match wins), consistent with
        the TS ``maxSim > bestSim`` strict-greater comparison.
        """
        best_cluster: Cluster | None = None
        best_sim: float = -1.0

        for c in self._clusters:
            if c.closed:
                continue
            if ts - c.last_edit_ts > CLUSTER_WINDOW_MS:
                continue
            max_sim = max(jaccard_similarity(f, file_path) for f in c.files)
            if max_sim > JACCARD_THRESHOLD and max_sim > best_sim:
                best_sim = max_sim
                best_cluster = c

        if best_cluster is not None:
            if file_path not in best_cluster.files:
                best_cluster.files.append(file_path)
            best_cluster.last_edit_ts = ts
        else:
            self._clusters.append(
                Cluster(
                    id=_new_cluster_id(),
                    files=[file_path],
                    last_edit_ts=ts,
                    closed=False,
                )
            )

    def close_idle(self, now: int) -> list[Cluster]:
        """Close clusters idle for >= CLUSTER_IDLE_MS.

        Returns the list of clusters that were just closed (for derived-event
        emission).  Faithful to TS ``closeIdleClusters``.
        """
        closed: list[Cluster] = []
        for c in self._clusters:
            if not c.closed and now - c.last_edit_ts >= CLUSTER_IDLE_MS:
                c.closed = True
                closed.append(c)
        return closed

    def open_clusters(self) -> list[Cluster]:
        """Return all clusters that are not yet closed."""
        return [c for c in self._clusters if not c.closed]
