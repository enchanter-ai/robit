"""boundary-segmenter — port of sylph W2 Jaccard sliding-window edit clustering.

Advisory plugin at ``post-session``.  Accumulates file-edit events into
Jaccard-similarity clusters; on post-session sweep, closes idle clusters and
emits a ``boundary-segmenter.boundary.closed`` derived event per closure.
"""

from .adapter import BoundarySegmenter, adapter
from .jaccard import jaccard_similarity
from .store import CLUSTER_IDLE_MS, CLUSTER_WINDOW_MS, JACCARD_THRESHOLD, Cluster, ClusterStore

__all__ = [
    "CLUSTER_IDLE_MS",
    "CLUSTER_WINDOW_MS",
    "JACCARD_THRESHOLD",
    "BoundarySegmenter",
    "Cluster",
    "ClusterStore",
    "adapter",
    "jaccard_similarity",
]
