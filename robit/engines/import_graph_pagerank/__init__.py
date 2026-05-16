"""import-graph-pagerank — Python import graph extraction + Tarjan SCC + PageRank.

Port of the TS gorgon adapter (gorgon.adapter.ts + gorgon/tarjan.ts +
gorgon/python-extractor.ts).

Phases: post-session, cross-session (compute + emit snapshot).
Advisory (required=False), fail-open.

Emits:
  gorgon.snapshot.ready    — top-N hotspots + Tarjan cycle list
  gorgon.hotspot.changed   — when a dirty file shifts rank by >= 3
"""

from .adapter import ImportGraphPagerank, adapter
from .store import ImportGraphStore
from .python_extractor import extract_imports
from .tarjan import tarjan_scc
from .pagerank import pagerank

__all__ = [
    "ImportGraphPagerank",
    "ImportGraphStore",
    "adapter",
    "extract_imports",
    "tarjan_scc",
    "pagerank",
]
