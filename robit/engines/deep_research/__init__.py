"""deep-research — E0 evidence-gathering pipeline.

6-phase pipeline: Decompose → Cast → Triangulate → Gap-fill → Synthesize → Verify.
Produces claims.json, sources.jsonl, trace.json in a configurable state_dir.

Public API:
    from robit.engines.deep_research import DeepResearch, adapter, run_pipeline, ResearchResult
"""

from .adapter import DeepResearch, adapter
from .pipeline import ResearchResult, run_pipeline

__all__ = [
    "DeepResearch",
    "ResearchResult",
    "adapter",
    "run_pipeline",
]
