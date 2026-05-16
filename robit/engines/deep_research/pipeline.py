"""deep_research.pipeline — the 6-phase deep research pipeline.

Entry point:
    result = await run_pipeline(topic, llm, tier_router, state_dir)

Phases:
    1. Decompose (Opus) — topic → sub-questions + seed queries
    2. Cast (Haiku × N, parallel) — fetchers → sources.jsonl
    3. Triangulate (Sonnet) — sources → claims with τ score
    4. Gap-fill (Opus decides, Haiku fetches) — if stop_recommended=False, fetch more
    5. Synthesize (Opus) — write claims.json
    6. Verify (Haiku) — every claim traces to sources; F02 guard

Returns:
    ResearchResult(verdict, claims_path, sources_path, trace_path, triangulation_score)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robit.llm import LlmClient
from robit.runtime.tier_router import TierRouter
from robit.engines.deep_research.artifacts import (
    Source,
    write_sources,
    write_trace,
)
from robit.engines.deep_research.phases.decompose import run_decompose
from robit.engines.deep_research.phases.cast import run_cast
from robit.engines.deep_research.phases.triangulate import run_triangulate
from robit.engines.deep_research.phases.gap_fill import run_gap_fill
from robit.engines.deep_research.phases.synthesize import run_synthesize
from robit.engines.deep_research.phases.verify import run_verify

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3


@dataclass
class ResearchResult:
    """Returned by run_pipeline() — paths + summary metadata."""

    verdict: str           # "READY" | "PARTIAL" | "FAIL"
    claims_path: Path
    sources_path: Path
    trace_path: Path
    triangulation_score: float


async def run_pipeline(
    topic: str,
    llm: LlmClient,
    tier_router: TierRouter,
    state_dir: Path,
) -> ResearchResult:
    """Execute the 6-phase deep research pipeline.

    Parameters
    ----------
    topic:
        The research topic (slug or free text).
    llm:
        An LlmClient instance (real or mock).
    tier_router:
        Resolves task_class → model_id.
    state_dir:
        Directory where artifacts are written (claims.json, sources.jsonl, trace.json).

    Returns
    -------
    ResearchResult
        Verdict, paths to artifacts, and the final triangulation score.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    claims_path = state_dir / "claims.json"
    sources_path = state_dir / "sources.jsonl"
    trace_path = state_dir / "trace.json"

    # Resolve model IDs from the tier router
    opus_model = tier_router.route("orchestrator")
    sonnet_model = tier_router.route("executor")
    haiku_model = tier_router.route("validator")

    trace: dict[str, Any] = {
        "topic": topic,
        "phases": {},
        "verdict": None,
        "triangulation_score": None,
    }

    # -----------------------------------------------------------------------
    # Phase 1 — Decompose
    # -----------------------------------------------------------------------
    logger.info("Pipeline: Phase 1 Decompose — topic=%r", topic)
    decompose_result = await run_decompose(topic=topic, model_id=opus_model, llm=llm)
    sub_questions: list[dict[str, Any]] = decompose_result.get("sub_questions", [])
    trace["phases"]["phase1"] = decompose_result

    # -----------------------------------------------------------------------
    # Phase 2 — Cast (first round)
    # -----------------------------------------------------------------------
    logger.info("Pipeline: Phase 2 Cast (round 1) — %d sub-questions", len(sub_questions))
    sources: list[Source] = await run_cast(
        sub_questions=sub_questions,
        model_id=haiku_model,
        llm=llm,
        existing_source_count=0,
    )
    write_sources(sources_path, sources)
    trace["phases"]["phase2_round1"] = {"source_count": len(sources)}

    # -----------------------------------------------------------------------
    # Triangulate → Gap-fill loop (Phases 3 + 4)
    # -----------------------------------------------------------------------
    triangulator_output: dict[str, Any] = {}
    round_num = 1
    prior_claim_count = 0

    while round_num <= MAX_ROUNDS:
        # Phase 3 — Triangulate
        logger.info("Pipeline: Phase 3 Triangulate (round %d)", round_num)
        triangulator_output = await run_triangulate(
            sources=sources,
            sub_questions=sub_questions,
            round_num=round_num,
            prior_claim_count=prior_claim_count,
            model_id=sonnet_model,
            llm=llm,
        )
        trace["phases"][f"phase3_round{round_num}"] = triangulator_output

        current_claim_count = len(triangulator_output.get("claims", []))
        stop = triangulator_output.get("stop_recommended", True)

        if stop or round_num >= MAX_ROUNDS:
            break

        # Phase 4 — Gap-fill
        logger.info("Pipeline: Phase 4 Gap-fill (round %d)", round_num)
        new_sources = await run_gap_fill(
            triangulator_output=triangulator_output,
            sub_questions=sub_questions,
            existing_sources=sources,
            model_id=haiku_model,
            llm=llm,
            round_num=round_num,
        )
        trace["phases"][f"phase4_round{round_num}"] = {
            "new_sources": len(new_sources) if new_sources else 0,
            "stop_reason": "stop_recommended" if stop else "gap_fill_ran",
        }

        if new_sources is None:
            # Gap-fill decided to stop
            break

        # Append new sources and continue
        sources.extend(new_sources)
        write_sources(sources_path, sources)
        prior_claim_count = current_claim_count
        round_num += 1

    # -----------------------------------------------------------------------
    # Phase 5 — Synthesize
    # -----------------------------------------------------------------------
    logger.info("Pipeline: Phase 5 Synthesize")
    claims_doc = run_synthesize(
        topic=topic,
        triangulator_output=triangulator_output,
        sub_questions=sub_questions,
        sources=sources,
        claims_path=claims_path,
    )
    trace["phases"]["phase5"] = {
        "claims_count": len(claims_doc.claims),
        "triangulation_score": claims_doc.triangulation_score,
        "initial_verdict": claims_doc.verdict,
    }

    # -----------------------------------------------------------------------
    # Phase 6 — Verify
    # -----------------------------------------------------------------------
    logger.info("Pipeline: Phase 6 Verify")
    verify_result = await run_verify(
        doc=claims_doc,
        sources=sources,
        claims_path=claims_path,
        model_id=haiku_model,
        llm=llm,
    )
    trace["phases"]["phase6"] = verify_result

    # Final verdict
    verify_passed = bool(verify_result.get("verify_passed", False))
    tau = claims_doc.triangulation_score
    contradictions = triangulator_output.get("unresolved_contradictions", [])

    if not verify_passed:
        verdict = "FAIL"
    elif tau >= 0.85 and not contradictions:
        verdict = "READY"
    else:
        verdict = "PARTIAL"

    # Update claims_doc verdict after verification
    claims_doc.verdict = verdict
    from robit.engines.deep_research.artifacts import write_claims
    write_claims(claims_path, claims_doc)

    trace["verdict"] = verdict
    trace["triangulation_score"] = tau
    write_trace(trace_path, trace)

    logger.info(
        "Pipeline: complete — verdict=%s τ=%.3f claims=%d sources=%d",
        verdict,
        tau,
        len(claims_doc.claims),
        len(sources),
    )

    return ResearchResult(
        verdict=verdict,
        claims_path=claims_path,
        sources_path=sources_path,
        trace_path=trace_path,
        triangulation_score=tau,
    )
