"""Phase 4 — Gap-fill (Opus decides, Haiku fetches).

Reads the triangulator's stop_recommended. If False, generates new seed queries
targeting coverage_gaps and unresolved_contradictions, then triggers another
Phase 2 + Phase 3 round.

Stop conditions (any one):
  - stop_recommended = True
  - round >= 3 (max_rounds cap)
  - saturation_delta < 0.1
"""

from __future__ import annotations

import logging
from typing import Any

from enchanter.llm import LlmClient
from enchanter.engines.deep_research.artifacts import Source
from enchanter.engines.deep_research.phases.cast import run_cast

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3


def generate_gap_queries(
    coverage_gaps: list[str],
    unresolved_contradictions: list[dict[str, Any]],
    sub_questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate 2-3 new seed queries for each uncovered sub-question.

    Returns a list of synthetic sub_question dicts (id, question, seed_queries)
    suitable for Phase 2's run_cast().
    """
    # Build set of gap sq IDs
    gap_sq_ids: set[str] = set()
    for gap in coverage_gaps:
        if isinstance(gap, str):
            gap_sq_ids.add(gap)
        elif isinstance(gap, dict):
            gap_sq_ids.add(gap.get("sq_id", ""))

    # Also add sub_questions whose text appears in contradictions
    contradiction_keywords: list[str] = []
    for contradiction in unresolved_contradictions:
        desc = contradiction.get("description", "")
        if desc:
            contradiction_keywords.append(desc[:80])

    # Build gap sub_question stubs
    gap_sub_questions: list[dict[str, Any]] = []
    for sq in sub_questions:
        sq_id = sq.get("id", "")
        if sq_id not in gap_sq_ids:
            continue
        question = sq.get("question", "")
        # Synthesize queries: "latest <question>" and "evidence <question>"
        gap_queries = [
            f"latest research {question}",
            f"evidence findings {question}",
            f"sources {question}",
        ]
        gap_sub_questions.append({
            "id": sq_id,
            "question": question,
            "acceptance": sq.get("acceptance", ""),
            "seed_queries": gap_queries[:3],
        })

    # If no specific gap sqs, create generic contradiction resolution queries
    if not gap_sub_questions and contradiction_keywords:
        for i, kw in enumerate(contradiction_keywords[:2]):
            gap_sub_questions.append({
                "id": f"gap_fill_{i + 1}",
                "question": kw,
                "acceptance": "Resolves the identified contradiction",
                "seed_queries": [
                    f"resolve {kw}",
                    f"clarify {kw}",
                ],
            })

    return gap_sub_questions


async def run_gap_fill(
    triangulator_output: dict[str, Any],
    sub_questions: list[dict[str, Any]],
    existing_sources: list[Source],
    model_id: str,
    llm: LlmClient,
    round_num: int,
) -> list[Source] | None:
    """Run Phase 4: decide whether to continue and fetch new sources if so.

    Returns:
      - A list of new Source objects to append (if gap-fill ran)
      - None if stop_recommended (caller should skip to Phase 5)
    """
    stop_recommended = triangulator_output.get("stop_recommended", True)
    saturation_delta = float(triangulator_output.get("saturation_delta", 0.0))

    # Stop conditions
    if stop_recommended:
        logger.info("Phase 4 Gap-fill: stop_recommended=True, skipping gap-fill")
        return None

    if round_num >= MAX_ROUNDS:
        logger.info(
            "Phase 4 Gap-fill: reached max rounds (%d), stopping", MAX_ROUNDS
        )
        return None

    if saturation_delta < 0.1 and round_num > 1:
        logger.info(
            "Phase 4 Gap-fill: saturation_delta=%.3f < 0.1, stopping", saturation_delta
        )
        return None

    # Generate gap queries
    coverage_gaps = triangulator_output.get("coverage_gaps", [])
    unresolved_contradictions = triangulator_output.get("unresolved_contradictions", [])

    gap_sub_questions = generate_gap_queries(
        coverage_gaps=coverage_gaps,
        unresolved_contradictions=unresolved_contradictions,
        sub_questions=sub_questions,
    )

    if not gap_sub_questions:
        logger.info("Phase 4 Gap-fill: no gap queries generated, proceeding to synthesize")
        return None

    logger.info(
        "Phase 4 Gap-fill: round=%d generating %d gap sub-questions",
        round_num,
        len(gap_sub_questions),
    )

    new_sources = await run_cast(
        sub_questions=gap_sub_questions,
        model_id=model_id,
        llm=llm,
        existing_source_count=len(existing_sources),
    )

    logger.info("Phase 4 Gap-fill: fetched %d new sources", len(new_sources))
    return new_sources
