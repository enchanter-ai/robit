"""Phase 5 — Synthesize (Opus, inline).

Writes claims.json from the final triangulator output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from enchanter.engines.deep_research.artifacts import (
    Claim,
    ClaimsDoc,
    Source,
    SubQuestion,
    today_str,
    write_claims,
)

logger = logging.getLogger(__name__)


def _compute_verdict(
    triangulator_output: dict[str, Any],
    verify_passed: bool = True,
) -> str:
    """Compute verdict per SKILL.md criteria.

    READY:   verify_passed=True AND τ ≥ 0.85 AND no unresolved contradictions
    PARTIAL: verify_passed=True AND (τ < 0.85 OR contradictions remain)
    FAIL:    verify_passed=False
    """
    if not verify_passed:
        return "FAIL"
    tau = float(triangulator_output.get("tau", 0.0))
    contradictions = triangulator_output.get("unresolved_contradictions", [])
    if tau >= 0.85 and not contradictions:
        return "READY"
    return "PARTIAL"


def run_synthesize(
    topic: str,
    triangulator_output: dict[str, Any],
    sub_questions: list[dict[str, Any]],
    sources: list[Source],
    claims_path: Path,
) -> ClaimsDoc:
    """Run Phase 5: write claims.json from the final triangulator output.

    Returns the ClaimsDoc written to disk.
    """
    tau = float(triangulator_output.get("tau", 0.0))
    verdict = _compute_verdict(triangulator_output, verify_passed=True)

    # Map triangulator claims to Claim dataclasses
    claims: list[Claim] = []
    for c in triangulator_output.get("claims", []):
        claims.append(
            Claim(
                id=c.get("id", ""),
                claim=c.get("claim", ""),
                sq=c.get("sq", ""),
                supporting=c.get("supporting", []),
                independent_count=c.get("independent_count", 0),
                confidence=c.get("confidence", "low"),
                contradicts=c.get("contradicts"),
            )
        )

    # Map sub_questions to SubQuestion dataclasses
    sqs: list[SubQuestion] = []
    for sq in sub_questions:
        sqs.append(
            SubQuestion(
                id=sq.get("id", ""),
                question=sq.get("question", ""),
                acceptance=sq.get("acceptance", ""),
            )
        )

    today = today_str()
    doc = ClaimsDoc(
        topic=topic,
        generated=today,
        freshness=today,
        triangulation_score=tau,
        verdict=verdict,
        source_count=len(sources),
        claims=claims,
        unresolved_contradictions=triangulator_output.get("unresolved_contradictions", []),
        coverage_gaps=triangulator_output.get("coverage_gaps", []),
        sub_questions=sqs,
    )

    write_claims(claims_path, doc)
    logger.info(
        "Phase 5 Synthesize: wrote %d claims, τ=%.3f, verdict=%s → %s",
        len(claims),
        tau,
        verdict,
        claims_path,
    )
    return doc
