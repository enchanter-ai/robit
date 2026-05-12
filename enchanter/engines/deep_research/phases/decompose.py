"""Phase 1 — Decompose (Opus, inline).

Expands raw topic into sub-questions and seed queries.

Stop condition: every sub-question has an acceptance criterion and ≥ 2 seed queries.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from enchanter.llm import CompletionRequest, LlmClient, Message

logger = logging.getLogger(__name__)

_TOPIC_TYPES = (
    "model-capability",
    "api-behavior",
    "benchmark",
    "library-usage",
    "competitive-landscape",
    "deprecation-status",
    "other",
)

_SYSTEM = """\
You are a research decomposition specialist. Your job is to break a topic into \
sub-questions and search queries for web research.
"""

_USER_TMPL = """\
Decompose this research topic: {topic}

1. Classify topic_type as one of: {topic_types}
2. Produce 3-7 sub-questions, each with a one-line acceptance_criterion.
3. Produce 2-5 seed_queries per sub-question.

Return ONLY this JSON object with no preamble, no markdown fences:
{{
  "topic_type": "<type>",
  "sub_questions": [
    {{"id": "sq1", "question": "<question>", "acceptance": "<criterion>",
      "seed_queries": ["<query1>", "<query2>"]}}
  ]
}}
"""


async def run_decompose(topic: str, model_id: str, llm: LlmClient) -> dict[str, Any]:
    """Run Phase 1: decompose topic into sub-questions and seed queries.

    Returns the parsed decomposition dict with keys:
      topic_type, sub_questions (list of {id, question, acceptance, seed_queries})
    """
    prompt = _USER_TMPL.format(topic=topic, topic_types=", ".join(_TOPIC_TYPES))
    req = CompletionRequest(
        model=model_id,
        messages=[Message(role="user", content=prompt)],
        system=_SYSTEM,
        max_tokens=2048,
        temperature=0.0,
    )
    resp = await llm.complete(req)
    text = resp.text.strip()

    # Strip markdown fences if the model added them despite instructions
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Phase 1 Decompose: JSON parse failed: %s\nRaw: %r", exc, text)
        raise ValueError(f"Phase 1 Decompose: invalid JSON from model: {exc}") from exc

    # Validate stop condition: every sub-question has acceptance + ≥ 2 seed queries
    sub_questions = result.get("sub_questions", [])
    if not sub_questions:
        raise ValueError("Phase 1 Decompose: no sub_questions returned")

    for sq in sub_questions:
        if not sq.get("acceptance"):
            raise ValueError(
                f"Phase 1 Decompose: sub_question {sq.get('id')} missing acceptance criterion"
            )
        if len(sq.get("seed_queries", [])) < 2:
            raise ValueError(
                f"Phase 1 Decompose: sub_question {sq.get('id')} has fewer than 2 seed queries"
            )

    logger.info(
        "Phase 1 Decompose: %d sub-questions, topic_type=%s",
        len(sub_questions),
        result.get("topic_type"),
    )
    return result
