"""Phase 2 — Cast (parallel Haiku fetchers).

One LLM call per seed query, all dispatched concurrently via asyncio.gather.
Aggregates fetcher JSON returns into a list of Source objects with assigned IDs.

Stop condition: every seed query has returned (or errored with "unfetchable").
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from enchanter.llm import CompletionRequest, LlmClient, Message
from enchanter.engines.deep_research.artifacts import Source

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a web research fetcher. Fetch sources for one search query and return \
structured findings as JSON. Every judgment is a boolean test.
"""

_USER_TMPL = """\
Fetch sources for this query and extract structured findings.

Query: {query}
Sub-question: {sub_question}

Run WebSearch with the query, take top 3 results, apply source-type and topicality filters, \
fetch each surviving page, and extract claim+quote findings.

Return ONLY a JSON array (no preamble, no markdown fences):
[
  {{
    "url": "<url>",
    "date": "<YYYY-MM-DD|YYYY-MM|YYYY|null>",
    "source_type": "official|third-party|community|paper|other",
    "findings": [
      {{"claim": "<paraphrase>", "quote": "<verbatim sentence ≤200 chars>"}}
    ]
  }}
]

Unfetchable pages: {{"url": "<url>", "error": "unfetchable"}}
"""


async def _fetch_one(
    query: str,
    sub_question: str,
    sq_id: str,
    model_id: str,
    llm: LlmClient,
    start_id: int,
) -> list[Source]:
    """Call the fetcher LLM for a single query; return parsed Source objects."""
    prompt = _USER_TMPL.format(query=query, sub_question=sub_question)
    req = CompletionRequest(
        model=model_id,
        messages=[Message(role="user", content=prompt)],
        system=_SYSTEM,
        max_tokens=1024,
        temperature=0.0,
    )
    try:
        resp = await llm.complete(req)
    except Exception as exc:
        logger.warning("Phase 2 Cast: fetcher call failed for query %r: %s", query, exc)
        return []

    text = resp.text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        raw_list = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Phase 2 Cast: JSON parse failed for query %r: %s\nRaw: %r", query, exc, text
        )
        return []

    if not isinstance(raw_list, list):
        logger.warning("Phase 2 Cast: fetcher returned non-list for query %r", query)
        return []

    # Normalize and assign IDs
    sources: list[Source] = []
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            continue
        url = item.get("url", f"unknown-{start_id + i}")
        error = item.get("error")
        sources.append(
            Source(
                id=f"S{start_id + i}",
                url=url,
                date=item.get("date"),
                source_type=item.get("source_type", "other"),
                findings=item.get("findings", []),
                error=error,
                sub_question_id=sq_id,
            )
        )
    return sources


async def run_cast(
    sub_questions: list[dict[str, Any]],
    model_id: str,
    llm: LlmClient,
    existing_source_count: int = 0,
) -> list[Source]:
    """Run Phase 2: fan out one fetcher per seed query in parallel.

    Returns a flat list of Source objects with sequential IDs starting after
    existing_source_count (so gap-fill sources get fresh IDs).
    """
    # Build (query, sub_question_text, sq_id) triples
    tasks: list[tuple[str, str, str]] = []
    for sq in sub_questions:
        sq_id = sq.get("id", "")
        sq_text = sq.get("question", "")
        for query in sq.get("seed_queries", []):
            tasks.append((query, sq_text, sq_id))

    if not tasks:
        logger.warning("Phase 2 Cast: no seed queries to dispatch")
        return []

    logger.info("Phase 2 Cast: dispatching %d fetchers in parallel", len(tasks))

    # Assign start_ids so IDs don't collide across rounds
    id_counter = existing_source_count + 1
    coros = []
    id_starts: list[int] = []
    for query, sq_text, sq_id in tasks:
        coros.append(
            _fetch_one(
                query=query,
                sub_question=sq_text,
                sq_id=sq_id,
                model_id=model_id,
                llm=llm,
                start_id=id_counter,
            )
        )
        id_starts.append(id_counter)
        id_counter += 10  # reserve 10 slots per fetcher (max ~10 pages returned)

    results = await asyncio.gather(*coros, return_exceptions=True)

    all_sources: list[Source] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Phase 2 Cast: fetcher %d raised: %s", i, result)
        elif isinstance(result, list):
            all_sources.extend(result)

    # Re-number sequentially to avoid gaps
    for idx, src in enumerate(all_sources):
        src.id = f"S{existing_source_count + idx + 1}"

    logger.info("Phase 2 Cast: collected %d sources", len(all_sources))
    return all_sources
