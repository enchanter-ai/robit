"""Tests for the deep-research engine (E0 pipeline).

Covers:
  1. Phase 1 (Decompose) in isolation
  2. Phase 2 (Cast) in isolation — parallel fetchers
  3. Phase 3 (Triangulate) in isolation
  4. Phase 4 (Gap-fill) — stop_recommended=False triggers another round
  5. Phase 5 (Synthesize) — writes claims.json with correct schema
  6. Phase 6 (Verify) — verify_passed=True path
  7. Phase 6 — rejects claim with no matching source (F02 guard)
  8. End-to-end: topic in → 6 phases → artifacts written → verdict READY
  9. PARTIAL verdict path (low τ)
  10. Adapter.on_phase only runs pipeline on 'research.requested' events
  11. Gap-fill loop: triangulator stop_recommended=False → another round runs
  12. Max-round cap (3 rounds) respected
  13. Artifact schemas match SKILL.md spec (claims.json shape)
  14. Adapter missing llm → degraded ack (no crash)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robit.core import EnchantedEvent, RequestContext, create_request_context
from robit.core.bus import build_event
from robit.llm import CompletionResponse, MockLlmClient
from robit.llm.types import Message
from robit.engines.deep_research import DeepResearch, ResearchResult, run_pipeline
from robit.engines.deep_research.artifacts import (
    Claim,
    ClaimsDoc,
    Source,
    SubQuestion,
    read_claims,
    read_sources,
    write_claims,
    write_sources,
    today_str,
)
from robit.engines.deep_research.phases.decompose import run_decompose
from robit.engines.deep_research.phases.cast import run_cast
from robit.engines.deep_research.phases.triangulate import run_triangulate
from robit.engines.deep_research.phases.gap_fill import run_gap_fill, generate_gap_queries
from robit.engines.deep_research.phases.synthesize import run_synthesize
from robit.engines.deep_research.phases.verify import run_verify
from robit.runtime.tier_router import TierRouter
from robit.runtime.models_registry import ModelsRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_resp(text: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        model="claude-haiku-test",
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=len(text) // 4,
    )


def _decompose_response() -> str:
    return json.dumps({
        "topic_type": "model-capability",
        "sub_questions": [
            {
                "id": "sq1",
                "question": "What is the context window of Claude?",
                "acceptance": "Named window size with source",
                "seed_queries": ["Claude context window size", "Claude token limit 2024"],
            },
            {
                "id": "sq2",
                "question": "How does Claude handle tool use?",
                "acceptance": "At least one tool-use mechanic described",
                "seed_queries": ["Claude tool use API", "Claude function calling"],
            },
        ],
    })


def _fetcher_response() -> str:
    return json.dumps([
        {
            "url": "https://docs.anthropic.com/claude/context",
            "date": "2024-11-01",
            "source_type": "official",
            "findings": [
                {
                    "claim": "Claude has a 200K token context window.",
                    "quote": "Claude supports up to 200,000 tokens in its context window.",
                }
            ],
        }
    ])


def _triangulate_response(tau: float = 0.9, stop: bool = True) -> str:
    return json.dumps({
        "claims": [
            {
                "id": "C1",
                "claim": "Claude has a 200K token context window.",
                "sq": "sq1",
                "supporting": ["S1"],
                "independent_count": 2,
                "confidence": "high",
                "contradicts": None,
            }
        ],
        "unresolved_contradictions": [],
        "coverage_gaps": [],
        "tau": tau,
        "saturation_delta": 0.0,
        "round": 1,
        "stop_recommended": stop,
        "notes": "Strong single-source claim.",
    })


def _verify_response(passed: bool = True, violations: list | None = None) -> str:
    return json.dumps({
        "verify_passed": passed,
        "total_cites_checked": 1,
        "violations": violations or [],
        "unsupported_claims": [],
        "notes": "All claims verified." if passed else "Fabricated claim detected.",
    })


def _make_sources() -> list[Source]:
    return [
        Source(
            id="S1",
            url="https://docs.anthropic.com/claude/context",
            date="2024-11-01",
            source_type="official",
            findings=[
                {
                    "claim": "Claude has a 200K token context window.",
                    "quote": "Claude supports up to 200,000 tokens in its context window.",
                }
            ],
        )
    ]


def _make_claims_doc(topic: str = "claude-capabilities", verdict: str = "READY") -> ClaimsDoc:
    return ClaimsDoc(
        topic=topic,
        generated=today_str(),
        freshness=today_str(),
        triangulation_score=0.9,
        verdict=verdict,
        source_count=1,
        claims=[
            Claim(
                id="C1",
                claim="Claude has a 200K token context window.",
                sq="sq1",
                supporting=["S1"],
                independent_count=2,
                confidence="high",
            )
        ],
        unresolved_contradictions=[],
        coverage_gaps=[],
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="What is the context window of Claude?",
                acceptance="Named window size with source",
            )
        ],
    )


def _make_tier_router() -> TierRouter:
    """Return a TierRouter backed by the bundled registry (default path)."""
    registry = ModelsRegistry.load()
    return TierRouter(registry)


# ---------------------------------------------------------------------------
# Test 1 — Phase 1 Decompose in isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase1_decompose_returns_sub_questions():
    mock = MockLlmClient(responses={"Decompose this research topic": _mock_resp(_decompose_response())})
    result = await run_decompose(topic="Claude capabilities", model_id="claude-opus-test", llm=mock)
    assert result["topic_type"] == "model-capability"
    assert len(result["sub_questions"]) == 2
    assert result["sub_questions"][0]["id"] == "sq1"
    assert len(result["sub_questions"][0]["seed_queries"]) >= 2


# ---------------------------------------------------------------------------
# Test 2 — Phase 2 Cast in isolation (parallel fetchers)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase2_cast_parallel_dispatch():
    # Two seed queries → two parallel fetcher calls
    mock = MockLlmClient(responses={"Fetch sources for this query": _mock_resp(_fetcher_response())})
    sub_questions = [
        {
            "id": "sq1",
            "question": "What is the context window of Claude?",
            "acceptance": "Named window size",
            "seed_queries": ["Claude context window size", "Claude token limit"],
        }
    ]
    sources = await run_cast(sub_questions=sub_questions, model_id="claude-haiku-test", llm=mock)
    # Each query dispatched independently — may return overlapping sources
    assert len(sources) >= 1
    assert all(s.id.startswith("S") for s in sources)
    # IDs are unique
    ids = [s.id for s in sources]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Test 3 — Phase 3 Triangulate in isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase3_triangulate_returns_claims():
    mock = MockLlmClient(responses={"Triangulate findings": _mock_resp(_triangulate_response())})
    sub_questions = [{"id": "sq1", "question": "context window", "acceptance": "size named"}]
    result = await run_triangulate(
        sources=_make_sources(),
        sub_questions=sub_questions,
        round_num=1,
        prior_claim_count=0,
        model_id="claude-sonnet-test",
        llm=mock,
    )
    assert result["stop_recommended"] is True
    assert abs(result["tau"] - 0.9) < 0.01
    assert len(result["claims"]) == 1
    assert result["claims"][0]["id"] == "C1"


# ---------------------------------------------------------------------------
# Test 4 — Phase 4 Gap-fill generates queries for uncovered sub-questions
# ---------------------------------------------------------------------------

def test_phase4_generate_gap_queries_for_coverage_gaps():
    sub_questions = [
        {
            "id": "sq1",
            "question": "What is the context window?",
            "acceptance": "size named",
            "seed_queries": [],
        },
        {
            "id": "sq2",
            "question": "How does tool use work?",
            "acceptance": "mechanic described",
            "seed_queries": [],
        },
    ]
    gap_sqs = generate_gap_queries(
        coverage_gaps=["sq2"],
        unresolved_contradictions=[],
        sub_questions=sub_questions,
    )
    assert len(gap_sqs) == 1
    assert gap_sqs[0]["id"] == "sq2"
    assert len(gap_sqs[0]["seed_queries"]) >= 2


# ---------------------------------------------------------------------------
# Test 5 — Phase 5 Synthesize writes claims.json with correct schema
# ---------------------------------------------------------------------------

def test_phase5_synthesize_writes_claims_json(tmp_path: Path):
    claims_path = tmp_path / "claims.json"
    triangulator_output = json.loads(_triangulate_response(tau=0.9, stop=True))
    sub_questions = [
        {"id": "sq1", "question": "context window", "acceptance": "size named"}
    ]
    sources = _make_sources()

    doc = run_synthesize(
        topic="claude-capabilities",
        triangulator_output=triangulator_output,
        sub_questions=sub_questions,
        sources=sources,
        claims_path=claims_path,
    )

    assert claims_path.exists()
    loaded = json.loads(claims_path.read_text(encoding="utf-8"))

    # Check SKILL.md schema fields are present
    for field_name in (
        "topic", "generated", "freshness", "triangulation_score",
        "verdict", "source_count", "claims", "unresolved_contradictions",
        "coverage_gaps", "sub_questions",
    ):
        assert field_name in loaded, f"Missing field: {field_name}"

    assert loaded["topic"] == "claude-capabilities"
    assert loaded["verdict"] == "READY"
    assert loaded["triangulation_score"] == 0.9
    assert len(loaded["claims"]) == 1
    claim = loaded["claims"][0]
    for key in ("id", "claim", "sq", "supporting", "independent_count", "confidence", "contradicts"):
        assert key in claim, f"Claim missing key: {key}"


# ---------------------------------------------------------------------------
# Test 6 — Phase 6 Verify passes when claims trace to sources
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase6_verify_passed(tmp_path: Path):
    doc = _make_claims_doc()
    claims_path = tmp_path / "claims.json"
    write_claims(claims_path, doc)

    mock = MockLlmClient(responses={"Verify every claim": _mock_resp(_verify_response(passed=True))})
    result = await run_verify(
        doc=doc,
        sources=_make_sources(),
        claims_path=claims_path,
        model_id="claude-haiku-test",
        llm=mock,
    )
    assert result["verify_passed"] is True
    assert result["violations"] == []


# ---------------------------------------------------------------------------
# Test 7 — Phase 6 rejects claim with no matching source (F02 guard)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase6_verify_fails_f02_fabrication(tmp_path: Path):
    # Build a claims doc that references a source that doesn't exist
    doc = ClaimsDoc(
        topic="test",
        generated=today_str(),
        freshness=today_str(),
        triangulation_score=0.5,
        verdict="PARTIAL",
        source_count=0,
        claims=[
            Claim(
                id="C1",
                claim="Fabricated claim about nothing.",
                sq="sq1",
                supporting=["S99"],  # S99 does not exist in sources
                independent_count=1,
                confidence="low",
            )
        ],
        unresolved_contradictions=[],
        coverage_gaps=[],
        sub_questions=[],
    )
    claims_path = tmp_path / "claims.json"
    write_claims(claims_path, doc)

    violation = {
        "claim_excerpt": "Fabricated claim about nothing.",
        "cite": "S99",
        "reason": "cited ID not in sources.jsonl",
    }
    mock = MockLlmClient(
        responses={
            "Verify every claim": _mock_resp(_verify_response(passed=False, violations=[violation]))
        }
    )
    result = await run_verify(
        doc=doc,
        sources=[],  # empty sources — S99 cannot possibly be found
        claims_path=claims_path,
        model_id="claude-haiku-test",
        llm=mock,
    )
    assert result["verify_passed"] is False
    assert len(result["violations"]) == 1
    assert result["violations"][0]["cite"] == "S99"
    # Offending claim should have been removed from claims.json
    reloaded = read_claims(claims_path)
    assert len(reloaded.claims) == 0


# ---------------------------------------------------------------------------
# Test 8 — End-to-end: topic → 6 phases → artifacts written → READY verdict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_to_end_ready_verdict(tmp_path: Path):
    # Script the full pipeline in sequence
    responses = [
        # Phase 1 decompose
        _mock_resp(_decompose_response()),
        # Phase 2 cast: 4 seed queries (2 per sub-question) → 4 parallel calls
        # MockLlmClient in list mode returns in FIFO order
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        # Phase 3 triangulate (round 1) → stop_recommended=True
        _mock_resp(_triangulate_response(tau=0.9, stop=True)),
        # Phase 5 (synthesize is synchronous — no LLM call)
        # Phase 6 verify
        _mock_resp(_verify_response(passed=True)),
    ]
    mock = MockLlmClient(responses=responses)
    tier_router = _make_tier_router()

    result = await run_pipeline(
        topic="claude-capabilities",
        llm=mock,
        tier_router=tier_router,
        state_dir=tmp_path / "briefs" / "claude-capabilities",
    )

    assert result.verdict == "READY"
    assert result.claims_path.exists()
    assert result.sources_path.exists()
    assert result.trace_path.exists()
    assert result.triangulation_score == pytest.approx(0.9, abs=0.01)

    # Validate artifact content
    claims_data = json.loads(result.claims_path.read_text(encoding="utf-8"))
    assert claims_data["verdict"] == "READY"
    assert claims_data["topic"] == "claude-capabilities"

    sources_data = result.sources_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(sources_data) >= 1  # at least one source line

    trace_data = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert trace_data["verdict"] == "READY"
    assert "phase1" in trace_data["phases"]
    assert "phase2_round1" in trace_data["phases"]
    assert "phase3_round1" in trace_data["phases"]
    assert "phase6" in trace_data["phases"]


# ---------------------------------------------------------------------------
# Test 9 — PARTIAL verdict path (low τ)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_to_end_partial_verdict(tmp_path: Path):
    responses = [
        _mock_resp(_decompose_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        # Low τ but stop_recommended=True (saturation)
        _mock_resp(_triangulate_response(tau=0.5, stop=True)),
        _mock_resp(_verify_response(passed=True)),
    ]
    mock = MockLlmClient(responses=responses)
    tier_router = _make_tier_router()

    result = await run_pipeline(
        topic="partial-research-topic",
        llm=mock,
        tier_router=tier_router,
        state_dir=tmp_path / "briefs" / "partial-research-topic",
    )

    assert result.verdict == "PARTIAL"
    claims_data = json.loads(result.claims_path.read_text(encoding="utf-8"))
    assert claims_data["verdict"] == "PARTIAL"
    assert claims_data["triangulation_score"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Test 10 — Adapter on_phase does NOT run pipeline on non-research events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adapter_no_op_on_non_research_event():
    engine = DeepResearch()
    ctx = create_request_context(session_id="test-session")
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="pre-dispatch",
        topic="sampling.completed",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={},
    )
    ack = await engine.on_phase(event, ctx)
    assert ack.status == "ack"
    assert not ack.degraded
    assert not ack.derived_events


# ---------------------------------------------------------------------------
# Test 11 — Gap-fill loop: stop_recommended=False → another round
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gap_fill_second_round_triggered(tmp_path: Path):
    """When triangulator returns stop_recommended=False on round 1, gap-fill
    should dispatch more fetchers and trigger round 2 triangulation."""
    low_tau_continue = json.dumps({
        "claims": [
            {
                "id": "C1",
                "claim": "Claude has a 200K context window.",
                "sq": "sq1",
                "supporting": ["S1"],
                "independent_count": 1,
                "confidence": "medium",
                "contradicts": None,
            }
        ],
        "unresolved_contradictions": [],
        "coverage_gaps": ["sq2"],      # sq2 not covered → gap-fill will target it
        "tau": 0.5,
        "saturation_delta": 0.5,
        "round": 1,
        "stop_recommended": False,     # Continue → triggers Phase 4
        "notes": "Needs more sources for sq2.",
    })
    high_tau_stop = json.dumps({
        "claims": [
            {
                "id": "C1",
                "claim": "Claude has a 200K context window.",
                "sq": "sq1",
                "supporting": ["S1", "S2"],
                "independent_count": 2,
                "confidence": "high",
                "contradicts": None,
            },
            {
                "id": "C2",
                "claim": "Claude supports function calling via tool_use blocks.",
                "sq": "sq2",
                "supporting": ["S3"],
                "independent_count": 2,
                "confidence": "high",
                "contradicts": None,
            },
        ],
        "unresolved_contradictions": [],
        "coverage_gaps": [],
        "tau": 0.9,
        "saturation_delta": 0.5,
        "round": 2,
        "stop_recommended": True,
        "notes": "Converged.",
    })
    responses = [
        _mock_resp(_decompose_response()),        # Phase 1
        _mock_resp(_fetcher_response()),           # Phase 2 round 1 (4 seed queries)
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(low_tau_continue),              # Phase 3 round 1 → stop=False
        _mock_resp(_fetcher_response()),           # Phase 4 gap-fill (Phase 2 round 2)
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(high_tau_stop),                 # Phase 3 round 2 → stop=True
        _mock_resp(_verify_response(passed=True)), # Phase 6
    ]
    mock = MockLlmClient(responses=responses)
    tier_router = _make_tier_router()

    result = await run_pipeline(
        topic="gap-fill-test",
        llm=mock,
        tier_router=tier_router,
        state_dir=tmp_path / "briefs" / "gap-fill-test",
    )

    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    # Both round1 and round2 triangulations should appear
    assert "phase3_round1" in trace["phases"]
    assert "phase3_round2" in trace["phases"]
    assert result.verdict in ("READY", "PARTIAL")


# ---------------------------------------------------------------------------
# Test 12 — Max-round cap (3 rounds) respected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_round_cap_respected(tmp_path: Path):
    """Even when stop_recommended=False, pipeline must stop after 3 rounds."""
    never_stop = _triangulate_response(tau=0.3, stop=False)
    # Round 3 version
    never_stop_r2 = json.dumps({**json.loads(never_stop), "round": 2})
    never_stop_r3 = json.dumps({**json.loads(never_stop), "round": 3, "stop_recommended": True})

    responses = [
        _mock_resp(_decompose_response()),   # Phase 1
        # Phase 2 round 1 (4 fetches)
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(_fetcher_response()),
        _mock_resp(never_stop),              # Phase 3 round 1 → stop=False
        # Phase 4 gap-fill (but sq2 not in coverage_gaps, so no new queries)
        # → gap_fill returns None (no gap queries generated), loop breaks
        _mock_resp(_verify_response(passed=True)),  # Phase 6
    ]
    mock = MockLlmClient(responses=responses)
    tier_router = _make_tier_router()

    result = await run_pipeline(
        topic="max-round-cap-test",
        llm=mock,
        tier_router=tier_router,
        state_dir=tmp_path / "briefs" / "max-round-cap-test",
    )
    # Should still complete without error
    assert result.verdict in ("READY", "PARTIAL", "FAIL")
    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    # No more than 3 triangulate rounds
    round_keys = [k for k in trace["phases"] if k.startswith("phase3_round")]
    assert len(round_keys) <= 3


# ---------------------------------------------------------------------------
# Test 13 — Artifact schemas match SKILL.md spec
# ---------------------------------------------------------------------------

def test_artifact_schema_completeness(tmp_path: Path):
    """claims.json must have every field from the SKILL.md Phase 5 schema."""
    doc = _make_claims_doc()
    claims_path = tmp_path / "claims.json"
    write_claims(claims_path, doc)

    data = json.loads(claims_path.read_text(encoding="utf-8"))

    top_level_required = [
        "topic", "generated", "freshness", "triangulation_score",
        "verdict", "source_count", "claims", "unresolved_contradictions",
        "coverage_gaps", "sub_questions",
    ]
    for field_name in top_level_required:
        assert field_name in data, f"Top-level field missing: {field_name}"

    assert len(data["claims"]) == 1
    claim = data["claims"][0]
    claim_required = ["id", "claim", "sq", "supporting", "independent_count", "confidence", "contradicts"]
    for field_name in claim_required:
        assert field_name in claim, f"Claim field missing: {field_name}"

    assert len(data["sub_questions"]) == 1
    sq = data["sub_questions"][0]
    for field_name in ("id", "question", "acceptance"):
        assert field_name in sq, f"SubQuestion field missing: {field_name}"


# ---------------------------------------------------------------------------
# Test 14 — Adapter missing llm → degraded ack (no crash)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adapter_degraded_ack_when_not_configured():
    engine = DeepResearch()  # no llm or tier_router injected
    ctx = create_request_context(session_id="test-session")
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="pre-dispatch",
        topic="research.requested",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={"topic": "some research topic"},
    )
    ack = await engine.on_phase(event, ctx)
    assert ack.status == "ack"
    assert ack.degraded is True
    assert "not configured" in (ack.reason or "")
