"""Tests for the structural-fingerprint (naga) engine.

13 test cases covering:
  Levenshtein (5): identical, empty-vs-N, substitution, insertion, deletion
  Levenshtein ratio (1): scales [0, 1]
  TF-IDF (2): universal term has low IDF; unique term has high IDF
  Cosine similarity (3): identical→1.0, orthogonal→0.0, partial→between
  Tokenizer (1): lowercases, splits non-alnum, drops empty and stop-words
  End-to-end (1): event feeds corpus; derived naga.schema.drift.detected fires
"""

from __future__ import annotations

import math
import pytest

from robit.core import (
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    SecurityVetoError,
    create_request_context,
)
from robit.core.bus import build_event
from robit.core.context import RequestContext
from robit.engines.structural_fingerprint import (
    StructuralFingerprint,
    StructuralFingerprintStore,
    tokenize,
    compute_tfidf,
    cosine_similarity,
    levenshtein,
    levenshtein_ratio,
)


# ===========================================================================
# Levenshtein — 5 cases
# ===========================================================================

def test_levenshtein_identical_strings_returns_zero():
    """Two identical strings have zero edit distance."""
    assert levenshtein("kitten", "kitten") == 0
    assert levenshtein("", "") == 0


def test_levenshtein_empty_vs_n_returns_n():
    """Edit distance from empty string to string of length N is N."""
    assert levenshtein("", "abc") == 3
    assert levenshtein("xyz", "") == 3


def test_levenshtein_one_substitution():
    """One character substitution costs 1."""
    assert levenshtein("kitten", "sitten") == 1


def test_levenshtein_one_insertion():
    """One character insertion costs 1."""
    assert levenshtein("abc", "abbc") == 1


def test_levenshtein_one_deletion():
    """One character deletion costs 1."""
    assert levenshtein("abbc", "abc") == 1


# ===========================================================================
# Levenshtein ratio — 1 case
# ===========================================================================

def test_levenshtein_ratio_scales_between_zero_and_one():
    """Ratio is in [0.0, 1.0]; identical→1.0; maximally different→0.0."""
    assert levenshtein_ratio("abc", "abc") == pytest.approx(1.0)
    # "abc" vs "xyz": 3 substitutions, max_len=3 → ratio = 1 - 3/3 = 0.0
    assert levenshtein_ratio("abc", "xyz") == pytest.approx(0.0)
    # partial overlap
    ratio = levenshtein_ratio("kitten", "sitting")
    assert 0.0 < ratio < 1.0, f"Expected ratio in (0,1), got {ratio}"


# ===========================================================================
# TF-IDF — 2 cases
# ===========================================================================

def test_tfidf_universal_term_has_low_idf():
    """A term present in every document gets a low IDF and low overall weight."""
    # "common" appears in all 3 docs; "unique" appears in only one.
    docs = [
        tokenize("common alpha beta"),
        tokenize("common gamma delta"),
        tokenize("common epsilon zeta"),
    ]
    vectors = compute_tfidf(docs)
    assert len(vectors) == 3

    # "common" weight in doc 0.
    common_weight = vectors[0].get("common", 0.0)
    # "alpha" appears only in doc 0 → higher IDF → higher weight.
    alpha_weight = vectors[0].get("alpha", 0.0)

    assert alpha_weight > common_weight, (
        f"Unique term 'alpha' ({alpha_weight:.4f}) should outweigh "
        f"universal term 'common' ({common_weight:.4f})"
    )


def test_tfidf_unique_term_has_high_idf():
    """A term appearing in only one of N docs gets a larger IDF than an N-doc term."""
    n = 5
    # "shared" in all 5 docs; "rare" only in doc 0.
    docs = [tokenize("shared rare word")] + [tokenize("shared other stuff")] * (n - 1)
    vectors = compute_tfidf(docs)

    shared_weight = vectors[0].get("shared", 0.0)
    rare_weight   = vectors[0].get("rare", 0.0)

    # IDF("shared") = log(1 + 5 / (1+5)) ≈ 0.154
    # IDF("rare")   = log(1 + 5 / (1+1)) ≈ 0.916
    assert rare_weight > shared_weight, (
        f"Rare term ({rare_weight:.4f}) should exceed shared term ({shared_weight:.4f})"
    )


# ===========================================================================
# Cosine similarity — 3 cases
# ===========================================================================

def test_cosine_similarity_identical_vectors_returns_one():
    """Cosine similarity of a vector with itself is 1.0."""
    v = {"a": 0.5, "b": 0.3, "c": 0.9}
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)


def test_cosine_similarity_orthogonal_vectors_returns_zero():
    """Vectors with no shared terms have cosine similarity 0.0."""
    a = {"x": 1.0, "y": 2.0}
    b = {"p": 3.0, "q": 4.0}
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cosine_similarity_partial_overlap_between_zero_and_one():
    """Partially overlapping vectors return a similarity strictly in (0, 1)."""
    a = {"foo": 1.0, "bar": 1.0}
    b = {"foo": 1.0, "baz": 1.0}
    sim = cosine_similarity(a, b)
    assert 0.0 < sim < 1.0, f"Expected (0, 1), got {sim}"
    # Exact: dot=1; |a|=sqrt(2); |b|=sqrt(2) → sim = 1/2 = 0.5
    assert sim == pytest.approx(0.5, abs=1e-9)


# ===========================================================================
# Tokenizer — 1 case
# ===========================================================================

def test_tokenizer_lowercases_splits_and_drops_stopwords():
    """tokenize(): lowercases, splits on non-alnum, drops empties and stop-words."""
    # "the" is a stop-word; "a" is filtered (len <= 1); others survive.
    result = tokenize("The quick Brown Fox jumps over the lazy Dog")
    assert "the" not in result, "'the' is a stop-word and must be dropped"
    # "quick", "brown", "fox", "lazy", "dog", "over", "jumps" are NOT stop-words.
    assert "quick" in result
    assert "brown" in result
    assert "fox" in result
    assert "lazy" in result
    assert "dog" in result
    assert "over" in result   # "over" is NOT in the stop-word list
    assert "jumps" in result
    # All tokens must be lowercase.
    for tok in result:
        assert tok == tok.lower(), f"Token '{tok}' is not lowercase"
    # No empty strings or single-char tokens.
    assert all(len(t) > 1 for t in result)


# ===========================================================================
# End-to-end — 1 case
# ===========================================================================

async def test_e2e_tool_registration_then_drift_fires_derived_event():
    """
    1. Fire mcp.tools.list.received with tool 'read_file' (description A).
       → naga.pattern.fingerprinted emitted; no drift.
    2. Fire again with 'read_file' whose description is radically different
       and whose inputSchema param count changed (N1 drift).
       → naga.schema.drift.detected with structural=True; ack is 'veto'.
    """
    engine = StructuralFingerprint()

    bus = InProcessBus()
    registry = {engine.name: engine}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    async def dispatch(ctx: RequestContext) -> str:
        return "ok"

    # --- Round 1: first registration ---
    ctx1 = create_request_context()

    event1 = build_event(
        correlation_id=ctx1.correlation_id,
        session_id=ctx1.session_id,
        phase="trust-gate",
        topic="mcp.tools.list.received",
        source="test-server",
        budget_tier=ctx1.budget_tier,
        payload={
            "server_id": "test-server",
            "tools": [
                {
                    "name": "read_file",
                    "description": "Reads the contents of a file from the filesystem",
                    "inputSchema": {
                        "properties": {
                            "file_path": {"type": "string"},
                        }
                    },
                }
            ],
        },
    )
    await bus.publish(event1.topic, event1)
    result1 = await orch.run(ctx1, dispatch)
    assert result1 == "ok"

    fingerprinted = [
        e for e in bus.tap(ctx1.correlation_id)
        if e.topic == "structural-fingerprint.pattern.fingerprinted"
    ]
    assert len(fingerprinted) == 1, "Expected structural-fingerprint.pattern.fingerprinted on first registration"
    assert fingerprinted[0].payload["qualified_name"] == "test-server.read_file"

    # --- Round 2: mutated schema → N1 + N2 drift ---
    ctx2 = create_request_context()

    event2 = build_event(
        correlation_id=ctx2.correlation_id,
        session_id=ctx2.session_id,
        phase="trust-gate",
        topic="mcp.tools.list.received",
        source="test-server",
        budget_tier=ctx2.budget_tier,
        payload={
            "server_id": "test-server",
            "tools": [
                {
                    "name": "read_file",
                    # Completely different description → N2 drift (Jaccard < 0.6)
                    "description": "Execute SQL queries against a relational database",
                    # Extra params → N1 shape hash changes
                    "inputSchema": {
                        "properties": {
                            "query":    {"type": "string"},
                            "database": {"type": "string"},
                            "timeout":  {"type": "integer"},
                        },
                        "outputSchema": {"type": "object"},
                    },
                }
            ],
        },
    )
    await bus.publish(event2.topic, event2)

    # N1 structural drift → veto → Orchestrator raises SecurityVetoError.
    with pytest.raises(SecurityVetoError) as exc_info:
        await orch.run(ctx2, dispatch)

    assert "structural-fingerprint" in str(exc_info.value), "Veto must name the structural-fingerprint plugin"

    # structural-fingerprint.schema.drift.detected must have been published to the bus before the veto.
    drift_events = [
        e for e in bus.tap(ctx2.correlation_id)
        if e.topic == "structural-fingerprint.schema.drift.detected"
    ]
    assert len(drift_events) == 1, (
        f"Expected structural-fingerprint.schema.drift.detected on schema mutation, got {len(drift_events)}"
    )
    drift_payload = drift_events[0].payload
    assert drift_payload["qualified_name"] == "test-server.read_file"
    assert drift_payload["structural"] is True, "N1 param-count change must be structural"
    assert "n1" in drift_payload["axes"], "N1 must appear in drift axes"
