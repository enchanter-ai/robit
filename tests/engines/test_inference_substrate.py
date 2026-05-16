"""Tests for the InferenceSubstrateEngine PluginAdapter wrapper.

Covers:
  - Adapter has correct name, phases, required, topics, budget_tier
  - post-session phase fires emit on accumulated artifacts
  - cross-session phase fires reconcile + render-briefing
  - Failure events (*.veto, *.warn) captured during a session are emitted at post-session
  - Empty session (no captured events) is a clean no-op at post-session
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robit.core import EnchantedEvent, PluginAck, create_request_context
from robit.core.bus import build_event
from robit.core.context import RequestContext
from robit.engines.inference_substrate import InferenceSubstrateEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(tmp_path: Path) -> InferenceSubstrateEngine:
    return InferenceSubstrateEngine(state_dir=tmp_path)


def _event(
    ctx: RequestContext,
    *,
    phase: str,
    topic: str,
    payload: dict | None = None,
) -> EnchantedEvent:
    return build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase=phase,  # type: ignore[arg-type]
        topic=topic,
        source="test-source",
        budget_tier=ctx.budget_tier,
        payload=payload or {},
    )


# ---------------------------------------------------------------------------
# Test 1 — Adapter has correct metadata
# ---------------------------------------------------------------------------


def test_adapter_metadata(tmp_path: Path):
    engine = _make_engine(tmp_path)
    assert engine.name == "inference-substrate"
    assert "post-session" in engine.phases
    assert "cross-session" in engine.phases
    assert engine.required is False
    assert engine.budget_tier == "always"
    assert "*.veto" in engine.topics.subscribes
    assert "*.warn" in engine.topics.subscribes
    assert "inference-substrate.emitted" in engine.topics.emits
    assert "inference-substrate.reconciled" in engine.topics.emits
    assert "inference-substrate.briefing-rendered" in engine.topics.emits


# ---------------------------------------------------------------------------
# Test 2 — post-session flushes buffered artifacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_session_flushes_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENCHANTER_INFERENCE_STATE", str(tmp_path))
    engine = _make_engine(tmp_path)
    ctx = create_request_context()

    # Inject a veto event to be captured into the buffer.
    veto_event = _event(ctx, phase="trust-gate", topic="trust-scorer.veto",
                        payload={"reason": "low trust score"})
    await engine.on_phase(veto_event, ctx)
    assert len(engine._buffer) == 1

    # Fire post-session.
    post_event = _event(ctx, phase="post-session", topic="lifecycle.post-session")
    ack = await engine.on_phase(post_event, ctx)

    assert ack.status == "ack"
    assert not ack.degraded
    # Buffer must be cleared after flush.
    assert len(engine._buffer) == 0
    # artifacts.jsonl must exist and contain one entry.
    artifact_path = tmp_path / "artifacts.jsonl"
    assert artifact_path.exists()
    lines = artifact_path.read_text().strip().splitlines()
    assert len(lines) == 1
    # At least one derived event should have been emitted.
    emitted = [e for e in ack.derived_events if e.topic == "inference-substrate.emitted"]
    assert len(emitted) == 1


# ---------------------------------------------------------------------------
# Test 3 — cross-session fires reconcile + render-briefing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_session_reconcile_and_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENCHANTER_INFERENCE_STATE", str(tmp_path))

    # Pre-populate the artifact log so reconcile has something to work with.
    from robit.inference.engine import emit_unconditional

    emit_unconditional(
        {
            "code": "F01",
            "tags": ["enchanter"],
            "title": "sycophancy",
            "session_id": "seed-session",
        },
        tmp_path,
    )

    engine = _make_engine(tmp_path)
    ctx = create_request_context()
    cross_event = _event(ctx, phase="cross-session", topic="lifecycle.cross-session")
    ack = await engine.on_phase(cross_event, ctx)

    assert ack.status == "ack"
    assert not ack.degraded
    reconciled = [e for e in ack.derived_events if e.topic == "inference-substrate.reconciled"]
    assert len(reconciled) == 1
    assert reconciled[0].payload["total_artifacts"] >= 1

    rendered = [e for e in ack.derived_events if e.topic == "inference-substrate.briefing-rendered"]
    assert len(rendered) == 1
    briefing_path = Path(rendered[0].payload["path"])
    assert briefing_path.exists()


# ---------------------------------------------------------------------------
# Test 4 — failure events (*.warn) are captured and emitted at post-session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warn_events_captured_and_emitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENCHANTER_INFERENCE_STATE", str(tmp_path))
    engine = _make_engine(tmp_path)
    ctx = create_request_context()

    # Fire two warn events in different "phases".
    for i in range(2):
        warn = _event(
            ctx,
            phase="pre-dispatch",
            topic=f"secret-mask.warn",
            payload={"reason": f"sensitive pattern {i}"},
        )
        await engine.on_phase(warn, ctx)

    assert len(engine._buffer) == 2

    # Flush.
    post = _event(ctx, phase="post-session", topic="lifecycle.post-session")
    ack = await engine.on_phase(post, ctx)

    assert ack.status == "ack"
    assert len(engine._buffer) == 0
    lines = (tmp_path / "artifacts.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    # Both derived events should be inference-substrate.emitted.
    emitted = [e for e in ack.derived_events if e.topic == "inference-substrate.emitted"]
    assert len(emitted) == 2


# ---------------------------------------------------------------------------
# Test 5 — Empty session: no captured events → clean no-op at post-session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_session_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENCHANTER_INFERENCE_STATE", str(tmp_path))
    engine = _make_engine(tmp_path)
    ctx = create_request_context()

    post = _event(ctx, phase="post-session", topic="lifecycle.post-session")
    ack = await engine.on_phase(post, ctx)

    assert ack.status == "ack"
    assert not ack.degraded
    # No artifacts written.
    assert not (tmp_path / "artifacts.jsonl").exists()
    # No derived events.
    assert ack.derived_events == []
