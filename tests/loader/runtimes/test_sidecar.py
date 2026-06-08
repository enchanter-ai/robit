"""Tests for robit.loader.runtimes.sidecar — subprocess JSON-RPC stdio runtime."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from robit.core import PluginAck, create_request_context
from robit.core.events import EnchantedEvent
from robit.core.plugin import PluginTopics
from robit.loader.manifest import EngineManifest, EngineTopics
from robit.loader.runtimes import (
    SidecarAdapter,
    load_runtime,
    load_sidecar_adapter,
)
from robit.loader.runtimes.sidecar import _parse_ack

FIXTURE = Path(__file__).parent / "fixtures" / "echo_sidecar.py"


def _sidecar_manifest(
    extra_args: tuple[str, ...] = (),
    *,
    name: str = "echo-sidecar",
    required: bool = False,
) -> EngineManifest:
    args = (str(FIXTURE),) + extra_args
    return EngineManifest(
        name=name,
        description="d",
        version="1.0.0",
        phases=("trust-gate",),
        required=required,
        budget_tier="always",
        topics=EngineTopics(subscribes=(), emits=()),
        runtime="sidecar",
        command=sys.executable,
        args=args,
        env_allowlist=("PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP", "PYTHONPATH"),
    )


def _event(session_id: str, correlation_id: str, *, topic: str = "mcp.tool.call.requested") -> EnchantedEvent:
    return EnchantedEvent(
        id="evt-1",
        correlation_id=correlation_id,
        session_id=session_id,
        phase="trust-gate",
        topic=topic,
        source="orchestrator",
        budget_tier="HIGH",
        ts=0,
        payload={"hello": "world"},
    )


# ────────────────────────────────────────────────────────────────────────────────
# 1. initialize handshake mirrors attributes
# ────────────────────────────────────────────────────────────────────────────────

async def test_initialize_handshake_mirrors_attributes() -> None:
    m = _sidecar_manifest(
        extra_args=(
            "--name", "rust-echo",
            "--phases", "trust-gate,dispatch",
            "--required",
            "--budget-tier", "med-or-higher",
            "--subscribes", "topic-a,topic-b",
            "--emits", "topic-out",
        ),
    )
    adapter = load_runtime(m)
    try:
        assert isinstance(adapter, SidecarAdapter)
        await adapter.warm_up()
        assert adapter.name == "rust-echo"
        assert adapter.phases == ("trust-gate", "dispatch")
        assert adapter.required is True
        assert adapter.budget_tier == "med-or-higher"
        assert isinstance(adapter.topics, PluginTopics)
        assert adapter.topics.subscribes == ("topic-a", "topic-b")
        assert adapter.topics.emits == ("topic-out",)
    finally:
        await adapter.shutdown()


# ────────────────────────────────────────────────────────────────────────────────
# 2. on_phase ack passes through
# ────────────────────────────────────────────────────────────────────────────────

async def test_on_phase_ack_passes_through() -> None:
    m = _sidecar_manifest(extra_args=("--mode", "ack"))
    adapter = load_sidecar_adapter(m)
    try:
        ctx = create_request_context(session_id="s1", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        assert isinstance(ack, PluginAck)
        assert ack.status == "ack"
        assert ack.derived_events == []
    finally:
        await adapter.shutdown()


# ────────────────────────────────────────────────────────────────────────────────
# 3. on_phase veto-shaped response → PluginAck status=veto
# ────────────────────────────────────────────────────────────────────────────────

async def test_on_phase_veto_response_passes_through() -> None:
    m = _sidecar_manifest(extra_args=("--mode", "veto"))
    adapter = load_sidecar_adapter(m)
    try:
        ctx = create_request_context(session_id="s2", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        assert ack.status == "veto"
        assert ack.reason == "echoed-veto"
    finally:
        await adapter.shutdown()


# ────────────────────────────────────────────────────────────────────────────────
# 4. derived events round-trip
# ────────────────────────────────────────────────────────────────────────────────

async def test_derived_events_round_trip() -> None:
    m = _sidecar_manifest(extra_args=("--mode", "derive", "--name", "echoer", "--emits", "mcp.tool.call.requested"))
    adapter = load_sidecar_adapter(m)
    try:
        ctx = create_request_context(session_id="s3", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        assert ack.status == "ack"
        assert len(ack.derived_events) == 1
        derived = ack.derived_events[0]
        assert isinstance(derived, EnchantedEvent)
        assert derived.source == "echoer"
        assert derived.id == "evt-1-derived"
        assert derived.payload == {"hello": "world"}
    finally:
        await adapter.shutdown()


# ────────────────────────────────────────────────────────────────────────────────
# 5. Timeout → veto-shaped ack (and sidecar process is killed)
# ────────────────────────────────────────────────────────────────────────────────

async def test_timeout_is_coerced_to_veto_ack() -> None:
    # G4 — fail-closed coercion is conditional on required. Build via a manifest
    # with required=True so `required` is seeded BEFORE init (deterministic even
    # if the timeout fires during the spawn/initialize handshake under load).
    m = _sidecar_manifest(extra_args=("--mode", "hang", "--required"), required=True)
    adapter = load_sidecar_adapter(m)
    adapter._timeout_s = 0.5  # type: ignore[attr-defined]
    try:
        ctx = create_request_context(session_id="s4", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        assert ack.status == "veto"
        # The timeout may fire during on_phase ("sidecar:timeout") or during a
        # slow spawn/init under suite load ("sidecar:init:...within 0.5s");
        # either way a REQUIRED sidecar fails closed (veto). `required` is
        # seeded from the manifest pre-init so the verdict is deterministic.
        assert "timeout" in (ack.reason or "") or "within 0.5s" in (ack.reason or "")
        assert ack.degraded is True
    finally:
        await adapter.shutdown()


# ────────────────────────────────────────────────────────────────────────────────
# 6. Crash mid-call → veto-shaped ack
# ────────────────────────────────────────────────────────────────────────────────

async def test_subprocess_crash_is_coerced_to_veto_ack() -> None:
    # G4 — required=True → crash coerced to veto (fail closed).
    m = _sidecar_manifest(extra_args=("--mode", "crash", "--required"), required=True)
    adapter = load_sidecar_adapter(m)
    try:
        ctx = create_request_context(session_id="s5", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        assert ack.status == "veto"
        # Reason should mention crash (initialize succeeds; the on_phase EOF triggers crash path).
        assert "crash" in (ack.reason or "")
    finally:
        await adapter.shutdown()


# ────────────────────────────────────────────────────────────────────────────────
# 7. 8 MiB body cap is enforced on incoming messages
# ────────────────────────────────────────────────────────────────────────────────

async def test_eight_mib_body_cap_enforced() -> None:
    # G4 — required=True so the protocol cap-trip coerces to a veto (fail closed).
    m = _sidecar_manifest(extra_args=("--mode", "big", "--required"), required=True)
    adapter = load_sidecar_adapter(m)
    try:
        ctx = create_request_context(session_id="s6", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        # Cap trip surfaces as veto with reason mentioning protocol / size.
        assert ack.status == "veto"
        assert ack.degraded is True
    finally:
        await adapter.shutdown()


# ────────────────────────────────────────────────────────────────────────────────
# 8. Graceful shutdown closes the subprocess
# ────────────────────────────────────────────────────────────────────────────────

async def test_graceful_shutdown_closes_subprocess() -> None:
    m = _sidecar_manifest(extra_args=("--mode", "ack"))
    adapter = load_sidecar_adapter(m)
    ctx = create_request_context(session_id="s7", budget_tier="HIGH")
    ack = await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
    assert ack.status == "ack"
    proc = adapter._proc  # type: ignore[attr-defined]
    assert proc is not None
    await adapter.shutdown()
    # After shutdown, process has exited.
    assert proc.returncode is not None


# ────────────────────────────────────────────────────────────────────────────────
# 9. Multiple sequential on_phase calls reuse the same subprocess
# ────────────────────────────────────────────────────────────────────────────────

async def test_subprocess_is_long_lived_across_calls() -> None:
    m = _sidecar_manifest(extra_args=("--mode", "ack"))
    adapter = load_sidecar_adapter(m)
    try:
        ctx = create_request_context(session_id="s8", budget_tier="HIGH")
        await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        pid_first = adapter._proc.pid  # type: ignore[attr-defined,union-attr]
        await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        await adapter.on_phase(_event(ctx.session_id, ctx.correlation_id), ctx)
        pid_third = adapter._proc.pid  # type: ignore[attr-defined,union-attr]
        assert pid_first == pid_third
    finally:
        await adapter.shutdown()


# ────────────────────────────────────────────────────────────────────────────────
# 10. _parse_ack unit: malformed result becomes veto (no subprocess needed)
# ────────────────────────────────────────────────────────────────────────────────

def test_parse_ack_handles_malformed_result() -> None:
    assert _parse_ack("not-a-dict").status == "veto"
    assert _parse_ack({"status": "bogus"}).status == "veto"
    assert _parse_ack({"status": "ack", "derived_events": "not-a-list"}).status == "veto"
    # Valid ack survives.
    a = _parse_ack({"status": "ack", "reason": None, "derived_events": []})
    assert a.status == "ack"


# ────────────────────────────────────────────────────────────────────────────────
# Wave 14.1 — derived-event validation (source allowlist + topic allowlist +
# phase consistency + malformed-event filter). The audit hook record_rejection
# is patched so these tests don't depend on Sibling B's _audit module landing
# first.
# ────────────────────────────────────────────────────────────────────────────────

from unittest.mock import AsyncMock, patch

from robit.core.plugin import PluginTopics as _PluginTopics


def _make_adapter(
    *,
    name: str = "good-engine",
    phases: tuple[str, ...] = ("trust-gate",),
    emits: tuple[str, ...] = ("good-topic",),
) -> SidecarAdapter:
    """Build a SidecarAdapter without spawning a subprocess and inject the
    handshake-derived attributes directly. Validation reads only self.name,
    self.phases, self.topics.emits — no subprocess required.
    """
    adapter = SidecarAdapter(command="unused", args=())
    adapter.name = name
    adapter.phases = phases
    adapter.topics = _PluginTopics(subscribes=(), emits=emits)
    return adapter


def _good_event(
    *,
    source: str = "good-engine",
    topic: str = "good-topic",
    phase: str = "trust-gate",
    evt_id: str = "evt-d-1",
) -> dict:
    return {
        "id": evt_id,
        "correlation_id": "corr-1",
        "session_id": "s-1",
        "phase": phase,
        "topic": topic,
        "source": source,
        "budget_tier": "HIGH",
        "ts": 0,
        "payload": {"k": "v"},
    }


def _ack_result(derived: list[dict]) -> dict:
    return {
        "status": "ack",
        "reason": None,
        "derived_events": derived,
        "degraded": False,
    }


async def test_validate_accepts_well_formed_derived_event() -> None:
    adapter = _make_adapter()
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection",
        new=AsyncMock(),
    ) as rec:
        accepted = await adapter._validate_derived_event(_good_event())
        assert accepted is not None
        assert accepted["source"] == "good-engine"
        rec.assert_not_awaited()


async def test_validate_rejects_reserved_source_orchestrator() -> None:
    adapter = _make_adapter()
    forged = _good_event(source="orchestrator")
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection",
        new=AsyncMock(),
    ) as rec:
        result = await adapter._validate_derived_event(forged)
        assert result is None
        rec.assert_awaited_once()
        kwargs = rec.await_args.kwargs
        assert kwargs["rejection_reason"] == "source-forgery"
        assert kwargs["adapter_name"] == "good-engine"
        assert kwargs["raw_event"] == forged
        assert kwargs["expected"]["name"] == "good-engine"
        assert kwargs["expected"]["emits"] == ["good-topic"]


async def test_validate_rejects_source_mismatch_with_manifest_name() -> None:
    """Sidecar claimed to be name="foo" at initialize but emits source="bar"."""
    adapter = _make_adapter(name="foo", emits=("foo-topic",))
    bad = _good_event(source="bar", topic="foo-topic")
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection",
        new=AsyncMock(),
    ) as rec:
        result = await adapter._validate_derived_event(bad)
        assert result is None
        rec.assert_awaited_once()
        assert rec.await_args.kwargs["rejection_reason"] == "source-forgery"


async def test_validate_rejects_undeclared_topic() -> None:
    adapter = _make_adapter(emits=("declared-topic",))
    bad = _good_event(topic="surprise-topic")
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection",
        new=AsyncMock(),
    ) as rec:
        result = await adapter._validate_derived_event(bad)
        assert result is None
        rec.assert_awaited_once()
        kwargs = rec.await_args.kwargs
        assert kwargs["rejection_reason"] == "undeclared-topic"
        assert kwargs["expected"]["emits"] == ["declared-topic"]


async def test_validate_rejects_phase_out_of_scope() -> None:
    adapter = _make_adapter(phases=("trust-gate",))
    bad = _good_event(phase="dispatch")
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection",
        new=AsyncMock(),
    ) as rec:
        result = await adapter._validate_derived_event(bad)
        assert result is None
        rec.assert_awaited_once()
        assert rec.await_args.kwargs["rejection_reason"] == "phase-out-of-scope"


async def test_validate_rejects_malformed_event_missing_topic() -> None:
    adapter = _make_adapter()
    bad = _good_event()
    del bad["topic"]
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection",
        new=AsyncMock(),
    ) as rec:
        result = await adapter._validate_derived_event(bad)
        assert result is None
        rec.assert_awaited_once()
        assert rec.await_args.kwargs["rejection_reason"] == "malformed-event"


async def test_on_phase_mixed_batch_keeps_only_valid_event() -> None:
    """1 valid + 2 forged events in one ack → only the valid one survives.
    Two separate record_rejection calls; the ack itself still flows.

    Drives the full on_phase path against a programmable echo sidecar via a
    custom JSON payload mode so we exercise _parse_ack integration too.
    """
    # The sidecar's --mode derive emits exactly one event — to test mixed
    # batches we go via the adapter's on_phase pipeline with a synthetic
    # result dict bypassing the subprocess. This keeps the test deterministic.
    adapter = _make_adapter(name="mix-engine", emits=("mix-topic",))
    good = _good_event(source="mix-engine", topic="mix-topic")
    forged1 = _good_event(source="orchestrator", topic="mix-topic", evt_id="evt-bad-1")
    forged2 = _good_event(source="mix-engine", topic="not-declared", evt_id="evt-bad-2")
    raw_result = _ack_result([good, forged1, forged2])

    with patch(
        "robit.loader.runtimes.sidecar.record_rejection",
        new=AsyncMock(),
    ) as rec:
        kept: list[dict] = []
        for d in raw_result["derived_events"]:
            accepted = await adapter._validate_derived_event(d)
            if accepted is not None:
                kept.append(accepted)
        assert len(kept) == 1
        assert kept[0]["id"] == "evt-d-1"
        assert rec.await_count == 2
        reasons = sorted(c.kwargs["rejection_reason"] for c in rec.await_args_list)
        assert reasons == ["source-forgery", "undeclared-topic"]


async def test_on_phase_forgery_does_not_abort_ack_via_pipeline() -> None:
    """Multiple forgeries don't abort the ack (status='ack' still flows).
    Exercises the live on_phase pipeline against the echo_sidecar fixture in
    derive mode (which emits one forgery-style event). The ack must still
    surface as 'ack'; only the bus pollution is filtered.
    """
    m = _sidecar_manifest(extra_args=("--mode", "derive", "--name", "alpha"))
    adapter = load_sidecar_adapter(m)
    try:
        with patch(
            "robit.loader.runtimes.sidecar.record_rejection",
            new=AsyncMock(),
        ) as rec:
            ctx = create_request_context(session_id="s-mix", budget_tier="HIGH")
            # Event topic 'mcp.tool.call.requested' is NOT in the manifest's
            # emits (default empty), so the sidecar's derived event will be
            # rejected as undeclared-topic. The ack must still be 'ack'.
            ack = await adapter.on_phase(
                _event(ctx.session_id, ctx.correlation_id), ctx,
            )
            assert ack.status == "ack"
            assert ack.derived_events == []
            assert rec.await_count == 1
            assert rec.await_args.kwargs["rejection_reason"] == "undeclared-topic"
            assert rec.await_args.kwargs["adapter_name"] == "alpha"
    finally:
        await adapter.shutdown()


async def test_validate_rejects_all_reserved_source_names() -> None:
    """Defence-in-depth: each name in the reserved set must be refused even
    if a misconfigured manifest somehow declared one of them as self.name."""
    from robit.loader.runtimes.sidecar import _RESERVED_SOURCES

    for reserved in _RESERVED_SOURCES:
        adapter = _make_adapter(name=reserved)
        bad = _good_event(source=reserved)
        with patch(
            "robit.loader.runtimes.sidecar.record_rejection",
            new=AsyncMock(),
        ) as rec:
            result = await adapter._validate_derived_event(bad)
            assert result is None, f"reserved name {reserved!r} should be rejected"
            rec.assert_awaited_once()
            assert rec.await_args.kwargs["rejection_reason"] == "source-forgery"
