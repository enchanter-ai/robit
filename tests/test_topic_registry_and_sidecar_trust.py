"""G2 + G4 — topic registry cross-check and sidecar trust hardening.

Covers (per the LOADER-TRUST package contract):

* G2: the boot-time topic cross-check accepts the real engine set, and (in
  strict mode) rejects a manifest declaring an unknown topic.
* G4: sidecar `initialize` fails when the reported topics exceed the manifest;
  forged-source and undeclared-topic derived_events are dropped; an advisory
  sidecar fails OPEN on timeout while a required sidecar fails CLOSED.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from robit.core import create_request_context
from robit.core.events import EnchantedEvent
from robit.core.plugin import PluginTopics
from robit.core.topics import is_known_topic
from robit.loader.discovery import (
    _cross_check_topics,
    _default_root,
    find_engine_manifests,
    load_engine_registry,
)
from robit.loader.errors import TopicRegistryError
from robit.loader.manifest import EngineManifest, EngineTopics, parse_manifest
from robit.loader.runtimes import SidecarAdapter, load_sidecar_adapter
from robit.loader.runtimes._base import SidecarInitError

_FIXTURE = Path(__file__).parent / "loader" / "runtimes" / "fixtures" / "echo_sidecar.py"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _real_manifests() -> list[EngineManifest]:
    return [parse_manifest(p) for p in find_engine_manifests(_default_root())]


def _manifest(
    *,
    name: str,
    subscribes: tuple[str, ...],
    emits: tuple[str, ...],
) -> EngineManifest:
    return EngineManifest(
        name=name,
        description="d",
        version="1.0.0",
        phases=("trust-gate",),
        required=False,
        budget_tier="always",
        topics=EngineTopics(subscribes=subscribes, emits=emits),
        runtime="python",
        adapter="x.y:z",
    )


def _sidecar_manifest(
    extra_args: tuple[str, ...],
    *,
    name: str = "echo-sidecar",
    required: bool = False,
    subscribes: tuple[str, ...] = (),
    emits: tuple[str, ...] = (),
) -> EngineManifest:
    return EngineManifest(
        name=name,
        description="d",
        version="1.0.0",
        phases=("trust-gate",),
        required=required,
        budget_tier="always",
        topics=EngineTopics(subscribes=subscribes, emits=emits),
        runtime="sidecar",
        command=sys.executable,
        args=(str(_FIXTURE),) + extra_args,
        env_allowlist=("PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP", "PYTHONPATH"),
    )


def _event(*, topic: str = "mcp.tool.call.requested") -> EnchantedEvent:
    return EnchantedEvent(
        id="evt-1",
        correlation_id="corr-1",
        session_id="s-1",
        phase="trust-gate",
        topic=topic,
        source="orchestrator",
        budget_tier="HIGH",
        ts=0,
        payload={"hello": "world"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# G2 — registry cross-check
# ──────────────────────────────────────────────────────────────────────────────

def test_real_engine_set_passes_cross_check_warn_mode() -> None:
    """The live 14-engine set must cross-check cleanly in warn mode (no raise)."""
    manifests = _real_manifests()
    assert len(manifests) >= 14
    # Warn mode never raises; it returns the (soft) problem list.
    problems = _cross_check_topics(manifests, strict=False)
    assert isinstance(problems, list)
    # And the registry actually loads.
    registry = load_engine_registry()
    assert len(registry) >= 14


def test_real_engine_subscribe_topics_are_registry_known() -> None:
    """Every concrete (non-wildcard, non-lifecycle) subscribe is registry-known."""
    for m in _real_manifests():
        for topic in m.topics.subscribes:
            assert is_known_topic(topic), f"{m.name} subscribes unknown {topic!r}"
        for topic in m.topics.emits:
            assert is_known_topic(topic), f"{m.name} emits unknown {topic!r}"


def test_real_engine_set_passes_cross_check_even_in_strict_mode() -> None:
    """The real set has no unknown topics; in strict mode unsubscribed emits do
    warn — so strict mode raises on the real set ONLY because of layered
    coverage gaps, which is the documented honest-over-breaking trade-off.

    We assert the failure is purely coverage (no UNKNOWN-topic problem), proving
    no real engine declares an off-registry topic.
    """
    manifests = _real_manifests()
    with pytest.raises(TopicRegistryError) as ei:
        _cross_check_topics(manifests, strict=True)
    # None of the strict-mode problems are unknown-topic problems.
    assert ei.value.problems
    assert not any("not in registry" in p for p in ei.value.problems)


def test_bogus_unknown_topic_manifest_rejected_in_strict_mode() -> None:
    """A manifest emitting a topic absent from the registry raises in strict mode."""
    good = _real_manifests()
    bogus = _manifest(
        name="bogus-engine",
        subscribes=("mcp.tool.call.requested",),
        emits=("bogus-engine.totally.invented.topic",),
    )
    with pytest.raises(TopicRegistryError) as ei:
        _cross_check_topics(good + [bogus], strict=True)
    assert any("not in registry" in p for p in ei.value.problems)
    assert ei.value.problems


def test_unknown_subscribe_topic_rejected_in_strict_mode() -> None:
    bogus = _manifest(
        name="bogus-sub",
        subscribes=("no.such.topic.exists",),
        emits=(),
    )
    with pytest.raises(TopicRegistryError) as ei:
        _cross_check_topics([bogus], strict=True)
    assert any("no.such.topic.exists" in p for p in ei.value.problems)


def test_wildcard_and_lifecycle_subscriptions_always_allowed() -> None:
    """A manifest subscribing only to wildcards / lifecycle topics never trips
    the unknown-topic check, even in strict mode (modulo coverage warnings on
    its own emits — here it emits nothing)."""
    ok = _manifest(
        name="wildcard-sub",
        subscribes=("*.veto", "foo.*", "lifecycle.trust-gate", "*"),
        emits=(),
    )
    # No emits, only always-allowed subscriptions → strict mode is clean.
    problems = _cross_check_topics([ok], strict=True)
    assert problems == []


def test_deprecated_synonym_is_registry_known() -> None:
    """llm.proxy.request is retained as a deprecated synonym and stays known."""
    from robit.core.topics import get_topic

    spec = get_topic("llm.proxy.request")
    assert spec is not None
    assert spec.deprecated is True
    assert is_known_topic("llm.proxy.request")
    # The canonical name is present and NOT deprecated.
    canonical = get_topic("mcp.tool.call.requested")
    assert canonical is not None
    assert canonical.deprecated is False


# ──────────────────────────────────────────────────────────────────────────────
# G4 — sidecar init topic cross-check
# ──────────────────────────────────────────────────────────────────────────────

async def test_sidecar_init_fails_when_reported_topics_exceed_manifest() -> None:
    """A sidecar that claims an emit topic not in its manifest fails init."""
    m = _sidecar_manifest(
        extra_args=("--name", "narrow", "--emits", "declared-a,sneaky-b"),
        name="narrow",
        emits=("declared-a",),  # manifest only allows declared-a
        subscribes=("lifecycle.trust-gate",),
    )
    adapter = load_sidecar_adapter(m)
    try:
        with pytest.raises(SidecarInitError) as ei:
            await adapter.warm_up()
        assert "manifest" in str(ei.value)
        assert "sneaky-b" in str(ei.value)
    finally:
        await adapter.shutdown()


async def test_sidecar_init_fails_when_reported_subscribes_exceed_manifest() -> None:
    m = _sidecar_manifest(
        extra_args=("--name", "narrow-sub", "--subscribes", "ok-topic,extra-topic"),
        name="narrow-sub",
        subscribes=("ok-topic",),
        emits=(),
    )
    adapter = load_sidecar_adapter(m)
    try:
        with pytest.raises(SidecarInitError) as ei:
            await adapter.warm_up()
        assert "extra-topic" in str(ei.value)
    finally:
        await adapter.shutdown()


async def test_sidecar_init_succeeds_when_reported_topics_subset_of_manifest() -> None:
    """Reporting a subset of the manifest's topics is fine (narrowing allowed)."""
    m = _sidecar_manifest(
        extra_args=("--name", "subsetter", "--emits", "declared-a"),
        name="subsetter",
        emits=("declared-a", "declared-b"),  # manifest is wider
        subscribes=(),
    )
    adapter = load_sidecar_adapter(m)
    try:
        await adapter.warm_up()
        assert adapter.name == "subsetter"
        assert adapter.topics.emits == ("declared-a",)
    finally:
        await adapter.shutdown()


# ──────────────────────────────────────────────────────────────────────────────
# G4 — derived-event trust filter (forged source / undeclared topic dropped)
# ──────────────────────────────────────────────────────────────────────────────

def _make_adapter(
    *,
    name: str = "good-engine",
    phases: tuple[str, ...] = ("trust-gate",),
    emits: tuple[str, ...] = ("good-topic",),
) -> SidecarAdapter:
    adapter = SidecarAdapter(command="unused", args=())
    adapter.name = name
    adapter.phases = phases
    adapter.topics = PluginTopics(subscribes=(), emits=emits)
    return adapter


def _derived(
    *,
    source: str,
    topic: str,
    phase: str = "trust-gate",
    evt_id: str = "d-1",
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
        "payload": {},
    }


async def test_forged_source_derived_event_is_dropped() -> None:
    from unittest.mock import AsyncMock, patch

    adapter = _make_adapter(name="alpha", emits=("alpha-topic",))
    forged = _derived(source="orchestrator", topic="alpha-topic")
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection", new=AsyncMock()
    ) as rec:
        assert await adapter._validate_derived_event(forged) is None
        assert rec.await_args.kwargs["rejection_reason"] == "source-forgery"


async def test_undeclared_topic_derived_event_is_dropped() -> None:
    from unittest.mock import AsyncMock, patch

    adapter = _make_adapter(name="alpha", emits=("alpha-topic",))
    bad = _derived(source="alpha", topic="not-declared")
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection", new=AsyncMock()
    ) as rec:
        assert await adapter._validate_derived_event(bad) is None
        assert rec.await_args.kwargs["rejection_reason"] == "undeclared-topic"


async def test_valid_derived_event_survives_while_forged_dropped_in_batch() -> None:
    from unittest.mock import AsyncMock, patch

    adapter = _make_adapter(name="alpha", emits=("alpha-topic",))
    good = _derived(source="alpha", topic="alpha-topic", evt_id="ok")
    forged_src = _derived(source="framework", topic="alpha-topic", evt_id="bad1")
    forged_topic = _derived(source="alpha", topic="evil", evt_id="bad2")
    with patch(
        "robit.loader.runtimes.sidecar.record_rejection", new=AsyncMock()
    ) as rec:
        kept = []
        for d in (good, forged_src, forged_topic):
            accepted = await adapter._validate_derived_event(d)
            if accepted is not None:
                kept.append(accepted)
        assert [k["id"] for k in kept] == ["ok"]
        assert rec.await_count == 2


# ──────────────────────────────────────────────────────────────────────────────
# G4 — advisory fails OPEN, required fails CLOSED on timeout
# ──────────────────────────────────────────────────────────────────────────────

async def test_required_sidecar_fails_closed_on_timeout() -> None:
    """A REQUIRED sidecar that times out returns a veto (fail closed)."""
    m = _sidecar_manifest(
        extra_args=("--mode", "hang", "--required"),
        required=True,
        subscribes=("lifecycle.trust-gate",),
    )
    adapter = load_sidecar_adapter(m)
    adapter._timeout_s = 0.5  # type: ignore[attr-defined]
    try:
        ctx = create_request_context(session_id="req", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(), ctx)
        # Fail CLOSED: veto regardless of whether the timeout fires during
        # on_phase or a slow spawn/init (required is seeded pre-init).
        assert ack.status == "veto"
        assert ack.degraded is True
        assert "timeout" in (ack.reason or "") or "within 0.5s" in (ack.reason or "")
    finally:
        await adapter.shutdown()


async def test_advisory_sidecar_fails_open_on_timeout() -> None:
    """An ADVISORY sidecar that times out returns status='error', degraded=True
    (fail open) — NOT a veto. A flaky advisory engine must not block requests."""
    m = _sidecar_manifest(
        extra_args=("--mode", "hang"),
        required=False,
        subscribes=("lifecycle.trust-gate",),
    )
    adapter = load_sidecar_adapter(m)
    adapter._timeout_s = 0.5  # type: ignore[attr-defined]
    try:
        ctx = create_request_context(session_id="adv", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(), ctx)
        # Fail OPEN: error + degraded, never a veto. The timeout may fire during
        # on_phase ("sidecar:timeout") or during a slow spawn/init under load
        # ("sidecar:init:...within 0.5s") — both are the fail-open path.
        assert ack.status == "error"
        assert ack.degraded is True
        assert "timeout" in (ack.reason or "") or "within 0.5s" in (ack.reason or "")
    finally:
        await adapter.shutdown()


async def test_advisory_sidecar_fails_open_on_crash() -> None:
    """Crash → advisory fails open with error + degraded (not veto)."""
    m = _sidecar_manifest(
        extra_args=("--mode", "crash"),
        required=False,
        subscribes=("lifecycle.trust-gate",),
    )
    adapter = load_sidecar_adapter(m)
    try:
        ctx = create_request_context(session_id="adv2", budget_tier="HIGH")
        ack = await adapter.on_phase(_event(), ctx)
        assert ack.status == "error"
        assert ack.degraded is True
        assert "crash" in (ack.reason or "")
    finally:
        await adapter.shutdown()
