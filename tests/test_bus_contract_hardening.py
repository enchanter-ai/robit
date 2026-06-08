"""Wave 1 (FOUNDATION) — bus & contract hardening.

Covers the four exit-criteria behaviours for the foundation wave:

* G1 — a structured :class:`Verdict` renders to HTTP 451 + ``X-Enchanter-Veto``.
* G1 — the structured verdict fields survive the proxy veto path with NO
  string-slicing (a reason that does NOT fit the ``plugin:pattern`` convention
  still carries the engine's real pattern_id when the ack attaches a Verdict).
* G5 — a runaway derived-event chain is capped by ``MAX_DERIVED_HOPS`` and the
  drop is recorded/observable, not silent.
* G6 — a failing subscriber is surfaced (recorded + callback) rather than
  swallowed, while the bus stays crash-isolated.
"""

from __future__ import annotations

from robit.core import (
    DroppedEvent,
    HandlerFailure,
    InProcessBus,
    MAX_DERIVED_HOPS,
    PluginAck,
    SecurityVetoError,
    Verdict,
    render_veto_http,
)
from robit.core.bus import build_event
from robit.core.events import CURRENT_SCHEMA_VERSION, EnchantedEvent
from robit.proxy.pipeline import VetoResult, _veto_from_error


# ---------------------------------------------------------------------------
# G1 — Verdict → HTTP 451 rendering.
# ---------------------------------------------------------------------------


def test_verdict_renders_to_http_451_with_header():
    verdict = Verdict(
        plugin="destructive-op-gate",
        phase="trust-gate",
        reason="destructive-op-gate:DOG-014",
        pattern_id="DOG-014",
        pattern_name="rm -rf /",
    )
    status, headers, body = render_veto_http(verdict)

    assert status == 451
    header = headers["X-Enchanter-Veto"]
    assert "plugin=destructive-op-gate" in header
    assert "phase=trust-gate" in header
    assert "pattern_id=DOG-014" in header
    # Body is built from structured fields.
    assert body["pattern_id"] == "DOG-014"
    assert body["pattern_name"] == "rm -rf /"
    assert body["plugin"] == "destructive-op-gate"


def test_veto_result_to_http_451_degrades_without_verdict():
    # Backwards-compat: a VetoResult built without a structured verdict still
    # renders a valid 451 (graceful degradation, no crash).
    vr = VetoResult(phase="trust-gate", plugin="some-gate", reason="blocked")
    status, headers, body = vr.to_http_451()
    assert status == 451
    assert "plugin=some-gate" in headers["X-Enchanter-Veto"]
    assert body["reason"] == "blocked"
    # No pattern data was available; the optional keys are omitted, not faked.
    assert "pattern_id" not in body


# ---------------------------------------------------------------------------
# G1 — structured fields survive the veto path (no string-slicing).
# ---------------------------------------------------------------------------


def test_structured_verdict_survives_veto_path_without_string_slicing():
    # The engine attaches a structured Verdict whose reason DOES NOT carry the
    # pattern id in the legacy "plugin:pattern" shape. Old code string-sliced
    # the reason and would have recovered pattern_id=None; the structured path
    # must preserve the real pattern id.
    verdict = Verdict(
        plugin="cve-pattern-gate",
        phase="trust-gate",
        reason="blocked for legal reasons",  # no colon-delimited pattern id
        pattern_id="CVE-2024-9999",
        pattern_name="log4shell",
    )
    err = SecurityVetoError(
        "cve-pattern-gate", "trust-gate", verdict.reason, verdict=verdict
    )
    result = _veto_from_error(err)

    assert isinstance(result, VetoResult)
    # String-slicing the reason would have yielded None here.
    assert result.pattern_id == "CVE-2024-9999"
    assert result.pattern_name == "log4shell"
    assert result.verdict is verdict


def test_security_veto_error_derives_verdict_from_legacy_reason():
    # When no structured verdict is attached, SecurityVetoError still produces
    # one by parsing the legacy reason at the single core site.
    err = SecurityVetoError("destructive-op-gate", "trust-gate", "destructive-op-gate:DOG-007")
    assert err.verdict is not None
    assert err.verdict.pattern_id == "DOG-007"
    result = _veto_from_error(err)
    assert result.pattern_id == "DOG-007"


# ---------------------------------------------------------------------------
# G3 — contract versioning.
# ---------------------------------------------------------------------------


def test_events_and_acks_carry_default_schema_version():
    ev = build_event(
        correlation_id="c",
        session_id="s",
        phase="anchor",
        topic="t",
        source="orchestrator",
        budget_tier="HIGH",
    )
    assert ev.schema_version == CURRENT_SCHEMA_VERSION == 1
    assert ev.hop_count == 0
    ack = PluginAck(status="ack")
    assert ack.schema_version == 1
    assert ack.verdict is None


# ---------------------------------------------------------------------------
# G5 — hop-count cap drops a runaway derived event (observably).
# ---------------------------------------------------------------------------


async def test_hop_count_cap_drops_runaway_derived_event():
    dropped_via_callback: list[DroppedEvent] = []
    bus = InProcessBus(on_event_dropped=dropped_via_callback.append)

    # A handler that always re-emits one derived event on the same topic →
    # an infinite cycle if uncapped.
    async def loop_handler(event: EnchantedEvent):
        derived = build_event(
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            phase=event.phase,
            topic="loop.topic",
            source="looper",
            budget_tier=event.budget_tier,
        )
        return [derived]

    bus.subscribe("loop.topic", loop_handler)

    root = build_event(
        correlation_id="c",
        session_id="s",
        phase="anchor",
        topic="loop.topic",
        source="orchestrator",
        budget_tier="HIGH",
    )

    # Must terminate (no RecursionError / hang) because of the hop cap.
    await bus.publish(root.topic, root)

    # The drop was recorded and surfaced via callback.
    assert len(bus.dropped_events) == 1
    assert bus.dropped_events[0].reason == "hop-cap"
    assert bus.dropped_events[0].hop_count == MAX_DERIVED_HOPS + 1
    assert dropped_via_callback == bus.dropped_events

    # The chain ran exactly MAX_DERIVED_HOPS deep before the drop: the buffer
    # holds the root + MAX_DERIVED_HOPS re-published derived events.
    looped = [e for e in bus.tap("c") if e.topic == "loop.topic"]
    assert len(looped) == MAX_DERIVED_HOPS + 1
    assert max(e.hop_count for e in looped) == MAX_DERIVED_HOPS


async def test_derived_event_inherits_incremented_hop_count():
    bus = InProcessBus()
    seen: list[int] = []

    async def chain_once(event: EnchantedEvent):
        seen.append(event.hop_count)
        if event.hop_count == 0:
            return [
                build_event(
                    correlation_id=event.correlation_id,
                    session_id=event.session_id,
                    phase=event.phase,
                    topic="child.topic",
                    source="x",
                    budget_tier=event.budget_tier,
                )
            ]
        return None

    bus.subscribe("*", chain_once)
    root = build_event(
        correlation_id="c",
        session_id="s",
        phase="anchor",
        topic="parent.topic",
        source="orchestrator",
        budget_tier="HIGH",
    )
    await bus.publish(root.topic, root)
    # Root seen at hop 0, the derived child seen at hop 1.
    assert seen == [0, 1]


# ---------------------------------------------------------------------------
# G6 — failing subscriber is surfaced, not swallowed.
# ---------------------------------------------------------------------------


async def test_failing_subscriber_is_surfaced_not_swallowed():
    failures_via_callback: list[HandlerFailure] = []
    bus = InProcessBus(on_handler_error=failures_via_callback.append)

    good_calls: list[str] = []

    async def crashing_handler(event: EnchantedEvent):
        raise ValueError("boom")

    async def good_handler(event: EnchantedEvent):
        good_calls.append(event.id)
        return None

    bus.subscribe("topic.x", crashing_handler)
    bus.subscribe("topic.x", good_handler)

    event = build_event(
        correlation_id="c",
        session_id="s",
        phase="anchor",
        topic="topic.x",
        source="orchestrator",
        budget_tier="HIGH",
    )

    # publish must NOT raise — the bus stays crash-isolated.
    await bus.publish(event.topic, event)

    # The crash was recorded + forwarded to the callback.
    assert len(bus.handler_failures) == 1
    assert "boom" in bus.handler_failures[0].error
    assert bus.handler_failures[0].topic == "topic.x"
    assert failures_via_callback == bus.handler_failures

    # The other subscriber still ran — isolation, not abort.
    assert good_calls == [event.id]
