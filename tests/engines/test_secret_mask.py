"""Tests for the secret-mask engine.

Six required scenarios:
  1. AWS access key in result is masked + event emitted.
  2. Bearer token in result is masked + event emitted.
  3. PEM block in result is masked + event emitted.
  4. Clean result passes through with no derived event.
  5. Result missing from payload is handled cleanly (no crash, just ack).
  6. String AND dict result shapes both supported.
"""

from __future__ import annotations

import pytest

from enchanter.core import (
    EnchantedEvent,
    InProcessBus,
    Orchestrator,
    OrchestratorConfig,
    PluginAck,
    RequestContext,
    create_request_context,
)
from enchanter.core.bus import build_event
from enchanter.engines.secret_mask import adapter as mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _publish_tool_result(
    bus: InProcessBus,
    ctx: RequestContext,
    result: object,
) -> None:
    """Publish an mcp.tool.result.received event at post-response phase."""
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="post-response",
        topic="mcp.tool.result.received",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={"result": result},
    )
    await bus.publish(event.topic, event)


async def _run_with_result(result: object) -> tuple[RequestContext, InProcessBus]:
    """Publish a post-response event and run the orchestrator; return ctx + bus."""
    bus = InProcessBus()
    registry = {mask.name: mask}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    ctx = create_request_context()
    await _publish_tool_result(bus, ctx, result)

    async def dispatch(c: RequestContext) -> str:
        return "ok"

    await orch.run(ctx, dispatch)
    return ctx, bus


def _matched_events(bus: InProcessBus, correlation_id: str) -> list[EnchantedEvent]:
    return [
        e for e in bus.tap(correlation_id)
        if e.topic == "secret-mask.matched"
    ]


# ---------------------------------------------------------------------------
# Test 1: AWS access key
# ---------------------------------------------------------------------------

async def test_aws_access_key_is_masked_and_event_emitted():
    """An AWS access key in the result triggers masking and a derived event."""
    secret = "AKIAIOSFODNN7EXAMPLE"  # 20-char AKIA key (AKIA + 16 chars)
    result_text = f"The tool output contains key={secret} for debugging."

    ctx, bus = await _run_with_result(result_text)

    events = _matched_events(bus, ctx.correlation_id)
    assert len(events) == 1, "Expected exactly one secret-mask.matched event"
    payload = events[0].payload
    assert "s-aws-key" in payload["matched_patterns"]
    # Redacted length should be shorter (or same) than the original — the
    # replacement "AKIA****[REDACTED]" is shorter than a 20-char key.
    assert payload["redacted_length"] > 0


# ---------------------------------------------------------------------------
# Test 2: Bearer token
# ---------------------------------------------------------------------------

async def test_bearer_token_is_masked_and_event_emitted():
    """A Bearer token in an Authorization header value triggers masking."""
    result_text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"

    ctx, bus = await _run_with_result(result_text)

    events = _matched_events(bus, ctx.correlation_id)
    assert len(events) == 1
    assert "s-bearer-token" in events[0].payload["matched_patterns"]


# ---------------------------------------------------------------------------
# Test 3: PEM private key block
# ---------------------------------------------------------------------------

async def test_pem_private_key_is_masked_and_event_emitted():
    """A PEM PRIVATE KEY block in the result is fully redacted."""
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Z3VS5JJcds3xHn/ygWep4PAtEsHAh0bEQ==\n"
        "-----END RSA PRIVATE KEY-----"
    )
    result_text = f"Key material:\n{pem}\nEnd of output."

    ctx, bus = await _run_with_result(result_text)

    events = _matched_events(bus, ctx.correlation_id)
    assert len(events) == 1
    assert "s-pem-private-key" in events[0].payload["matched_patterns"]


# ---------------------------------------------------------------------------
# Test 4: Clean result passes through with no derived event
# ---------------------------------------------------------------------------

async def test_clean_result_passes_with_no_derived_event():
    """A result containing no secrets produces no secret-mask.matched event."""
    result_text = "The calculation result is 42. No secrets here."

    ctx, bus = await _run_with_result(result_text)

    events = _matched_events(bus, ctx.correlation_id)
    assert len(events) == 0, "Expected no derived event on clean result"


# ---------------------------------------------------------------------------
# Test 5: Missing result field handled cleanly (no crash)
# ---------------------------------------------------------------------------

async def test_missing_result_field_does_not_crash():
    """A payload with no 'result' key is handled gracefully — just acks."""
    bus = InProcessBus()
    registry = {mask.name: mask}
    orch = Orchestrator(OrchestratorConfig(registry=registry, bus=bus))

    ctx = create_request_context()
    # Publish without a 'result' key in the payload.
    event = build_event(
        correlation_id=ctx.correlation_id,
        session_id=ctx.session_id,
        phase="post-response",
        topic="mcp.tool.result.received",
        source="test",
        budget_tier=ctx.budget_tier,
        payload={"status": "ok"},  # no 'result' key
    )
    await bus.publish(event.topic, event)

    async def dispatch(c: RequestContext) -> str:
        return "ok"

    # Must not raise.
    result = await orch.run(ctx, dispatch)
    assert result == "ok"
    assert len(_matched_events(bus, ctx.correlation_id)) == 0


# ---------------------------------------------------------------------------
# Test 6: String AND dict result shapes both supported
# ---------------------------------------------------------------------------

async def test_dict_result_shape_is_scanned():
    """A dict result (JSON-serialised corpus) is also scanned for secrets."""
    secret = "AKIAIOSFODNN7EXAMPLE"
    result_dict = {"output": f"key={secret}", "status": 200}

    ctx, bus = await _run_with_result(result_dict)

    events = _matched_events(bus, ctx.correlation_id)
    assert len(events) == 1, "Expected one event when secret is inside a dict result"
    assert "s-aws-key" in events[0].payload["matched_patterns"]


async def test_string_result_shape_is_scanned():
    """A plain string result is scanned directly (no JSON serialisation needed)."""
    result_text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789ABCD"

    ctx, bus = await _run_with_result(result_text)

    events = _matched_events(bus, ctx.correlation_id)
    assert len(events) == 1
    assert "s-bearer-token" in events[0].payload["matched_patterns"]
