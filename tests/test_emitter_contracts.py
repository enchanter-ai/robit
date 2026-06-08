"""Tests for the EMITTER-CONTRACTS package (bus-contract-hardening).

Covers two consolidations:

Task 1 — Scratch-bucket consolidation (audit §5 Q4)
    * Per-emitter scratch isolation: two emitters writing the same key land in
      structurally-separate buckets and cannot collide.
    * The cost-ledger emitter surfaces ``cents`` through a typed scratch field
      (``EmitterScratch.cents``) — NOT the old ``payload["score"]`` smuggling.
    * The deprecated ``ctx.scratch`` dict view still routes legacy keys
      correctly (namespace -> isolated bucket, other -> shared).

Task 2 — Pricing source-of-truth (audit §5 Q2)
    * Registry-driven pricing yields the SAME cents as the historical
      hardcoded table for anthropic / openai / google samples.
    * The unknown-model fallback still works.
"""

from __future__ import annotations

import pytest

from robit.core import EmitterScratch, RequestScratchpad
from robit.proxy.canonical import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    Message,
    TextPart,
)
from robit.core import InProcessBus
from robit.proxy.events._types import EmitContext
from robit.proxy.events.cost_ledger import (
    _DEFAULT_PRICE_INPUT_CENTS_PER_M,
    _DEFAULT_PRICE_OUTPUT_CENTS_PER_M,
    _PRICE_CENTS_PER_M_TOKENS,
    _compute_cents,
    _load_registry_prices,
    _price_for,
    emitter as cost_emitter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(model: str = "gpt-4o-mini") -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=(Message(role="user", content=(TextPart(text="hi"),)),),
        max_tokens=64,
    )


def _resp(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> CanonicalResponse:
    return CanonicalResponse(
        model=model,
        content=(TextPart(text="ok"),),
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


async def _make_ctx(response: CanonicalResponse | None, model: str) -> tuple[EmitContext, list]:
    bus = InProcessBus()
    captured: list = []

    async def _capture(event):
        captured.append(event)
        return None

    bus.subscribe("*", _capture)
    ctx = EmitContext(
        req=_req(model=model),
        bus=bus,
        correlation_id="cid",
        session_id="sid",
        response=response,
    )
    return ctx, captured


# ---------------------------------------------------------------------------
# Task 1 — scratch isolation
# ---------------------------------------------------------------------------


def test_per_emitter_buckets_are_structurally_isolated():
    """Two emitters writing the SAME key cannot collide — separate buckets."""
    pad = RequestScratchpad()
    a = pad.for_emitter("alpha")
    b = pad.for_emitter("beta")

    a.data["shared_key"] = "alpha-value"
    b.data["shared_key"] = "beta-value"

    assert a is not b
    assert pad.for_emitter("alpha").data["shared_key"] == "alpha-value"
    assert pad.for_emitter("beta").data["shared_key"] == "beta-value"
    # Typed fields are likewise isolated.
    a.cents = 5
    b.cents = 99
    assert pad.for_emitter("alpha").cents == 5
    assert pad.for_emitter("beta").cents == 99


def test_for_emitter_is_idempotent():
    """Repeated for_emitter calls return the same bucket (created up front)."""
    pad = RequestScratchpad()
    first = pad.for_emitter("cost-ledger")
    first.cents = 7
    second = pad.for_emitter("cost-ledger")
    assert first is second
    assert second.cents == 7


def test_scratch_compat_view_routes_namespace_vs_shared():
    """Legacy ctx.scratch keys route: emitter-name -> bucket, other -> shared."""
    ctx = EmitContext(
        req=_req(),
        bus=InProcessBus(),
        correlation_id="cid",
        session_id="sid",
    )
    # Cross-cutting key -> shared.
    ctx.scratch["budget_tier"] = "HIGH"
    assert ctx.scratchpad.shared["budget_tier"] == "HIGH"
    # Emitter-namespace key -> that emitter's isolated bucket.
    ctx.scratch.setdefault("cost-ledger", {})["cents"] = 3
    assert ctx.scratchpad.for_emitter("cost-ledger").cents == 3
    # And reading back through the view works.
    assert ctx.scratch["cost-ledger"]["cents"] == 3
    assert "cost-ledger" in ctx.scratch
    assert ctx.scratch.get("budget_tier") == "HIGH"


def test_two_emitter_namespaces_dont_collide_through_compat_view():
    """The smell the refactor fixes: same key in two emitter namespaces."""
    ctx = EmitContext(
        req=_req(),
        bus=InProcessBus(),
        correlation_id="cid",
        session_id="sid",
    )
    ctx.scratch.setdefault("cost-ledger", {})["model"] = "gpt-4o"
    ctx.scratch.setdefault("trust-scorer", {})["model"] = "claude-opus-4-6"
    assert ctx.scratch["cost-ledger"]["model"] == "gpt-4o"
    assert ctx.scratch["trust-scorer"]["model"] == "claude-opus-4-6"


def test_emitcontext_replace_preserves_scratch_state():
    """dataclasses.replace must carry the scratchpad state forward."""
    from dataclasses import replace

    ctx = EmitContext(
        req=_req(),
        bus=InProcessBus(),
        correlation_id="cid",
        session_id="sid",
    )
    ctx.scratch["budget_tier"] = "MED"
    ctx.scratchpad.for_emitter("cost-ledger").cents = 42

    ctx2 = replace(ctx, pre_dispatch_done=True)
    assert ctx2.pre_dispatch_done is True
    assert ctx2.scratchpad.shared["budget_tier"] == "MED"
    assert ctx2.scratchpad.for_emitter("cost-ledger").cents == 42
    # The compat view stays bound to the same scratchpad.
    assert ctx2.scratch["cost-ledger"]["cents"] == 42


# ---------------------------------------------------------------------------
# Task 1 — cost-ledger surfaces cents WITHOUT the score-key hack
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_ledger_surfaces_cents_via_typed_scratch_field():
    """cents lands in the typed EmitterScratch.cents field, not a score key."""
    # claude-3-5-sonnet: 300/1500 c per M. 10k in + 5k out -> 10.5 -> 11 cents.
    resp = _resp("claude-3-5-sonnet-20241022", input_tokens=10_000, output_tokens=5_000)
    ctx, _captured = await _make_ctx(resp, model="claude-3-5-sonnet-20241022")

    await cost_emitter.emit(cost_emitter.phases[0], ctx)

    bucket = ctx.scratchpad.for_emitter("cost-ledger")
    assert isinstance(bucket, EmitterScratch)
    # The realised cost is reachable as a TYPED attribute — no "score" key.
    assert bucket.cents == 11
    assert bucket.model == "claude-3-5-sonnet-20241022"
    # The scratch bucket's free-form data never carried a "score" smuggle.
    assert "score" not in bucket.data
    # Reachable through the deprecated compat view too.
    assert ctx.scratch["cost-ledger"]["cents"] == 11


# ---------------------------------------------------------------------------
# Task 2 — registry-driven pricing == old hardcoded table
# ---------------------------------------------------------------------------


def test_registry_prices_match_seed_table_exactly():
    """The registry pricing map mirrors the historical hardcoded table."""
    assert _load_registry_prices() == _PRICE_CENTS_PER_M_TOKENS


@pytest.mark.parametrize(
    "model, expected",
    [
        # Anthropic — longest-prefix wins over claude-3-sonnet.
        ("claude-3-5-sonnet-20241022", (300, 1500)),
        ("claude-3-haiku-20240307", (25, 125)),
        # OpenAI.
        ("gpt-4o-mini", (15, 60)),
        ("gpt-4o-2024-08-06", (250, 1000)),
        ("o1-mini", (300, 1200)),
        # Google.
        ("gemini-1.5-pro-latest", (125, 500)),
        ("gemini-1.5-flash", (8, 30)),
    ],
)
def test_price_for_registry_driven_matches_old_values(model, expected):
    assert _price_for(model) == expected


@pytest.mark.parametrize(
    "model, input_tokens, output_tokens, expected_cents",
    [
        # Same arithmetic the hardcoded table produced before the migration.
        ("claude-3-5-sonnet-20241022", 10_000, 5_000, 11),
        ("gpt-4o-mini", 100, 50, 1),
        ("gemini-1.5-pro", 1_000_000, 0, 125),
    ],
)
def test_compute_cents_unchanged_after_registry_migration(
    model, input_tokens, output_tokens, expected_cents
):
    assert _compute_cents(input_tokens, output_tokens, model) == expected_cents


def test_unknown_model_falls_back_to_default_price():
    in_rate, out_rate = _price_for("zephyr-mystery-7b")
    assert (in_rate, out_rate) == (
        _DEFAULT_PRICE_INPUT_CENTS_PER_M,
        _DEFAULT_PRICE_OUTPUT_CENTS_PER_M,
    )
    # And the fallback still computes a non-zero, defined cost.
    assert _compute_cents(1, 0, "zephyr-mystery-7b") == 1
