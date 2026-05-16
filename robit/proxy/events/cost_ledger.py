"""robit.proxy.events.cost_ledger — Wave 13.1 cost-ledger emitter.

Publishes one ``cost.ledger.recorded`` bus event per request at
:attr:`EmitPhase.POST_SESSION`.  The event carries the computed cost in
US-cent units (rounded up to the next whole cent) plus the upstream model
identifier, so the proxy's HTTP frontend can surface them as
``X-Enchanter-Cost-Cents`` on unary responses.

Why a dedicated emitter (the cost-ledger engine already subscribes to
``mcp.tool.result.received``)
====================================================================

The :mod:`robit.engines.cost_ledger` engine consumes the builtin
``mcp.tool.result.received`` event and emits ``cost-ledger.appended`` as a
derived event.  Two gaps stop that engine-emitted event from satisfying
this contract:

#. The engine emits with ``source="cost-ledger"`` and topic
   ``cost-ledger.appended`` — neither value matches
   :attr:`robit.proxy.pipeline._BusRecorder._INTERESTING_SOURCES`
   nor the ``.veto/.warn/.matched/accumulator-truncated`` suffix gate, so
   the recorder drops it.
#. The engine's payload contains ``input_tokens``/``output_tokens``, never
   a computed cost — pricing is not part of its responsibility.

This emitter closes both gaps without touching the engine or
:mod:`robit.proxy.pipeline`:

* It publishes with ``source="proxy-pipeline"``, which the recorder marks
  interesting (see ``_INTERESTING_SOURCES``).
* It computes the cost from a small per-million-token price map.  The
  numeric cost lands under ``payload["score"]`` because ``score`` is one
  of the keys :func:`_BusRecorder._summarise_payload` whitelists for
  ``payload_summary``.  Reusing ``score`` is a known compromise; see the
  "Known limitations" section below.

Token counts
------------

For unary requests we read :attr:`CanonicalResponse.usage` (the upstream
authority).  For streaming requests the pipeline does not currently
surface ``usage`` — until that lands we fall back to a chars-per-token
estimate of ``len(ctx.accumulated_text) // 4`` for output and ``0`` for
input.  Operators consuming the cost header for streaming responses
should treat the figure as a lower bound, not a precise charge.

Pricing
-------

The pricing table is a tiny, hand-curated map keyed by model-name
prefix.  Unknown models fall back to a conservative middle estimate so
``X-Enchanter-Cost-Cents`` is always defined for non-zero usage but the
caller can detect "unknown model" by joining the value against the
table.  Real per-vendor accounting belongs in the cost-ledger engine —
this emitter is intentionally lightweight.

Known limitations
-----------------

#. **Pricing-table staleness.**  Hardcoded prices in
   :data:`_PRICE_CENTS_PER_M_TOKENS` will drift as vendors change rates.
   Operators should refresh the table when a recurring discrepancy with
   vendor invoices appears.
#. **Missing-model fallback.**  Any model not matched by the table uses
   :data:`_DEFAULT_PRICE_INPUT_CENTS_PER_M` and
   :data:`_DEFAULT_PRICE_OUTPUT_CENTS_PER_M` — fine for telemetry, not
   fine for chargeback.  Add the model explicitly when accuracy matters.
#. **Streaming-output estimate.**  Output tokens for streaming requests
   are estimated at ``chars / 4``.  Real tokenisation is provider-
   specific and the actual count typically differs by ±15%.
#. **Score-field reuse.**  ``cents`` is carried in
   ``payload["score"]`` because :class:`_BusRecorder` whitelists ``score``
   but not ``cents``.  A Wave 13.2 pipeline change should add ``cents``
   to the whitelist and migrate this emitter to write its own key.
#. **Rounding.**  Sub-cent costs round UP to one cent so a tiny request
   still surfaces a header — easier for downstream alarms to detect.
   Strict zero-cost flows therefore land at ``X-Enchanter-Cost-Cents:
   1``.  Operators wanting strict floor-zero should subtract one from
   single-cent observations during analysis.
"""

from __future__ import annotations

from robit.core.bus import build_event

from ._types import EmitContext, EmitPhase


# ---------------------------------------------------------------------------
# Pricing table — cents per 1,000,000 tokens, keyed by model-name prefix.
# Values are illustrative middle-of-2024 list-price snapshots; refresh as
# vendors update rates.  Keys are matched longest-prefix-first.
# ---------------------------------------------------------------------------

# (input_cents_per_million, output_cents_per_million)
_PRICE_CENTS_PER_M_TOKENS: dict[str, tuple[int, int]] = {
    # Anthropic
    "claude-3-5-sonnet": (300, 1500),
    "claude-3-5-haiku": (80, 400),
    "claude-3-opus": (1500, 7500),
    "claude-3-sonnet": (300, 1500),
    "claude-3-haiku": (25, 125),
    # OpenAI
    "gpt-4o-mini": (15, 60),
    "gpt-4o": (250, 1000),
    "gpt-4-turbo": (1000, 3000),
    "gpt-4": (3000, 6000),
    "gpt-3.5-turbo": (50, 150),
    "o1-preview": (1500, 6000),
    "o1-mini": (300, 1200),
    # Google
    "gemini-1.5-pro": (125, 500),
    "gemini-1.5-flash": (8, 30),
    "gemini-1.0-pro": (50, 150),
}

# Fallbacks for unknown models — conservative middle estimates so headers
# remain populated.  See "Missing-model fallback" in the module docstring.
_DEFAULT_PRICE_INPUT_CENTS_PER_M: int = 100
_DEFAULT_PRICE_OUTPUT_CENTS_PER_M: int = 300

# Chars-per-token estimate for streaming requests where the pipeline has not
# surfaced upstream usage yet.  See "Streaming-output estimate" above.
_CHARS_PER_TOKEN_ESTIMATE: int = 4


def _price_for(model: str) -> tuple[int, int]:
    """Return ``(input_cents_per_million, output_cents_per_million)`` for *model*.

    Longest-prefix match wins so e.g. ``claude-3-5-sonnet-20241022`` picks the
    ``claude-3-5-sonnet`` row rather than the broader ``claude-3-sonnet``
    fallback.
    """
    best_key = ""
    for key in _PRICE_CENTS_PER_M_TOKENS:
        if model.startswith(key) and len(key) > len(best_key):
            best_key = key
    if best_key:
        return _PRICE_CENTS_PER_M_TOKENS[best_key]
    return (_DEFAULT_PRICE_INPUT_CENTS_PER_M, _DEFAULT_PRICE_OUTPUT_CENTS_PER_M)


def _compute_cents(input_tokens: int, output_tokens: int, model: str) -> int:
    """Return the integer-cents charge for the given usage.

    Rounds UP to the next whole cent for any non-zero usage so downstream
    consumers can rely on ``cents > 0`` whenever real work happened.
    """
    if input_tokens <= 0 and output_tokens <= 0:
        return 0
    in_rate, out_rate = _price_for(model)
    # Integer math to keep the result deterministic across platforms.
    # cents = (in_tokens * in_rate + out_tokens * out_rate) / 1_000_000
    numerator = input_tokens * in_rate + output_tokens * out_rate
    # Ceiling division: any leftover sub-cent rounds to the next integer.
    cents = numerator // 1_000_000
    if numerator % 1_000_000 != 0:
        cents += 1
    return max(0, int(cents))


class CostLedgerEmitter:
    """POST_SESSION emitter that surfaces per-request cost.

    Discovery contract: module-level :data:`emitter` instance picked up by
    :func:`robit.proxy.events.load_emitters`.  Module name sorts
    alphabetically AFTER ``builtin`` so the builtin's
    ``mcp.tool.result.received`` publish runs first — that's the event the
    :class:`~robit.engines.cost_ledger.adapter.CostLedger` engine
    subscribes to.  This emitter is independent telemetry, not a
    replacement for the engine's bookkeeping.
    """

    name = "cost-ledger"
    phases = (EmitPhase.POST_SESSION,)

    async def emit(self, phase: str, ctx: EmitContext) -> None:
        # Be defensive: the pipeline only drives POST_SESSION for this
        # emitter, but the protocol allows multi-phase emitters so guard
        # against an unexpected dispatch.
        if phase != EmitPhase.POST_SESSION:
            return

        # Pull the authoritative usage from the unary response when present.
        if ctx.response is not None:
            input_tokens = int(ctx.response.usage.input_tokens)
            output_tokens = int(ctx.response.usage.output_tokens)
            model = ctx.response.model
        else:
            # Streaming path — pipeline does not currently feed usage.
            input_tokens = 0
            text = ctx.accumulated_text or ""
            output_tokens = len(text) // _CHARS_PER_TOKEN_ESTIMATE
            model = ctx.req.model

        cents = _compute_cents(input_tokens, output_tokens, model)
        if cents <= 0:
            # Nothing to surface — don't pollute the bus with zero-value
            # observations (the secret-mask path uses the same pattern).
            return

        # Stash the figure in scratch so siblings (rate-limiter, trust-
        # scorer) can read the realised cost without re-deriving it.
        # Namespace by emitter name per the EmitContext scratch convention.
        ctx.scratch.setdefault("cost-ledger", {})["cents"] = cents
        ctx.scratch["cost-ledger"]["model"] = model

        # ``score`` carries the cents — see "Score-field reuse" in the
        # module docstring.  Source is ``proxy-pipeline`` so
        # ``_BusRecorder.is_interesting`` accepts the event.
        event = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="post-response",
            topic="cost.ledger.recorded",
            source="proxy-pipeline",
            budget_tier=ctx.scratch.get("budget_tier", "always"),
            payload={
                "score": cents,
                "model": model,
                # The raw fields are retained for engines that subscribe
                # directly to the bus and bypass _BusRecorder.  They are
                # dropped from payload_summary by the recorder's
                # whitelist — see _BusRecorder._summarise_payload.
                "cents": cents,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        await ctx.bus.publish(event.topic, event)


emitter = CostLedgerEmitter()


__all__ = ["CostLedgerEmitter", "emitter"]
