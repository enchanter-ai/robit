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
* It computes the cost from the registry's per-million-token price map
  (``pricing.by_prefix`` in models-registry.json).  The numeric cost is
  published as ``payload["cents"]`` and also mirrored under
  ``payload["score"]`` because the pipeline-side
  ``_BusRecorder._summarise_payload`` whitelist (owned by another package)
  currently surfaces ``score`` but not ``cents`` into ``payload_summary``.
  See the "Known limitations" section below.

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

Prices come from the models registry (``pricing.by_prefix`` in
models-registry.json — the single source of truth for model data), read
read-only via :func:`_load_registry_prices`.  The map is keyed by
model-name prefix; unknown models fall back to a conservative middle
estimate so ``X-Enchanter-Cost-Cents`` is always defined for non-zero
usage but the caller can detect "unknown model" by joining the value
against the table.  The in-module :data:`_PRICE_CENTS_PER_M_TOKENS` is a
seed/fallback only, used when the registry has no pricing section.  Real
per-vendor accounting belongs in the cost-ledger engine — this emitter is
intentionally lightweight.

Known limitations
-----------------

#. **Pricing staleness.**  The registry's ``pricing.by_prefix`` prices
   will drift as vendors change rates.  Operators should refresh the
   registry section when a recurring discrepancy with vendor invoices
   appears.
#. **Missing-model fallback.**  Any model not matched by the registry map
   uses :data:`_DEFAULT_PRICE_INPUT_CENTS_PER_M` and
   :data:`_DEFAULT_PRICE_OUTPUT_CENTS_PER_M` — fine for telemetry, not
   fine for chargeback.  Add the model's prefix to the registry when
   accuracy matters.
#. **Streaming-output estimate.**  Output tokens for streaming requests
   are estimated at ``chars / 4``.  Real tokenisation is provider-
   specific and the actual count typically differs by ±15%.
#. **Score-field mirror.**  ``cents`` is published under its own
   ``payload["cents"]`` field, but is ALSO mirrored under
   ``payload["score"]`` because :class:`_BusRecorder` (in proxy/pipeline.py,
   owned by another package) whitelists ``score`` but not ``cents`` into the
   header-facing ``payload_summary``.  The scratch path no longer touches
   ``score``; drop the bus mirror once the recorder whitelists ``cents``.
#. **Rounding.**  Sub-cent costs round UP to one cent so a tiny request
   still surfaces a header — easier for downstream alarms to detect.
   Strict zero-cost flows therefore land at ``X-Enchanter-Cost-Cents:
   1``.  Operators wanting strict floor-zero should subtract one from
   single-cent observations during analysis.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from robit.core.bus import build_event
from robit.runtime.models_registry import _DEFAULT_REGISTRY

from ._types import EmitContext, EmitPhase

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing — cents per 1,000,000 tokens, keyed by model-name prefix.
#
# The models-registry.json ``pricing.by_prefix`` map is now the source of
# truth (the registry is robit's single source of truth for all model data).
# :func:`_load_registry_prices` reads it; the module-level
# :data:`_PRICE_CENTS_PER_M_TOKENS` below is kept ONLY as a seed/fallback for
# environments where the registry is absent or carries no pricing section, so
# behaviour never silently regresses.  Keys are matched longest-prefix-first.
# ---------------------------------------------------------------------------

# (input_cents_per_million, output_cents_per_million)
# Seed/fallback ONLY — must stay in sync with registry ``pricing.by_prefix``.
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


@lru_cache(maxsize=1)
def _load_registry_prices() -> dict[str, tuple[int, int]]:
    """Load ``pricing.by_prefix`` from the models registry (read-only).

    Reads the registry JSON directly via the loader's default path constant
    (``robit.runtime.models_registry._DEFAULT_REGISTRY``) — the registry is the
    source of truth for model data, including pricing.  Falls back to the
    in-module :data:`_PRICE_CENTS_PER_M_TOKENS` seed if the file is missing,
    malformed, or has no pricing section, so behaviour never regresses to a
    crash.  Cached because the registry is immutable for a process lifetime.
    """
    try:
        data = json.loads(_DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        by_prefix = data["pricing"]["by_prefix"]
        prices: dict[str, tuple[int, int]] = {}
        for prefix, pair in by_prefix.items():
            prices[prefix] = (int(pair[0]), int(pair[1]))
        if prices:
            return prices
    except (OSError, KeyError, ValueError, TypeError) as exc:
        _log.warning(
            "cost-ledger: falling back to seed price table (registry pricing "
            "unavailable: %s)",
            exc,
        )
    return dict(_PRICE_CENTS_PER_M_TOKENS)


def _price_for(model: str) -> tuple[int, int]:
    """Return ``(input_cents_per_million, output_cents_per_million)`` for *model*.

    Prices come from the models registry's ``pricing.by_prefix`` map (source of
    truth), with longest-prefix match — so e.g. ``claude-3-5-sonnet-20241022``
    picks the ``claude-3-5-sonnet`` row rather than the broader
    ``claude-3-sonnet`` fallback.  Unpriced models fall back to the
    ``_DEFAULT_PRICE_*`` constants.
    """
    prices = _load_registry_prices()
    best_key = ""
    for key in prices:
        if model.startswith(key) and len(key) > len(best_key):
            best_key = key
    if best_key:
        return prices[best_key]
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

        # Stash the figure in this emitter's OWN typed scratch bucket so
        # siblings (rate-limiter, trust-scorer) can read the realised cost
        # without re-deriving it.  The bucket is structurally isolated — no
        # ``score``-key smuggling, no risk of colliding with another emitter's
        # namespace (see robit.core.RequestScratchpad).
        bucket = ctx.scratchpad.for_emitter(self.name)
        bucket.cents = cents
        bucket.model = model

        # Bus payload.  ``cents`` is the real field every direct bus subscriber
        # reads.  ``score`` mirrors it ONLY because the pipeline-side
        # ``_BusRecorder._summarise_payload`` whitelist (in proxy/pipeline.py,
        # owned by another package) surfaces ``score`` but not ``cents`` into
        # the header-facing ``payload_summary``.  The scratch path above is now
        # ``score``-free; this remaining mirror is a documented cross-package
        # constraint, not a scratch-bucket hack — drop it once the recorder
        # whitelists ``cents``.  Source is ``proxy-pipeline`` so
        # ``_BusRecorder.is_interesting`` accepts the event.
        event = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="post-response",
            topic="cost.ledger.recorded",
            source="proxy-pipeline",
            budget_tier=ctx.scratch.get("budget_tier", "always"),
            payload={
                "cents": cents,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                # Recorder-whitelist mirror of ``cents`` — see comment above.
                "score": cents,
            },
        )
        await ctx.bus.publish(event.topic, event)


emitter = CostLedgerEmitter()


__all__ = ["CostLedgerEmitter", "emitter"]
