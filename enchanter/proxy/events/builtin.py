"""enchanter.proxy.events.builtin — the three pre-refactor bus publishes.

Ports the exact publishes that used to live inline in
:mod:`enchanter.proxy.pipeline`:

* ``mcp.tool.call.requested`` — what destructive-op-gate / cve-pattern-gate
  subscribe to (PRE_DISPATCH).
* ``llm.proxy.request`` — branded variant for general listeners (PRE_DISPATCH).
* ``mcp.tool.result.received`` — what secret-mask subscribes to (POST_SESSION).
* ``llm.proxy.response`` — branded variant; now also carries
  ``mid_stream_redactions`` so Wave 13.1 observers (cost-ledger, trust-
  scorer) can detect mid-stream leak events without re-scanning text.

Behaviour MUST stay identical to the pre-refactor pipeline so the smoke at
``scripts/smoke_proxy.py`` and all existing tests pass unchanged.
"""

from __future__ import annotations

from enchanter.core.bus import build_event

from ..canonical import CanonicalRequest, CanonicalResponse, TextPart
from ._types import EmitContext, EmitPhase


# Mirrors enchanter.proxy.pipeline._PROMPT_SUMMARY_LIMIT.  Duplicated to keep
# the events package import-light (pipeline imports events, not the other
# way round) — bump in both places if the trust-gate corpus shape changes.
_PROMPT_SUMMARY_LIMIT = 1024


def _first_user_text(req: CanonicalRequest) -> str:
    for msg in req.messages:
        if msg.role != "user":
            continue
        parts: list[str] = []
        for part in msg.content:
            if isinstance(part, TextPart):
                parts.append(part.text)
        if parts:
            return "".join(parts)
    return ""


def _prompt_summary(req: CanonicalRequest) -> str:
    text = _first_user_text(req)
    if len(text) > _PROMPT_SUMMARY_LIMIT:
        return text[:_PROMPT_SUMMARY_LIMIT]
    return text


def _trust_gate_payload(req: CanonicalRequest) -> dict:
    """Tool/args-shaped payload for destructive-op-gate / cve-pattern-gate."""
    summary = _prompt_summary(req)
    return {
        "tool": "llm.completion",
        "args": [req.model, summary],
        "model": req.model,
        "prompt_summary": summary,
    }


def _budget_tier_of_ctx(bus_ctx) -> str:
    """Extract the budget_tier label from the ctx.  Default to 'always'."""
    # ctx (RequestContext) carries .budget_tier; the EmitContext doesn't,
    # so we accept either flavour.  Wave 13.1 emitters typically read
    # ctx.scratch["budget_tier"] populated by the builtin.
    return getattr(bus_ctx, "budget_tier", None) or "always"


class BuiltinEmitter:
    """Ports the three pre-refactor bus publishes into the emitter framework.

    Phases registered: ``PRE_DISPATCH`` and ``POST_SESSION``.  The discovery
    order makes this emitter the first one to fire — Wave 13.1 emitters
    that depend on the trust-gate / post-response events landing first can
    sort alphabetically after ``"builtin"`` (modules starting with c-z).
    """

    name = "builtin"
    phases = (EmitPhase.PRE_DISPATCH, EmitPhase.POST_SESSION)

    async def emit(self, phase: str, ctx: EmitContext) -> None:
        if phase == EmitPhase.PRE_DISPATCH:
            await self._emit_pre_dispatch(ctx)
        elif phase == EmitPhase.POST_SESSION:
            await self._emit_post_session(ctx)
        # Other phases are silently ignored — keeps the contract permissive.

    async def _emit_pre_dispatch(self, ctx: EmitContext) -> None:
        payload = _trust_gate_payload(ctx.req)
        # Stash the budget_tier so the matching post-session call uses the
        # same value without re-deriving it.
        budget_tier = ctx.scratch.get("budget_tier", "always")

        event_a = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="trust-gate",
            topic="mcp.tool.call.requested",
            source="proxy-pipeline",
            budget_tier=budget_tier,
            payload=payload,
        )
        await ctx.bus.publish(event_a.topic, event_a)

        event_b = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="trust-gate",
            topic="llm.proxy.request",
            source="proxy-pipeline",
            budget_tier=budget_tier,
            payload={
                "model": ctx.req.model,
                "prompt_summary": payload["prompt_summary"],
            },
        )
        await ctx.bus.publish(event_b.topic, event_b)

    async def _emit_post_session(self, ctx: EmitContext) -> None:
        # Prefer accumulated_text (works for both unary and streaming).
        result_text = ctx.accumulated_text or ""
        input_tokens = 0
        output_tokens = 0
        model = ctx.req.model
        if ctx.response is not None:
            input_tokens = ctx.response.usage.input_tokens
            output_tokens = ctx.response.usage.output_tokens
            model = ctx.response.model

        budget_tier = ctx.scratch.get("budget_tier", "always")

        result_payload = {
            "result": result_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
        }
        event_a = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="post-response",
            topic="mcp.tool.result.received",
            source="proxy-pipeline",
            budget_tier=budget_tier,
            payload=result_payload,
        )
        await ctx.bus.publish(event_a.topic, event_a)

        response_payload: dict = {
            "result_length": len(result_text),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
        }
        # Surface mid-stream redactions so Wave 13.1 observers can react
        # without having to re-scan the corpus.  Empty list for unary.
        if ctx.redactions:
            response_payload["mid_stream_redactions"] = list(ctx.redactions)

        event_b = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="post-response",
            topic="llm.proxy.response",
            source="proxy-pipeline",
            budget_tier=budget_tier,
            payload=response_payload,
        )
        await ctx.bus.publish(event_b.topic, event_b)


emitter = BuiltinEmitter()


__all__ = ["BuiltinEmitter", "emitter"]
