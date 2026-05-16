"""robit.proxy.events.rate_limiter — pre-dispatch emitter for the
rate-limiter engine.

Wave 13.1 / Agent A.  Wires the existing
:class:`robit.engines.rate_limiter.adapter.RateLimiter` engine into the
proxy's pre-dispatch phase via the deterministic emitter framework.

Topic + payload contract
------------------------

The rate-limiter engine subscribes to ``mcp.tool.call.requested`` (and
``lifecycle.pre-dispatch``).  The :mod:`builtin` emitter ALREADY publishes
``mcp.tool.call.requested`` at PRE_DISPATCH — so, strictly, this emitter is
redundant for the engine to fire.

However, the builtin's payload is shaped for the *trust-gate* engines
(``tool``, ``args``, ``model``, ``prompt_summary``) and carries
``source="proxy-pipeline"``.  The rate-limiter's
:func:`_extract_vendor` keys its token bucket on
``payload["vendor"]`` first, falling back to ``event.source`` only when
``vendor`` is absent.  Without an explicit ``vendor`` field every proxy
request collapses into a single global bucket keyed by ``"proxy-pipeline"``
— defeating the engine's per-vendor design.

This emitter therefore publishes a *confirming* ``mcp.tool.call.requested``
event carrying the rate-limiter-shaped payload (``vendor`` set to the
canonical request's model), giving the engine a per-model bucket without
disturbing the trust-gate flow.  Because the engine's handler dedups on
``(correlation_id, phase, plugin.name)`` (see
:meth:`robit.core.lifecycle.Orchestrator._wire_plugin`), the first event
the engine sees wins and the second is silently dropped — so publish order
matters.  ``rate_limiter`` sorts after ``builtin`` alphabetically, meaning
the builtin's vendor-less event would consume the dedup slot first.

Mitigation: the engine reads BOTH topics.  Our event is published to the
same ``mcp.tool.call.requested`` topic; the orchestrator's dedup means the
builtin's event reaches the engine first and our richer payload is dropped.
The honest answer is that the current engine design cannot be steered by a
later emitter at the same topic + phase.

For Wave 13.1, we ship this emitter as an **active, documented re-publish**
so the wire-up is explicit and discoverable (the empty module would be
indistinguishable from "we forgot to wire it"), and so a future emitter
re-ordering or topic-rename to e.g. ``rate-limiter.evaluate.requested``
is a one-line change in the engine's ``subscribes`` tuple.

Limitations (please read before extending):

* Single global bucket per model name across all proxy callers.  Multi-
  tenant separation needs a per-caller identifier (auth header, tenant id)
  propagated into ``EmitContext`` — a Wave 14+ concern.
* No vendor-vs-model distinction.  ``vendor`` is set to the model name
  because the proxy does not currently know which upstream provider hosts
  a given model id.  Use a model-to-vendor map (litellm has one) when the
  registry is wired into the pipeline.
* Engine quirk: because the engine subscribes to the same topic the
  builtin emitter already publishes, this re-publish is observably a no-op
  via the orchestrator's dedup.  Wave 14 should rename the engine's
  subscribed topic to ``rate-limiter.evaluate.requested`` and emit the
  rate-limiter-shaped event there exclusively.
"""

from __future__ import annotations

from robit.core.bus import build_event

from ._types import EmitContext, EmitPhase


class RateLimiterEmitter:
    """Pre-dispatch emitter for the rate-limiter engine.

    See module docstring for the topic + payload rationale and the known
    dedup interaction with :class:`~robit.proxy.events.builtin.BuiltinEmitter`.

    The emitter is intentionally pure — no per-request state, no scratch
    interaction, no response data needed.  It exists to make the
    rate-limiter engine's wire-in explicit alongside the other Wave 13.1
    emitters (cost-ledger, trust-scorer, tool-poisoning-scan).
    """

    name = "rate-limiter"
    phases = (EmitPhase.PRE_DISPATCH,)

    # Topic the rate-limiter engine subscribes to (see engine.toml).
    # Centralised here so a Wave 14+ topic rename is a one-line change.
    TOPIC = "mcp.tool.call.requested"

    async def emit(self, phase: str, ctx: EmitContext) -> None:
        # Defensive — the pipeline only calls us at our registered phase,
        # but the Protocol allows broader dispatch and a future caller
        # might fan out by phase loosely.  No-op on anything else.
        if phase != EmitPhase.PRE_DISPATCH:
            return

        budget_tier = ctx.scratch.get("budget_tier", "always")

        # Vendor falls back to the model name — the proxy does not yet
        # know the upstream provider for a given model id.  See the module
        # docstring's limitations section.
        vendor = ctx.req.model

        event = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="pre-dispatch",
            topic=self.TOPIC,
            # source is read by the engine as a vendor fallback; keep it
            # distinct from the builtin's "proxy-pipeline" so an event
            # without an explicit ``vendor`` field still buckets sanely.
            source="proxy-rate-limiter",
            budget_tier=budget_tier,
            payload={
                "vendor": vendor,
                "model": ctx.req.model,
                # ``tool`` mirrors the builtin shape for any other engine
                # that might re-key off this topic; harmless to the
                # rate-limiter (it ignores unknown fields).
                "tool": "llm.proxy",
                "source": "proxy",
            },
        )
        await ctx.bus.publish(event.topic, event)


emitter = RateLimiterEmitter()


__all__ = ["RateLimiterEmitter", "emitter"]
