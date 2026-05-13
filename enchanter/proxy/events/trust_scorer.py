"""enchanter.proxy.events.trust_scorer — wire proxy completions into the
trust-scorer Beta-Bernoulli posterior.

The trust-scorer engine (``enchanter.engines.trust_scorer``) maintains a
per-(server_id, tool_name) posterior.  At ``trust-gate`` the engine *reads*
the posterior and emits ``trust-scorer.trust.scored`` so callers can inspect
the current score; it does NOT auto-record observations from any bus topic.

This emitter is the recording side of that contract.  At POST_SESSION it
calls ``adapter.record_success`` on the trust-scorer singleton, mapping the
proxy's natural key shape ``(model_id, "completion")`` onto the engine's
``(server_id, tool_name)`` key.  A companion ``trust-scorer.observation.recorded``
bus event is also published so the recorder can surface "we ticked the
posterior" telemetry — handy for Agent E response headers and for tests
that prefer asserting on bus events over engine internals.

Phase ordering
--------------

The pipeline fires PRE_DISPATCH emitters (where the built-in publishes
``mcp.tool.call.requested`` at phase ``trust-gate``, which is what the
trust-scorer engine subscribes to *for the read path*).  Only AFTER the
upstream call succeeds does the pipeline fire POST_SESSION emitters, where
this module's emitter records the observation.

Success-only limitation
-----------------------

Today's pipeline only reaches POST_SESSION when the upstream call
succeeded — vetoes return synchronously and upstream errors raise before
POST_SESSION runs.  So this emitter is a *success counter* only; failures
are not yet recorded against the posterior.  When the pipeline grows a
"POST_SESSION even on upstream error" branch, switch this emitter to
inspect ``ctx.response`` / a future failure flag and call
``record_failure`` accordingly.  Until then, the posterior is biased
upward — document this when consumers compare scores across models.

Multi-tenant note
-----------------

``server_id`` here is the LiteLLM model id (``gpt-4o-mini``,
``claude-3-5-sonnet-latest``, ...).  When the proxy grows tenant isolation
the key should compose tenant + model (``"<tenant>:<model>"``) so a noisy
tenant cannot poison another tenant's trust signal.  The engine's key
shape is opaque ``str`` on both sides so this is a pure proxy-side change.
"""

from __future__ import annotations

from enchanter.core.bus import build_event

# Import the engine singleton directly.  ``adapter`` is module-level in the
# engine package, so this is the *same* instance that ``load_engine_registry``
# returns — Python's module cache guarantees the identity.  The engine has
# no subscription that records observations; the public ``record_success``/
# ``record_failure`` methods are the documented hook.
from enchanter.engines.trust_scorer.adapter import adapter as _trust_engine

from ._types import EmitContext, EmitPhase


# Synthetic ``tool_name`` used for every completion observation — the engine
# expects a string, and "completion" matches the trust-gate corpus emitted
# by the builtin emitter (``tool="llm.completion"``).  Kept short so the
# bus event payload stays small.
_COMPLETION_TOOL = "completion"

# Topic this emitter publishes.  Not subscribed by the engine; it exists so
# the ``_BusRecorder`` in :mod:`enchanter.proxy.pipeline` (which lists
# ``trust-scorer`` as an interesting source) can surface "an observation
# was recorded" telemetry.  Sister emitters / Agent E may key headers on
# this topic name; treat it as part of the public bus surface.
_TOPIC_OBSERVATION = "trust-scorer.observation.recorded"


class TrustScorerEmitter:
    """Post-session emitter recording ``(model, completion, success)``
    observations into the trust-scorer engine's Beta-Bernoulli posterior.

    The emitter is intentionally idempotent in its side-effects per call:
    one POST_SESSION fire → one ``record_success`` call → one bus event.
    Tests can assert posterior updates via ``_trust_engine.store`` or via
    the published bus event, whichever is more convenient.
    """

    name = "trust-scorer"
    phases = (EmitPhase.POST_SESSION,)

    async def emit(self, phase: str, ctx: EmitContext) -> None:
        # Guard for callers that route every phase to every emitter — the
        # discovery layer respects ``phases`` but tests sometimes invoke
        # ``emit`` directly with a non-matching phase.
        if phase != EmitPhase.POST_SESSION:
            return

        server_id = ctx.req.model
        tool_name = _COMPLETION_TOOL

        # Pipeline contract: POST_SESSION only runs when the upstream call
        # succeeded.  Record success.  See module docstring for the
        # known-limitation around failure counting.
        _trust_engine.record_success(server_id, tool_name)

        # Publish a small bus event so the recorder / Agent E can surface
        # the observation.  Payload is intentionally tiny — pattern-id-ish
        # scalars only, no request content.
        score = _trust_engine.score(server_id, tool_name)
        n = _trust_engine.store.observation_count((server_id, tool_name))
        budget_tier = ctx.scratch.get("budget_tier", "always")

        event = build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase="post-response",
            topic=_TOPIC_OBSERVATION,
            source=self.name,
            budget_tier=budget_tier,
            payload={
                "server_id": server_id,
                "tool_name": tool_name,
                "outcome": "success",
                "score": score,
                "observation_count": n,
            },
        )
        await ctx.bus.publish(event.topic, event)


emitter = TrustScorerEmitter()


__all__ = ["TrustScorerEmitter", "emitter"]
