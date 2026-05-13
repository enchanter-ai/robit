"""enchanter.proxy.events.tool_poisoning_scan — pre-dispatch tool-schema scanner wire-in.

Publishes a ``mcp.tool.registered`` bus event for every :class:`~enchanter.
proxy.canonical.Tool` carried on the inbound request.  The ``tool-poisoning-
scan`` engine (``enchanter.engines.tool_poisoning_scan.adapter``) subscribes
to that topic and runs its M1 static scan against the tool schema; a
suspicion score at or above ``VETO_THRESHOLD`` causes the engine to ack
``veto`` for the ``post-response`` phase, which the orchestrator surfaces as
:class:`~enchanter.proxy.pipeline.VetoResult` from :func:`pipeline.run`.

Phase mapping
-------------
The engine declares ``phases=("post-response",)`` and the orchestrator's
wiring (`enchanter.core.lifecycle._wire_plugin`) short-circuits any inbound
event whose ``event.phase`` is not in the plugin's ``phases`` tuple.  This
emitter therefore stamps the published events with ``phase="post-response"``
even though the EMITTER itself fires at ``PRE_DISPATCH`` — the goal is to
have the engine's veto landed in the bus ack tracker BEFORE the
orchestrator reaches its own post-response wait, so a poisoned tool
short-circuits the upstream call.

Payload shape
-------------
The engine extracts text corpora from ``payload["tool_schema"]`` via its
:func:`_extract_corpora` helper, which expects:

* ``description`` (str)
* ``parameters`` (dict) OR ``inputSchema.properties`` (dict-of-dicts)
* ``errorTemplates`` (str|object, optional)
* ``name`` / ``displayName`` (str, optional)

We map :class:`Tool` → engine schema as:

    {"name":         tool.name,
     "description":  tool.description,
     "inputSchema":  tool.input_schema}

— ``inputSchema`` is the JSON-Schema-ish dict already on the Tool dataclass
and matches the engine's ``inputSchema.properties`` extraction branch.

Multi-tool semantics
--------------------
The engine's handler dedups by ``(correlation_id, phase, plugin)`` — only
the FIRST event the engine processes for a given phase actually runs M1;
subsequent events on the same phase are short-circuited by the orchestrator
wiring.  We still publish one event per tool because:

  1. The bus event log is the audit trail; one event per tool surfaces each
     tool in the recorder + Agent E's response headers.
  2. If/when the engine moves to per-tool ack keys (or post-response acks
     become per-event), the wire is already in place.

For now, the practical effect is "scan the first tool; veto if poisoned".
Wave 13.2 should either re-key the engine ack tracker per tool or shift
the per-tool fan-out into the engine itself.
"""

from __future__ import annotations

from enchanter.core.bus import build_event

from ._types import EmitContext, EmitPhase


class ToolPoisoningScanEmitter:
    """Pre-dispatch emitter — publishes a tool-poisoning scan event per tool.

    For requests with no ``tools`` array, this is a no-op.  For requests
    with one or more :class:`Tool`s, publishes ``mcp.tool.registered``
    events (one per tool) carrying ``tool_schema`` in the payload so the
    ``tool-poisoning-scan`` engine can run its M1 static scan.

    Discovery name: ``tool-poisoning-scan`` — sorts after ``builtin`` and
    after any sibling Wave 13.1 emitter whose module name precedes
    ``tool_poisoning_scan`` (alphabetical, e.g. ``cost_ledger``,
    ``rate_limiter``, ``trust_scorer``).
    """

    name = "tool-poisoning-scan"
    phases = (EmitPhase.PRE_DISPATCH,)

    async def emit(self, phase: str, ctx: EmitContext) -> None:
        if phase != EmitPhase.PRE_DISPATCH:
            return
        tools = ctx.req.tools
        if not tools:
            return

        budget_tier = ctx.scratch.get("budget_tier", "always")

        for tool in tools:
            event = build_event(
                correlation_id=ctx.correlation_id,
                session_id=ctx.session_id,
                # Stamp the engine's declared phase so its handler does not
                # short-circuit on the ``event.phase not in plugin.phases``
                # check in ``enchanter.core.lifecycle._wire_plugin``.
                phase="post-response",
                topic="mcp.tool.registered",
                source="proxy-pipeline",
                budget_tier=budget_tier,
                payload={
                    "tool_schema": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                    },
                },
            )
            await ctx.bus.publish(event.topic, event)


emitter = ToolPoisoningScanEmitter()


__all__ = ["ToolPoisoningScanEmitter", "emitter"]
