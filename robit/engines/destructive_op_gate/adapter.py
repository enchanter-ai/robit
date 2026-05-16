"""DestructiveOpGate engine — fires at trust-gate, fails closed on a match.

Scans two corpus views of the event payload (faithful to the TS sylph
adapter):
  1. JSON-stringified payload (catches inline string args)
  2. Reconstructed command line `<tool> <args.join(' ')>` so patterns
     anchored on tool-name boundaries match even when MCP splits the
     tool name from its argument array.

Required: True. Phase: trust-gate. On critical match: veto + derived event.
On advisory match (plain git push): ack with degraded=True.
"""

from __future__ import annotations

import json
import time

from robit.core import EnchantedEvent, PluginAck, RequestContext
from robit.core.plugin import PluginTopics
from robit.core.bus import new_event_id

from .patterns import DESTRUCTIVE_OP_PATTERNS


def _now_ms() -> int:
    return int(time.time() * 1000)


def _corpora_from_payload(payload: dict[str, object]) -> list[str]:
    """Return both corpus views (JSON dump + reconstructed command line)."""
    corpora: list[str] = [json.dumps(payload, default=str)]
    tool = payload.get("tool")
    args = payload.get("args")

    tool_str = tool if isinstance(tool, str) else ""
    if isinstance(args, list) and all(isinstance(a, str) for a in args):
        arg_str = " ".join(args)  # type: ignore[arg-type]
    elif isinstance(args, str):
        arg_str = args
    else:
        arg_str = ""

    if tool_str or arg_str:
        corpora.append(f"{tool_str} {arg_str}".strip())
    return corpora


class DestructiveOpGate:
    """Required at trust-gate. Fail-closed veto on W5 patterns."""

    name = "destructive-op-gate"
    phases = ("trust-gate",)
    required = True
    topics = PluginTopics(
        subscribes=(
            "mcp.tool.call.requested",
            "lifecycle.trust-gate",
        ),
        emits=("destructive-op-gate.veto",),
    )
    budget_tier = "always"

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        if event.phase != "trust-gate":
            return PluginAck(status="ack")

        payload = dict(event.payload or {})
        corpora = _corpora_from_payload(payload)

        for pattern in DESTRUCTIVE_OP_PATTERNS:
            hit = any(pattern.regex.search(c) for c in corpora)
            if not hit:
                continue

            if not pattern.requires_consent:
                # Advisory only — ack with degraded flag.
                return PluginAck(
                    status="ack",
                    degraded=True,
                    reason=f"{self.name}:{pattern.id} (advisory)",
                )

            # Veto + derived event.
            derived = EnchantedEvent(
                id=new_event_id(),
                correlation_id=event.correlation_id,
                session_id=event.session_id,
                phase=event.phase,
                topic="destructive-op-gate.veto",
                source=self.name,
                budget_tier=event.budget_tier,
                ts=_now_ms(),
                payload={
                    "pattern_id": pattern.id,
                    "pattern_name": pattern.name,
                },
            )
            return PluginAck(
                status="veto",
                reason=f"{self.name}:{pattern.id}",
                derived_events=[derived],
            )

        return PluginAck(status="ack")


adapter = DestructiveOpGate()
