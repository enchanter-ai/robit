"""CvePatternGate engine — fires at trust-gate, fails closed on critical CVE patterns.

Scans two corpus views of the event payload (faithful to the TS hydra adapter):
  1. JSON-stringified payload (catches inline string args and deep nested fields)
  2. Reconstructed command line `<tool> <args.join(' ')>` so patterns anchored on
     tool-name boundaries match even when MCP splits the tool name from its argument
     array.

Severity routing:
  critical → veto + derived cve-pattern-gate.veto event
  high / medium → ack with degraded=True + derived cve-pattern-gate.warn event
  low → ack with degraded=True (no derived event; informational only)
  no hit → ack
"""

from __future__ import annotations

import json
import time

from robit.core import EnchantedEvent, PluginAck, RequestContext
from robit.core.plugin import PluginTopics
from robit.core.bus import new_event_id

from .patterns import CVE_PATTERNS, CvePattern


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


class CvePatternGate:
    """Required at trust-gate. Fail-closed veto on critical CVE patterns."""

    name = "cve-pattern-gate"
    phases = ("trust-gate",)
    required = True
    topics = PluginTopics(
        subscribes=(
            "mcp.tool.call.requested",
            "lifecycle.trust-gate",
        ),
        emits=("cve-pattern-gate.veto", "cve-pattern-gate.warn"),
    )
    budget_tier = "always"

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        if event.phase != "trust-gate":
            return PluginAck(status="ack")

        payload = dict(event.payload or {})
        corpora = _corpora_from_payload(payload)

        # Collect unique hits across all corpus views.
        seen_ids: set[str] = set()
        hits: list[CvePattern] = []
        for pattern in CVE_PATTERNS:
            if pattern.id in seen_ids:
                continue
            if any(pattern.match.search(c) for c in corpora):
                hits.append(pattern)
                seen_ids.add(pattern.id)

        if not hits:
            return PluginAck(status="ack")

        # Critical takes precedence over everything.
        critical = next((h for h in hits if h.severity == "critical"), None)
        if critical is not None:
            derived = EnchantedEvent(
                id=new_event_id(),
                correlation_id=event.correlation_id,
                session_id=event.session_id,
                phase=event.phase,
                topic="cve-pattern-gate.veto",
                source=self.name,
                budget_tier=event.budget_tier,
                ts=_now_ms(),
                payload={
                    "pattern_id": critical.id,
                    "severity": critical.severity,
                    "cve_anchor": critical.cve_anchor,
                },
            )
            return PluginAck(
                status="veto",
                reason=(
                    f"{self.name}:{critical.id} ({critical.cve_anchor}): {critical.rationale}"
                ),
                derived_events=[derived],
            )

        # High / medium: advisory warn.
        warn_hits = [h for h in hits if h.severity in ("high", "medium")]
        if warn_hits:
            warn_ids = ",".join(h.id for h in warn_hits)
            derived = EnchantedEvent(
                id=new_event_id(),
                correlation_id=event.correlation_id,
                session_id=event.session_id,
                phase=event.phase,
                topic="cve-pattern-gate.warn",
                source=self.name,
                budget_tier=event.budget_tier,
                ts=_now_ms(),
                payload={
                    "pattern_ids": [h.id for h in warn_hits],
                    "severities": [h.severity for h in warn_hits],
                },
            )
            return PluginAck(
                status="ack",
                degraded=True,
                reason=f"cve-pattern-gate-warn: {warn_ids}",
                derived_events=[derived],
            )

        # Low hits only — ack degraded, no derived event.
        low_ids = ",".join(h.id for h in hits)
        return PluginAck(
            status="ack",
            degraded=True,
            reason=f"cve-pattern-gate-warn: {low_ids}",
        )


adapter = CvePatternGate()
