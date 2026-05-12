"""SecretMask engine — fires at post-response, redacts secrets in tool results.

Mirrors `scanResultAtPostResponse` + `maskSecrets` from the TS
`hydra.adapter.ts`, isolated as a standalone engine.

Corpus construction: the `result` field of the event payload is the
primary scan target.
  - string result → scan directly.
  - dict/other result → JSON-stringified, then scanned (mirrors the
    TS branch `JSON.stringify(result)`).
  - missing result → pass through with no derived event (no crash).

Required: True.  Security plugin — never silenced.  Phase: post-response.
"""

from __future__ import annotations

import json
import re
import time

from enchanter.core import EnchantedEvent, PluginAck, RequestContext
from enchanter.core.plugin import PluginTopics
from enchanter.core.bus import new_event_id

from .patterns import SECRET_PATTERNS


def _now_ms() -> int:
    return int(time.time() * 1000)


def _mask_secrets(corpus: str) -> tuple[str, list[str]]:
    """Return (redacted_corpus, list_of_matched_pattern_ids).

    Iterates patterns in declaration order.  For the bearer-token pattern
    the redaction string contains a back-reference (`\\1`) so re.sub is
    used directly; for all other patterns re.sub replaces with the literal
    redaction string.
    """
    masked = corpus
    matched: list[str] = []
    for p in SECRET_PATTERNS:
        if p.match.search(masked):
            matched.append(p.id)
            masked = p.match.sub(p.redaction, masked)
    return masked, matched


class SecretMask:
    """Required at post-response.  Redacts secrets found in tool results."""

    name = "secret-mask"
    phases = ("post-response",)
    required = True
    topics = PluginTopics(
        subscribes=(
            "mcp.tool.result.received",
            "lifecycle.post-response",
        ),
        emits=("secret-mask.matched",),
    )
    budget_tier = "always"

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        if event.phase != "post-response":
            return PluginAck(status="ack")

        payload = dict(event.payload or {})
        result = payload.get("result")

        if result is None:
            # Missing result — pass through cleanly per spec.
            return PluginAck(status="ack")

        corpus: str = result if isinstance(result, str) else json.dumps(result, default=str)

        masked, matched_ids = _mask_secrets(corpus)

        if not matched_ids:
            return PluginAck(status="ack")

        derived = EnchantedEvent(
            id=new_event_id(),
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            phase=event.phase,
            topic="secret-mask.matched",
            source=self.name,
            budget_tier=event.budget_tier,
            ts=_now_ms(),
            payload={
                "matched_patterns": matched_ids,
                "redacted_length": len(masked),
            },
        )
        return PluginAck(
            status="ack",
            reason=f"secret-mask: {','.join(matched_ids)}",
            derived_events=[derived],
        )


adapter = SecretMask()
