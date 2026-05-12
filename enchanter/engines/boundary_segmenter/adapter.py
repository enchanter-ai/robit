"""BoundarySegmenter engine — W2 post-session boundary segmentation.

Port of ``handleW2PostSession`` (and supporting state) from
``src/plugins/sylph.adapter.ts``.

Advisory (required=False), post-session phase only.  Consumes
``filesystem.write.completed`` events to accumulate clusters, then on
``lifecycle.post-session`` closes idle clusters and emits a derived
``boundary-segmenter.boundary.closed`` event per closure.
"""

from __future__ import annotations

import time

from enchanter.core import EnchantedEvent, PluginAck, RequestContext
from enchanter.core.plugin import PluginTopics
from enchanter.core.bus import new_event_id

from .store import ClusterStore


def _now_ms() -> int:
    return int(time.time() * 1000)


class BoundarySegmenter:
    """Advisory at post-session.  Fail-open on errors."""

    name = "boundary-segmenter"
    phases = ("post-session",)
    required = False
    topics = PluginTopics(
        subscribes=(
            "filesystem.write.completed",
            "lifecycle.post-session",
        ),
        emits=("boundary-segmenter.boundary.closed",),
    )
    budget_tier = "always"

    def __init__(self) -> None:
        self._store = ClusterStore()

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        try:
            return await self._handle(event)
        except Exception:
            # Fail-open: advisory plugin must not block the session.
            return PluginAck(status="ack", degraded=True, reason=f"{self.name}:error")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle(self, event: EnchantedEvent) -> PluginAck:
        if event.topic == "filesystem.write.completed":
            file_path = str(event.payload.get("file_path", ""))
            if file_path:
                self._store.record_edit(file_path, event.ts)
            return PluginAck(status="ack")

        if event.topic == "lifecycle.post-session" or event.phase == "post-session":
            return self._handle_post_session(event)

        return PluginAck(status="ack")

    def _handle_post_session(self, event: EnchantedEvent) -> PluginAck:
        now = event.ts if event.ts else _now_ms()
        closed_clusters = self._store.close_idle(now)

        if not closed_clusters:
            return PluginAck(status="ack")

        derived: list[EnchantedEvent] = [
            EnchantedEvent(
                id=new_event_id(),
                correlation_id=event.correlation_id,
                session_id=event.session_id,
                phase=event.phase,
                topic="boundary-segmenter.boundary.closed",
                source=self.name,
                budget_tier=event.budget_tier,
                ts=now,
                payload={
                    "cluster_id": c.id,
                    "files": list(c.files),
                    "closed_at": now,
                },
            )
            for c in closed_clusters
        ]

        return PluginAck(
            status="ack",
            reason=f"{self.name}: closed {len(closed_clusters)} cluster(s)",
            derived_events=derived,
        )


adapter = BoundarySegmenter()
