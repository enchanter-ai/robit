"""deep_research.adapter — DeepResearch engine adapter.

Event-driven (phases=()), purely subscriber-based. Runs the 6-phase pipeline
when a 'research.requested' event arrives carrying a topic string.

Required: False — advisory engine; pipeline failures are logged and surfaced
via a degraded ack rather than blocking dispatch.

Topics subscribed: research.requested
Topics emitted:
  deep-research.started       — pipeline kicked off
  deep-research.completed     — pipeline finished (READY|PARTIAL verdict)
  deep-research.failed        — pipeline failed (FAIL verdict or exception)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from enchanter.core import EnchantedEvent, PluginAck, RequestContext
from enchanter.core.plugin import PluginTopics
from enchanter.core.bus import new_event_id
from enchanter.llm import LlmClient

from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


def _make_derived(
    base: EnchantedEvent,
    topic: str,
    payload: dict,
) -> EnchantedEvent:
    return EnchantedEvent(
        id=new_event_id(),
        correlation_id=base.correlation_id,
        session_id=base.session_id,
        phase=base.phase,
        topic=topic,
        source="deep-research",
        budget_tier=base.budget_tier,
        ts=_now_ms(),
        payload=payload,
    )


class DeepResearch:
    """Purely event-driven deep research engine.

    Phases = () — does not fire on lifecycle phases.  Activates only when the
    orchestrator delivers a 'research.requested' event.

    The adapter requires a wired LlmClient and TierRouter to function. These
    are injected at construction time (or via .configure()) so tests can swap
    in mocks without patching global state.
    """

    name = "deep-research"
    phases: tuple[()] = ()
    required = False   # fail-open; pipeline failure should not block dispatch
    topics = PluginTopics(
        subscribes=("research.requested",),
        emits=(
            "deep-research.started",
            "deep-research.completed",
            "deep-research.failed",
        ),
    )
    budget_tier = "high-only"

    def __init__(
        self,
        llm: LlmClient | None = None,
        tier_router=None,
        state_dir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._tier_router = tier_router
        self._state_dir = state_dir or Path("state/briefs")

    def configure(
        self,
        llm: LlmClient,
        tier_router,
        state_dir: Path | None = None,
    ) -> None:
        """Inject dependencies after construction (useful for runtime wiring)."""
        self._llm = llm
        self._tier_router = tier_router
        if state_dir is not None:
            self._state_dir = state_dir

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        """Handle research.requested events; no-op for all other events."""
        if event.topic != "research.requested":
            return PluginAck(status="ack")

        if self._llm is None or self._tier_router is None:
            logger.error(
                "DeepResearch.on_phase: llm or tier_router not configured; "
                "call .configure() before dispatching research.requested events"
            )
            return PluginAck(
                status="ack",
                degraded=True,
                reason="deep-research: llm/tier_router not configured",
            )

        topic = str(event.payload.get("topic") or "")
        if not topic:
            return PluginAck(
                status="ack",
                degraded=True,
                reason="deep-research: research.requested event missing 'topic' payload field",
            )

        # Derive state_dir from topic slug
        slug = topic.lower().replace(" ", "-")[:64]
        run_dir = self._state_dir / slug

        started_event = _make_derived(
            event,
            "deep-research.started",
            {"topic": topic, "slug": slug, "state_dir": str(run_dir)},
        )

        try:
            result = await run_pipeline(
                topic=topic,
                llm=self._llm,
                tier_router=self._tier_router,
                state_dir=run_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("DeepResearch pipeline failed for topic %r: %s", topic, exc)
            failed_event = _make_derived(
                event,
                "deep-research.failed",
                {"topic": topic, "error": str(exc)},
            )
            return PluginAck(
                status="ack",
                degraded=True,
                reason=f"deep-research: pipeline failed — {exc}",
                derived_events=[started_event, failed_event],
            )

        completed_event = _make_derived(
            event,
            "deep-research.completed",
            {
                "topic": topic,
                "verdict": result.verdict,
                "triangulation_score": result.triangulation_score,
                "claims_path": str(result.claims_path),
                "sources_path": str(result.sources_path),
                "trace_path": str(result.trace_path),
            },
        )

        return PluginAck(
            status="ack",
            derived_events=[started_event, completed_event],
        )


# Module-level singleton (mirrors other engine adapters).
adapter = DeepResearch()
