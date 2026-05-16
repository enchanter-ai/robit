"""InferenceSubstrateEngine — PluginAdapter wrapper for the inference substrate.

Phase:        post-session, cross-session
Required:     False (advisory — the substrate enriches future sessions but its
              absence never gates the current one)
Topics sub:   lifecycle.post-session, lifecycle.cross-session, *.veto, *.warn
Topics emit:  inference-substrate.emitted
              inference-substrate.reconciled
              inference-substrate.briefing-rendered

Lifecycle
---------
During a session the engine collects failure-shaped events (*.veto, *.warn)
into an in-process buffer.  At ``post-session`` it flushes the buffer to
artifacts.jsonl via :func:`emit_unconditional`.  At ``cross-session`` it runs
reconcile() + render_briefing() so the next session opens with a fresh
catalog and briefing.

The engine never fails-closed.  IO errors are caught and surfaced as
``degraded=True`` in the PluginAck so the orchestrator can log them without
blocking the session.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from robit.core import EnchantedEvent, PluginAck, RequestContext
from robit.core.plugin import PluginTopics
from robit.core.bus import new_event_id
from robit.inference.engine import (
    emit_unconditional,
    iso_now,
    reconcile,
    render_briefing,
)


def _now_ms() -> int:
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
        source="inference-substrate",
        budget_tier=base.budget_tier,
        ts=_now_ms(),
        payload=payload,
    )


def _state_dir_from_env() -> Path | None:
    from robit._compat import get_env

    override = get_env("ROBIT_INFERENCE_STATE")
    return Path(override) if override else None


def _event_to_artifact(event: EnchantedEvent) -> dict:
    """Convert a bus event to an inference artifact record.

    Derives code and category from the topic name and payload heuristics so
    the artifact is useful without requiring the emitting plugin to construct
    an explicit artifact dict.
    """
    payload = dict(event.payload)

    # Best-effort code derivation: use explicit code in payload, or map topic
    # suffixes to the failure taxonomy.
    code = str(payload.get("code") or "")
    if not code:
        topic_lower = event.topic.lower()
        if "veto" in topic_lower:
            code = "F18"
        elif "warn" in topic_lower:
            code = "F07"
        else:
            code = "F00"

    tags: list[str] = []
    if isinstance(payload.get("tags"), list):
        tags = [str(t) for t in payload["tags"]]
    else:
        tags = [event.source, event.topic.split(".")[0]]

    return {
        "code": code,
        "category": str(payload.get("category") or "operational-discipline"),
        "title": str(payload.get("title") or f"{event.topic} at {event.phase}"),
        "cause": str(payload.get("cause") or payload.get("reason") or event.topic),
        "counter": str(payload.get("counter") or ""),
        "signal": str(payload.get("signal") or ""),
        "tags": tags,
        "scope": str(payload.get("scope") or "enchanter"),
        "evidence": payload.get("evidence") or {},
        "ts": iso_now(),
        "session_id": event.session_id,
        "plugin": event.source,
    }


class InferenceSubstrateEngine:
    """Advisory engine that accumulates failure events into the inference substrate.

    One instance per session.  Construct fresh per test — no global state.
    """

    name = "inference-substrate"
    phases = ("post-session", "cross-session")
    required = False
    topics = PluginTopics(
        subscribes=(
            "lifecycle.post-session",
            "lifecycle.cross-session",
            "*.veto",
            "*.warn",
        ),
        emits=(
            "inference-substrate.emitted",
            "inference-substrate.reconciled",
            "inference-substrate.briefing-rendered",
        ),
    )
    budget_tier = "always"

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        plugin_for_briefing: str = "all",
    ) -> None:
        # state_dir=None → picks up ROBIT_INFERENCE_STATE (or the legacy
        # ENCHANTER_INFERENCE_STATE via robit._compat) or the default.
        self._state_dir: Path | None = state_dir if state_dir is not None else _state_dir_from_env()
        self._plugin_for_briefing = plugin_for_briefing
        self._buffer: list[dict] = []

    # ------------------------------------------------------------------
    # Event collection (called on every subscribed topic)
    # ------------------------------------------------------------------

    def _capture(self, event: EnchantedEvent) -> None:
        """Buffer a failure-shaped event for later flush."""
        topic = event.topic
        # Capture *.veto and *.warn events; ignore lifecycle control events.
        if "veto" in topic or "warn" in topic:
            self._buffer.append(_event_to_artifact(event))

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_post_session(self, event: EnchantedEvent) -> PluginAck:
        """Flush buffered artifacts to disk."""
        derived: list[EnchantedEvent] = []
        errors: list[str] = []

        for artifact in self._buffer:
            try:
                emit_unconditional(artifact, self._state_dir)
                derived.append(
                    _make_derived(
                        event,
                        "inference-substrate.emitted",
                        {
                            "code": artifact.get("code", ""),
                            "title": artifact.get("title", ""),
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

        self._buffer.clear()

        if errors:
            return PluginAck(
                status="ack",
                degraded=True,
                reason=f"inference-substrate: {len(errors)} emit error(s) — {errors[0]}",
                derived_events=derived,
            )
        return PluginAck(status="ack", derived_events=derived)

    def _handle_cross_session(self, event: EnchantedEvent) -> PluginAck:
        """Reconcile catalog + render briefing."""
        derived: list[EnchantedEvent] = []
        errors: list[str] = []

        try:
            cat = reconcile(self._state_dir)
            derived.append(
                _make_derived(
                    event,
                    "inference-substrate.reconciled",
                    {
                        "total_artifacts": cat.get("total_artifacts", 0),
                        "total_patterns": cat.get("total_patterns", 0),
                        "elevated_count": cat.get("elevated_count", 0),
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"reconcile: {exc}")

        try:
            out = render_briefing(self._plugin_for_briefing, self._state_dir)
            derived.append(
                _make_derived(
                    event,
                    "inference-substrate.briefing-rendered",
                    {"plugin": self._plugin_for_briefing, "path": str(out)},
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"render-briefing: {exc}")

        if errors:
            return PluginAck(
                status="ack",
                degraded=True,
                reason=f"inference-substrate: {'; '.join(errors)}",
                derived_events=derived,
            )
        return PluginAck(status="ack", derived_events=derived)

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        # Capture failure-shaped events regardless of phase.
        self._capture(event)

        if event.phase == "post-session":
            return self._handle_post_session(event)
        if event.phase == "cross-session":
            return self._handle_cross_session(event)
        return PluginAck(status="ack")


# Module-level singleton (parallel to cost_ledger.adapter.adapter).
adapter = InferenceSubstrateEngine()
