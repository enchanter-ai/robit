"""Central bus-topic registry — port of `src/bus/topic-registry.ts`.

Historically robit had **no** central catalogue of bus topics: each engine
declared its own ``[topics] subscribes/emits`` in ``engine.toml`` and the bus
matched them by string equality at publish time. A typo in an emit or a
subscribe simply never matched anything — silently, with no boot-time error
(roadmap G2 / audit §5 Q5).

This module is the single source of truth for the **canonical** topic set. For
each topic it records:

* ``name``      — the canonical dotted topic string.
* ``owner``     — who is authoritative for it: an engine name, or one of the
                  framework owners ``"orchestrator"`` / ``"framework"``.
* ``kind``      — ``"emit"`` | ``"subscribe"`` | ``"both"``: how the *owner*
                  relates to the topic.  (A topic emitted by the framework and
                  consumed by engines is ``"emit"`` from the framework's view.)
* ``phase``     — the lifecycle phase the topic is expected to flow on, or
                  ``None`` for cross-phase / lifecycle-agnostic topics.
* ``deprecated``— ``True`` for retained-but-discouraged synonyms (e.g.
                  ``llm.proxy.request``).  A deprecated topic is still
                  *registry-known* so the boot-time cross-check tolerates it.

The registry deliberately includes **both** ``mcp.tool.call.requested`` and its
historical synonym ``llm.proxy.request``: the proxy pipeline publishes both.
Collapsing the synonym is another package's territory (roadmap PIPELINE-OPS) —
here we only *record* the deprecation so :func:`is_known_topic` stays tolerant.

Wildcard subscriptions (``*``, ``foo.*``) and ``lifecycle.*`` phase
subscriptions are always permitted (see :func:`is_wildcard` /
:func:`is_known_topic`); engines legitimately fan-in across many emitters
(``inference-substrate`` subscribes to ``*.veto`` / ``*.warn``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .context import LIFECYCLE_PHASES, LifecyclePhase


TopicKind = Literal["emit", "subscribe", "both"]
TopicOwner = str  # engine name, or "orchestrator" / "framework"


@dataclass(frozen=True)
class TopicSpec:
    """A single canonical topic and its ownership metadata."""

    name: str
    owner: TopicOwner
    kind: TopicKind
    phase: LifecyclePhase | None = None
    deprecated: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Framework / orchestrator-emitted topics
# ──────────────────────────────────────────────────────────────────────────────
#
# These are NOT declared in any engine.toml — the orchestrator / pipeline /
# transport layer publishes them and engines subscribe. They must be in the
# registry so an engine subscribing to (say) ``mcp.tool.call.requested`` does
# not trip the "no one emits this" cross-check.

_FRAMEWORK_TOPICS: tuple[TopicSpec, ...] = (
    # Lifecycle phase ticks — one per phase, emitted by the orchestrator as it
    # walks the 7-phase pipeline. Engines subscribe to ``lifecycle.<phase>``.
    *(
        TopicSpec(
            name=f"lifecycle.{phase}",
            owner="orchestrator",
            kind="emit",
            phase=phase,
        )
        for phase in LIFECYCLE_PHASES
    ),
    # The canonical "a tool call was requested" topic. Published by the proxy
    # pipeline at trust-gate / pre-dispatch.
    TopicSpec(
        name="mcp.tool.call.requested",
        owner="orchestrator",
        kind="emit",
        phase="trust-gate",
    ),
    # Deprecated synonym of mcp.tool.call.requested. The pipeline historically
    # publishes BOTH; retiring it is PIPELINE-OPS territory. Recorded here only
    # so the cross-check tolerates engines/pipelines still referencing it.
    TopicSpec(
        name="llm.proxy.request",
        owner="orchestrator",
        kind="emit",
        phase="trust-gate",
        deprecated=True,
    ),
    # Tool result returned from the upstream model/provider.
    TopicSpec(
        name="mcp.tool.result.received",
        owner="orchestrator",
        kind="emit",
        phase="post-response",
    ),
    # Tool registry / discovery events emitted by the MCP transport layer.
    TopicSpec(
        name="mcp.tools.list.received",
        owner="framework",
        kind="emit",
        phase="trust-gate",
    ),
    TopicSpec(
        name="mcp.tool.registered",
        owner="framework",
        kind="emit",
        phase="post-response",
    ),
    # Sampling lifecycle (token accounting source for cost-ledger).
    TopicSpec(
        name="sampling.completed",
        owner="orchestrator",
        kind="emit",
        phase="post-response",
    ),
    # Session lifecycle bookends.
    TopicSpec(
        name="session.start",
        owner="orchestrator",
        kind="emit",
        phase="anchor",
    ),
    TopicSpec(
        name="user.prompt.submit",
        owner="orchestrator",
        kind="emit",
        phase="anchor",
    ),
    TopicSpec(
        name="compact.requested",
        owner="orchestrator",
        kind="emit",
        phase="post-session",
    ),
    # Filesystem write notifications (host-tool observation surface).
    TopicSpec(
        name="filesystem.write.completed",
        owner="framework",
        kind="emit",
        phase="post-session",
    ),
    # Standalone deep-research trigger (not a lifecycle phase).
    TopicSpec(
        name="research.requested",
        owner="framework",
        kind="emit",
        phase=None,
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Engine-emitted topics (mirrors robit/engines/*/engine.toml [topics].emits)
# ──────────────────────────────────────────────────────────────────────────────
#
# Kept in sync with the manifests by hand (no new deps). The boot-time
# cross-check in loader/discovery.py reconciles the *live* manifest set against
# this table, so a drift between this list and a manifest surfaces as an
# unknown-topic warning (or error in strict mode) rather than silently.

_ENGINE_EMITTED_TOPICS: tuple[TopicSpec, ...] = (
    TopicSpec("boundary-segmenter.boundary.closed", "boundary-segmenter", "emit", "post-session"),
    TopicSpec("cost-ledger.appended", "cost-ledger", "emit", "post-response"),
    TopicSpec("cost-ledger.threshold.crossed", "cost-ledger", "emit", "post-response"),
    TopicSpec("cost-ledger.vendor.exhausted", "cost-ledger", "emit", "post-response"),
    TopicSpec("cve-pattern-gate.veto", "cve-pattern-gate", "emit", "trust-gate"),
    TopicSpec("cve-pattern-gate.warn", "cve-pattern-gate", "emit", "trust-gate"),
    TopicSpec("deep-research.started", "deep-research", "emit", None),
    TopicSpec("deep-research.completed", "deep-research", "emit", None),
    TopicSpec("deep-research.failed", "deep-research", "emit", None),
    TopicSpec("destructive-op-gate.veto", "destructive-op-gate", "emit", "trust-gate"),
    TopicSpec("import-graph-pagerank.snapshot.ready", "import-graph-pagerank", "emit", "post-session"),
    TopicSpec("import-graph-pagerank.hotspot.changed", "import-graph-pagerank", "emit", "post-session"),
    TopicSpec("inference-substrate.emitted", "inference-substrate", "emit", "post-session"),
    TopicSpec("inference-substrate.reconciled", "inference-substrate", "emit", "cross-session"),
    TopicSpec("inference-substrate.briefing-rendered", "inference-substrate", "emit", "cross-session"),
    TopicSpec("intent-anchor.anchor.set", "intent-anchor", "emit", "anchor"),
    TopicSpec("intent-anchor.drift.detected", "intent-anchor", "emit", "post-session"),
    TopicSpec("rate-limiter.bucket-exhausted", "rate-limiter", "emit", "pre-dispatch"),
    TopicSpec("secret-mask.matched", "secret-mask", "emit", "post-response"),
    TopicSpec("structural-fingerprint.pattern.fingerprinted", "structural-fingerprint", "emit", "trust-gate"),
    TopicSpec("structural-fingerprint.schema.drift.detected", "structural-fingerprint", "emit", "trust-gate"),
    TopicSpec("token-runway.runway.forecast", "token-runway", "emit", "pre-dispatch"),
    TopicSpec("token-runway.compression.applied", "token-runway", "emit", "post-response"),
    TopicSpec("token-runway.drift.pattern", "token-runway", "emit", "post-response"),
    TopicSpec("tool-poisoning-scan.suspicion.flagged", "tool-poisoning-scan", "emit", "post-response"),
    TopicSpec("tool-poisoning-scan.sandbox.executed", "tool-poisoning-scan", "emit", "post-response"),
    TopicSpec("tool-poisoning-scan.rubric.verdict", "tool-poisoning-scan", "emit", "post-response"),
    TopicSpec("trust-scorer.trust.scored", "trust-scorer", "emit", "trust-gate"),
    TopicSpec("trust-scorer.review.ordered", "trust-scorer", "emit", "trust-gate"),
)


# All canonical specs, framework first.
TOPIC_REGISTRY: tuple[TopicSpec, ...] = _FRAMEWORK_TOPICS + _ENGINE_EMITTED_TOPICS

# Fast lookup by canonical name.
_BY_NAME: dict[str, TopicSpec] = {spec.name: spec for spec in TOPIC_REGISTRY}


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_topic(name: str) -> TopicSpec | None:
    """Return the :class:`TopicSpec` for *name*, or ``None`` if not registered."""
    return _BY_NAME.get(name)


def is_wildcard(topic: str) -> bool:
    """True for fan-in subscription patterns the cross-check always allows.

    Covers the bare ``*`` catch-all and any dotted prefix wildcard such as
    ``*.veto`` or ``foo.*``. Engines legitimately subscribe to these to fan in
    across many emitters (e.g. inference-substrate on ``*.veto``).
    """
    return topic == "*" or "*" in topic


def is_lifecycle_subscription(topic: str) -> bool:
    """True for ``lifecycle.*`` phase subscriptions, which are always allowed."""
    return topic.startswith("lifecycle.")


def is_known_topic(topic: str) -> bool:
    """True if *topic* is registry-known or an always-allowed pattern.

    A topic is acceptable when it is one of:

    * a registered canonical topic (including deprecated synonyms),
    * a wildcard subscription pattern (``*`` / ``foo.*``),
    * a ``lifecycle.<phase>`` subscription.
    """
    return (
        topic in _BY_NAME
        or is_wildcard(topic)
        or is_lifecycle_subscription(topic)
    )


def all_emitted_topics() -> frozenset[str]:
    """Canonical topics that something (framework or an engine) emits."""
    return frozenset(
        spec.name for spec in TOPIC_REGISTRY if spec.kind in ("emit", "both")
    )


__all__ = [
    "TOPIC_REGISTRY",
    "TopicKind",
    "TopicOwner",
    "TopicSpec",
    "all_emitted_topics",
    "get_topic",
    "is_known_topic",
    "is_lifecycle_subscription",
    "is_wildcard",
]
