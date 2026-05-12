"""Tests for the boundary-segmenter engine (W2 Jaccard sliding-window clustering).

Eight required tests:
  1. Jaccard: identical paths → 1.0
  2. Jaccard: disjoint paths → 0.0
  3. Jaccard: partial overlap → strictly between 0 and 1
  4. First edit creates a new cluster
  5. Similar-named edit joins the existing open cluster
  6. Dissimilar edit opens a new cluster
  7. Idle cluster (>10 min) closes on post-session sweep
  8. End-to-end: filesystem.write events feed clusters; post-session emits
     boundary.closed derived events
"""

from __future__ import annotations

import pytest

from enchanter.core.bus import build_event
from enchanter.engines.boundary_segmenter import (
    CLUSTER_IDLE_MS,
    CLUSTER_WINDOW_MS,
    JACCARD_THRESHOLD,
    BoundarySegmenter,
    ClusterStore,
    jaccard_similarity,
)


# ---------------------------------------------------------------------------
# 1–3  Jaccard pure-function tests
# ---------------------------------------------------------------------------


def test_jaccard_identical_paths() -> None:
    """Identical paths share all tokens → similarity == 1.0."""
    path = "src/plugins/sylph/adapter.py"
    assert jaccard_similarity(path, path) == 1.0


def test_jaccard_disjoint_paths() -> None:
    """Completely different tokens → similarity == 0.0."""
    assert jaccard_similarity("alpha/beta/gamma.py", "delta/epsilon/zeta.ts") == 0.0


def test_jaccard_partial_overlap() -> None:
    """Shared tokens but not identical → strictly between 0 and 1."""
    # "src/plugins/foo.py"  tokens: {src, plugins, foo, py}
    # "src/plugins/bar.py"  tokens: {src, plugins, bar, py}
    # intersection: {src, plugins, py} = 3;  union: {src, plugins, foo, py, bar} = 5
    result = jaccard_similarity("src/plugins/foo.py", "src/plugins/bar.py")
    assert 0.0 < result < 1.0
    assert pytest.approx(result, rel=1e-6) == 3 / 5


# ---------------------------------------------------------------------------
# 4  First edit creates a new cluster
# ---------------------------------------------------------------------------


def test_first_edit_creates_cluster() -> None:
    store = ClusterStore()
    now = 1_000_000

    store.record_edit("src/engines/foo.py", now)

    open_clusters = store.open_clusters()
    assert len(open_clusters) == 1
    assert "src/engines/foo.py" in open_clusters[0].files
    assert not open_clusters[0].closed


# ---------------------------------------------------------------------------
# 5  Similar-named edit joins the existing open cluster
# ---------------------------------------------------------------------------


def test_similar_edit_joins_existing_cluster() -> None:
    store = ClusterStore()
    t0 = 1_000_000

    store.record_edit("src/engines/boundary_segmenter/store.py", t0)
    # Same directory, same "engine" token family — Jaccard should be well above 0.4.
    store.record_edit("src/engines/boundary_segmenter/adapter.py", t0 + 1_000)

    open_clusters = store.open_clusters()
    assert len(open_clusters) == 1, "Similar paths must merge into one cluster"
    assert len(open_clusters[0].files) == 2


# ---------------------------------------------------------------------------
# 6  Dissimilar edit opens a new cluster
# ---------------------------------------------------------------------------


def test_dissimilar_edit_opens_new_cluster() -> None:
    store = ClusterStore()
    t0 = 1_000_000

    store.record_edit("src/engines/boundary_segmenter/store.py", t0)
    # Completely different token set → new cluster.
    store.record_edit("docs/reference/changelog.md", t0 + 1_000)

    open_clusters = store.open_clusters()
    assert len(open_clusters) == 2, "Dissimilar paths must produce two separate clusters"


# ---------------------------------------------------------------------------
# 7  Idle cluster closes on post-session sweep
# ---------------------------------------------------------------------------


def test_idle_cluster_closes_on_sweep() -> None:
    store = ClusterStore()
    t0 = 1_000_000

    store.record_edit("src/alpha.py", t0)

    # Advance clock beyond the idle threshold.
    now_after_idle = t0 + CLUSTER_IDLE_MS + 1

    closed = store.close_idle(now_after_idle)
    assert len(closed) == 1
    assert closed[0].closed is True
    assert store.open_clusters() == []


# ---------------------------------------------------------------------------
# 8  End-to-end: filesystem.write events → clusters → post-session derived events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_filesystem_write_to_boundary_events() -> None:
    """Full adapter path: two filesystem.write.completed events → one cluster closed
    on post-session → one boundary-segmenter.boundary.closed derived event."""

    engine = BoundarySegmenter()

    t0 = 2_000_000
    # Simulate two closely related writes arriving before post-session.
    write_event_1 = build_event(
        correlation_id="corr-001",
        session_id="sess-001",
        phase="post-session",          # phase field required by EnchantedEvent
        topic="filesystem.write.completed",
        source="test",
        budget_tier="always",
        payload={"file_path": "src/engines/boundary_segmenter/store.py"},
    )
    # Patch ts to t0 (build_event stamps wall-clock; we need deterministic ts)
    import dataclasses
    write_event_1 = dataclasses.replace(write_event_1, ts=t0)

    write_event_2 = build_event(
        correlation_id="corr-001",
        session_id="sess-001",
        phase="post-session",
        topic="filesystem.write.completed",
        source="test",
        budget_tier="always",
        payload={"file_path": "src/engines/boundary_segmenter/adapter.py"},
    )
    write_event_2 = dataclasses.replace(write_event_2, ts=t0 + 5_000)

    from enchanter.core.context import create_request_context
    ctx = create_request_context(
        session_id="sess-001",
        budget_tier="HIGH",
    )

    ack1 = await engine.on_phase(write_event_1, ctx)
    assert ack1.status == "ack"
    assert not ack1.derived_events

    ack2 = await engine.on_phase(write_event_2, ctx)
    assert ack2.status == "ack"
    assert not ack2.derived_events

    # Both writes should be in one open cluster.
    assert len(engine._store.open_clusters()) == 1

    # Fire post-session beyond the idle threshold (measured from last edit at t0+5_000).
    post_session_ts = t0 + 5_000 + CLUSTER_IDLE_MS + 1
    post_event = build_event(
        correlation_id="corr-001",
        session_id="sess-001",
        phase="post-session",
        topic="lifecycle.post-session",
        source="test",
        budget_tier="always",
        payload={},
    )
    post_event = dataclasses.replace(post_event, ts=post_session_ts)

    ack3 = await engine.on_phase(post_event, ctx)
    assert ack3.status == "ack"
    assert len(ack3.derived_events) == 1

    derived = ack3.derived_events[0]
    assert derived.topic == "boundary-segmenter.boundary.closed"
    assert derived.source == "boundary-segmenter"
    assert derived.payload["closed_at"] == post_session_ts
    files = derived.payload["files"]
    assert "src/engines/boundary_segmenter/store.py" in files
    assert "src/engines/boundary_segmenter/adapter.py" in files

    # Cluster is now closed; no open clusters remain.
    assert engine._store.open_clusters() == []
