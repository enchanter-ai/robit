"""ImportGraphPagerank engine — Python port of gorgon.adapter.ts.

Tracks Python-file import edges across events, builds a cumulative import
graph, and periodically computes PageRank centrality to surface high-centrality
modules as potential code-poisoning surface area.

Phase:       post-session  — compute + emit snapshot after each session ends
             cross-session — compute + emit snapshot at the cross-session tick
Required:    False — advisory, fail-open.  Never vetoes.

Topics sub:  session.start, filesystem.write.completed
Topics emit:
  import-graph-pagerank.snapshot.ready    — top-N hotspots + cycle list
  import-graph-pagerank.hotspot.changed   — when a dirty file shifts rank by >= 3

Deviations from TS:
  • Topic prefix is "import-graph-pagerank." (renamed from TS gorgon.).
  • TS uses a module-level singleton STATE; Python uses per-instance state.
  • TS phases are "cross-session" + "post-response"; Python adds "post-session"
    as a natural fit for the Python runtime's phase vocabulary, and keeps
    "cross-session" for parity.  Both fire the same snapshot logic.
  • Python import extraction uses ``ast.parse`` (stdlib, exact); TS uses regex.
  • Tarjan cycle detection included in v0 (TS deferred it to v0.3).
  • pyproject.toml resolver deferred to v2 per the porting spec.
"""

from __future__ import annotations

import time

from robit.core import EnchantedEvent, PluginAck, RequestContext
from robit.core.plugin import PluginTopics
from robit.core.bus import new_event_id

from .store import ImportGraphStore
from .tarjan import tarjan_scc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOP_N_DEFAULT = 10
_RANK_SHIFT_THRESHOLD = 3   # hotspot.changed fires when rank shifts >= this


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# ImportGraphPagerank adapter
# ---------------------------------------------------------------------------

class ImportGraphPagerank:
    """Advisory import-graph PageRank engine.

    Instance-isolated: each ``ImportGraphPagerank()`` owns its own
    ``ImportGraphStore``, dirty-path set, and previous-rank snapshot.
    """

    name = "import-graph-pagerank"
    phases = ("post-session", "cross-session")
    required = False   # advisory — fail-open
    topics = PluginTopics(
        subscribes=(
            "session.start",
            "filesystem.write.completed",
        ),
        emits=(
            "import-graph-pagerank.snapshot.ready",
            "import-graph-pagerank.hotspot.changed",
        ),
    )
    budget_tier = "high-only"

    def __init__(self, top_n: int = _TOP_N_DEFAULT) -> None:
        self._store = ImportGraphStore()
        self._dirty: set[str] = set()
        # previous ranks: node → 1-based rank (1 = highest score)
        self._prev_ranks: dict[str, int] = {}
        self._top_n = top_n

    # ------------------------------------------------------------------
    # Public test-seam helpers
    # ------------------------------------------------------------------

    @property
    def store(self) -> ImportGraphStore:
        return self._store

    def add_file(self, path: str, source: str) -> None:
        """Directly feed a file into the graph — useful for tests and setup."""
        self._store.add_file(path, source)

    def reset(self) -> None:
        """Clear all per-instance state — test teardown helper."""
        self._store.reset()
        self._dirty.clear()
        self._prev_ranks.clear()

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_snapshot(self, event: EnchantedEvent) -> PluginAck:
        """Compute PageRank + Tarjan SCCs; emit snapshot (and optionally hotspot.changed)."""
        scores = self._store.compute_centrality()

        if not scores:
            return PluginAck(
                status="ack",
                degraded=True,
                reason="import-graph-pagerank: graph is empty; snapshot skipped",
            )

        # Build ranked list descending by score.
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        rank_map: dict[str, int] = {node: i + 1 for i, (node, _) in enumerate(ranked)}

        # Tarjan SCC — filter to multi-node components (cycles only).
        graph = self._store.graph
        all_scc = tarjan_scc(graph)
        cycles = [c for c in all_scc if len(c) > 1]

        # Detect hotspot.changed: dirty files that shifted rank >= threshold.
        hotspot_changed = False
        changed_files: list[str] = []

        if self._prev_ranks and self._dirty:
            for path in self._dirty:
                prev = self._prev_ranks.get(path)
                curr = rank_map.get(path)
                if prev is not None and curr is not None:
                    if abs(curr - prev) >= _RANK_SHIFT_THRESHOLD:
                        hotspot_changed = True
                        changed_files.append(path)

        # Persist current ranks for the next snapshot.
        self._prev_ranks = rank_map
        self._dirty.clear()

        top_hotspots = [
            {"file": node, "score": score, "rank": rank_map[node]}
            for node, score in ranked[: self._top_n]
        ]

        derived: list[EnchantedEvent] = []
        ts_now = _now_ms()

        derived.append(
            EnchantedEvent(
                id=new_event_id(),
                correlation_id=event.correlation_id,
                session_id=event.session_id,
                phase=event.phase,
                topic="import-graph-pagerank.snapshot.ready",
                source=self.name,
                budget_tier=event.budget_tier,
                ts=ts_now,
                payload={
                    "file_count": len(scores),
                    "top_hotspots": top_hotspots,
                    "cycles": cycles,
                },
            )
        )

        if hotspot_changed:
            derived.append(
                EnchantedEvent(
                    id=new_event_id(),
                    correlation_id=event.correlation_id,
                    session_id=event.session_id,
                    phase=event.phase,
                    topic="import-graph-pagerank.hotspot.changed",
                    source=self.name,
                    budget_tier=event.budget_tier,
                    ts=ts_now,
                    payload={"changed_files": changed_files},
                )
            )

        return PluginAck(status="ack", derived_events=derived)

    def _handle_write_completed(self, event: EnchantedEvent) -> PluginAck:
        """Track dirty paths when a filesystem write is reported.

        Mirrors the TS post-response handler that checks write_path.
        """
        write_path = event.payload.get("write_path")
        if isinstance(write_path, str) and write_path in self._store.graph:
            self._dirty.add(write_path)
        return PluginAck(status="ack")

    # ------------------------------------------------------------------
    # PluginAdapter protocol
    # ------------------------------------------------------------------

    async def on_phase(self, event: EnchantedEvent, ctx: RequestContext) -> PluginAck:
        try:
            if event.phase in ("post-session", "cross-session"):
                return self._handle_snapshot(event)
            if (
                event.phase == "post-response"
                and event.topic == "filesystem.write.completed"
            ):
                return self._handle_write_completed(event)
            # session.start and other topics: no-op ack.
            return PluginAck(status="ack")
        except Exception as exc:  # noqa: BLE001
            return PluginAck(
                status="ack",
                degraded=True,
                reason=f"import-graph-pagerank: unexpected error — {exc}",
            )


# Module-level default instance (mirrors gorgonAdapter export in TS).
adapter = ImportGraphPagerank()
