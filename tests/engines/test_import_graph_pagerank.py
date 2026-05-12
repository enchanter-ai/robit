"""Tests for the import-graph-pagerank engine.

Covers:
  extract_imports: simple import, from-import, relative imports, malformed source
  tarjan_scc: linear chain, cycle, disconnected components
  pagerank: uniform sink, hub-and-spoke, convergence, damping=0
  End-to-end: event feeds a file; centrality returns sensible scores
"""

from __future__ import annotations

import pytest

from enchanter.core import EnchantedEvent, PluginAck, create_request_context
from enchanter.core.bus import build_event
from enchanter.core.context import RequestContext
from enchanter.engines.import_graph_pagerank import (
    ImportGraphPagerank,
    ImportGraphStore,
    extract_imports,
    pagerank,
    tarjan_scc,
)


# ===========================================================================
# extract_imports
# ===========================================================================


class TestExtractImports:
    def test_simple_import_returns_module_name(self):
        result = extract_imports("import foo")
        assert result == ["foo"]

    def test_from_import_returns_module_not_name(self):
        result = extract_imports("from foo import bar")
        assert result == ["foo"]

    def test_dotted_import_returns_full_dotted_name(self):
        result = extract_imports("import foo.bar")
        assert result == ["foo.bar"]

    def test_from_dotted_import_returns_module_path(self):
        result = extract_imports("from foo.bar import baz")
        assert result == ["foo.bar"]

    def test_relative_import_labeled_as_sentinel(self):
        result = extract_imports("from . import sibling")
        assert "__relative__" in result

    def test_relative_dotted_import_labeled_as_sentinel(self):
        result = extract_imports("from .pkg import something")
        assert "__relative__" in result

    def test_multiple_relative_imports_single_sentinel(self):
        source = "from . import a\nfrom .pkg import b"
        result = extract_imports(source)
        assert result.count("__relative__") == 1

    def test_malformed_source_returns_empty_list(self):
        result = extract_imports("def (((broken syntax ===")
        assert result == []

    def test_empty_source_returns_empty_list(self):
        result = extract_imports("")
        assert result == []

    def test_deduplication_across_repeated_imports(self):
        source = "import os\nimport os\nfrom os import path"
        result = extract_imports(source)
        assert result.count("os") == 1

    def test_multi_module_import_returns_each(self):
        # import foo, bar  → both "foo" and "bar"
        result = extract_imports("import foo, bar")
        assert "foo" in result
        assert "bar" in result

    def test_result_is_sorted(self):
        source = "import zoo\nimport alpha\nfrom beta import x"
        result = extract_imports(source)
        assert result == sorted(result)


# ===========================================================================
# tarjan_scc
# ===========================================================================


class TestTarjanScc:
    def test_linear_chain_gives_three_singleton_sccs(self):
        # a → b → c, no back-edges → 3 SCCs each of size 1
        graph = {"a": ["b"], "b": ["c"], "c": []}
        sccs = tarjan_scc(graph)
        sizes = sorted(len(s) for s in sccs)
        assert sizes == [1, 1, 1]

    def test_simple_cycle_gives_one_scc_of_size_two(self):
        # a → b → a
        graph = {"a": ["b"], "b": ["a"]}
        sccs = tarjan_scc(graph)
        assert len(sccs) == 1
        assert sorted(sccs[0]) == ["a", "b"]

    def test_three_node_cycle(self):
        # a → b → c → a
        graph = {"a": ["b"], "b": ["c"], "c": ["a"]}
        sccs = tarjan_scc(graph)
        assert len(sccs) == 1
        assert sorted(sccs[0]) == ["a", "b", "c"]

    def test_disconnected_graph_gives_multiple_sccs(self):
        # Two independent chains: (a→b) and (c→d)
        graph = {"a": ["b"], "b": [], "c": ["d"], "d": []}
        sccs = tarjan_scc(graph)
        assert len(sccs) == 4  # all singletons

    def test_mixed_cycle_and_chain(self):
        # x → a → b → a (cycle a-b), x is a singleton
        graph = {"x": ["a"], "a": ["b"], "b": ["a"]}
        sccs = tarjan_scc(graph)
        cycle_sccs = [s for s in sccs if len(s) > 1]
        assert len(cycle_sccs) == 1
        assert sorted(cycle_sccs[0]) == ["a", "b"]

    def test_empty_graph_returns_empty(self):
        assert tarjan_scc({}) == []

    def test_self_loop_is_singleton_scc(self):
        graph = {"a": ["a"]}
        sccs = tarjan_scc(graph)
        assert len(sccs) == 1
        assert sccs[0] == ["a"]


# ===========================================================================
# pagerank
# ===========================================================================


class TestPagerank:
    def test_empty_graph_returns_empty_dict(self):
        assert pagerank({}) == {}

    def test_uniform_sink_graph_all_equal(self):
        # All nodes are sinks (no out-edges) → dangling mass redistributed
        # uniformly → all converge to 1/N.
        graph = {"a": [], "b": [], "c": []}
        scores = pagerank(graph)
        vals = list(scores.values())
        assert len(vals) == 3
        # Each score should be close to 1/3
        for v in vals:
            assert abs(v - 1.0 / 3) < 1e-4

    def test_hub_and_spoke_hub_has_highest_pr(self):
        # All spokes point to the hub; hub points nowhere.
        # Hub should accumulate the most incoming mass.
        graph = {
            "hub": [],
            "s1": ["hub"],
            "s2": ["hub"],
            "s3": ["hub"],
        }
        scores = pagerank(graph)
        assert scores["hub"] == max(scores.values())

    def test_converges_within_max_iter(self):
        # A balanced graph should converge well within 100 iterations.
        graph = {"a": ["b"], "b": ["c"], "c": ["a"]}
        # Run with a high tolerance to confirm it terminates early.
        scores = pagerank(graph, tol=1e-4, max_iter=100)
        total = sum(scores.values())
        assert abs(total - 1.0) < 1e-3

    def test_damping_zero_gives_uniform_distribution(self):
        # With d=0: PR(p) = 1/N + 0 * ... = 1/N for all p
        # (dangling redistribution also scaled by d, so with d=0 the only
        # contribution is (1-0)/N = 1/N regardless of graph topology).
        graph = {"a": ["b"], "b": ["c"], "c": []}
        scores = pagerank(graph, damping=0.0)
        for v in scores.values():
            assert abs(v - 1.0 / len(scores)) < 1e-6

    def test_scores_sum_to_one(self):
        graph = {"a": ["b", "c"], "b": ["c"], "c": ["a"]}
        scores = pagerank(graph)
        assert abs(sum(scores.values()) - 1.0) < 1e-5

    def test_single_node_graph(self):
        graph = {"a": []}
        scores = pagerank(graph)
        assert abs(scores["a"] - 1.0) < 1e-6


# ===========================================================================
# End-to-end: adapter feeds files, centrality reflects graph structure
# ===========================================================================


class TestEndToEnd:
    def _make_event(
        self,
        ctx: RequestContext,
        phase: str = "post-session",
    ) -> EnchantedEvent:
        return build_event(
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            phase=phase,  # type: ignore[arg-type]
            topic="session.start",
            source="test",
            budget_tier=ctx.budget_tier,
            payload={},
        )

    async def test_single_file_snapshot_emits_ready_event(self):
        engine = ImportGraphPagerank()
        engine.add_file("main.py", "import os\nimport sys")

        ctx = create_request_context()
        event = self._make_event(ctx)

        ack = await engine.on_phase(event, ctx)
        assert ack.status == "ack"
        assert ack.derived_events is not None
        topics = [e.topic for e in ack.derived_events]
        assert "import-graph-pagerank.snapshot.ready" in topics

    def test_centrality_scores_reflect_import_structure(self):
        # Build a graph where "utils.py" is imported by three files.
        # It should end up with the highest (or tied-highest) PageRank.
        engine = ImportGraphPagerank()
        engine.add_file("a.py", "from utils import helper")
        engine.add_file("b.py", "from utils import helper")
        engine.add_file("c.py", "from utils import helper")
        engine.add_file("utils.py", "import os")  # utils has no Python project imports

        scores = engine.store.compute_centrality()
        # "utils" should have a high score because three nodes point to it.
        # (The module name extracted is "utils", matching the edge targets.)
        assert "utils" in scores
        assert scores["utils"] == max(scores.values())

    def test_top_n_returns_requested_count(self):
        engine = ImportGraphPagerank()
        for i in range(10):
            engine.add_file(f"mod{i}.py", "import os")
        top = engine.store.top_n(3)
        assert len(top) <= 3

    async def test_empty_graph_snapshot_returns_degraded_ack(self):
        engine = ImportGraphPagerank()
        ctx = create_request_context()
        event = self._make_event(ctx)

        ack = await engine.on_phase(event, ctx)
        assert ack.status == "ack"
        assert ack.degraded is True

    async def test_snapshot_includes_cycles(self):
        engine = ImportGraphPagerank()
        # Manually insert a cycle into the store's graph —
        # simulate two files that mutually import each other.
        engine.store._graph["a.py"] = ["b.py"]
        engine.store._graph["b.py"] = ["a.py"]

        ctx = create_request_context()
        event = self._make_event(ctx)

        ack = await engine.on_phase(event, ctx)
        assert ack.status == "ack"

        snapshot = next(
            e for e in (ack.derived_events or [])
            if e.topic == "import-graph-pagerank.snapshot.ready"
        )
        cycles = snapshot.payload["cycles"]
        # Expect at least one cycle containing both files.
        assert any(
            sorted(c) == ["a.py", "b.py"]
            for c in cycles
        )
