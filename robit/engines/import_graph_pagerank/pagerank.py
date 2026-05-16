"""pagerank — power-iteration PageRank on a directed graph.

Port of ``computePageRank`` in gorgon.adapter.ts (Brin & Page 1998).

Public API
----------
pagerank(
    graph: dict[str, list[str]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]

    Standard formula:
        PR(p) = (1 - d) / N + d * Σ( PR(q) / out_deg(q)  for q → p )

    Dangling nodes (out-degree 0) distribute their mass uniformly —
    matches the TS implementation's ``danglingMass / n`` redistribution.

    Stops when ``max_delta < tol`` or after ``max_iter`` iterations.
    Returns an empty dict for an empty graph.

    Nodes referenced only as edge targets are added as zero-out-degree sinks.
"""

from __future__ import annotations


def pagerank(
    graph: dict[str, list[str]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Power-iteration PageRank.

    Parameters
    ----------
    graph:
        Adjacency list of out-edges: ``graph[u]`` = list of nodes *u* points to.
    damping:
        Damping factor *d* (canonical default 0.85).
    max_iter:
        Maximum number of power iterations.
    tol:
        Convergence threshold on the L1 delta between iterations.

    Returns
    -------
    dict[str, float]
        Node → PageRank score.  Scores sum to approximately 1.0.
    """
    if not graph:
        return {}

    # Ensure every referenced target node exists in the node set.
    all_nodes: set[str] = set(graph.keys())
    for targets in graph.values():
        all_nodes.update(targets)

    node_list = sorted(all_nodes)        # stable ordering for reproducibility
    n = len(node_list)
    if n == 0:
        return {}

    idx: dict[str, int] = {node: i for i, node in enumerate(node_list)}

    # out_degree[i] = number of out-edges from node i
    out_degree = [0] * n
    # in_links[j] = list of node indices that point to j
    in_links: list[list[int]] = [[] for _ in range(n)]

    for node, targets in graph.items():
        i = idx[node]
        out_degree[i] = len(targets)
        for t in targets:
            j = idx.get(t)
            if j is not None:
                in_links[j].append(i)

    # For target-only nodes (added but not in graph), out_degree stays 0.

    scores = [1.0 / n] * n
    next_scores = [0.0] * n

    for _ in range(max_iter):
        # Mass from dangling nodes (out-degree 0).
        dangling_mass = sum(scores[i] for i in range(n) if out_degree[i] == 0)

        delta = 0.0
        for j in range(n):
            in_sum = sum(scores[i] / out_degree[i] for i in in_links[j])
            nv = (1.0 - damping) / n + damping * (in_sum + dangling_mass / n)
            next_scores[j] = nv
            delta += abs(nv - scores[j])

        scores, next_scores = next_scores, scores   # swap buffers

        if delta < tol:
            break

    return {node: scores[idx[node]] for node in node_list}
