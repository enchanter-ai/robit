"""tarjan — iterative Tarjan strongly-connected-components algorithm.

Port of gorgon/tarjan.ts (tarjanScc).  Iterative variant so deep graphs
don't blow the Python call stack.

Public API
----------
tarjan_scc(graph: dict[str, list[str]]) -> list[list[str]]
    Returns SCCs in reverse topological order (leaves first), matching the
    classic Tarjan output order.  Callers that want topological order should
    reverse the result.

    Nodes referenced only as edge targets (not as keys) are treated as
    terminal singletons.
"""

from __future__ import annotations


def tarjan_scc(graph: dict[str, list[str]]) -> list[list[str]]:
    """Iterative Tarjan SCC on a directed graph.

    Parameters
    ----------
    graph:
        Adjacency list: ``graph[u]`` is the list of nodes *u* has edges to
        (i.e. out-edges / imports-of).  Nodes mentioned only as targets but
        absent as keys are still included in the output as singleton SCCs.

    Returns
    -------
    list[list[str]]
        Each inner list is one SCC.  Order: reverse topological (leaves first).
    """
    # Collect every node, including edge-only targets.
    nodes: set[str] = set(graph.keys())
    for outs in graph.values():
        nodes.update(outs)

    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []
    next_index = 0

    def successors(v: str) -> list[str]:
        return graph.get(v, [])

    for start in nodes:
        if start in index:
            continue

        # --- Initialise the root frame ---
        index[start] = next_index
        lowlink[start] = next_index
        next_index += 1
        stack.append(start)
        on_stack.add(start)

        # work stack: each frame is (node, successor_index, successor_list)
        work: list[tuple[str, list[str], list[int]]] = [
            (start, successors(start), [0])
        ]

        while work:
            v, succs, pos = work[-1]
            i = pos[0]

            if i < len(succs):
                w = succs[i]
                pos[0] += 1                   # advance successor pointer

                if w not in index:
                    # Tree edge — push new frame.
                    index[w] = next_index
                    lowlink[w] = next_index
                    next_index += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, successors(w), [0]))
                elif w in on_stack:
                    # Back edge — tighten lowlink.
                    if index[w] < lowlink[v]:
                        lowlink[v] = index[w]
            else:
                # All successors processed.
                work.pop()

                if lowlink[v] == index[v]:
                    # v is the root of an SCC — pop the component.
                    component: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        component.append(w)
                        if w == v:
                            break
                    result.append(component)

                # Propagate lowlink to the parent frame.
                if work:
                    parent_v = work[-1][0]
                    if lowlink[v] < lowlink[parent_v]:
                        lowlink[parent_v] = lowlink[v]

    return result
