"""K-median placement — the textbook facility-location baseline.

Objective: choose ``budget`` controllers minimising the **total** assignment distance
``sum_v min_j d(v, c_j)``.  This is the standard latency-oriented placement, and it is
the baseline most likely to be requested because every SDN placement paper compares
against it.

Exact k-median is NP-hard, so this uses the standard **greedy** construction: start from
the single best node, then repeatedly add the node that yields the largest reduction in
total distance.  Greedy k-median is within a well-known approximation factor of the
optimum and is what practitioners use at this scale.

Complexity is ``O(budget * n^2)`` with numpy vectorisation — a few seconds at n ~ 1.5k.
The result is deterministic (no RNG), so ``seed`` is accepted for interface parity only.

Note: unlike k-center (MWCL-sim, which minimises the *maximum* distance), k-median can
leave a few nodes very far from any controller.  That distinction matters for the
tail of the attack — worth reporting both.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

__all__ = ["NAME", "select"]

NAME = "KMedian"


def _distance_matrix(graph: nx.Graph) -> tuple[list[int], np.ndarray]:
    nodes = sorted(graph.nodes())
    index = {node: position for position, node in enumerate(nodes)}
    unreachable = graph.number_of_nodes() + 1
    distances = np.full((len(nodes), len(nodes)), float(unreachable), dtype=np.float64)
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        row = index[source]
        for target, distance in lengths.items():
            distances[row, index[target]] = float(distance)
    return nodes, distances


def select(graph: nx.Graph, budget: int, seed: int = 0) -> list[int]:
    """Greedy k-median: iteratively add the node cutting total distance the most."""
    nodes, distances = _distance_matrix(graph)
    count = len(nodes)
    if budget >= count:
        return nodes

    # current best distance from every node to the chosen set
    nearest = np.full(count, np.inf, dtype=np.float64)
    chosen: list[int] = []
    for _ in range(budget):
        # total distance if candidate j were added to the current set
        candidate_cost = np.minimum(nearest[:, None], distances).sum(axis=0)
        if chosen:
            candidate_cost[chosen] = np.inf
        best = int(np.argmin(candidate_cost))
        chosen.append(best)
        nearest = np.minimum(nearest, distances[best])

    return [nodes[index] for index in chosen]
