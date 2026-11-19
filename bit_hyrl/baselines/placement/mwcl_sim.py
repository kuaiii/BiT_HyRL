"""MWCL-sim placement proxy — ported from ``scripts/evaluate_physical_control_metrics.py``.

``select_mwcl_proxy``: a seeded-free k-center latency proxy for the MWCL objective.
Honestly labelled ``-sim`` — it is a proxy, not the original method.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .._common import seeded_legacy

__all__ = ["NAME", "select"]


NAME = "MWCL-sim"


def _all_pairs_distance_matrix(graph: nx.Graph) -> tuple[list[int], np.ndarray]:
    """Sorted node list + dense all-pairs shortest-path distance matrix."""
    nodes = sorted(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    unreachable = graph.number_of_nodes() + 1
    distances = np.full((len(nodes), len(nodes)), unreachable, dtype=np.float64)
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        row = index[source]
        for target, distance in lengths.items():
            distances[row, index[target]] = distance
    return nodes, distances


def select(graph: nx.Graph, budget: int, seed: int = 0) -> list[int]:
    """Return ``budget`` distinct controller nodes (k-center latency proxy)."""
    with seeded_legacy(seed):
        nodes, distances = _all_pairs_distance_matrix(graph)
        if not nodes:
            return []
        budget = min(int(budget), len(nodes))
        first = int(np.argmin(np.max(distances, axis=1)))
        selected = [first]
        nearest = distances[first].copy()
        while len(selected) < budget:
            candidates = [i for i in range(len(nodes)) if i not in selected]
            next_index = max(
                candidates, key=lambda i: (nearest[i], graph.degree(nodes[i]), -i)
            )
            selected.append(next_index)
            nearest = np.minimum(nearest, distances[next_index])
    return [nodes[i] for i in selected]
