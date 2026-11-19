"""K-median controller placement baseline."""

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

    nearest = np.full(count, np.inf, dtype=np.float64)
    chosen: list[int] = []
    for _ in range(budget):
        candidate_cost = np.minimum(nearest[:, None], distances).sum(axis=0)
        if chosen:
            candidate_cost[chosen] = np.inf
        best = int(np.argmin(candidate_cost))
        chosen.append(best)
        nearest = np.minimum(nearest, distances[best])

    return [nodes[index] for index in chosen]
