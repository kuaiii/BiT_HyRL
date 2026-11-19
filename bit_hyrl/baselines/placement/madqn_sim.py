"""MADQN-sim placement proxy.

``select_madqn_proxy`` runs a single seeded greedy rollout that approximates MADQN
placement; the ``-sim`` suffix marks it as a proxy rather than the original method.
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np

from .._common import seeded_legacy

__all__ = ["NAME", "select"]


NAME = "MADQN-sim"


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
    """Return ``budget`` distinct controller nodes (one seeded greedy rollout)."""
    with seeded_legacy(seed):
        nodes, distances = _all_pairs_distance_matrix(graph)
        if not nodes:
            return []
        budget = min(int(budget), len(nodes))
        rng = random.Random(seed)
        degree = np.array([graph.degree(node) for node in nodes], dtype=np.float64)
        degree /= max(1.0, float(degree.max()))

        first = int(np.argmin(np.sum(distances, axis=1)))
        selected = [first]
        nearest = distances[first].copy()
        while len(selected) < budget:
            candidates = [i for i in range(len(nodes)) if i not in selected]
            gains = np.array(
                [np.sum(nearest - np.minimum(nearest, distances[i])) for i in candidates],
                dtype=np.float64,
            )
            separations = np.array([nearest[i] for i in candidates], dtype=np.float64)
            if gains.max() > 0:
                gains /= gains.max()
            if separations.max() > 0:
                separations /= separations.max()
            scores = np.array(
                [0.70 * gains[j] + 0.20 * separations[j] + 0.10 * degree[i]
                 for j, i in enumerate(candidates)],
                dtype=np.float64,
            )
            scores += np.array([rng.random() * 1e-12 for _ in candidates])
            next_index = candidates[int(np.argmax(scores))]
            selected.append(next_index)
            nearest = np.minimum(nearest, distances[next_index])
    return [nodes[i] for i in selected]
