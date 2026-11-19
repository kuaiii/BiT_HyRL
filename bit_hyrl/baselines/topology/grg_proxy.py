"""Regular, high-expansion GRG proxy used for the early comparisons.

Superseded by :mod:`grg`; kept so the older numbers remain reproducible.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .._common import finalize_construct, seeded_legacy
from ._rewire import propose_swap

__all__ = ["NAME", "construct"]

NAME = "GRG-proxy"


def _lambda2(graph: nx.Graph) -> float:
    """Algebraic connectivity (Fiedler value) -- spectral proxy of edge expansion."""
    if graph.number_of_nodes() < 3:
        return 0.0
    laplacian = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    return float(np.linalg.eigvalsh(laplacian.astype(float))[1])


def _near_regular(n: int, m: int, seed: int = 0) -> nx.Graph:
    """A connected graph on ``n`` nodes with exactly ``m`` edges, as regular as possible."""
    k = max(1, (2 * m) // n)
    if (k * n) % 2:
        k -= 1
    k = max(1, k)
    graph = None
    for degree in (k, k + 1, k - 1):
        if degree < 1 or degree >= n:
            continue
        try:
            graph = nx.random_regular_graph(degree, n, seed=seed)
            break
        except nx.NetworkXError:
            continue
    if graph is None:
        return nx.gnp_random_graph(n, min(1.0, 2 * m / (n * (n - 1))), seed=seed)

    while graph.number_of_edges() < m:
        candidates = sorted(graph.nodes(), key=lambda v: (graph.degree(v), v))[: 2 * n]
        added = False
        for i, u in enumerate(candidates):
            for v in candidates[i + 1:]:
                if u != v and not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    added = True
                    break
            if added:
                break
        if not added:
            break
    while graph.number_of_edges() > m:
        u, v = max(graph.edges(), key=lambda e: (graph.degree(e[0]) + graph.degree(e[1]), e))
        graph.remove_edge(u, v)
    return graph


def construct(G, *, seed: int = 0, swaps: int = 200, **params):
    """High-expansion near-regular graph with the input's ``N`` and exact ``M``."""
    with seeded_legacy(seed):
        n, m = G.number_of_nodes(), G.number_of_edges()
        current = _near_regular(n, m, seed=int(seed))
        if not nx.is_connected(current):
            current = finalize_construct(current, NAME)
        edges = list(current.edges())
        current_l2 = _lambda2(current)
        best, best_l2, best_edges = current, current_l2, edges
        for _ in range(swaps):
            candidate, candidate_edges = propose_swap(current, edges)
            if candidate is None:
                continue
            value = _lambda2(candidate)
            if value > current_l2 + 1e-9:
                current, current_l2, edges = candidate, value, candidate_edges
                if value > best_l2:
                    best, best_l2, best_edges = candidate, value, candidate_edges

    out = finalize_construct(best, NAME)
    out.graph["search_evals"] = swaps
    out.graph["lambda2"] = best_l2
    return out
