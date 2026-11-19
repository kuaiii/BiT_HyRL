"""Graph variants used for mechanism attribution.

Not used by any production path.
"""

from __future__ import annotations

import networkx as nx

from ..rng import derive
from .builder import build_bimodal_graph, repair_connectivity
from .degree_sequence import plan_degree_sequence

__all__ = [
    "configuration_model_like",
    "bit_degree_random_realize",
    "bit_degree_hh_realize",
    "regular_reference_like",
]


def _repair_degree_preserving(graph: nx.Graph, seed: int) -> nx.Graph:
    """Connect with degree-preserving 2-switches (N, M and degree seq preserved)."""
    if graph.number_of_nodes() == 0 or nx.is_connected(graph):
        return graph
    working = graph.copy()
    repair_connectivity(working, derive(seed, "mechanism", "repair"), max_swaps=2_000_000)
    if not nx.is_connected(working):  # pragma: no cover - extremely defensive
        raise RuntimeError("configuration-model variant could not be made connected")
    return working


def configuration_model_like(G: nx.Graph, seed: int = 0) -> nx.Graph:
    """Same **degree sequence** as ``G``, topology randomised (configuration-model-like)."""
    base = nx.Graph(G)
    n = base.number_of_nodes()
    m = base.number_of_edges()
    if m < 2 or n < 4:
        return base
    working = base.copy()
    swaps = max(1, 10 * m)
    try:
        nx.double_edge_swap(working, nswap=swaps, max_tries=100 * m + 1000, seed=seed)
    except Exception:  # pragma: no cover - pathological inputs
        pass
    out = _repair_degree_preserving(working, seed)
    if out.number_of_nodes() != n or out.number_of_edges() != m:
        raise RuntimeError("configuration_model_like changed N or M")
    return out


def bit_degree_random_realize(G: nx.Graph, c: float = 1.6, h: int = 1, seed: int = 0) -> nx.Graph:
    """BiT's near-regular degree sequence with a **randomised** topology."""
    base = nx.Graph(G)
    n = base.number_of_nodes()
    m = base.number_of_edges()
    plan = plan_degree_sequence(n, m, c=c, h=h)
    hh = build_bimodal_graph(plan, seed, realize="hh", enforce_connected=True).graph
    working = nx.Graph(hh)
    if working.number_of_edges() >= 2:
        try:
            nx.double_edge_swap(working, nswap=max(1, 10 * m), max_tries=100 * m + 1000, seed=seed)
        except Exception:  # pragma: no cover - pathological inputs
            pass
    out = _repair_degree_preserving(working, seed)
    if out.number_of_nodes() != n or out.number_of_edges() != m:
        raise RuntimeError("bit_degree_random_realize changed N or M")
    return out


def bit_degree_hh_realize(G: nx.Graph, c: float = 1.6, h: int = 1, seed: int = 0) -> nx.Graph:
    """BiT's near-regular degree sequence realised by Havel–Hakimi, **no decorrelation**."""
    base = nx.Graph(G)
    plan = plan_degree_sequence(base.number_of_nodes(), base.number_of_edges(), c=c, h=h)
    result = build_bimodal_graph(plan, seed, realize="hh", enforce_connected=True)
    return nx.Graph(result.graph)


def regular_reference_like(G: nx.Graph, seed: int = 0) -> nx.Graph:
    """Exact ``k̄``-regular circulant (ring) reference — the near-regular upper bound."""
    base = nx.Graph(G)
    n = base.number_of_nodes()
    if n < 4:
        return base
    k_bar = int(round(2.0 * base.number_of_edges() / max(1, n)))
    half = max(1, k_bar // 2)
    ring = nx.Graph()
    ring.add_nodes_from(range(n))
    for i in range(n):
        for d in range(1, half + 1):
            ring.add_edge(i, (i + d) % n)
    return ring
