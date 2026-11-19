"""GRG circulant construction (Huang et al., IEEE ToN 34, 2026).

Used for every comparison that cites the paper; :mod:`grg_proxy` is kept only
for the early runs that predate the full description.
"""

from __future__ import annotations

import networkx as nx

from .._common import finalize_construct

__all__ = ["construct", "egrg", "grg_regular", "jump_set"]


def jump_set(n: int, degree: int, *, variant: str = "eq18") -> list[int]:
    """Return the circulant jump set for ``n`` nodes of even ``degree = 2k``."""
    if degree % 2:
        raise ValueError(f"degree must be even, got {degree}")
    k = degree // 2
    if variant not in {"eq18", "ladder"}:
        raise ValueError(f"variant must be 'eq18' or 'ladder', got {variant!r}")
    side = n ** (1.0 / k)
    jumps: list[int] = [1]
    for j in range(1, k):
        raw = (j * n) // k if variant == "ladder" else int(round(j * side))
        t = min(raw % n, n - (raw % n))
        if t > 1 and t not in jumps:
            jumps.append(t)
    return sorted(jumps)


def grg_regular(n: int, degree: int, *, variant: str = "eq18") -> nx.Graph:
    """``d``-regular circulant on ``0..n-1`` with the GRG jump set."""
    jumps = jump_set(n, degree, variant=variant)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for t in jumps:
            graph.add_edge(i, (i + t) % n)
    return graph


def _top_up(graph: nx.Graph, target_edges: int) -> nx.Graph:
    """Spread the ``M - n·d/2`` remainder over the ring as evenly spaced chords."""
    n = graph.number_of_nodes()
    remainder = int(target_edges) - graph.number_of_edges()
    if remainder <= 0 or n < 4:
        return graph
    span = max(1, n // remainder)
    for t in range(remainder):
        u = (t * n) // remainder
        v = (u + span) % n
        if u != v and not graph.has_edge(u, v):
            graph.add_edge(u, v)
    while graph.number_of_edges() < target_edges:
        placed = False
        for u in sorted(graph.nodes(), key=lambda v: (graph.degree(v), v)):
            for jump in range(n // 2, 0, -1):
                v = (u + jump) % n
                if u != v and not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            break
    return graph


def egrg(n: int, edges: int) -> nx.Graph:
    """EGRG (Algorithm 4) for a degree-2 leaf network: ring + evenly spread chords."""
    graph = nx.cycle_graph(n) if n >= 3 else nx.path_graph(max(n, 1))
    extra = max(0, int(edges) - graph.number_of_edges())
    if extra == 0 or n < 4:
        return graph
    span = max(1, n // extra)
    for t in range(extra):
        u = (t * n) // extra
        v = (u + span) % n
        if u != v and not graph.has_edge(u, v):
            graph.add_edge(u, v)
    return graph


def construct(G: nx.Graph, *, seed: int = 0, variant: str = "eq18", **_: object) -> nx.Graph:
    """Return the GRG reconstruction of ``G`` at the same ``(N, M)`` budget."""
    del seed
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 3:
        return finalize_construct(G.copy(), "GRG")
    degree = 2 * (m // n)
    if degree >= 4:
        g = grg_regular(n, degree, variant=variant)
    else:
        g = egrg(n, m)
    g = _top_up(g, m)
    g.graph["grg_degree"] = degree
    g.graph["grg_jumps"] = jump_set(n, degree, variant=variant) if degree >= 4 else []
    return finalize_construct(g, "GRG")
