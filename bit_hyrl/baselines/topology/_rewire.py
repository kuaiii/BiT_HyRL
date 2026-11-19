"""Degree-preserving rewiring helpers shared by the rewiring baselines."""

from __future__ import annotations

import random
from collections.abc import Iterable

import networkx as nx

__all__ = ["propose_swap", "degree_product_sum", "r_paper", "FAST_R_STEPS"]

FAST_R_STEPS = 25


def degree_product_sum(graph: nx.Graph) -> float:
    """``sum_{(u,v) in E} k_u * k_v`` -- the assortativity proxy of a 2-switch."""
    return float(sum(graph.degree(u) * graph.degree(v) for u, v in graph.edges()))


def propose_swap(
    graph: nx.Graph,
    edges: Iterable[tuple[int, int]],
    rng: random.Random | None = None,
    tries: int = 40,
) -> tuple[nx.Graph, list[tuple[int, int]]] | tuple[None, None]:
    """One legal degree-preserving 2-switch, or ``(None, None)`` if none was found."""
    rng = rng or random
    pool = list(edges)
    if len(pool) < 2:
        return None, None
    for _ in range(tries):
        (a, b), (c, d) = rng.sample(pool, 2)
        if len({a, b, c, d}) < 4:
            continue
        if graph.has_edge(a, d) or graph.has_edge(c, b):
            continue
        candidate = graph.copy()
        candidate.remove_edge(a, b)
        candidate.remove_edge(c, d)
        candidate.add_edge(a, d)
        candidate.add_edge(c, b)
        if not nx.is_connected(candidate):
            continue
        new_edges = [(u, v) for (u, v) in pool if (u, v) != (a, b) and (u, v) != (c, d)]
        new_edges.extend([(a, d), (c, b)])
        return candidate, new_edges
    return None, None


def r_paper(graph: nx.Graph, steps: int = FAST_R_STEPS) -> float:
    """Schneider's ``R`` on a coarse grid, with the **paper's** ``/N`` normalisation."""
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    order = list(graph.nodes())
    random.Random(20261115).shuffle(order)
    order.sort(key=lambda v: -graph.degree(v))
    step = max(1, n // steps)
    work = graph.copy()
    curve = [max((len(c) for c in nx.connected_components(work)), default=0) / n]
    for start in range(0, n, step):
        for node in order[start:start + step]:
            if work.has_node(node):
                work.remove_node(node)
        curve.append(max((len(c) for c in nx.connected_components(work)), default=0) / n)
    area = 0.0
    for index in range(len(curve) - 1):
        area += 0.5 * (curve[index] + curve[index + 1]) / (len(curve) - 1)
    return area
