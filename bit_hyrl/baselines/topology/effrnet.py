"""Approximate implementation of Eff-R-Net (Huang, Wu & Li, Inf. Sci. 690, 2024).

The original code is not public; this port follows the published description of
the three operators (two-switch, onion-swap and three-part crossover) and is
labelled as an approximation wherever its numbers are reported.
"""

from __future__ import annotations

import random

import networkx as nx

from .._common import finalize_construct, seeded_legacy

__all__ = ["construct"]


def _largest_cc_fraction(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    biggest = max((len(c) for c in nx.connected_components(graph)), default=0)
    return biggest / graph.number_of_nodes()


def _fast_robustness(graph: nx.Graph, steps: int) -> float:
    """Cheap ``R``: GCC/N area on a coarse degree-targeted removal grid in ``[0,1]``."""
    n0 = graph.number_of_nodes()
    if n0 == 0:
        return 0.0
    order = sorted(graph.nodes(), key=lambda v: (-graph.degree(v), v))
    step = max(1, n0 // steps)
    work = graph.copy()
    curve: list[float] = []
    for index in range(steps + 1):
        curve.append(_largest_cc_fraction(work))
        for node in order[index * step:(index + 1) * step]:
            if work.has_node(node):
                work.remove_node(node)
    area = 0.0
    for index in range(steps):
        area += 0.5 * (curve[index] + curve[index + 1]) / steps
    return area


def _degree_assort(graph: nx.Graph) -> float:
    try:
        return nx.degree_assortativity_coefficient(graph)
    except Exception:  # pragma: no cover - degenerate graphs
        return 0.0


def _two_switch(graph: nx.Graph) -> nx.Graph:
    """One random degree-preserving 2-switch; keep simple + connected (else return input)."""
    edges = list(graph.edges())
    if len(edges) < 2:
        return graph
    for _ in range(40):
        (a, b), (c, d) = random.sample(edges, 2)
        if len({a, b, c, d}) < 4:
            continue
        if graph.has_edge(a, d) or graph.has_edge(c, b):
            continue
        candidate = graph.copy()
        candidate.remove_edge(a, b)
        candidate.remove_edge(c, d)
        candidate.add_edge(a, d)
        candidate.add_edge(c, b)
        if nx.is_connected(candidate):
            return candidate
    return graph


def _onion_swap(graph: nx.Graph, trials: int = 8) -> nx.Graph:
    """Prefer a swap that raises degree assortativity (more onion-like)."""
    best, best_score = graph, _degree_assort(graph)
    for _ in range(trials):
        candidate = _two_switch(graph)
        score = _degree_assort(candidate)
        if score > best_score:
            best, best_score = candidate, score
    return best


def _three_part_crossover(child: nx.Graph, best: nx.Graph) -> nx.Graph:
    """Approximate three-part composite crossover."""
    target = _degree_assort(best)
    improved = child
    for _ in range(6):
        candidate = _two_switch(improved)
        if abs(_degree_assort(candidate) - target) < abs(_degree_assort(improved) - target):
            improved = candidate
            break
    improved = _two_switch(improved)
    improved = _onion_swap(improved, trials=4)
    return improved


def _greedy_local(graph: nx.Graph, steps: int) -> nx.Graph:
    """LocalSearchA: accept a neighbour only if the cheap fitness improves."""
    best = graph
    best_fit = _fast_robustness(graph, steps)
    for _ in range(5):
        candidate = _two_switch(best)
        fit = _fast_robustness(candidate, steps)
        if fit > best_fit:
            best, best_fit = candidate, fit
    return best


def construct(G: nx.Graph, *, seed: int = 0, pop: int = 20, gen: int = 30,
              steps: int = 10, **params) -> nx.Graph:
    """Return an Eff-R-Net-approximate reconstruction of ``G`` (same N and M)."""
    with seeded_legacy(int(seed)):
        g0 = finalize_construct(G, "Eff-R-Net")
        n = g0.number_of_nodes()
        m = g0.number_of_edges()
        if n < 4 or m < 2:
            return g0

        pop = max(2, int(pop))
        gen = max(1, int(gen))

        population: list[nx.Graph] = []
        for _ in range(pop):
            candidate = g0.copy()
            for _ in range(max(1, m // 20)):
                candidate = _two_switch(candidate)
            population.append(candidate)
        fitness = [_fast_robustness(g, steps) for g in population]

        best_index = max(range(len(population)), key=lambda i: fitness[i])
        best_graph, best_fit = population[best_index], fitness[best_index]

        weights = {"cross": 0.40, "mut": 0.30, "lsA": 0.15, "lsB": 0.15}
        names = list(weights)

        def _pick() -> str:
            draw = random.random()
            acc = 0.0
            for name in names:
                acc += weights[name]
                if draw <= acc:
                    return name
            return names[-1]

        for _ in range(gen):
            survivors = [best_graph]
            survivor_fit = [best_fit]
            while len(survivors) < pop:
                parent = random.choice(population)
                parent_fit = _fast_robustness(parent, steps)
                child = parent.copy()
                operator = _pick()
                if operator == "cross":
                    child = _three_part_crossover(child, best_graph)
                elif operator == "mut":
                    child = _onion_swap(child)
                elif operator == "lsA":
                    child = _greedy_local(child, steps)
                else:
                    child = _onion_swap(_onion_swap(child))
                child_fit = _fast_robustness(child, steps)
                survivors.append(child)
                survivor_fit.append(child_fit)
                if child_fit > parent_fit:
                    weights[operator] *= 1.10
                else:
                    weights[operator] *= 0.95
                total = sum(weights.values())
                for name in names:
                    weights[name] /= total

            order = sorted(range(len(survivors)), key=lambda i: survivor_fit[i], reverse=True)
            population = [survivors[i] for i in order[:pop]]
            fitness = [survivor_fit[i] for i in order[:pop]]
            if fitness[0] > best_fit:
                best_graph, best_fit = population[0], fitness[0]

        return finalize_construct(best_graph, "Eff-R-Net")
