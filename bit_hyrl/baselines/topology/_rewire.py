"""Shared machinery for the *rewiring* baselines (R-GREEDY / SA-R / ASSORT-SWAP / RAND-SWAP).

All four baselines move through the **same** feasible set: degree-preserving
double-edge swaps (2-switch) that keep the graph simple and connected.  Only the
**acceptance rule** differs, which is exactly what makes them a clean ablation of
"how much does search buy you":

* ``RAND-SWAP``    -- accept every legal swap (null model: same N, M, degree sequence)
* ``ASSORT-SWAP``  -- Xulvi-Brunet & Sokolov style: accept swaps that raise the
                      degree-product sum (i.e. connect high to high, low to low)
* ``SA-R``         -- accept by Metropolis on the robustness ``R``
* ``R-GREEDY``     -- accept only strict ``R`` improvements picked from a candidate pool

The robustness objective is :func:`effrnet._fast_robustness` -- the **same** cheap
``R`` (GCC/N area over a degree-targeted removal grid) that the Eff-R-Net baseline
already optimises, so the search baselines are compared on one fitness definition.
Nothing here touches the core metrics: the returned graph is scored afterwards by
the ordinary pipeline.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import networkx as nx

__all__ = ["propose_swap", "degree_product_sum", "r_paper", "FAST_R_STEPS"]

#: Removal grid used by the cheap fitness.  Deliberately finer than Eff-R-Net's 12:
#: on a 100-node graph a 12-point grid quantises ``R`` to ~1/12 and a strict-improvement
#: greedy stalls after a handful of rounds (measured: 49 evaluations, R=0.188 vs 0.238
#: at 25 points).  25 points is the coarsest grid on which greedy keeps moving.
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
    """One legal degree-preserving 2-switch, or ``(None, None)`` if none was found.

    Picks two disjoint edges ``(a,b)``, ``(c,d)`` and reconnects them as
    ``(a,d)``, ``(c,b)``.  Rejected when the new edges already exist (would create a
    multi-edge) or when the swap disconnects the graph -- connectivity is a hard
    constraint for every baseline, otherwise "robustness" would be bought by
    fragmenting the network.

    Returns the **new graph** plus its **new edge list** so a caller that keeps the
    candidate can reuse the list instead of recomputing it.
    """
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
    """Schneider's ``R`` on a coarse grid, with the **paper's** ``/N`` normalisation.

    ``R = (1/N) Σ_i MCS_i / N`` (GRG paper Eq. 1, ceiling 0.5) — not the
    survivor-normalised ``effrnet._fast_robustness``, which divides by the *alive*
    count, can exceed 0.5, and weights late steps (``f → 1``) much more heavily.  The
    two orderings disagree: a swap that keeps 90 % of a 10-node remnant alive looks
    excellent under ``/alive`` and irrelevant under ``/N``.

    Args:
        graph: The evaluated graph.
        steps: Removal-grid resolution (coarse: this is a search fitness).

    Returns:
        ``R ∈ [0, 0.5]``.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    # Tie-breaking MUST be random: on a regular graph every degree is equal, and an
    # id-ordered tie-break removes nodes along the construction's own index order
    # (e.g. around the GRG ring), which leaves the rest perfectly connected and
    # reports the impossible ceiling R = 0.5.  The paper also breaks ties at random.
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
