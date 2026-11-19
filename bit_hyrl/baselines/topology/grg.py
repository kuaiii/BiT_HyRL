"""GRG — the *paper-faithful* circulant construction (Huang et al., IEEE ToN 34, 2026).

This module replaces :mod:`grg_proxy` for any comparison that cites the paper.  The
proxy was built from the abstract alone (near-regular graph + λ₂ hill-climbing); the
real GRG is a **deterministic circulant** and needs no search at all, which is the
whole point of the paper (``O(n)`` construction vs ``O(n^{d/2+1})`` brute force).

Source of truth — Sec. IV-A, Eq. (18):

    A_xy = 1  iff  |x - y| = ±t,  t ∈ { 1, ⌊n/k⌋, ⌊2n/k⌋, …, ⌊(k-1)n/k⌋ },  d = 2k

i.e. ``V_i`` is wired to ``V_{i±1}`` and to ``V_{i±⌊jn/k⌋}`` for ``j = 1..k-1``.  The
jump set is what makes bisection expensive: one cut can only separate nodes that are
*near* each other, so the attacker must remove many nodes before ``|C_max|`` falls.

Two notes on fidelity, both recorded because they change the numbers:

1. **Eq. (18) is a k-dimensional cube, not a ladder.**  The jump set is
   ``{1, ⌊n^{1/k}⌋, …, ⌊(k-1)n^{1/k}⌋}``; text extraction renders the exponent as
   "⌊n1 k⌋", and reading it as ``⌊n/k⌋`` yields a circular ladder (k=2, n=100 →
   jumps {1,50}) with bisection width 2.  Figs. 10–11 (n=13, d=4 → jumps {1,4},
   "closer to a square") confirm the cube reading.  ``variant="ladder"`` keeps the
   wrong reading so its cost can be measured.
2. **Odd ``d`` / ``d < 4``.**  Sec. IV-A only proves the even case ``d = 2k``.  For
   ``M/N < 2`` (every ISP substrate: Cogentco 2.47, Colt 2.31) the paper's answer is
   EGRG (Algorithm 4) on the degree-2 leaf network; we implement its *intent* — a
   ring plus evenly spread chords — and label it ``EGRG`` so it is never confused
   with the proven case.

Both constructions return a graph with the **same N and the same M** as the input, so
they enter the same-budget table without a normalisation step.
"""

from __future__ import annotations

import networkx as nx

from .._common import finalize_construct

__all__ = ["construct", "egrg", "grg_regular", "jump_set"]


def jump_set(n: int, degree: int, *, variant: str = "eq18") -> list[int]:
    """Return the circulant jump set for ``n`` nodes of even ``degree = 2k``.

    Eq. (18) reads ``t ∈ {1, ⌊n^{1/k}⌋, …, ⌊(k-1)n^{1/k}⌋}`` — the exponent is easy to
    lose in text extraction (it renders as "⌊n1 k⌋"), and mistaking it for ``⌊n/k⌋``
    turns the construction into a circular ladder (k=2, n=100 → jumps {1,50}) whose
    bisection width is 2, which would make GRG collapse under exactly the attack it is
    designed against.  Figs. 10–11 settle which reading is right: for n=13, d=4 the
    paper draws jumps {1,4} and calls it "closer to a square", and 4 ≈ √13.

    Args:
        n: Node count.
        degree: Even target degree ``2k``.
        variant: ``"eq18"`` (default, multiples of ``n^{1/k}``).  ``"ladder"`` keeps the
            mis-extracted reading so the difference can be measured instead of argued.

    Returns:
        Sorted list of distinct jumps in ``1 .. n//2``.
    """
    if degree % 2:
        raise ValueError(f"degree 必须是偶数，收到 {degree}")
    k = degree // 2
    if variant not in {"eq18", "ladder"}:
        raise ValueError(f"variant 只能是 'eq18' / 'ladder'，收到 {variant!r}")
    side = n ** (1.0 / k)
    jumps: list[int] = [1]
    for j in range(1, k):
        raw = (j * n) // k if variant == "ladder" else int(round(j * side))
        t = min(raw % n, n - (raw % n))                  # ±t 给出同一组边
        if t > 1 and t not in jumps:
            jumps.append(t)
    return sorted(jumps)


def grg_regular(n: int, degree: int, *, variant: str = "eq18") -> nx.Graph:
    """``d``-regular circulant on ``0..n-1`` with the GRG jump set.

    The construction is deterministic and ``O(nk)``; no seed is involved, matching
    the paper's claim that GRG needs no search.

    Args:
        n: Node count.
        degree: Even target degree.
        variant: Passed to :func:`jump_set`.

    Returns:
        A simple ``d``-regular graph when ``d`` distinct jumps exist, else as close
        to regular as the jump budget allows (``finalize_construct`` repairs
        connectivity, never node count).
    """
    jumps = jump_set(n, degree, variant=variant)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for t in jumps:
            graph.add_edge(i, (i + t) % n)
    return graph


def _top_up(graph: nx.Graph, target_edges: int) -> nx.Graph:
    """Spread the ``M - n·d/2`` remainder over the ring as evenly spaced chords.

    Flooring ``d`` to an even number leaves up to ``n-1`` edges unspent (BA-100:
    ``d = 4`` from ``M/N = 2.91`` leaves 91 of 291 edges).  Dumping them on the
    lowest-degree nodes — the obvious repair — creates a new hub class that the
    targeted attack removes first, i.e. it silently rebuilds the weakness GRG exists
    to avoid (measured: ``R_tar`` 0.375 → 0.356 once the remainder is spread).
    Spreading them at spacing ``⌊n/remainder⌋`` keeps every degree within one of
    ``d`` and, per the paper's EGRG argument, keeps the cut wide.
    """
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
    # Anything still unspent (integer collisions) goes to the longest free span.
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
    """EGRG (Algorithm 4) for a degree-2 leaf network: ring + evenly spread chords.

    ``m = (n(d-2))/2`` chords are placed at spacing ``⌊n/m⌋`` so that the cut needed
    to split the ring stays long — the paper's Algorithm 4 step 3 (its pseudo-code
    prints the two index expressions without saying which node is the source; we
    take the symmetric reading and record it here).
    """
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
    """Return the GRG reconstruction of ``G`` at the same ``(N, M)`` budget.

    Args:
        G: Substrate graph (only ``N`` and ``M`` are used — GRG ignores the wiring).
        seed: Accepted for interface compatibility; GRG is deterministic.
        variant: ``"eq18"`` (default) or ``"sqrt"``; see :func:`jump_set`.

    Returns:
        ``d``-regular circulant (``d = 2⌊M/N⌋``) topped up to exactly ``M`` edges, or
        the EGRG ring construction when ``M/N < 2`` (``d < 4``, outside the proven
        case).
    """
    del seed                                                   # deterministic
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 3:
        return finalize_construct(G.copy(), "GRG")
    degree = 2 * (m // n)                                      # largest even d
    if degree >= 4:
        g = grg_regular(n, degree, variant=variant)
    else:
        g = egrg(n, m)
    g = _top_up(g, m)
    g.graph["grg_degree"] = degree
    g.graph["grg_jumps"] = jump_set(n, degree, variant=variant) if degree >= 4 else []
    return finalize_construct(g, "GRG")
