"""Deterministic construction of the bimodal graph, decorrelation and repair.

The builder has exactly four jobs, and it is deliberately *not* allowed to do
anything else:

1. realise the planned degree sequence with deterministic Havel–Hakimi;
2. **decorrelate** the edge set with degree-preserving 2-switches
   (``realize="hh+rewire"``, the default);
3. if the result is disconnected, repair it with **degree-preserving
   2-switches** (``a-b, c-d → a-c, b-d``) — never by adding or removing edges;
4. prove the result: the realised degree histogram must equal the planned
   histogram by strict equality, the edge count must still be ``m``, and the
   graph hash must be reproducible.

Why step 2 is *correctness*, not polish
---------------------------------------
Plain Havel–Hakimi on a near-regular sequence does not produce a network: it
produces a "hub plus a handful of very long chains".  Measured at ``c = 1.6``,
``seed = 0`` — and the sharpest statement is not "the diameter is large" but
"**pure HH does not even connect**":

===========  ==============  ====================  ============  ==============
graph        pure HH         pure HH giant         ``hh+rewire`` published
             components      D / L                 D / L         D / L
===========  ==============  ====================  ============  ==============
BA-400       4 (376+8+8+8)   61 / 29.91            5 / 3.15      5 / 3.15
BA-500       4 (482+6+6+6)   102 / 50.88           6 / 3.76      ≈4 / 3.75
Cogentco     36 (92+35×3)    35 / 17.21            16 / 8.23     18 / 8.18
===========  ==============  ====================  ============  ==============

The repair alone cannot rescue this: joining four chains with three 2-switches
gives a connected graph whose diameter is still two orders of magnitude off.  A
5-regular random graph on 500 nodes has ``L ≈ 3.75``, so ``L = 50.88`` can only
come from an ordered construction.  Only the decorrelated version reproduces the
project's published ``BA-400 = 5 / 3.15`` **digit for digit**, which settles the
question: ``realize="hh"`` is kept only so the *pure* HH graph can still be
reproduced for comparison, and every production path uses ``realize="hh+rewire"``.

Cogentco's diameter is the one published number that does not reproduce at
``seed = 0`` (16 measured, 18 published): the sparse graph's diameter spans 16–19
over eight seeds at a 10 m budget, so ``D = 18`` is a single sample of the same
ensemble rather than a contradiction.

Where the connectivity guarantee lives
--------------------------------------
:func:`rewire_preserving_connectivity` is the **only** decorrelation path, and it
is *connectivity-preserving by construction*: a pass is accepted only if its
result is connected, otherwise it is discarded and re-run on the same derived
stream (a deterministic retry — no re-seeding, no re-drawing from a fresh
stream).  There is deliberately no second, older path of the form "rewire
freely, then check, then repair": one semantics, one place to reason about.

Consequently ``repairs`` is an **upper-bound witness** rather than the usual
mechanism — at the 10 m default it is ``0`` on all nine graphs, because no
accepted pass ever handed back a disconnected graph, and ``rewire_restarts``
records how many passes had to be thrown away.  The final
:func:`repair_connectivity` step remains as a defensive net (it is a no-op
whenever the pass succeeded), which is exactly why it may be called without
weakening the guarantee.

The default budget is ``10 * m`` switches, and it is measured rather than picked.
``tests/test_rewire_convergence.py`` checks two things:

* **mixing** — the edge set after ``1 m`` switches already differs from the
  ``40 m`` edge set by as much as two *independent seeds* differ from each other
  (Jaccard 0.036 vs 0.028 on BA-100, 0.009 vs 0.003 on Cogentco), i.e. the chain
  is fully mixed after one swap per edge and ``10 m`` carries a 10× margin;
* **stationarity** — the paired mean path length at ``10 m`` and ``40 m`` agrees
  to well under 1 % at 40 seeds.

The reason a naive "same diameter and < 1 % mean path at 4m/10m/20m" test would be
wrong is that a 2-switch chain is sampl*ing* an ensemble, not converging to a fixed
point: over 16 seeds σ(``avg_path``) is 0.006 (BA-500) to 0.45 (Colt) and the
diameter spans 16–19 on Cogentco at every budget.  Point-wise cross-budget
equality is below the noise floor and can never hold; the tests state the
ensemble question instead.

Why step 3 is degree-preserving
-------------------------------
The repair exists because the earlier implementation "fixed" connectivity with
helpers that added and removed edges, which silently destroyed both ``Σk = 2M``
and the degree distribution — the two properties the whole construction exists to
guarantee.

The repair used here is *component merging*: pick one edge from the smallest
component and one edge from the second smallest, and rewire them so that the two
components join.  Because the two edges live in different components, no
cross edge can already exist, so the rewire is always legal and each successful
switch reduces the component count by exactly one.  At most ``n_components - 1``
switches are therefore ever needed.

Randomness discipline
---------------------
Every random draw comes from a named stream derived from the single root seed:

* ``("topology", dataset, "rewire")`` — decorrelation;
* ``("topology", dataset, "repair")`` — connectivity repair.

There is no module-level seeding call, no use of the legacy global entry
point of ``numpy.random``, and no bare ``default_rng()``.  Consequently ``(plan, seed, realize, dataset)`` in → the
same ``graph_sha256`` out, always.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..io.graph_io import edge_list_sha256, graph_fingerprint
from ..rng import derive
from .degree_sequence import DegreePlan, DegreeSequenceError
from .graphicality import GraphicalityLog, havel_hakimi, realize_with_downgrade

__all__ = [
    "ConstructionError",
    "ConstructionResult",
    "REALIZE_MODES",
    "DEFAULT_REWIRE_SWAPS_PER_EDGE",
    "build_bimodal_graph",
    "build_from_sequence",
    "degree_preserving_swaps",
    "repair_connectivity",
    "rewire_preserving_connectivity",
    "realized_histogram",
]

#: How many random edge-pair samples to try before declaring a component pair
#: unmergeable.  With the component-merging strategy the first sample always
#: works, so this is a defensive bound rather than a working parameter.
_MAX_SWAP_TRIES = 64

#: Decorrelation budget, in 2-switches per edge.  Pinned by the convergence test
#: (``4m``/``10m``/``20m`` must already agree to 1% on the mean path length).
DEFAULT_REWIRE_SWAPS_PER_EDGE = 10

#: Accepted values of the ``realize`` parameter.
REALIZE_MODES: tuple[str, ...] = ("hh", "hh+rewire")


class ConstructionError(RuntimeError):
    """Raised when a construction cannot satisfy its own invariants."""


@dataclass(frozen=True)
class ConstructionResult:
    """Outcome of one BiT construction.

    Attributes:
        graph: The realised simple undirected graph, node ids ``0 .. n-1``.
        plan: The (possibly downgraded) plan that was realised.
        graphicality: Attempt log from the downgrade loop.
        connected: Whether the graph is connected after repair.
        realize: Which realisation mode produced the graph — ``"hh"`` (pure,
            deterministic, chain-like) or ``"hh+rewire"`` (decorrelated).
        repairs: Degree-preserving 2-switches spent on **connectivity**.  Because
            :func:`rewire_preserving_connectivity` only accepts connected
            results, this is an upper-bound witness and measures ``0`` on all
            nine evaluation graphs at the default ``10 m`` budget; it is non-zero
            only on the degraded paths (``realize="hh"`` on a sequence whose HH
            realisation is disconnected, or a rewire pass whose every retry
            failed).  Every one of these switches preserves ``Σk`` and the degree
            histogram.
        final_edge_count: Edge count of ``graph``; asserted to equal
            ``plan.m`` before the result is returned.
        graph_sha256: SHA-256 over the canonical sorted edge list.  Two
            constructions from the same plan + seed + mode must agree.
        realized_degree_histogram: ``{degree: count}`` of the realised graph.
        histogram_matches_plan: Always ``True``; kept as a field so a caller can
            assert on it without recomputing.
        elapsed_sec: Wall-clock construction time.
        rewire_swaps_used: Decorrelation switches actually applied.
        rewire_restarts: How many decorrelation passes were discarded because
            they disconnected the graph (deterministic retry, not a random
            re-seed).
        seed: Root seed used to derive every stream.
        fingerprint: Loader-style fingerprint record of the realised graph.
    """

    graph: nx.Graph
    plan: DegreePlan
    graphicality: GraphicalityLog
    connected: bool
    realize: str
    repairs: int
    final_edge_count: int
    graph_sha256: str
    realized_degree_histogram: dict[int, int]
    histogram_matches_plan: bool
    elapsed_sec: float
    rewire_swaps_used: int = 0
    rewire_restarts: int = 0
    seed: int = 0
    fingerprint: dict[str, Any] = field(default_factory=dict)

    # -- backwards-compatible view ----------------------------------------- #
    @property
    def swaps_used(self) -> int:
        """Alias of :attr:`repairs` (the pre-D3 field name)."""
        return self.repairs

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` summary; the graph itself is excluded on purpose."""
        return {
            "n": self.plan.n,
            "m": self.final_edge_count,
            "plan": self.plan.to_dict(),
            "graphicality": self.graphicality.to_dict(),
            "connected": self.connected,
            "realize": self.realize,
            "repairs": self.repairs,
            "final_edge_count": self.final_edge_count,
            "graph_sha256": self.graph_sha256,
            "rewire_swaps_used": self.rewire_swaps_used,
            "rewire_restarts": self.rewire_restarts,
            "realized_degree_histogram": {
                str(k): v for k, v in sorted(self.realized_degree_histogram.items())
            },
            "histogram_matches_plan": self.histogram_matches_plan,
            "elapsed_sec": self.elapsed_sec,
            "seed": self.seed,
            "fingerprint": dict(self.fingerprint),
        }


def realized_histogram(graph: nx.Graph) -> dict[int, int]:
    """Return ``{degree: count}`` for ``graph``, sorted by degree."""
    counts = Counter(dict(graph.degree()).values())
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# 2-switch machinery
# --------------------------------------------------------------------------- #
def _swap_is_legal(graph: nx.Graph, a: int, b: int, c: int, d: int,
                   new_pairs: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """Check that replacing ``a-b, c-d`` by ``new_pairs`` is a legal 2-switch."""
    removed = {(a, b) if a <= b else (b, a), (c, d) if c <= d else (d, c)}
    for u, v in new_pairs:
        if u == v:
            return False  # would create a self loop
        key = (u, v) if u <= v else (v, u)
        if key in removed:
            continue
        if graph.has_edge(u, v):
            return False  # would create a parallel edge
    return True


def repair_connectivity(graph: nx.Graph, rng: Any,
                        max_swaps: int = 20000) -> tuple[nx.Graph, int]:
    """Make ``graph`` connected using only degree-preserving 2-switches.

    The graph is modified **in place** (and returned for convenience).  Degrees
    are preserved exactly, so the caller can still assert the realised histogram
    against the plan afterwards.

    Args:
        graph: Connected or disconnected simple graph.
        rng: A :class:`numpy.random.Generator`; only this stream is consumed.
        max_swaps: Budget of successful switches.

    Returns:
        ``(graph, swaps_used)``.

    Raises:
        ConstructionError: If the graph has an isolated node (a 2-switch can
            never attach it), if a component pair cannot be merged, or if the
            budget runs out before the graph becomes connected.
    """
    node_count = graph.number_of_nodes()
    if node_count <= 1:
        return graph, 0
    if min(degree for _, degree in graph.degree()) < 1:
        raise ConstructionError(
            "存在度数为 0 的节点：保度 2-switch 无法把它接入主分量，"
            "该度序列本身不可能连通"
        )

    swaps = 0
    while True:
        components = sorted(
            (sorted(component) for component in nx.connected_components(graph)),
            key=lambda component: component[0],
        )
        if len(components) == 1:
            return graph, swaps
        if swaps >= max_swaps:
            raise ConstructionError(
                f"连通修复预算耗尽（{max_swaps} 次 swap），仍剩 {len(components)} 个分量"
            )

        left, right = components[0], components[1]
        left_edges = sorted(tuple(sorted(edge)) for edge in graph.subgraph(left).edges())
        right_edges = sorted(tuple(sorted(edge)) for edge in graph.subgraph(right).edges())
        if not left_edges or not right_edges:
            raise ConstructionError(
                f"分量 {left[0]} 或 {right[0]} 无边，无法用 2-switch 合并"
            )

        merged = False
        for _ in range(_MAX_SWAP_TRIES):
            a, b = left_edges[int(rng.integers(len(left_edges)))]
            c, d = right_edges[int(rng.integers(len(right_edges)))]
            for new_pairs in (((a, c), (b, d)), ((a, d), (b, c))):
                if not _swap_is_legal(graph, a, b, c, d, new_pairs):
                    continue
                graph.remove_edge(a, b)
                graph.remove_edge(c, d)
                graph.add_edge(*new_pairs[0])
                graph.add_edge(*new_pairs[1])
                merged = True
                break
            if merged:
                break
        if not merged:  # pragma: no cover - cannot happen with the merge strategy
            raise ConstructionError(
                f"无法合并分量 {left[0]} 与 {right[0]}：{_MAX_SWAP_TRIES} 次尝试均被拒绝"
            )
        swaps += 1


def degree_preserving_swaps(graph: nx.Graph, target_swaps: int, rng: Any, *,
                            max_attempts_factor: int = 8) -> tuple[nx.Graph, int]:
    """Apply up to ``target_swaps`` random degree-preserving 2-switches.

    Degrees are preserved *exactly*, so the degree histogram and ``Σk = 2M``
    are untouched; only the edge arrangement changes.  Connectivity is **not**
    guarded here — use :func:`rewire_preserving_connectivity` when it matters.

    The edge list and adjacency sets are maintained incrementally (an earlier
    version re-sorted the whole edge list after every switch, which made a
    ``10m`` pass on BA-500 take minutes instead of milliseconds).

    Args:
        graph: Input graph (copied, not modified).
        target_swaps: Number of switches to attempt.
        rng: Random stream; the only source of randomness used.
        max_attempts_factor: Attempt budget per switch; a rejected proposal
            (would create a parallel edge) consumes an attempt but no switch.

    Returns:
        ``(new_graph, applied_swaps)``.
    """
    working = graph.copy()
    edges = sorted(tuple(sorted(edge)) for edge in working.edges())
    if len(edges) < 2 or target_swaps <= 0:
        return working, 0

    neighbours: dict[int, set[int]] = {
        node: set(working.neighbors(node)) for node in working
    }

    def _is_edge(u: int, v: int) -> bool:
        return v in neighbours.get(u, ())

    applied = 0
    attempts = 0
    max_attempts = max(64, target_swaps * max_attempts_factor)
    edge_slots = len(edges)

    while applied < target_swaps and attempts < max_attempts:
        attempts += 1
        first = int(rng.integers(edge_slots))
        second = int(rng.integers(edge_slots))
        if first == second:
            continue
        a, b = edges[first]
        c, d = edges[second]
        if a == c or a == d or b == c or b == d:
            continue  # a shared endpoint would create a self loop or a no-op

        chosen: tuple[tuple[int, int], tuple[int, int]] | None = None
        for candidate in (((a, c), (b, d)), ((a, d), (b, c))):
            (p, q), (s, t) = candidate
            if _is_edge(p, q) or _is_edge(s, t):
                continue
            chosen = ((p, q), (s, t))
            break
        if chosen is None:
            continue

        neighbours[a].discard(b)
        neighbours[b].discard(a)
        neighbours[c].discard(d)
        neighbours[d].discard(c)
        for u, v in chosen:
            neighbours[u].add(v)
            neighbours[v].add(u)
        edges[first] = tuple(sorted(chosen[0]))
        edges[second] = tuple(sorted(chosen[1]))
        applied += 1

    rebuilt = nx.Graph()
    rebuilt.add_nodes_from(working.nodes())
    rebuilt.add_edges_from(edges)
    return rebuilt, applied


def rewire_preserving_connectivity(graph: nx.Graph, target_swaps: int, rng: Any, *,
                                   max_restarts: int = 8,
                                   require_connected: bool = True
                                   ) -> tuple[nx.Graph, int, int]:
    """Decorrelation pass that refuses to hand back a disconnected graph.

    This is the **only** decorrelation entry point in the construction, so there
    is exactly one semantics to reason about: the pass never returns a
    disconnected graph when ``require_connected=True``.  Connectivity is
    guaranteed by *acceptance*, not by post-hoc repair — a chain of 2-switches
    can walk the realisation into a disconnected corner of the degree-sequence
    space, and such a pass is **discarded and re-run on the same deterministic
    stream** rather than patched afterwards.

    Consequence for the audit fields: ``repairs`` (the final
    :func:`repair_connectivity` net) is a **upper-bound witness** that is ``0``
    on all nine evaluation graphs at the default budget, while ``restarts``
    records how many candidate passes were thrown away.  When every retry fails,
    the last candidate is still returned and the caller's connectivity repair —
    also degree-preserving — closes the gap, so the invariant ``edge_count == M``
    and the degree histogram hold on every path.

    Args:
        graph: Connected input graph (Havel–Hakimi output).
        target_swaps: Number of switches per pass.
        rng: The derived decorrelation stream.
        max_restarts: How many additional passes may be discarded.
        require_connected: When ``False``, the first pass is accepted as-is; this
            is used only by the ``realize="hh"`` comparison path and by tests
            that deliberately want the disconnected realisation.

    Returns:
        ``(graph, swaps_applied, restarts)`` where ``restarts`` counts discarded
        passes (``0`` on a clean first try).
    """
    if target_swaps <= 0 or graph.number_of_nodes() <= 2 or not require_connected:
        candidate, applied = degree_preserving_swaps(graph, target_swaps, rng)
        return candidate, applied, 0

    candidate = graph
    applied = 0
    restarts = 0
    for attempt in range(max_restarts + 1):
        candidate, applied = degree_preserving_swaps(graph, target_swaps, rng)
        if nx.is_connected(candidate):
            return candidate, applied, attempt
        restarts = attempt + 1
    return candidate, applied, restarts


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def _resolve_rewire_budget(realize: str, rewire_swaps: int | None, m: int) -> int:
    """Translate ``(realize, rewire_swaps)`` into an explicit switch budget.

    Args:
        realize: ``"hh"`` or ``"hh+rewire"``.
        rewire_swaps: Explicit budget, or ``None`` for the mode default.
        m: Planned edge count (the default budget is ``10 * m``).

    Returns:
        The number of 2-switches to attempt (``0`` means pure Havel–Hakimi).

    Raises:
        ConstructionError: For an unknown ``realize`` value or a negative budget.
    """
    if realize not in REALIZE_MODES:
        raise ConstructionError(
            f"realize={realize!r} 不受支持，只能是 {list(REALIZE_MODES)}"
        )
    if rewire_swaps is not None:
        if isinstance(rewire_swaps, bool) or not isinstance(rewire_swaps, int):
            raise ConstructionError(f"rewire_swaps 必须是整数，收到 {rewire_swaps!r}")
        if rewire_swaps < 0:
            raise ConstructionError(f"rewire_swaps 不能为负，收到 {rewire_swaps}")
        return int(rewire_swaps)
    if realize == "hh":
        return 0
    return DEFAULT_REWIRE_SWAPS_PER_EDGE * int(m)


def build_bimodal_graph(plan: DegreePlan, seed: int, *,
                        realize: str = "hh+rewire",
                        rewire_swaps: int | None = None,
                        enforce_connected: bool = True,
                        max_repair_swaps: int = 20000,
                        max_hh_attempts: int = 64,
                        max_rewire_restarts: int = 8,
                        dataset: str = "") -> ConstructionResult:
    """Realise ``plan`` as a graph, deterministically, and prove the invariants.

    Args:
        plan: Degree budget from
            :func:`~bit_hyrl.topology.degree_sequence.plan_degree_sequence`.
        seed: Root seed.  Only derived streams are used — the decorrelation
            (``("topology", dataset, "rewire")``) and the connectivity repair
            (``("topology", dataset, "repair")``).
        realize: ``"hh+rewire"`` (default, decorrelated — what the paper
            measures) or ``"hh"`` (pure Havel–Hakimi, kept for comparison with
            the earlier fingerprints).
        rewire_swaps: Decorrelation budget; ``None`` means ``10 * plan.m`` for
            ``"hh+rewire"`` and ``0`` for ``"hh"``.
        enforce_connected: Repair disconnection with 2-switches and raise if the
            repair is impossible within budget.
        max_repair_swaps: 2-switch budget for the repair.
        max_hh_attempts: Downgrade attempts allowed in the graphicality loop.
        max_rewire_restarts: How many decorrelation passes may be discarded
            because they disconnected the graph.
        dataset: Dataset name, used only to name the derived streams (so two
            datasets built with the same seed get independent randomness).

    Returns:
        A :class:`ConstructionResult`.  Same ``(plan, seed, realize, flags)`` in
        → same ``graph_sha256`` out.

    Raises:
        ConstructionError: If the realised graph violates the plan (edge count,
            degree histogram, connectivity under ``enforce_connected``).
        InfeasibleDegreeSequence: If no downgrade of ``plan`` is realisable.
    """
    started = time.perf_counter()

    realized_plan, graphicality_log = realize_with_downgrade(plan, max_attempts=max_hh_attempts)
    edges = havel_hakimi(realized_plan.degree_sequence)
    if edges is None:  # pragma: no cover - realize_with_downgrade guarantees this
        raise ConstructionError("Havel–Hakimi 在已判定成功之后返回了 None")

    graph = nx.Graph()
    graph.add_nodes_from(range(realized_plan.n))
    graph.add_edges_from(edges)

    if graph.number_of_edges() != realized_plan.m:
        raise ConstructionError(
            f"实现后的边数 {graph.number_of_edges()} != 计划的 m={realized_plan.m}"
        )

    # ---- step 2: degree-preserving decorrelation -------------------------- #
    budget = _resolve_rewire_budget(realize, rewire_swaps, realized_plan.m)
    rewire_used = 0
    rewire_restarts = 0
    if budget > 0:
        rewire_rng = derive(seed, "topology", dataset, "rewire")
        graph, rewire_used, rewire_restarts = rewire_preserving_connectivity(
            graph,
            budget,
            rewire_rng,
            max_restarts=max_rewire_restarts,
            require_connected=enforce_connected,
        )

    # ---- step 3: connectivity, still degree-preserving -------------------- #
    repairs = 0
    if enforce_connected:
        repair_rng = derive(seed, "topology", dataset, "repair")
        graph, repairs = repair_connectivity(graph, repair_rng, max_swaps=max_repair_swaps)
    connected = nx.is_connected(graph) if graph.number_of_nodes() > 0 else False

    # ---- step 4: prove it ------------------------------------------------ #
    histogram = realized_histogram(graph)
    planned = realized_plan.degree_histogram()
    matches = histogram == planned
    if not matches:
        raise ConstructionError(
            f"实现后的度直方图与计划不一致：实测 {histogram}，计划 {planned}"
        )
    final_edge_count = graph.number_of_edges()
    if final_edge_count != realized_plan.m:
        raise ConstructionError(
            f"边数被改变：final_edge_count={final_edge_count} != 计划的 m={realized_plan.m}"
        )
    if enforce_connected and not connected:
        raise ConstructionError("enforce_connected=True 但修复后仍不连通")

    digest = edge_list_sha256(graph)
    fingerprint = graph_fingerprint(graph)
    elapsed = time.perf_counter() - started

    return ConstructionResult(
        graph=graph,
        plan=realized_plan,
        graphicality=graphicality_log,
        connected=connected,
        realize=realize,
        repairs=repairs,
        final_edge_count=final_edge_count,
        graph_sha256=digest,
        realized_degree_histogram=histogram,
        histogram_matches_plan=matches,
        elapsed_sec=elapsed,
        rewire_swaps_used=rewire_used,
        rewire_restarts=rewire_restarts,
        seed=int(seed),
        fingerprint=fingerprint,
    )


def build_from_sequence(sequence: Any, seed: int = 0, *,
                        enforce_connected: bool = True,
                        max_repair_swaps: int = 20000) -> ConstructionResult:
    """Realise an explicit degree sequence (bypassing the plan solver).

    Provided for tests and for the ``realize="hh"`` comparison path: it lets a
    caller attempt a sequence that
    :func:`~bit_hyrl.topology.degree_sequence.plan_degree_sequence` would never
    emit, e.g. a deliberately non-graphical one.

    No decorrelation is applied here — the whole point of this entry point is to
    hand back the *pure* realisation of the given sequence.

    Args:
        sequence: Degree per node id.
        seed: Root seed for the repair stream.
        enforce_connected: Same meaning as in :func:`build_bimodal_graph`.
        max_repair_swaps: Same meaning as in :func:`build_bimodal_graph`.

    Returns:
        A :class:`ConstructionResult` whose ``plan`` is a synthetic plan carrying
        the given sequence.

    Raises:
        DegreeSequenceError: If the sequence is not graphical.
    """
    from .degree_sequence import DegreePlan as _DegreePlan

    degrees = [int(value) for value in sequence]
    total = sum(degrees)
    if total % 2 != 0:
        raise DegreeSequenceError(f"Σk={total} 为奇数，不可能实现为简单图")
    edges = havel_hakimi(degrees)
    if edges is None:
        raise DegreeSequenceError("给定度序列不可图化")

    counts = Counter(degrees)
    ordered = sorted(counts, reverse=True)
    k_max = ordered[0]
    k_low = ordered[-1]
    synthetic = _DegreePlan(
        n=len(degrees),
        m=total // 2,
        c=0.0,
        h=sum(1 for value in degrees if value == k_max),
        k_bar=total / len(degrees) if degrees else 0.0,
        k_max=k_max,
        k_low=k_low,
        r=sum(1 for value in degrees if value == k_low + 1) if k_low + 1 in counts else 0,
        degree_sequence=tuple(sorted(degrees, reverse=True)),
        negative_degree_count=counts.get(k_low, 0),
        degenerate_fallback=False,
        sum_degree=total,
        parity_touched=False,
        capped_at_n_minus_1=False,
        degenerate_reason=None,
    )

    graph = nx.Graph()
    graph.add_nodes_from(range(len(degrees)))
    graph.add_edges_from(edges)
    repairs = 0
    if enforce_connected:
        rng = derive(seed, "topology", "", "repair")
        graph, repairs = repair_connectivity(graph, rng, max_swaps=max_repair_swaps)
    connected = nx.is_connected(graph)
    histogram = realized_histogram(graph)
    return ConstructionResult(
        graph=graph,
        plan=synthetic,
        graphicality=GraphicalityLog(),
        connected=connected,
        realize="hh",
        repairs=repairs,
        final_edge_count=graph.number_of_edges(),
        graph_sha256=edge_list_sha256(graph),
        realized_degree_histogram=histogram,
        histogram_matches_plan=histogram == synthetic.degree_histogram(),
        elapsed_sec=0.0,
        rewire_swaps_used=0,
        rewire_restarts=0,
        seed=int(seed),
        fingerprint=graph_fingerprint(graph),
    )
