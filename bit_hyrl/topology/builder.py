"""Bimodal graph construction: degree realisation, decorrelation and repair.

Havel-Hakimi realises the planned sequence, degree-preserving 2-switches
decorrelate it, and the result is checked against the plan before it is
returned.
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

_MAX_SWAP_TRIES = 64

DEFAULT_REWIRE_SWAPS_PER_EDGE = 10

#: Accepted values of the ``realize`` parameter.
REALIZE_MODES: tuple[str, ...] = ("hh", "hh+rewire")


class ConstructionError(RuntimeError):
    """Raised when a construction cannot satisfy its own invariants."""


@dataclass(frozen=True)
class ConstructionResult:
    """Outcome of one BiT construction."""

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


def _swap_is_legal(graph: nx.Graph, a: int, b: int, c: int, d: int,
                   new_pairs: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """Check that replacing ``a-b, c-d`` by ``new_pairs`` is a legal 2-switch."""
    removed = {(a, b) if a <= b else (b, a), (c, d) if c <= d else (d, c)}
    for u, v in new_pairs:
        if u == v:
            return False
        key = (u, v) if u <= v else (v, u)
        if key in removed:
            continue
        if graph.has_edge(u, v):
            return False
    return True


def repair_connectivity(graph: nx.Graph, rng: Any,
                        max_swaps: int = 20000) -> tuple[nx.Graph, int]:
    """Make ``graph`` connected using only degree-preserving 2-switches."""
    node_count = graph.number_of_nodes()
    if node_count <= 1:
        return graph, 0
    if min(degree for _, degree in graph.degree()) < 1:
        raise ConstructionError(
            "the graph has an isolated node: a degree-preserving 2-switch cannot attach it "
            "to the giant component, so this degree sequence cannot be connected"
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
                f"connectivity repair exhausted after {max_swaps} swaps, {len(components)} components left"
            )

        left, right = components[0], components[1]
        left_edges = sorted(tuple(sorted(edge)) for edge in graph.subgraph(left).edges())
        right_edges = sorted(tuple(sorted(edge)) for edge in graph.subgraph(right).edges())
        if not left_edges or not right_edges:
            raise ConstructionError(
                f"component {left[0]} or {right[0]} has no edge, a 2-switch cannot merge them"
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
                f"cannot merge components {left[0]} and {right[0]}: all {_MAX_SWAP_TRIES} attempts were rejected"
            )
        swaps += 1


def degree_preserving_swaps(graph: nx.Graph, target_swaps: int, rng: Any, *,
                            max_attempts_factor: int = 8) -> tuple[nx.Graph, int]:
    """Apply up to ``target_swaps`` random degree-preserving 2-switches."""
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
            continue

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
    """Decorrelation pass that refuses to hand back a disconnected graph."""
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


def _resolve_rewire_budget(realize: str, rewire_swaps: int | None, m: int) -> int:
    """Translate ``(realize, rewire_swaps)`` into an explicit switch budget."""
    if realize not in REALIZE_MODES:
        raise ConstructionError(
            f"realize={realize!r} is not supported, expected one of {list(REALIZE_MODES)}"
        )
    if rewire_swaps is not None:
        if isinstance(rewire_swaps, bool) or not isinstance(rewire_swaps, int):
            raise ConstructionError(f"rewire_swaps must be an integer, got {rewire_swaps!r}")
        if rewire_swaps < 0:
            raise ConstructionError(f"rewire_swaps cannot be negative, got {rewire_swaps}")
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
    """Realise ``plan`` as a graph, deterministically, and prove the invariants."""
    started = time.perf_counter()

    realized_plan, graphicality_log = realize_with_downgrade(plan, max_attempts=max_hh_attempts)
    edges = havel_hakimi(realized_plan.degree_sequence)
    if edges is None:  # pragma: no cover - realize_with_downgrade guarantees this
        raise ConstructionError("Havel-Hakimi returned None after being found graphical")

    graph = nx.Graph()
    graph.add_nodes_from(range(realized_plan.n))
    graph.add_edges_from(edges)

    if graph.number_of_edges() != realized_plan.m:
        raise ConstructionError(
            f"realised {graph.number_of_edges()} edges, the plan asks for m={realized_plan.m}"
        )

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

    repairs = 0
    if enforce_connected:
        repair_rng = derive(seed, "topology", dataset, "repair")
        graph, repairs = repair_connectivity(graph, repair_rng, max_swaps=max_repair_swaps)
    connected = nx.is_connected(graph) if graph.number_of_nodes() > 0 else False

    histogram = realized_histogram(graph)
    planned = realized_plan.degree_histogram()
    matches = histogram == planned
    if not matches:
        raise ConstructionError(
            f"realised degree histogram differs from the plan: got {histogram}, planned {planned}"
        )
    final_edge_count = graph.number_of_edges()
    if final_edge_count != realized_plan.m:
        raise ConstructionError(
            f"edge count changed: final_edge_count={final_edge_count} != planned m={realized_plan.m}"
        )
    if enforce_connected and not connected:
        raise ConstructionError("enforce_connected=True but the graph is still disconnected after repair")

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
    """Realise an explicit degree sequence (bypassing the plan solver)."""
    from .degree_sequence import DegreePlan as _DegreePlan

    degrees = [int(value) for value in sequence]
    total = sum(degrees)
    if total % 2 != 0:
        raise DegreeSequenceError(f"sum of degrees is {total}, odd, so no simple graph realises it")
    edges = havel_hakimi(degrees)
    if edges is None:
        raise DegreeSequenceError("the degree sequence is not graphical")

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
