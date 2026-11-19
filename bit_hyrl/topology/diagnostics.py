"""Topology diagnostics: κ, the three bimodality evidence fields, distances.

κ is the second-moment ratio ``⟨k²⟩ / ⟨k⟩`` — *not* ``⟨k²⟩ / ⟨k⟩²``.  Written as a ratio of raw moments it can exceed 1, and it is
exactly the numerator of the percolation threshold ``1 / (⟨k²⟩/⟨k⟩ − 1)``, which
is why it is the quantity the robustness argument needs.  Sanity check on BA-100
with the new rule: ``⟨k²⟩/⟨k⟩ = 34.14/5.82 = 5.866``, matching the 5.87 in the
numeric contract; the normalised variant gives 1.008, which matches nothing.
The code therefore never divides by ``mean(k) ** 2``.

Bimodality evidence — three hard fields, **no** BC
---------------------------------------------------
D8 is settled: the Sarle bimodality coefficient is **abandoned**, not merely
demoted.  It is computed here and nowhere else, under the name
:func:`bc_internal`, and never reaches a paper figure, a table, or a public
``EvaluationRecord`` field.  Reason (kept here for the record): the new
construction scores ``bc_internal ≈ 0.21`` on BA graphs while the *old, broken*
star construction scores ``1.0`` — so quoting BC as evidence of bimodality would
argue that the correct construction is the *less* bimodal one.

The three replacement fields are:

1. :attr:`TopologyDiagnostics.n_distinct_degrees` — the degree set has at most
   three values;
2. :attr:`TopologyDiagnostics.peak_structure` — the explicit two-peak mass
   structure ``(h, r, N − h − r)`` and the peak ratio ``k_max / k_low``;
3. :attr:`TopologyDiagnostics.kappa` — 5.87 (new) versus 20.91 (old).

``bc_internal`` is confined to ``results/diagnostics/**`` — not even the
sweep tables carry it.  The record builder uses :func:`as_public` (an explicit
whitelist) so the value is physically unreadable from a figure/table data
source, and ``tests/test_no_bc_in_output.py`` asserts that no public payload
has a key containing ``bc`` at any nesting depth.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import networkx as nx

__all__ = [
    "TopologyDiagnostics",
    "kappa",
    "bc_internal",
    "c_star",
    "peak_structure",
    "peak_structure_from_graph",
    "degree_statistics",
    "path_statistics",
    "diagnose",
    "public_diagnostic_keys",
    "as_public",
    "UNIFORM_BC_INTERNAL",
]

#: Analytic BC of a *continuous* uniform distribution: the conventional
#: ``BC > 5/9`` reference value.  Internal diagnostic only (see module docstring).
UNIFORM_BC_INTERNAL = 5.0 / 9.0


@dataclass(frozen=True)
class TopologyDiagnostics:
    """Everything the topology layer reports about one graph.

    Attributes:
        n: Node count.
        m: Edge count.
        k_bar: Mean degree ``2m/n``.
        k_max: Maximum degree.
        k_min: Minimum degree.
        n_distinct_degrees: Bimodality evidence ① — number of distinct degrees.
        peak_structure: Bimodality evidence ② — ``{"h", "r", "mass_low",
            "ratio_kmax_klow"}``.
        kappa: Bimodality evidence ③ — ``⟨k²⟩/⟨k⟩`` (see module docstring).
        diameter: Graph diameter, or ``None`` when disconnected.
        avg_path_length: Mean shortest-path length, or ``None`` when disconnected.
        degree_histogram: ``{degree: count}``.
        is_connected: Connectivity flag.
        bc_internal: Sarle bimodality coefficient.  **Internal diagnostic only**
            (D8): never published, never in a figure or table.
        bc_internal_betweenness: Same statistic over the betweenness
            distribution, or ``None`` when not requested.
        degree_std: Population standard deviation of the degree sequence
            (additive audit field, not used in the paper).
        c_star: ``(n - 1) / k_bar`` — the ``c`` at which ``k_max`` saturates at
            ``n - 1``.  Spans 17…84 over the nine evaluation graphs, which is why
            raw ``c`` is not comparable across graphs.
        c_norm: ``c / c_star`` for the ``c`` the graph was built at, or ``None``
            when the graph's provenance is unknown.
        rewire_swaps_used: Decorrelation 2-switches spent on this graph, or
            ``None`` when the caller did not supply the
            :class:`~.builder.ConstructionResult`.  **Copied**, never
            recomputed — ``ConstructionResult`` stays the single source of
            truth; this field exists so an audit record carries the provenance
            of the edge arrangement it describes.
        rewire_restarts: How many decorrelation passes were discarded because
            they disconnected the graph, or ``None`` (same provenance rule).
    """

    n: int
    m: int
    k_bar: float
    k_max: int
    k_min: int
    n_distinct_degrees: int
    peak_structure: dict[str, Any]
    kappa: float
    diameter: int | None
    avg_path_length: float | None
    degree_histogram: dict[int, int]
    is_connected: bool
    bc_internal: float
    bc_internal_betweenness: float | None = None
    degree_std: float = 0.0
    c_star: float = 0.0
    c_norm: float | None = None
    rewire_swaps_used: int | None = None
    rewire_restarts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Public ``dict`` view — **excludes** every ``bc_*`` field.

        D8 requires BC to be unreadable from anything a figure or table consumes,
        so the public projection drops it and
        :meth:`to_internal_dict` is the only way to obtain it.
        """
        return {
            "n": self.n,
            "m": self.m,
            "k_bar": self.k_bar,
            "k_max": self.k_max,
            "k_min": self.k_min,
            "n_distinct_degrees": self.n_distinct_degrees,
            "peak_structure": dict(self.peak_structure),
            "kappa": self.kappa,
            "diameter": self.diameter,
            "avg_path_length": self.avg_path_length,
            "degree_histogram": {str(k): v for k, v in sorted(self.degree_histogram.items())},
            "is_connected": self.is_connected,
            "degree_std": self.degree_std,
            "c_star": self.c_star,
            "c_norm": self.c_norm,
            "rewire_swaps_used": self.rewire_swaps_used,
            "rewire_restarts": self.rewire_restarts,
        }

    def to_internal_dict(self) -> dict[str, Any]:
        """``{"bc_internal", "bc_internal_betweenness"}`` — internal artefacts only."""
        return {
            "bc_internal": self.bc_internal,
            "bc_internal_betweenness": self.bc_internal_betweenness,
        }


def kappa(graph: nx.Graph) -> float:
    """Return ``⟨k²⟩ / ⟨k⟩`` for ``graph``.

    Physical meaning (to be quoted verbatim in the paper): κ is the numerator of
    the percolation threshold ``1/(κ−1)`` — the closer to 1, the closer to a
    regular graph (homogeneous, robust but inefficient); the larger, the more
    concentrated the connectivity in a few hubs.

    Args:
        graph: Any undirected graph.

    Returns:
        The second-moment ratio, or ``0.0`` when the graph has no edges.
    """
    degrees = [degree for _, degree in graph.degree()]
    if not degrees:
        return 0.0
    mean_degree = sum(degrees) / len(degrees)
    if mean_degree == 0.0:
        return 0.0
    mean_square = sum(degree * degree for degree in degrees) / len(degrees)
    return mean_square / mean_degree


def bc_internal(values: Sequence[float]) -> float:
    """Return the Sarle bimodality coefficient of ``values``.

    **Internal diagnostic only** (D8) — see the module docstring for why this
    number must never appear in the paper.

    ``BC = (skew² + 1) / (excess_kurtosis + 3)`` with population moments
    (``ddof = 0``).  Anchors asserted by the test-suite: continuous uniform →
    ``5/9``, symmetric two-point → ``1.0``, constant → ``0.0``.

    Args:
        values: Sample.

    Returns:
        The coefficient, or ``0.0`` for the documented degenerate cases
        (``len < 4``, or zero variance).
    """
    data = [float(value) for value in values]
    count = len(data)
    if count < 4:
        return 0.0
    mean = sum(data) / count
    deviations = [value - mean for value in data]
    second = sum(delta * delta for delta in deviations) / count
    if second <= 0.0:
        return 0.0
    third = sum(delta**3 for delta in deviations) / count
    fourth = sum(delta**4 for delta in deviations) / count
    skew = third / second**1.5
    excess_kurtosis = fourth / (second * second) - 3.0
    denominator = excess_kurtosis + 3.0
    if abs(denominator) < 1e-12:
        return 0.0
    return (skew * skew + 1.0) / denominator


def peak_structure(plan: Any) -> dict[str, Any]:
    """Return the explicit two-peak mass structure of a :class:`DegreePlan`.

    This is bimodality evidence ②: it states *which* nodes carry which peak
    instead of reducing the shape to a single number.

    Args:
        plan: A ``DegreePlan`` (duck-typed: needs ``h``, ``r``, ``n``, ``k_max``,
            ``k_low``).

    Returns:
        ``{"h": 1, "r": 78, "mass_low": 21, "ratio_kmax_klow": 1.8}`` for
        BA-100 at ``c = 1.6``.
    """
    h = int(getattr(plan, "h"))
    r = int(getattr(plan, "r"))
    n = int(getattr(plan, "n"))
    k_max = int(getattr(plan, "k_max"))
    k_low = int(getattr(plan, "k_low"))
    return {
        "h": h,
        "r": r,
        "mass_low": n - h - r,
        "ratio_kmax_klow": (float(k_max) / float(k_low)) if k_low > 0 else None,
    }


def peak_structure_from_graph(graph: nx.Graph) -> dict[str, Any]:
    """Recover the two-peak mass structure from a realised graph's degrees.

    Used when only the graph is available (e.g. ``diagnose`` on a loaded network).
    For a BiT construction this reproduces :func:`peak_structure` exactly, because
    the construction assigns degrees explicitly by node id.

    Args:
        graph: Any undirected graph.

    Returns:
        ``{"h", "r", "mass_low", "ratio_kmax_klow"}``.  ``r`` counts nodes at
        ``k_min + 1``; it is ``0`` when the graph has no such intermediate layer.
    """
    histogram = Counter(degree for _, degree in graph.degree())
    if not histogram:
        return {"h": 0, "r": 0, "mass_low": 0, "ratio_kmax_klow": None}
    k_max = max(histogram)
    k_low = min(histogram)
    return {
        "h": int(histogram[k_max]),
        "r": int(histogram.get(k_low + 1, 0)),
        "mass_low": int(histogram[k_low]),
        "ratio_kmax_klow": (float(k_max) / float(k_low)) if k_low > 0 else None,
    }


def c_star(n: int, k_bar: float) -> float:
    """Return ``(n - 1) / k_bar``, the ``c`` at which ``k_max`` saturates.

    Args:
        n: Node count.
        k_bar: Mean degree ``2m/n``.

    Returns:
        The crossover factor, or ``0.0`` when ``k_bar`` is non-positive.
    """
    if k_bar <= 0.0:
        return 0.0
    return (n - 1) / k_bar


def degree_statistics(graph: nx.Graph) -> dict[str, Any]:
    """Return the degree-distribution summary used by diagnostics and sweeps.

    Args:
        graph: Any undirected graph.

    Returns:
        A dict with ``n``, ``m``, ``k_bar``, ``k_max``, ``k_min``, ``kappa``,
        ``bc_internal``, ``degree_histogram``, ``n_distinct_degrees``,
        ``peak_structure``, ``degree_std`` and ``c_star``.
    """
    degrees = [degree for _, degree in graph.degree()]
    histogram = dict(sorted(Counter(degrees).items()))
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    mean = (sum(degrees) / len(degrees)) if degrees else 0.0
    variance = (
        sum((degree - mean) ** 2 for degree in degrees) / len(degrees) if degrees else 0.0
    )
    return {
        "n": n,
        "m": m,
        "k_bar": mean,
        "k_max": max(degrees) if degrees else 0,
        "k_min": min(degrees) if degrees else 0,
        "kappa": kappa(graph),
        "bc_internal": bc_internal(degrees),
        "degree_histogram": histogram,
        "n_distinct_degrees": len(histogram),
        "peak_structure": peak_structure_from_graph(graph),
        "degree_std": math.sqrt(variance),
        "c_star": c_star(n, mean),
    }


def path_statistics(graph: nx.Graph) -> tuple[int | None, float | None]:
    """Return ``(diameter, average_path_length)`` in a single BFS sweep.

    Computing both from one ``all_pairs_shortest_path_length`` pass is cheaper
    than two separate networkx calls, which matters across the 108-construction
    ``c``-sweep.

    Args:
        graph: Any undirected graph.

    Returns:
        ``(diameter, avg_path_length)``, or ``(None, None)`` when the graph is
        empty or disconnected.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return None, None
    if n == 1:
        return 0, 0.0
    if not nx.is_connected(graph):
        return None, None

    diameter = 0
    total = 0
    for _, distances in nx.all_pairs_shortest_path_length(graph):
        if len(distances) != n:  # pragma: no cover - defensive
            return None, None
        values = distances.values()
        local_max = max(values)
        if local_max > diameter:
            diameter = local_max
        total += sum(values)
    return diameter, total / (n * (n - 1))


def diagnose(graph: nx.Graph, *, plan: Any = None, construction: Any = None,
             exact_paths_limit: int = 2000,
             include_betweenness: bool = True,
             c: float | None = None) -> TopologyDiagnostics:
    """Compute the full diagnostic record of ``graph``.

    Args:
        graph: Any undirected graph.
        plan: The :class:`~.degree_sequence.DegreePlan` the graph was built
            from, if known.  Supplying it makes ``peak_structure`` the *planned*
            two-peak mass structure (the explicit per-node assignment) instead of
            one recovered from the realised degree multiset — for a BiT
            construction the two agree exactly, but the planned form also reports
            the node masses even when a degree layer happens to be empty.  It
            also supplies ``c`` when that argument is left as ``None``.
        construction: The :class:`~.builder.ConstructionResult` the graph came
            from, if known.  Only its provenance counters are **copied** into
            ``rewire_swaps_used`` / ``rewire_restarts``; nothing is recomputed,
            so ``ConstructionResult`` remains the single source of truth.
        exact_paths_limit: Above this node count, exact all-pairs distances are
            skipped and ``diameter`` / ``avg_path_length`` become ``None``.  All
            nine evaluation graphs (N <= 500) sit well inside the exact regime.
        include_betweenness: Also compute ``bc_internal_betweenness``.  This costs
            one Brandes pass (0.74 s on the 500-node BA-500), so callers that only
            need the public fields can switch it off.
        c: The hub factor the graph was constructed at, if known.  Supplying it
            fills ``c_norm``; ``c_star`` is always computed from ``(n, m)``.

    Returns:
        A :class:`TopologyDiagnostics`.  ``plan`` and ``construction`` are
        optional, so ``diagnose(graph)`` remains valid for an arbitrary loaded
        network.
    """
    stats = degree_statistics(graph)
    n = stats["n"]
    exact_paths = n <= exact_paths_limit
    diameter, avg_path = path_statistics(graph) if exact_paths else (None, None)

    betweenness_bc: float | None = None
    if include_betweenness and n > 1:
        centrality = nx.betweenness_centrality(graph)
        betweenness_bc = bc_internal(list(centrality.values()))

    # The planned structure wins when available: it is the explicit assignment,
    # not a recovery from the degree multiset.
    structure = peak_structure(plan) if plan is not None else stats["peak_structure"]

    effective_c = c
    if effective_c is None and plan is not None:
        planned_c = float(getattr(plan, "c", 0.0) or 0.0)
        effective_c = planned_c if planned_c > 0.0 else None

    star = stats["c_star"]
    return TopologyDiagnostics(
        n=n,
        m=stats["m"],
        k_bar=stats["k_bar"],
        k_max=stats["k_max"],
        k_min=stats["k_min"],
        n_distinct_degrees=stats["n_distinct_degrees"],
        peak_structure=structure,
        kappa=stats["kappa"],
        diameter=diameter,
        avg_path_length=avg_path,
        degree_histogram=stats["degree_histogram"],
        is_connected=nx.is_connected(graph) if n > 0 else False,
        bc_internal=stats["bc_internal"],
        bc_internal_betweenness=betweenness_bc,
        degree_std=stats["degree_std"],
        c_star=star,
        c_norm=(float(effective_c) / star)
        if (effective_c is not None and star > 0.0)
        else None,
        rewire_swaps_used=(
            int(construction.rewire_swaps_used) if construction is not None else None
        ),
        rewire_restarts=(
            int(construction.rewire_restarts) if construction is not None else None
        ),
    )


def public_diagnostic_keys() -> tuple[str, ...]:
    """Whitelist of the keys a figure or table may read from a diagnostics dict.

    D8 enforcement aid: ``figures.py`` / ``tables.py`` must consult this and
    refuse anything else (in particular anything starting with ``bc`` or named
    ``_internal``).
    """
    return (
        "n", "m", "k_bar", "k_max", "k_min", "n_distinct_degrees",
        "peak_structure", "kappa", "diameter", "avg_path_length",
        "degree_histogram", "is_connected", "degree_std", "c_star", "c_norm",
        "rewire_swaps_used", "rewire_restarts",
    )


def as_public(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a diagnostics payload onto :func:`public_diagnostic_keys`.

    Args:
        payload: A ``TopologyDiagnostics.to_dict()`` output or a superset.

    Returns:
        The whitelisted subset, dropping every ``bc_*`` / ``_internal`` entry.

    Raises:
        KeyError: If a whitelisted key is missing, so a broken projection fails
            loudly instead of silently producing a partial table.
    """
    allowed = set(public_diagnostic_keys())
    projected = {key: payload[key] for key in allowed if key in payload}
    missing = allowed - set(projected)
    if missing:
        raise KeyError(f"诊断载荷缺少公开字段 {sorted(missing)}")
    return projected
