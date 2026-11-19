"""Topology diagnostics: kappa, the bimodality fields and distances.

``kappa`` is the second-moment ratio ``<k^2> / <k>``, used as a robustness proxy
throughout the paper.
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

UNIFORM_BC_INTERNAL = 5.0 / 9.0


@dataclass(frozen=True)
class TopologyDiagnostics:
    """Everything the topology layer reports about one graph."""

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
        """Public ``dict`` view — **excludes** every ``bc_*`` field."""
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
    """Return ``⟨k²⟩ / ⟨k⟩`` for ``graph``."""
    degrees = [degree for _, degree in graph.degree()]
    if not degrees:
        return 0.0
    mean_degree = sum(degrees) / len(degrees)
    if mean_degree == 0.0:
        return 0.0
    mean_square = sum(degree * degree for degree in degrees) / len(degrees)
    return mean_square / mean_degree


def bc_internal(values: Sequence[float]) -> float:
    """Return the Sarle bimodality coefficient of ``values``."""
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
    """Return the explicit two-peak mass structure of a :class:`DegreePlan`."""
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
    """Recover the two-peak mass structure from a realised graph's degrees."""
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
    """Return ``(n - 1) / k_bar``, the ``c`` at which ``k_max`` saturates."""
    if k_bar <= 0.0:
        return 0.0
    return (n - 1) / k_bar


def degree_statistics(graph: nx.Graph) -> dict[str, Any]:
    """Return the degree-distribution summary used by diagnostics and sweeps."""
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
    """Return ``(diameter, average_path_length)`` in a single BFS sweep."""
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
    """Compute the full diagnostic record of ``graph``."""
    stats = degree_statistics(graph)
    n = stats["n"]
    exact_paths = n <= exact_paths_limit
    diameter, avg_path = path_statistics(graph) if exact_paths else (None, None)

    betweenness_bc: float | None = None
    if include_betweenness and n > 1:
        centrality = nx.betweenness_centrality(graph)
        betweenness_bc = bc_internal(list(centrality.values()))

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
    """Whitelist of the keys a figure or table may read from a diagnostics dict."""
    return (
        "n", "m", "k_bar", "k_max", "k_min", "n_distinct_degrees",
        "peak_structure", "kappa", "diameter", "avg_path_length",
        "degree_histogram", "is_connected", "degree_std", "c_star", "c_norm",
        "rewire_swaps_used", "rewire_restarts",
    )


def as_public(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a diagnostics payload onto :func:`public_diagnostic_keys`."""
    allowed = set(public_diagnostic_keys())
    projected = {key: payload[key] for key in allowed if key in payload}
    missing = allowed - set(projected)
    if missing:
        raise KeyError(f"diagnostic payload is missing public fields {sorted(missing)}")
    return projected
