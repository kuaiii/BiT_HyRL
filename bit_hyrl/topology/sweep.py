"""Sensitivity sweep over the shape parameter ``c``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx

from ..io import paths as io_paths
from ..io.graph_io import load_graph
from .builder import ConstructionResult, build_bimodal_graph
from .degree_sequence import DegreePlan, plan_degree_sequence
from .diagnostics import diagnose

__all__ = [
    "C_NORM_GRID",
    "DEFAULT_C_GRID",
    "SWEEP_FIELDS",
    "sweep_row",
    "sweep_c",
    "sweep_c_norm",
    "sweep_dataset",
    "sweep_dataset_c_norm",
    "write_sweep_outputs",
]

DEFAULT_C_GRID: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3, 4, 6, 10, 20, 50)

C_NORM_GRID: tuple[float, ...] = (
    0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5,
)

SWEEP_FIELDS: tuple[str, ...] = (
    "dataset",
    "c_norm",
    "c",
    "c_star",
    "n",
    "m",
    "k_bar",
    "k_max",
    "k_low",
    "r",
    "negative_degree_count",
    "n_distinct_degrees",
    "peak_structure",
    "degenerate_fallback",
    "degenerate",
    "degenerate_reason",
    "capped_at_n_minus_1",
    "capped",
    "kappa",
    "diameter",
    "avg_path_length",
    "degree_std",
    "degree_histogram",
    "realize",
    "connected",
    "repairs",
    "rewire_swaps_used",
    "rewire_restarts",
    "final_edge_count",
    "graph_sha256",
    "density",
    "graphicality_attempts",
    "parity_touched",
    "sum_degree",
    "histogram_matches_plan",
    "fingerprint_sha256",
)


def _resolve(n_or_graph: Any, m: int | None, dataset: str | None) -> tuple[int, int, str]:
    """Accept either ``(n, m)`` or an ``nx.Graph`` as the sweep subject."""
    if isinstance(n_or_graph, nx.Graph):
        graph = n_or_graph
        name = dataset or str(graph.graph.get("_bit_hyrl_meta", {}).get("name", ""))
        return graph.number_of_nodes(), graph.number_of_edges(), name
    if m is None:
        raise TypeError("sweep_c(n, m, ...) needs both n and m, or an nx.Graph")
    return int(n_or_graph), int(m), dataset or ""


def sweep_row(plan: DegreePlan, result: ConstructionResult | None, *,
              dataset: str = "", include_betweenness: bool = False) -> dict[str, Any]:
    """Assemble one sweep row from a plan and (optionally) its realisation."""
    row: dict[str, Any] = {
        "dataset": dataset,
        "c_norm": plan.c_norm(),
        "c": float(plan.c),
        "c_star": plan.c_star(),
        "n": plan.n,
        "m": plan.m,
        "k_bar": plan.k_bar,
        "k_max": plan.k_max,
        "k_low": plan.k_low,
        "r": plan.r,
        "negative_degree_count": plan.negative_degree_count,
        "n_distinct_degrees": plan.n_distinct_degrees(),
        "peak_structure": _peak_structure_of(plan),
        "degenerate_fallback": plan.degenerate_fallback,
        "degenerate": plan.degenerate_fallback,
        "degenerate_reason": plan.degenerate_reason,
        "capped_at_n_minus_1": plan.capped_at_n_minus_1,
        "capped": plan.capped_at_n_minus_1,
        "kappa": None,
        "diameter": None,
        "avg_path_length": None,
        "degree_std": None,
        "degree_histogram": plan.degree_histogram(),
        "realize": None,
        "connected": None,
        "repairs": None,
        "rewire_swaps_used": None,
        "rewire_restarts": None,
        "final_edge_count": None,
        "graph_sha256": None,
        "density": None,
        "graphicality_attempts": None,
        "parity_touched": plan.parity_touched,
        "sum_degree": plan.sum_degree,
        "histogram_matches_plan": None,
        "fingerprint_sha256": None,
        "_internal": {},
    }
    if result is None:
        return row

    diagnostics = diagnose(
        result.graph,
        plan=plan,
        construction=result,
        include_betweenness=include_betweenness,
        c=plan.c,
    )
    n = result.graph.number_of_nodes()
    density = (
        2.0 * result.graph.number_of_edges() / (n * (n - 1)) if n > 1 else 0.0
    )
    row.update(
        {
            "kappa": diagnostics.kappa,
            "diameter": diagnostics.diameter,
            "avg_path_length": diagnostics.avg_path_length,
            "n_distinct_degrees": diagnostics.n_distinct_degrees,
            "peak_structure": dict(diagnostics.peak_structure),
            "degree_std": diagnostics.degree_std,
            "degree_histogram": diagnostics.degree_histogram,
            "realize": result.realize,
            "connected": result.connected,
            "repairs": result.repairs,
            "rewire_swaps_used": diagnostics.rewire_swaps_used,
            "rewire_restarts": diagnostics.rewire_restarts,
            "final_edge_count": result.final_edge_count,
            "graph_sha256": result.graph_sha256,
            "density": density,
            "graphicality_attempts": len(result.graphicality.attempts),
            "sum_degree": result.plan.sum_degree,
            "histogram_matches_plan": result.histogram_matches_plan,
            "fingerprint_sha256": result.fingerprint.get("sha256"),
            "_internal": diagnostics.to_internal_dict(),
        }
    )
    return row


def _peak_structure_of(plan: DegreePlan) -> dict[str, Any]:
    """Local copy of :func:`~.diagnostics.peak_structure` for the plan (no import cycle)."""
    return {
        "h": int(plan.h),
        "r": int(plan.r),
        "mass_low": int(plan.n - plan.h - plan.r),
        "ratio_kmax_klow": (
            float(plan.k_max) / float(plan.k_low) if plan.k_low > 0 else None
        ),
    }


def sweep_c(n_or_graph: Any, m: int | None = None,
            c_grid: Sequence[float] = DEFAULT_C_GRID, h: int = 1, seed: int = 0,
            dataset: str | None = None, *, build_graphs: bool = True,
            realize: str = "hh+rewire", rewire_swaps: int | None = None,
            enforce_connected: bool = True, max_repair_swaps: int = 20000,
            max_hh_attempts: int = 64, max_rewire_restarts: int = 8,
            include_betweenness: bool = False) -> list[dict[str, Any]]:
    """Run the ``c`` sweep over one subject graph."""
    n, m_resolved, name = _resolve(n_or_graph, m, dataset)
    rows: list[dict[str, Any]] = []
    for c in c_grid:
        plan = plan_degree_sequence(n, m_resolved, c=float(c), h=h)
        result = None
        if build_graphs:
            result = build_bimodal_graph(
                plan,
                seed,
                realize=realize,
                rewire_swaps=rewire_swaps,
                enforce_connected=enforce_connected,
                max_repair_swaps=max_repair_swaps,
                max_hh_attempts=max_hh_attempts,
                max_rewire_restarts=max_rewire_restarts,
                dataset=name,
            )
        rows.append(
            sweep_row(plan, result, dataset=name, include_betweenness=include_betweenness)
        )
    return rows


def c_star_for(n: int, m: int) -> float:
    """Return ``(n - 1) / k̄`` with ``k̄ = 2m/n``, the saturation crossover."""
    if n < 1 or m <= 0:
        return 0.0
    k_bar = 2.0 * m / n
    return (n - 1) / k_bar


def sweep_c_norm(n_or_graph: Any, m: int | None = None,
                 c_norm_grid: Sequence[float] = C_NORM_GRID, h: int = 1,
                 seed: int = 0, dataset: str | None = None, *,
                 build_graphs: bool = True, realize: str = "hh+rewire",
                 rewire_swaps: int | None = None, enforce_connected: bool = True,
                 max_repair_swaps: int = 20000, max_hh_attempts: int = 64,
                 max_rewire_restarts: int = 8,
                 include_betweenness: bool = False) -> list[dict[str, Any]]:
    """Run the sweep on the **normalised** grid: ``c = c_norm · c_star``."""
    n, m_resolved, name = _resolve(n_or_graph, m, dataset)
    star = c_star_for(n, m_resolved)
    if star <= 0.0:
        raise ValueError(f"{name or 'graph'} has no edges (m={m_resolved}), c_star=0, cannot normalise")

    rows: list[dict[str, Any]] = []
    for c_norm in c_norm_grid:
        c = float(c_norm) * star
        plan = plan_degree_sequence(n, m_resolved, c=c, h=h)
        result = None
        if build_graphs:
            result = build_bimodal_graph(
                plan,
                seed,
                realize=realize,
                rewire_swaps=rewire_swaps,
                enforce_connected=enforce_connected,
                max_repair_swaps=max_repair_swaps,
                max_hh_attempts=max_hh_attempts,
                max_rewire_restarts=max_rewire_restarts,
                dataset=name,
            )
        row = sweep_row(
            plan, result, dataset=name, include_betweenness=include_betweenness
        )
        rows.append(row)
    return rows


def sweep_dataset_c_norm(dataset: str, c_norm_grid: Sequence[float] = C_NORM_GRID,
                         h: int = 1, seed: int = 0, *, build_graphs: bool = True,
                         realize: str = "hh+rewire", rewire_swaps: int | None = None,
                         data_root: str | Path | None = None,
                         include_betweenness: bool = False) -> list[dict[str, Any]]:
    """Run the normalised sweep on a pinned dataset."""
    source = io_paths.dataset_path(dataset, data_root=data_root)
    graph = load_graph(source)
    return sweep_c_norm(
        graph, None, c_norm_grid, h=h, seed=seed, dataset=dataset,
        build_graphs=build_graphs, realize=realize, rewire_swaps=rewire_swaps,
        include_betweenness=include_betweenness,
    )


def sweep_dataset(dataset: str, c_grid: Sequence[float] = DEFAULT_C_GRID,
                  h: int = 1, seed: int = 0, *, build_graphs: bool = True,
                  realize: str = "hh+rewire", rewire_swaps: int | None = None,
                  data_root: str | Path | None = None,
                  include_betweenness: bool = False) -> list[dict[str, Any]]:
    """Run the ``c`` sweep on a pinned dataset in ``dataset/testdata``."""
    source = io_paths.dataset_path(dataset, data_root=data_root)
    graph = load_graph(source)
    return sweep_c(
        graph, None, c_grid, h=h, seed=seed, dataset=dataset, build_graphs=build_graphs,
        realize=realize, rewire_swaps=rewire_swaps, include_betweenness=include_betweenness,
    )


def write_sweep_outputs(rows: Iterable[dict[str, Any]], dataset: str, *,
                        out_dir: str | Path | None = None,
                        stem: str = "sweep_c") -> dict[str, Path]:
    """Write the sweep as canonical JSON and as CSV."""
    materialised = [dict(row) for row in rows]
    if not materialised:
        raise ValueError("empty sweep result, refusing to write")

    directory = Path(out_dir) if out_dir is not None else io_paths.sweep_dir(dataset)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / f"{stem}.json"
    io_paths.atomic_write_json(json_path, {"dataset": dataset, "rows": materialised})

    csv_path = directory / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SWEEP_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in materialised:
            flat = {
                key: ("" if value is None else value)
                for key, value in row.items()
                if key not in ("degree_histogram", "peak_structure")
            }
            flat["degree_histogram"] = ";".join(
                f"{degree}:{count}" for degree, count in sorted(row["degree_histogram"].items())
            )
            peaks = row.get("peak_structure") or {}
            flat["peak_structure"] = ";".join(
                f"{key}={peaks[key]}" for key in ("h", "r", "mass_low", "ratio_kmax_klow")
                if peaks.get(key) is not None
            )
            writer.writerow(flat)

    return {"json": json_path, "csv": csv_path}
