"""The ``c`` sensitivity sweep.

``c`` is the single shape knob of the BiT construction, so the honest way to
report it is a full sweep: one row per grid point, carrying the solved
``(k_max, k_low, r)`` *and* the realised statistics.

Three boundaries show up in the sweep and all are reported explicitly rather
than silently absorbed:

* **``c`` is not comparable across graphs.**  ``k_max = round(c·k̄)`` saturates at
  ``N − 1``, so the crossover ``c_star = (N−1)/k̄`` ranges from ≈17 (BA-100) to
  ≈84 (BA-500).  Every row therefore carries ``c_star`` and ``c_norm = c/c_star``,
  and the sensitivity table is keyed on ``c_norm`` with raw ``c`` as a side
  column.  Holding ``c`` fixed across the nine graphs is *not* a controlled
  comparison.

  Note that ``c_norm >= 1`` and ``capped_at_n_minus_1`` are not literally the
  same predicate: rounding means the ceiling is reached once
  ``c·k̄ >= N − 1.5``, i.e. ``c_norm >= (N − 1.5)/(N − 1)``, which is
  ``1 − 0.5/(N−1)``.  The test in ``tests/test_sweep_c.py`` states that
  tolerance explicitly instead of pretending the boundary is exact.

* **``c`` above its effective upper bound.**  Once ``k_max`` reaches ``n - 1``
  the construction cannot grow any further, so every larger ``c`` yields the same
  graph.  Each row carries ``capped_at_n_minus_1`` (alias ``capped``).

* **``c`` below its effective lower bound.**  On low mean degree graphs a small
  ``c`` makes ``k_max <= k_low``, which means the two modes have merged.  The
  plan solver then rolls back to ``k_max = k_low + 1`` while still conserving
  ``Σk = 2m``, and the row records ``degenerate_fallback`` (alias
  ``degenerate``) plus a human-readable ``degenerate_reason``.  The fallback is
  *recorded*, never hidden: quietly changing ``c`` to something admissible would
  misreport what was run.

BC is not in this file's output
-------------------------------
The Sarle coefficient is **abandoned** (D8).  Rows never carry a ``bc_*`` key;
the internal values live under a single ``"_internal"`` sub-object that no table
or figure reads, and are only persisted to ``results/diagnostics/**``.
"""

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

#: The 12 grid points used in the paper.  Deliberately geometric-ish near 1 and
#: coarse far above it: the interesting behaviour is ``c`` slightly above 1.
#:
#: ⚠️ This grid is **not** a controlled comparison across graphs — see
#: :data:`C_NORM_GRID`, which is the grid the sensitivity table is actually keyed
#: on.  ``DEFAULT_C_GRID`` is retained because the design's measured table
#: is quoted on it and the numeric contract is pinned there.
DEFAULT_C_GRID: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3, 4, 6, 10, 20, 50)

#: The normalised grid: 11 points of ``c_norm = c/c_star``, spanning
#: "narrow band → star → past saturation".
#:
#: Sampling on ``c`` is wrong because ``k_max = round(c·k̄)`` saturates at
#: ``N − 1``, so the *same* ``c`` means a different shape on every graph: on
#: BA-500 (``c_star = 83.67``) the whole 12-point ``DEFAULT_C_GRID`` only reaches
#: ``c_norm = 0.598`` and never touches saturation, while BA-100
#: (``c_star = 17.01``) is already capped at ``c = 20``.  Sampling on ``c_norm``
#: puts all nine graphs at the **same position on one shape spectrum**, which is
#: what a controlled comparison requires.
#:
#: The last two points (``1.2``, ``1.5``) are deliberately past saturation: they
#: demonstrate the degeneracy of the ``c``-only view (identical degree sequences,
#: different raw ``c``).
C_NORM_GRID: tuple[float, ...] = (
    0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5,
)

#: Column order of the CSV export (and the key set of every row).
#: ``c_norm`` leads because it, not ``c``, is the comparable quantity.
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
        raise TypeError("sweep_c(n, m, ...) 需要同时给出 n 与 m（或直接传 nx.Graph）")
    return int(n_or_graph), int(m), dataset or ""


def sweep_row(plan: DegreePlan, result: ConstructionResult | None, *,
              dataset: str = "", include_betweenness: bool = False) -> dict[str, Any]:
    """Assemble one sweep row from a plan and (optionally) its realisation.

    Args:
        plan: Solved degree plan for this ``c``.
        result: Construction result, or ``None`` to report the plan alone
            (cheap mode, no graph is built).
        dataset: Dataset label written into the row.
        include_betweenness: Also compute the betweenness BC diagnostic.  The
            value is stored under ``row["_internal"]`` and never in a flat
            column, so no table can consume it by accident.

    Returns:
        A flat ``dict`` with the keys of :data:`SWEEP_FIELDS`, plus a single
        ``"_internal"`` sub-object (empty when no graph was built).
    """
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
    """Run the ``c`` sweep over one subject graph.

    Two calling conventions are supported so that the design document's
    ``sweep_c(n, m, ...)`` signature and the task's ``sweep_c(graph, grid)`` both
    work: pass ``(n, m)`` or pass an :class:`networkx.Graph`.

    Args:
        n_or_graph: Node count, or the graph itself.
        m: Edge count (ignored when a graph is passed).
        c_grid: Grid of hub factors, evaluated in the given order.
        h: Number of hubs.
        seed: Root seed for the derived decorrelation/repair streams.
        dataset: Dataset label used in the row and in the stream tags.
        build_graphs: If ``False``, only the plan is solved (no graph is built).
            This is the fast mode used to assert conservation across the grid.
        realize: Realisation mode, see :func:`~.builder.build_bimodal_graph`.
        rewire_swaps: Decorrelation budget; ``None`` means the mode default.
        enforce_connected: Passed through to the builder.
        max_repair_swaps: Passed through to the builder.
        max_hh_attempts: Passed through to the builder.
        max_rewire_restarts: Passed through to the builder.
        include_betweenness: Compute the betweenness BC diagnostic per row
            (stored under ``"_internal"`` only).

    Returns:
        One row per grid point, in grid order.
    """
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
    """Return ``(n - 1) / k̄`` with ``k̄ = 2m/n``, the saturation crossover.

    ``k_max = round_half_up(c·k̄)`` hits the ceiling ``n - 1`` once ``c`` reaches
    this value, so ``c_norm = c / c_star_for(n, m)`` is the comparable coordinate.

    Args:
        n: Node count.
        m: Edge count.

    Returns:
        The crossover ``c``, or ``0.0`` when ``m == 0``.
    """
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
    """Run the sweep on the **normalised** grid: ``c = c_norm · c_star``.

    This is the controlled comparison.  On ``c_norm`` all nine graphs sit at the
    same position on one shape spectrum, so a row at ``c_norm = 0.5`` means the
    same structural situation on BA-100 and on BA-500 — which is not true of a row
    at ``c = 50`` (capped on BA-100, ``c_norm = 0.598`` on BA-500).

    Because ``k_max = round_half_up(c·k̄)`` and ``c·k̄ = c_norm·(n−1)``, the grid
    is equivalent to "a hub degree that is a fixed fraction of ``n − 1``".  That
    identity is what makes the table readable: a column of ``k_max`` across the
    nine graphs is, up to rounding, the same fraction of ``n − 1``.

    Args:
        n_or_graph: Node count, or the graph itself.
        m: Edge count (ignored when a graph is passed).
        c_norm_grid: Grid of ``c_norm`` values.
        h: Number of hubs.
        seed: Root seed.
        dataset: Dataset label (also names the derived streams).
        build_graphs: Also construct and diagnose each graph.
        realize: Realisation mode.
        rewire_swaps: Decorrelation budget (``None`` = mode default).
        enforce_connected: Passed through to the builder.
        max_repair_swaps: Passed through to the builder.
        max_hh_attempts: Passed through to the builder.
        max_rewire_restarts: Passed through to the builder.
        include_betweenness: Compute the betweenness BC diagnostic per row.

    Returns:
        One row per ``c_norm`` grid point, in grid order.  Each row carries
        ``c_norm`` (the key), the implied ``c``, ``c_star`` and ``capped``.

    Raises:
        ValueError: If ``c_star`` is ``0`` (a graph with no edges has no shape
            spectrum to sweep).
    """
    n, m_resolved, name = _resolve(n_or_graph, m, dataset)
    star = c_star_for(n, m_resolved)
    if star <= 0.0:
        raise ValueError(f"{name or 'graph'} 没有边（m={m_resolved}），c_star=0，无法归一化扫描")

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
        # `c_norm` is the grid coordinate and `c` its side column; the rows carry
        # exactly :data:`SWEEP_FIELDS` so the CSV schema stays single-sourced.
        # The identity k_max = round_half_up(c_norm · (n−1)) is asserted in
        # tests/test_sweep_c_norm.py rather than duplicated into the row.
        rows.append(row)
    return rows


def sweep_dataset_c_norm(dataset: str, c_norm_grid: Sequence[float] = C_NORM_GRID,
                         h: int = 1, seed: int = 0, *, build_graphs: bool = True,
                         realize: str = "hh+rewire", rewire_swaps: int | None = None,
                         data_root: str | Path | None = None,
                         include_betweenness: bool = False) -> list[dict[str, Any]]:
    """Run the normalised sweep on a pinned dataset.

    Args:
        dataset: Dataset name, e.g. ``"Cogentco"``.
        c_norm_grid: ``c_norm`` grid.
        h: Number of hubs.
        seed: Root seed.
        build_graphs: Whether to construct and diagnose the graphs.
        realize: Realisation mode.
        rewire_swaps: Decorrelation budget (``None`` = mode default).
        data_root: Override of the default data directory (tests only).
        include_betweenness: Compute the betweenness BC diagnostic per row.

    Returns:
        One row per ``c_norm`` grid point.
    """
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
    """Run the ``c`` sweep on a pinned dataset in ``dataset/testdata``.

    Args:
        dataset: Dataset name, e.g. ``"Cogentco"``.
        c_grid: Hub-factor grid.
        h: Number of hubs.
        seed: Root seed.
        build_graphs: Whether to construct and diagnose the graphs.
        realize: Realisation mode.
        rewire_swaps: Decorrelation budget (``None`` = mode default).
        data_root: Override of the default data directory (tests only).
        include_betweenness: Compute the betweenness BC diagnostic per row.

    Returns:
        One row per grid point.
    """
    source = io_paths.dataset_path(dataset, data_root=data_root)
    graph = load_graph(source)
    return sweep_c(
        graph, None, c_grid, h=h, seed=seed, dataset=dataset, build_graphs=build_graphs,
        realize=realize, rewire_swaps=rewire_swaps, include_betweenness=include_betweenness,
    )


def write_sweep_outputs(rows: Iterable[dict[str, Any]], dataset: str, *,
                        out_dir: str | Path | None = None,
                        stem: str = "sweep_c") -> dict[str, Path]:
    """Write the sweep as canonical JSON and as CSV.

    The JSON keeps the nested structures (``degree_histogram``, ``peak_structure``)
    intact and keeps ``"_internal"``; the CSV flattens everything a *table* may
    read and drops the ``"_internal"`` sub-object entirely.

    Args:
        rows: Sweep rows (as produced by :func:`sweep_c` or :func:`sweep_c_norm`).
        dataset: Dataset label, used for the output directory.
        out_dir: Override of ``results/sweep/<dataset>``.
        stem: File stem.  ``c``-keyed and ``c_norm``-keyed sweeps must not
            overwrite each other, so :func:`sweep_c_norm` callers pass
            ``"sweep_c_norm"``.

    Returns:
        ``{"json": path, "csv": path}``.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    materialised = [dict(row) for row in rows]
    if not materialised:
        raise ValueError("sweep 结果为空，拒绝写文件")

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
