"""Per-step robustness decomposition and the ``R_res`` integral."""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

from .metrics import calculate_csa, calculate_h_c_norm

__all__ = [
    "GCC_COLLAPSE_THRESHOLD",
    "HALF_WINDOW",
    "compute_robustness",
    "trapezoidal",
]

#: GCC fraction below which the graph counts as collapsed (fixed, H_C-independent).
GCC_COLLAPSE_THRESHOLD: float = 0.2

#: Upper bound of the half-window AUC (where the meaningful differences live).
HALF_WINDOW: float = 0.5


def trapezoidal(x: Sequence[float], y: Sequence[float]) -> float:
    """Composite trapezoidal area under ``(x, y)``; zero for < 2 points."""
    if len(x) < 2:
        return 0.0
    area = 0.0
    for index in range(1, len(x)):
        area += 0.5 * (y[index] + y[index - 1]) * (x[index] - x[index - 1])
    return area


def _half_area(f_grid: Sequence[float], curve: Sequence[float], upto: float) -> float:
    """``2 x`` trapezoidal area of the ``f <= upto`` segment (scaled to a unit window)."""
    xs = [f for f in f_grid if f <= upto]
    if len(xs) < 2:
        return 0.0
    ys = [curve[i] for i, f in enumerate(f_grid) if f <= upto]
    return 2.0 * trapezoidal(xs, ys)


def _collapse_f(f_grid: Sequence[float], gcc: Sequence[float], threshold: float) -> float:
    """First ``f`` where ``GCC < threshold``; ``1.0`` if it never does."""
    for f, value in zip(f_grid, gcc):
        if value < threshold:
            return float(f)
    return 1.0


def compute_robustness(
    graph: nx.Graph,
    trace,
    n_nodes: int,
    n_controllers_initial: int,
) -> dict:
    """Compute per-step GCC/CSA/H_C curves, three AUCs, ``r_res`` and sub-metrics."""
    f_grid = list(trace.f_grid)
    gcc_curve = list(trace.r_curve)

    csa_curve: list[float] = []
    hc_curve: list[float] = []
    for step in trace.steps:
        subgraph = graph.subgraph(set(step.alive_nodes))
        csa_curve.append(
            calculate_csa(subgraph, list(step.controllers_alive), initial_num_switches=n_nodes)
        )
        hc_curve.append(
            calculate_h_c_norm(step.loads, n_c0=n_controllers_initial)
        )

    hc_curve = list(np.minimum.accumulate(np.asarray(hc_curve, dtype=float)))

    auc_gcc = trapezoidal(f_grid, gcc_curve)
    auc_csa = trapezoidal(f_grid, csa_curve)
    auc_hc = trapezoidal(f_grid, hc_curve)
    r_res = (auc_gcc + auc_csa + auc_hc) / 3.0

    half_gcc = _half_area(f_grid, gcc_curve, HALF_WINDOW)
    half_csa = _half_area(f_grid, csa_curve, HALF_WINDOW)
    half_hc = _half_area(f_grid, hc_curve, HALF_WINDOW)
    r_res_half = (half_gcc + half_csa + half_hc) / 3.0

    return {
        "gcc_curve": gcc_curve,
        "csa_curve": csa_curve,
        "hc_curve": hc_curve,
        "auc_gcc": auc_gcc,
        "auc_csa": auc_csa,
        "auc_hc": auc_hc,
        "r_res": r_res,
        "collapse_f_gcc02": _collapse_f(f_grid, gcc_curve, GCC_COLLAPSE_THRESHOLD),
        "r_res_half": r_res_half,
        "auc_gcc_half": half_gcc,
        "auc_csa_half": half_csa,
        "auc_hc_half": half_hc,
    }
