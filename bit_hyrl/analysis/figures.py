"""Figure generation from the matrix long table (T12).

Uses the non-interactive ``Agg`` backend so it runs head-less (no display).  Three
figures are produced from a single matrix CSV:

1. **robustness** — mean ``GCC/N`` vs ``f`` per dataset, ``original`` vs ``bit``;
2. **ablation** — the 2x2 (topology x placement) bar with the three effects;
3. **F10** — the conflict-#22 panel: 2 topologies x 3 metrics (GCC / CSA / H_C)
   over 2 representative datasets.

Every figure degrades gracefully: a missing dataset or column logs a warning and
skips that figure, but the others are still written.  Both ``.pdf`` and ``.png``
are saved.  These library functions never ``print`` (the CLI reports the files).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # head-less backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..io.paths import RESULTS_ROOT, ensure_dir  # noqa: E402

__all__ = ["make_all_figures", "figure_robustness", "figure_ablation", "figure_f10"]

LOGGER = logging.getLogger("bit_hyrl.analysis.figures")


def _as_list(value: Any) -> list[float]:
    """Coerce a matrix-cell (JSON string, list, or NaN) into a float list."""
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return []
        return [float(v) for v in parsed] if isinstance(parsed, list) else []
    return []


def _save(fig, out_dir: Path, name: str) -> dict[str, Any]:
    """Save ``fig`` as PDF + PNG, close it, return the artefact descriptor."""
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {
        "name": name,
        "pdf": str(pdf_path),
        "png": str(png_path),
        "size_png": int(png_path.stat().st_size),
    }


def _mean_curve(frame, topology: str, dataset: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Element-wise mean ``(f_grid, r_curve)`` for one ``(dataset, topology)``."""
    sub = frame[(frame["dataset"] == dataset) & (frame["topology"] == topology)]
    pairs = []
    for grid_raw, curve_raw in zip(sub["f_grid"], sub["r_curve"]):
        grid = _as_list(grid_raw)
        curve = _as_list(curve_raw)
        if grid and len(grid) == len(curve):
            pairs.append((grid, curve))
    if not pairs:
        return None
    lengths = {len(grid) for grid, _ in pairs}
    if len(lengths) == 1:
        grids = np.mean([grid for grid, _ in pairs], axis=0)
        curves = np.mean([curve for _, curve in pairs], axis=0)
        return grids, curves
    return pairs[0]  # mixed lengths (shouldn't happen within a dataset): use first


def figure_robustness(frame, out_dir: Path, *, max_datasets: int = 6) -> dict[str, Any] | None:
    """Figure 1: robustness curves, ``original`` vs ``bit``, one panel per dataset."""
    if "r_curve" not in frame.columns or "f_grid" not in frame.columns:
        return None
    datasets = sorted(str(d) for d in frame["dataset"].dropna().unique())[:max_datasets]
    if not datasets:
        return None
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.0), squeeze=False)
    drew = False
    for ax, dataset in zip(axes[0], datasets):
        for topology, color in (("original", "C3"), ("bit", "C0")):
            curve = _mean_curve(frame, topology, dataset)
            if curve is None:
                continue
            grids, curves = curve
            ax.plot(grids, curves, color=color, label=topology)
            drew = True
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("f (direct removal)")
        ax.set_ylabel("GCC / N")
        ax.legend(fontsize=7)
    if not drew:
        plt.close(fig)
        return None
    fig.suptitle("Robustness: original vs BiT reconstruction", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_robustness")


def _effects_from_cells(cells: dict[tuple[str, str], float]) -> dict[str, float] | None:
    """Three effects from a 2x2 cell dict (None if a cell is missing)."""
    try:
        od = cells[("original", "degree_ranked")]
        op = cells[("original", "ppo")]
        bd = cells[("bit", "degree_ranked")]
        bp = cells[("bit", "ppo")]
    except KeyError:
        return None
    return {
        "topology": (bd + bp) / 2.0 - (od + op) / 2.0,
        "policy": (op + bp) / 2.0 - (od + bd) / 2.0,
        "interaction": (bp - bd) - (op - od),
    }


def figure_ablation(frame, out_dir: Path) -> dict[str, Any] | None:
    """Figure 2: the 2x2 ablation bar with the three effects annotated."""
    if "auc_r" not in frame.columns:
        return None
    cells: dict[tuple[str, str], float] = {}
    for topology in ("original", "bit"):
        for policy in ("degree_ranked", "ppo"):
            sub = frame[(frame["topology"] == topology) & (frame["policy"] == policy)]
            if len(sub):
                cells[(topology, policy)] = float(sub["auc_r"].mean())
    if not cells:
        return None
    order = [("original", "degree_ranked"), ("original", "ppo"),
             ("bit", "degree_ranked"), ("bit", "ppo")]
    labels = [f"{t}\n{p}" for t, p in order if (t, p) in cells]
    values = [cells[(t, p)] for t, p in order if (t, p) in cells]
    colors = ["#c44", "#f90", "#4a7", "#08c"][: len(values)]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean auc_r")
    ax.set_title("2x2 ablation (topology x placement)", fontsize=10)
    effects = _effects_from_cells(cells)
    if effects is not None:
        text = (
            f"topology={effects['topology']:+.3f}\n"
            f"policy={effects['policy']:+.3f}\n"
            f"interaction={effects['interaction']:+.3f}"
        )
        ax.text(0.99, 0.99, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.tight_layout()
    return _save(fig, out_dir, "fig_ablation")


def figure_f10(frame, out_dir: Path) -> dict[str, Any] | None:
    """Figure 3 (conflict #22): 2 topologies x 3 metrics over 2 datasets."""
    required = {"auc_r", "csa_final", "h_c_norm_final", "r_curve", "f_grid"}
    if not required.issubset(set(frame.columns)):
        return None
    present = set(str(d) for d in frame["dataset"].dropna().unique())
    candidates = [d for d in ("Cogentco", "Colt") if d in present]
    if not candidates:
        candidates = sorted(present)[:2]
    if not candidates:
        return None
    fig, axes = plt.subplots(len(candidates), 3, figsize=(11, 3.0 * len(candidates)), squeeze=False)
    for row, dataset in enumerate(candidates):
        sub = frame[frame["dataset"] == dataset]
        for topology, color in (("original", "C3"), ("bit", "C0")):
            curve = _mean_curve(frame, topology, dataset)
            if curve is not None:
                grids, curves = curve
                axes[row, 0].plot(grids, curves, color=color, label=topology)
        axes[row, 0].set_title(f"{dataset}: GCC(f)", fontsize=9)
        axes[row, 0].set_xlabel("f")
        axes[row, 0].legend(fontsize=7)

        topo_labels = ["original", "bit"]
        csa = [float(sub[sub["topology"] == t]["csa_final"].mean())
               if len(sub[sub["topology"] == t]) else 0.0 for t in topo_labels]
        hc = [float(sub[sub["topology"] == t]["h_c_norm_final"].mean())
              if len(sub[sub["topology"] == t]) else 0.0 for t in topo_labels]
        axes[row, 1].bar(topo_labels, csa, color=["C3", "C0"])
        axes[row, 1].set_title(f"{dataset}: CSA", fontsize=9)
        axes[row, 2].bar(topo_labels, hc, color=["C3", "C0"])
        axes[row, 2].set_title(f"{dataset}: H_C norm", fontsize=9)
    fig.suptitle("F10: topology x metric panel (GCC / CSA / H_C)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_f10")


def make_all_figures(frame, out_dir: str | Path | None = None, *, smoke: bool = False) -> list[dict[str, Any]]:
    """Render every figure that the data supports; return artefact descriptors.

    Args:
        frame: The long-table ``DataFrame`` from ``run_matrix`` (or a matrix CSV).
        out_dir: Output directory; defaults to ``results/figures``.
        smoke: Present for symmetry with the other commands (no behaviour change).

    Returns:
        A list of ``{"name", "pdf", "png", "size_png"}`` descriptors (one per
        figure that was successfully written).
    """
    target = ensure_dir(Path(out_dir) if out_dir is not None else RESULTS_ROOT / "figures")
    if frame is None or len(frame) == 0:
        LOGGER.warning("matrix 为空，无法出图")
        return []
    figures: list[dict[str, Any]] = []
    for builder in (figure_robustness, figure_ablation, figure_f10):
        try:
            result = builder(frame, target)
        except Exception as exc:  # pragma: no cover - defensive degrade
            LOGGER.warning("图生成失败（%s），降级跳过：%s", builder.__name__, exc)
            result = None
        if result:
            figures.append(result)
    return figures
