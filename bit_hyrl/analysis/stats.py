"""Paired statistics for the matrix table (T12).

For each requested pair of levels (e.g. ``bit`` vs ``original`` within the
``topology`` dimension) the module computes the **paired** difference on matched
rows (same ``dataset/attack/mode/seed`` and the *other* design factor), then
reports:

* the per-pair mean and standard deviation of the differences;
* an **exact Wilcoxon signed-rank** test for small samples (falls back to the
  normal approximation when exact is not defined / ties present);
* a **stratified paired bootstrap** 95% CI (1000 resamples, seed derived from
  :mod:`bit_hyrl.rng`).

Pairs are matched, never pooled, so a topology effect is not contaminated by the
placement factor (and vice-versa).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from .. import rng as _rng
from ..io.paths import RESULTS_ROOT, ensure_dir

__all__ = [
    "DEFAULT_PAIRS",
    "paired_stats",
    "write_stats_outputs",
    "holm_adjust",
    "seed_statistics",
    "significance",
    "method_significance",
    "write_metrics_outputs",
]

#: Default contrasts: topology then placement.
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("bit", "original"),
    ("ppo", "degree_ranked"),
)

_TOPOLOGY_LEVELS = {"original", "bit"}
_POLICY_LEVELS = {"degree_ranked", "ppo"}


def _pair_dimension(level_a: str) -> str:
    """Return which column the pair lives in (``topology`` or ``policy``)."""
    if level_a in _TOPOLOGY_LEVELS:
        return "topology"
    if level_a in _POLICY_LEVELS:
        return "policy"
    raise ValueError(f"无法判定配对维度：{level_a!r} 不在 topology/policy 的取值里")


def _paired_diffs(frame: pd.DataFrame, level_a: str, level_b: str, metric: str) -> list[float]:
    """Return the matched ``level_a - level_b`` differences along their dimension."""
    dim = _pair_dimension(level_a)
    others = [c for c in ("dataset", "topology", "policy", "attack", "mode", "seed")
              if c != dim and c in frame.columns]
    subset = frame[frame[dim].isin([level_a, level_b])]
    diffs: list[float] = []
    if subset.empty:
        return diffs
    for _key, group in subset.groupby(others, dropna=False):
        va = group.loc[group[dim] == level_a, metric]
        vb = group.loc[group[dim] == level_b, metric]
        if len(va) == 1 and len(vb) == 1:
            diffs.append(float(va.iloc[0]) - float(vb.iloc[0]))
    return diffs


def _wilcoxon(diffs: np.ndarray) -> tuple[float | None, float | None]:
    """Exact Wilcoxon signed-rank (small n) with a graceful approximation fallback."""
    if diffs.size == 0:
        return (None, None)
    if np.allclose(diffs, 0.0):
        return (0.0, 1.0)
    method = "exact" if diffs.size <= 25 else "approx"
    try:
        result = _scipy_stats.wilcoxon(diffs, method=method)
        return (float(result.statistic), float(result.pvalue))
    except Exception:
        try:
            result = _scipy_stats.wilcoxon(diffs)
            return (float(result.statistic), float(result.pvalue))
        except Exception:
            return (None, None)


def _bootstrap_ci(
    diffs: np.ndarray, *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI of the mean of ``diffs`` (paired, stratified by row)."""
    if diffs.size == 0:
        return (None, None)
    generator = _rng.derive(seed, "analysis", "bootstrap")
    n = diffs.size
    means = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = generator.integers(0, n, size=n)
        means[index] = float(np.mean(diffs[sample]))
    low = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (low, high)


def paired_stats(
    frame: pd.DataFrame,
    metric: str = "auc_r",
    pairs: Iterable[tuple[str, str]] = DEFAULT_PAIRS,
    *,
    seed: int = 0,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Compute paired effects for each contrast; return a JSON-ready dict.

    Args:
        frame: The long-table frame from :func:`bit_hyrl.analysis.matrix.run_matrix`.
        metric: Column to compare.
        pairs: ``(level_a, level_b)`` contrasts; the difference is ``a - b``.
        seed: Root seed for the bootstrap.
        n_boot: Bootstrap resamples.

    Returns:
        ``{"metric": ..., "pairs": { "<a>-<b>": {n, mean, std, wilcoxon_stat,
        wilcoxon_p, ci_low, ci_high} }}``.
    """
    if metric not in frame.columns:
        raise ValueError(f"metric {metric!r} 不在列 {list(frame.columns)} 中")
    report: dict[str, Any] = {"metric": metric, "pairs": {}}
    for level_a, level_b in pairs:
        diffs = np.asarray(_paired_diffs(frame, level_a, level_b, metric), dtype=float)
        key = f"{level_a}-{level_b}"
        if diffs.size == 0:
            report["pairs"][key] = {"n": 0}
            continue
        stat, pvalue = _wilcoxon(diffs)
        ci_low, ci_high = _bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
        report["pairs"][key] = {
            "n": int(diffs.size),
            "mean": float(diffs.mean()),
            "std": float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
            "wilcoxon_stat": stat,
            "wilcoxon_p": pvalue,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return report


def holm_adjust(pvalues: Sequence[float | None]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (``None`` treated as 1.0).

    Controls the family-wise error rate across the requested metric family (e.g.
    the three robustness metrics GCC / CSA / H_C, the paper's reporting口径).
    """
    pvals = [1.0 if p is None else float(p) for p in pvalues]
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[index]))
        adjusted[index] = running
    return adjusted


def seed_statistics(
    frame: pd.DataFrame,
    metrics: Sequence[str] = ("auc_gcc", "auc_csa", "auc_hc", "auc_r", "r_res"),
    *,
    group_keys: Sequence[str] = ("dataset", "topology", "policy", "attack", "mode"),
    seed: int = 0,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Aggregate each metric over seeds per ``(dataset, topology, policy, attack, mode)``.

    Returns a long ``DataFrame`` with ``group_keys + ["metric", "n", "mean", "std",
    "ci_low", "ci_high"]``; the 95% CI is a bootstrap of the mean over the seeds.
    """
    keys = [k for k in group_keys if k in frame.columns]
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=keys + ["metric", "n", "mean", "std", "ci_low", "ci_high"])
    for key, group in frame.groupby(keys, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(keys, key_tuple))
        for metric in metrics:
            if metric not in frame.columns:
                continue
            values = group[metric].dropna().to_numpy(dtype=float)
            n = int(values.size)
            if n == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            low, high = _bootstrap_ci(values, n_boot=n_boot, seed=seed)
            rows.append({**base, "metric": metric, "n": n, "mean": mean, "std": std,
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def significance(
    frame: pd.DataFrame,
    metrics: Sequence[str] = ("auc_gcc", "auc_csa", "auc_hc"),
    contrast: tuple[str, str] = ("bit", "original"),
    *,
    seed: int = 0,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Paired Wilcoxon vs a reference level + Holm correction across ``metrics``.

    Args:
        frame: Long-table frame.
        metrics: Metric family corrected together (paper口径: GCC / CSA / H_C).
        contrast: ``(level_a, level_b)``; the paired difference is ``a - b`` and the
            reference level is ``b`` (``original`` by default).
        seed / n_boot: Bootstrap CI controls.
        alpha: Significance threshold on the Holm-adjusted p.

    Returns:
        One dict per metric: ``{contrast, metric, n, mean_diff, wilcoxon_stat,
        wilcoxon_p, holm_p, ci_low, ci_high, significant}``.
    """
    level_a, level_b = contrast
    records = []
    pvalues: list[float | None] = []
    for metric in metrics:
        diffs = np.asarray(_paired_diffs(frame, level_a, level_b, metric), dtype=float)
        stat, pvalue = _wilcoxon(diffs)
        low, high = _bootstrap_ci(diffs, n_boot=n_boot, seed=seed) if diffs.size else (None, None)
        records.append({
            "contrast": f"{level_a}-{level_b}",
            "metric": metric,
            "n": int(diffs.size),
            "mean_diff": float(diffs.mean()) if diffs.size else None,
            "wilcoxon_stat": stat,
            "wilcoxon_p": pvalue,
            "ci_low": low,
            "ci_high": high,
        })
        pvalues.append(pvalue)
    adjusted = holm_adjust(pvalues)
    for record, holm_p in zip(records, adjusted):
        record["holm_p"] = holm_p
        record["significant"] = bool(holm_p < alpha)
    return records


def _wilcoxon_exact(diffs: np.ndarray) -> tuple[float | None, float | None]:
    """Exact Wilcoxon signed-rank (small n), normal-approximation fallback."""
    if diffs.size == 0:
        return (None, None)
    if np.allclose(diffs, 0.0):
        return (0.0, 1.0)
    method = "exact" if diffs.size <= 25 else "approx"
    try:
        result = _scipy_stats.wilcoxon(diffs, method=method)
        return (float(result.statistic), float(result.pvalue))
    except Exception:
        try:
            result = _scipy_stats.wilcoxon(diffs)
            return (float(result.statistic), float(result.pvalue))
        except Exception:
            return (None, None)


def _paired_method_diffs(
    frame: pd.DataFrame, method: str, reference: str, metric: str
) -> np.ndarray:
    """``reference - method`` paired on ``(dataset, attack, seed)`` (higher is better)."""
    keys = [k for k in ("dataset", "attack", "seed") if k in frame.columns]
    subset = frame[frame["policy"].isin([method, reference])]
    if subset.empty or metric not in frame.columns:
        return np.asarray([], dtype=float)
    ref = subset[subset["policy"] == reference][keys + [metric]]
    other = subset[subset["policy"] == method][keys + [metric]]
    merged = ref.merge(other, on=keys, suffixes=("_ref", "_m"))
    if merged.empty:
        return np.asarray([], dtype=float)
    return (merged[f"{metric}_ref"] - merged[f"{metric}_m"]).to_numpy(dtype=float)


def method_significance(
    frame: pd.DataFrame,
    *,
    reference: str = "BiT-HyRL",
    core: Sequence[str] = ("auc_gcc", "auc_csa", "auc_hc"),
) -> pd.DataFrame:
    """Method-vs-reference significance: exact Wilcoxon + Holm across the core metrics.

    For each non-reference method, pairs rows with ``reference`` on
    ``(dataset, attack, seed)`` (same graph, same attack, same seed) and tests the
    paired difference ``reference - method`` (higher is better) per core metric, with
    Holm correction over the three core metrics (GCC/CSA/H_C); ``r_res`` is reported
    but kept **out** of the Holm family (a summary column).  A ``__summary__`` row per
    method carries ``r_res`` mean diff / p and whether all three core metrics are
    significant after Holm.

    Returns:
        A ``DataFrame`` with columns ``method, metric, n_pairs, mean_diff,
        median_diff, wilcoxon_p, holm_p, direction`` (``metric="__summary__"`` rows
        summarise each method).
    """
    methods = sorted(m for m in frame["policy"].unique() if m != reference)
    rows: list[dict[str, Any]] = []
    for method in methods:
        pvalues: list[float | None] = []
        details: list[tuple[str, np.ndarray, float | None, float | None]] = []
        for metric in core:
            diffs = _paired_method_diffs(frame, method, reference, metric)
            stat, pvalue = _wilcoxon_exact(diffs)
            details.append((metric, diffs, stat, pvalue))
            pvalues.append(pvalue)
        adjusted = holm_adjust(pvalues)
        for (metric, diffs, stat, pvalue), holm_p in zip(details, adjusted):
            mean_diff = float(np.mean(diffs)) if diffs.size else float("nan")
            median_diff = float(np.median(diffs)) if diffs.size else float("nan")
            direction = "+" if diffs.size and mean_diff > 0 else ("-" if diffs.size and mean_diff < 0 else "0")
            rows.append({
                "method": method, "metric": metric, "n_pairs": int(diffs.size),
                "mean_diff": mean_diff, "median_diff": median_diff,
                "wilcoxon_p": pvalue, "holm_p": holm_p, "direction": direction,
            })
        diffs_r = _paired_method_diffs(frame, method, reference, "r_res")
        _stat_r, p_r = _wilcoxon_exact(diffs_r)
        mean_r = float(np.mean(diffs_r)) if diffs_r.size else float("nan")
        rows.append({
            "method": method, "metric": "r_res", "n_pairs": int(diffs_r.size),
            "mean_diff": mean_r, "median_diff": float(np.median(diffs_r)) if diffs_r.size else float("nan"),
            "wilcoxon_p": p_r, "holm_p": float("nan"),
            "direction": "+" if diffs_r.size and mean_r > 0 else ("-" if diffs_r.size and mean_r < 0 else "0"),
        })
        all_sig = all(hp is not None and hp < 0.05 for hp in adjusted)
        rows.append({
            "method": method, "metric": "__summary__", "n_pairs": len(core),
            "mean_diff": mean_r, "median_diff": float("nan"),
            "wilcoxon_p": p_r, "holm_p": float("nan"),
            "direction": f"core_all_sig={all_sig}",
        })
    return pd.DataFrame(
        rows,
        columns=["method", "metric", "n_pairs", "mean_diff", "median_diff",
                 "wilcoxon_p", "holm_p", "direction"],
    )


def write_metrics_outputs(
    seed_df: pd.DataFrame,
    sig_rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path | None = None,
    *,
    name: str = "stats",
) -> dict[str, Path]:
    """Write seed-aggregated stats + the significance table (CSV + JSON)."""
    target = ensure_dir(Path(out_dir) if out_dir is not None else RESULTS_ROOT / "stats")
    seed_path = target / f"{name}_seed.csv"
    sig_json = target / f"{name}_significance.json"
    sig_csv = target / f"{name}_significance.csv"
    seed_df.to_csv(seed_path, index=False)
    sig_json.write_text(json.dumps(list(sig_rows), indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["contrast", "metric", "n", "mean_diff", "wilcoxon_stat", "wilcoxon_p",
              "holm_p", "ci_low", "ci_high", "significant"]
    with sig_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sig_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return {"seed_csv": seed_path, "significance_json": sig_json, "significance_csv": sig_csv}


def write_stats_outputs(report: Mapping[str, Any], out_dir: str | Path | None = None,
                        *, name: str = "stats") -> dict[str, Path]:
    """Write the stats report as JSON and a flat CSV; return their paths."""
    target = ensure_dir(Path(out_dir) if out_dir is not None else RESULTS_ROOT / "stats")
    json_path = target / f"{name}.json"
    csv_path = target / f"{name}.csv"
    json_path.write_text(json.dumps(dict(report), indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for pair, values in report.get("pairs", {}).items():
        row = {"pair": pair, "metric": report.get("metric", "")}
        row.update(values)
        rows.append(row)
    fields = ["pair", "metric", "n", "mean", "std", "wilcoxon_stat", "wilcoxon_p",
              "ci_low", "ci_high"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return {"json": json_path, "csv": csv_path}
