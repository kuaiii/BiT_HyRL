"""Experiment matrix runner (T12): the long table behind every figure/test.

:func:`run_matrix` sweeps ``dataset x topology x policy x attack x mode x seed``
by calling :func:`~bit_hyrl.evaluation.runner.run_evaluation` once per point and
collecting the scalar metrics into a ``pandas.DataFrame``.  The robustness curve
(``f_grid`` / ``r_curve``) is kept as JSON-encoded columns so a single CSV is a
complete, loss-free record of every run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..control.capacity import capacity_model
from ..control.feasibility import controllers_degree_ranked
from ..evaluation.runner import build_eval_graph, run_evaluation
from ..io.paths import RESULTS_ROOT, ensure_dir
from ..rl.transfer import load_policy

__all__ = ["MATRIX_COLUMNS", "run_matrix", "write_matrix_outputs"]

#: Scalar + JSON column order of the long table.
MATRIX_COLUMNS: tuple[str, ...] = (
    "dataset", "topology", "policy", "attack", "mode", "seed",
    "n_controllers", "auc_r",
    "auc_gcc", "auc_csa", "auc_hc", "r_res",
    "collapse_f", "collapse_f_gcc02", "r_res_half",
    "eval_seconds", "placement_seconds",
    "csa_final", "h_c_final", "h_c_norm_final", "audit_ok",
    "f_grid", "r_curve", "csa_curve", "hc_curve",
)


def _record_row(record) -> dict[str, Any]:
    """Flatten one :class:`EvaluationRecord` into a long-table row."""
    return {
        "dataset": record.dataset,
        "topology": record.topology,
        "policy": record.policy,
        "attack": record.attack,
        "mode": record.mode,
        "seed": int(record.seed),
        "n_controllers": int(record.n_controllers),
        "auc_r": float(record.auc_r),
        "auc_gcc": float(record.auc_gcc),
        "auc_csa": float(record.auc_csa),
        "auc_hc": float(record.auc_hc),
        "r_res": float(record.r_res),
        "collapse_f": float(record.collapse_f),
        "collapse_f_gcc02": float(record.collapse_f_gcc02),
        "r_res_half": float(record.r_res_half),
        "eval_seconds": float(record.eval_seconds),
        "placement_seconds": float(record.placement_seconds),
        "csa_final": float(record.csa_final),
        "h_c_final": float(record.h_c_final),
        "h_c_norm_final": float(record.h_c_norm_final),
        "audit_ok": bool(record.audit_ok),
        "f_grid": json.dumps(list(record.f_grid)),
        "r_curve": json.dumps(list(record.r_curve)),
        "csa_curve": json.dumps(list(record.csa_curve)),
        "hc_curve": json.dumps(list(record.hc_curve)),
    }


def run_matrix(
    specs: Sequence[Mapping[str, Any]],
    *,
    policies: Sequence[str] = ("degree_ranked", "ppo"),
    topologies: Sequence[str] = ("original", "bit"),
    attacks: Sequence[str] = ("degree", "random"),
    modes: Sequence[str] = ("structural",),
    seeds: Sequence[int] = (0,),
    checkpoint: str | Path | None = None,
    c: float = 1.6,
    h: int = 1,
    realize: str = "hh+rewire",
    policy_selector: Any = None,
) -> pd.DataFrame:
    """Run the full matrix and return a long-table ``DataFrame``."""
    need_ppo = "ppo" in policies
    policy_obj = None
    if need_ppo:
        if policy_selector is not None:
            policy_obj = policy_selector
        elif checkpoint is not None:
            policy_obj = load_policy(checkpoint)
        else:
            raise ValueError("run_matrix was asked for the ppo policy, which needs a checkpoint or a policy_selector")

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for topology in topologies:
            for seed in seeds:
                graph = build_eval_graph(
                    spec, topology=topology, c=c, h=h, seed=seed, realize=realize
                )
                model = capacity_model(graph.number_of_nodes())
                n_c = model.n_controllers

                controller_sets: dict[str, list[int]] = {
                    "degree_ranked": controllers_degree_ranked(graph, n_c),
                }
                placement_secs: dict[str, float] = {"degree_ranked": 0.0}
                if policy_obj is not None:
                    _t_select = time.perf_counter()
                    controller_sets["ppo"] = policy_obj.select(graph, n_c, greedy=True)
                    placement_secs["ppo"] = time.perf_counter() - _t_select

                for policy in policies:
                    controllers = controller_sets.get(policy)
                    for mode in modes:
                        for attack in attacks:
                            record = run_evaluation(
                                spec,
                                topology=topology,
                                c=c,
                                h=h,
                                seed=seed,
                                attack=attack,
                                mode=mode,
                                realize=realize,
                                controllers=controllers,
                                policy=policy,
                                placement_seconds=placement_secs.get(policy, 0.0),
                            )
                            rows.append(_record_row(record))

    frame = pd.DataFrame(rows)
    for column in MATRIX_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame[list(MATRIX_COLUMNS)]


def write_matrix_outputs(frame: pd.DataFrame, out_dir: str | Path | None = None,
                         *, name: str = "matrix") -> dict[str, Path]:
    """Persist the long table as CSV; return its path."""
    target = ensure_dir(Path(out_dir) if out_dir is not None else RESULTS_ROOT / "matrix")
    csv_path = target / f"{name}.csv"
    frame.to_csv(csv_path, index=False)
    return {"csv": csv_path}
