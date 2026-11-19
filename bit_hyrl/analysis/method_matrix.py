"""Method-level experiment matrix (Task #4 glue).

Unlike :func:`bit_hyrl.analysis.matrix.run_matrix` (which only sweeps
``{original, bit} x {degree_ranked, ppo}``), this module evaluates the **paper's
method baselines**: each of the 5 topology baselines + Eff-R-Net (construct +
MADQN-sim placement), plus the two placement proxies on the original graph, plus
BiT-HyRL (BiT reconstruction + PPO placement).  The long-table output reuses the
matrix columns exactly, so ``stats --in`` consumes it unchanged.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..evaluation.runner import run_evaluation
from ..io.graph_io import load_graph_spec
from ..io.paths import RESULTS_ROOT, ensure_dir
from ..rl.transfer import load_policy
from .matrix import MATRIX_COLUMNS, _record_row
from .methods import METHODS, build_method_graph, method_controllers

__all__ = ["run_method_matrix", "write_method_matrix_outputs"]


def _topology_label(construct: str | None) -> str:
    """Record label: ``"bit"`` / ``"original"`` / the baseline construct name."""
    if construct == "bit":
        return "bit"
    if not construct:
        return "original"
    return construct


def _evaluate_one(
    spec: Mapping[str, Any],
    method_name: str,
    *,
    attacks: Sequence[str],
    mode: str,
    seeds: Sequence[int],
    checkpoint: str | Path | None,
    c: float,
    h: int,
) -> list[dict[str, Any]]:
    """Worker body for one ``(spec, method)``: construct -> place -> evaluate per attack.

    Module-level and pickle-friendly (``spec`` is a plain dict, ``checkpoint`` a
    path string); a worker loads the PPO policy from ``checkpoint`` itself so the
    main process need not.  Order of returned rows is ``seed`` then ``attack``,
    matching the serial loop exactly.
    """
    entry = METHODS.get(method_name)
    if entry is None:
        raise KeyError(f"未知方法 {method_name!r}；可用 {list(METHODS)}")
    construct = entry["construct"]
    place = entry["place"]
    topology_label = _topology_label(construct)
    dataset = str(spec.get("dataset", ""))
    policy_obj = load_policy(checkpoint) if place == "ppo" else None
    original, _meta = load_graph_spec(spec)

    rows: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            graph = build_method_graph(original, construct, seed=seed, c=c, h=h)
            controllers = method_controllers(
                graph, place, checkpoint=checkpoint, seed=seed, policy_obj=policy_obj
            )
            for attack in attacks:
                record = run_evaluation(
                    spec,
                    graph=graph,
                    controllers=controllers,
                    policy=method_name,
                    topology=topology_label,
                    attack=attack,
                    mode=mode,
                    seed=seed,
                    c=c,
                    h=h,
                )
                rows.append(_record_row(record))
    except Exception as exc:  # surface dataset + method for easy localisation
        raise RuntimeError(
            f"method-matrix worker 失败: dataset={dataset!r} method={method_name!r}: {exc}"
        ) from exc
    return rows


def _worker_entry(
    task: tuple[Mapping[str, Any], str],
    *,
    attacks: Sequence[str],
    mode: str,
    seeds: Sequence[int],
    checkpoint: str | Path | None,
    c: float,
    h: int,
) -> list[dict[str, Any]]:
    """Picklable ``(spec, method)`` task adapter for :meth:`ProcessPoolExecutor.map`."""
    spec, method_name = task
    return _evaluate_one(
        spec, method_name, attacks=attacks, mode=mode, seeds=seeds,
        checkpoint=checkpoint, c=c, h=h,
    )


def run_method_matrix(
    specs: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = tuple(METHODS),
    attacks: Sequence[str] = ("degree", "random"),
    mode: str = "structural",
    seeds: Sequence[int] = (0,),
    checkpoint: str | Path | None = None,
    c: float = 1.6,
    h: int = 1,
    workers: int = 1,
) -> pd.DataFrame:
    """Evaluate every ``(graph, method, attack, seed)`` and return the long table.

    ``workers <= 1`` runs the original serial double loop verbatim.  ``workers > 1``
    fans the ``(spec, method)`` tasks out to a process pool (``executor.map`` keeps
    order, so the flattened rows are identical to the serial run).
    """
    need_ppo = any(METHODS[name]["place"] == "ppo" for name in methods if name in METHODS)
    if need_ppo and checkpoint is None:
        raise ValueError("methods 含 ppo 放置（如 BiT-HyRL），需要 checkpoint")

    if workers <= 1:
        policy_obj = load_policy(checkpoint) if need_ppo else None
        rows: list[dict[str, Any]] = []
        for spec in specs:
            original, _meta = load_graph_spec(spec)
            dataset = str(spec.get("dataset", ""))
            for name in methods:
                entry = METHODS.get(name)
                if entry is None:
                    raise KeyError(f"未知方法 {name!r}；可用 {list(METHODS)}")
                construct = entry["construct"]
                place = entry["place"]
                topology_label = _topology_label(construct)
                for seed in seeds:
                    graph = build_method_graph(original, construct, seed=seed, c=c, h=h)
                    controllers = method_controllers(
                        graph, place, checkpoint=checkpoint, seed=seed, policy_obj=policy_obj
                    )
                    for attack in attacks:
                        record = run_evaluation(
                            spec,
                            graph=graph,
                            controllers=controllers,
                            policy=name,
                            topology=topology_label,
                            attack=attack,
                            mode=mode,
                            seed=seed,
                            c=c,
                            h=h,
                        )
                        rows.append(_record_row(record))
    else:
        tasks = [(spec, name) for spec in specs for name in methods]
        worker = partial(
            _worker_entry, attacks=attacks, mode=mode, seeds=seeds,
            checkpoint=checkpoint, c=c, h=h,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = list(executor.map(worker, tasks, chunksize=1))
        rows = [row for batch in batches for row in batch]

    frame = pd.DataFrame(rows)
    for column in MATRIX_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame[list(MATRIX_COLUMNS)]


def write_method_matrix_outputs(
    frame: pd.DataFrame, out_dir: str | Path | None = None, *, name: str = "matrix"
) -> dict[str, Path]:
    """Persist the method long table as CSV; return its path."""
    target = ensure_dir(Path(out_dir) if out_dir is not None else RESULTS_ROOT / "method_matrix")
    csv_path = target / f"{name}.csv"
    frame.to_csv(csv_path, index=False)
    return {"csv": csv_path}
