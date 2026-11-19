"""2x2 ablation: topology (original/bit) x placement (degree_ranked/ppo) (T11).

The central question of the paper is whether the BiT topology change and the RL
placement change are **separable** — i.e. whether the interaction term is zero.
Each cell is the mean ``auc_r`` of the robustness curve over ``attack x seed``;
the three reported effects are

* **topology**   = mean(bit, .) - mean(original, .)
* **policy**     = mean(., ppo) - mean(., degree_ranked)
* **interaction**= (bit,ppo - bit,degree) - (original,ppo - original,degree)

computed per graph and then averaged across graphs.  The interaction is reported
*as measured* — the paper expects ~0, but this module never rounds it to zero.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..control.capacity import capacity_model
from ..control.feasibility import controllers_degree_ranked
from ..evaluation.runner import build_eval_graph, run_evaluation
from ..io.paths import RESULTS_ROOT, ensure_dir
from ..rl.transfer import load_policy

__all__ = ["TOPOLOGIES", "POLICIES", "EFFECT_KEYS", "run_ablation", "write_ablation_outputs"]

TOPOLOGIES: tuple[str, ...] = ("original", "bit")
POLICIES: tuple[str, ...] = ("degree_ranked", "ppo")
EFFECT_KEYS: tuple[str, ...] = ("topology", "policy", "interaction")


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean that tolerates an empty sequence (returns ``nan``)."""
    values = [float(v) for v in values if v == v]  # drop NaN
    return sum(values) / len(values) if values else float("nan")


def _cell_effects(cell: Mapping[tuple[str, str], float]) -> dict[str, float] | None:
    """Decompose a 4-cell 2x2 table into the three effects (None if incomplete)."""
    try:
        od = cell[("original", "degree_ranked")]
        op = cell[("original", "ppo")]
        bd = cell[("bit", "degree_ranked")]
        bp = cell[("bit", "ppo")]
    except KeyError:
        return None
    topology = (bd + bp) / 2.0 - (od + op) / 2.0
    policy = (op + bp) / 2.0 - (od + bd) / 2.0
    interaction = (bp - bd) - (op - od)
    return {"topology": topology, "policy": policy, "interaction": interaction}


def _resolve_policy(checkpoint, policy_selector):
    """Return a PPOPolicy from an explicit selector or a checkpoint path."""
    if policy_selector is not None:
        return policy_selector
    if checkpoint is None:
        raise ValueError("run_ablation 需要 checkpoint（PPO 检查点）或 policy_selector")
    return load_policy(checkpoint)


def run_ablation(
    specs: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str | Path | None = None,
    attacks: Sequence[str] = ("degree", "random"),
    mode: str = "structural",
    seeds: Sequence[int] = (0,),
    c: float = 1.6,
    h: int = 1,
    realize: str = "hh+rewire",
    policy_selector: Any = None,
) -> dict[str, Any]:
    """Evaluate the 2x2 ablation and return cells, effects and the long table.

    Args:
        specs: ``graphs.json``-style specs.
        checkpoint: PPO checkpoint path (used when ``policy_selector`` is None).
        attacks: Attack methods averaged over within a cell.
        mode: Cascade mode.
        seeds: Seeds; the BiT graph itself depends on the seed, so controllers are
            recomputed per seed.
        c: Hub factor.
        h: Number of hubs.
        realize: BiT realisation mode.
        policy_selector: Optional pre-loaded :class:`PPOPolicy` (tests / reuse).

    Returns:
        ``{"rows", "per_dataset_cells", "overall_cell_means", "effects",
        "per_dataset_effects"}``.
    """
    policy_obj = _resolve_policy(checkpoint, policy_selector)
    rows: list[dict[str, Any]] = []
    per_dataset_cells: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for spec in specs:
        dataset = str(spec.get("dataset") or spec.get("name") or "")
        for topology in TOPOLOGIES:
            for seed in seeds:
                graph = build_eval_graph(
                    spec, topology=topology, c=c, h=h, seed=seed, realize=realize
                )
                model = capacity_model(graph.number_of_nodes())
                n_c = model.n_controllers
                controllers = {
                    "degree_ranked": controllers_degree_ranked(graph, n_c),
                    "ppo": policy_obj.select(graph, n_c, greedy=True),
                }
                for policy in POLICIES:
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
                            controllers=controllers[policy],
                            policy=policy,
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "topology": topology,
                                "policy": policy,
                                "attack": attack,
                                "seed": int(seed),
                                "auc_r": float(record.auc_r),
                            }
                        )
                        per_dataset_cells[dataset][(topology, policy)].append(record.auc_r)

    per_dataset_means: dict[str, dict[tuple[str, str], float]] = {
        ds: {cell: _mean(values) for cell, values in cells.items()}
        for ds, cells in per_dataset_cells.items()
    }
    per_dataset_effects: dict[str, dict[str, float]] = {}
    for ds, cell in per_dataset_means.items():
        effects = _cell_effects(cell)
        if effects is not None:
            per_dataset_effects[ds] = effects

    overall_cell_means = {
        cell: _mean([means[cell] for means in per_dataset_means.values() if cell in means])
        for cell in ((t, p) for t in TOPOLOGIES for p in POLICIES)
    }
    overall_effects = {
        key: _mean([e[key] for e in per_dataset_effects.values() if key in e])
        for key in EFFECT_KEYS
    }

    return {
        "rows": rows,
        "per_dataset_cells": {
            ds: {f"{k[0]}|{k[1]}": v for k, v in cells.items()}
            for ds, cells in per_dataset_means.items()
        },
        "overall_cell_means": {f"{t}|{p}": v for (t, p), v in overall_cell_means.items()},
        "effects": overall_effects,
        "per_dataset_effects": per_dataset_effects,
    }


def write_ablation_outputs(result: Mapping[str, Any], out_dir: str | Path | None = None,
                           *, name: str = "ablation") -> dict[str, Path]:
    """Write the long table (CSV) and the summary (JSON); return their paths."""
    target = ensure_dir(Path(out_dir) if out_dir is not None else RESULTS_ROOT / "ablation")
    csv_path = target / f"{name}.csv"
    json_path = target / f"{name}_summary.json"

    rows = result.get("rows", [])
    fields = ["dataset", "topology", "policy", "attack", "seed", "auc_r"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    summary = {
        "overall_cell_means": result.get("overall_cell_means", {}),
        "effects": result.get("effects", {}),
        "per_dataset_effects": result.get("per_dataset_effects", {}),
        "per_dataset_cells": result.get("per_dataset_cells", {}),
        "n_rows": len(rows),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}
