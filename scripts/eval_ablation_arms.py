"""Fill the missing cells of the 2x2 topology x placement ablation."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bit_hyrl.analysis.matrix import MATRIX_COLUMNS, _record_row  # noqa: E402
from bit_hyrl.analysis.methods import build_method_graph  # noqa: E402
from bit_hyrl.topology.variants import configuration_model_like  # noqa: E402
from bit_hyrl.evaluation.runner import run_evaluation  # noqa: E402
from bit_hyrl.io.graph_io import load_graph_spec  # noqa: E402
from bit_hyrl.rl.transfer import load_policy  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "baselines" / "ablation"
LABELS = {"original": "BiT-norecon", "degree_only": "BiT-degOnly", "bit_c10": "BiT-c1.0"}


def _variant_graph(original: nx.Graph, variant: str) -> nx.Graph:
    if variant == "original":
        return original
    if variant == "degree_only":
        return configuration_model_like(original, seed=0)
    if variant == "bit_c10":
        return build_method_graph(original, "bit", seed=0, c=1.0, h=1)
    raise KeyError(variant)


def _run_one(task: tuple[dict[str, Any], str, tuple[str, ...], tuple[int, ...], str]) -> dict[str, Any]:
    spec, variant, attacks, seeds, checkpoint = task
    dataset = spec["dataset"]
    original, _meta = load_graph_spec(spec)
    graph = _variant_graph(original, variant)
    n = graph.number_of_nodes()
    n_c = max(1, round(0.1 * n))
    policy_obj = load_policy(checkpoint)
    controllers = [int(v) for v in policy_obj.select(graph, n_c, greedy=True)]
    label = LABELS[variant]

    rows: list[dict[str, Any]] = []
    for attack in attacks:
        for seed in seeds:
            record = run_evaluation(
                spec, graph=graph, controllers=controllers, policy=label,
                topology="original" if variant == "original" else variant,
                attack=attack, mode="structural", seed=seed, c=1.0 if variant == "bit_c10" else 1.6,
                h=1,
            )
            rows.append(_record_row(record))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = list(MATRIX_COLUMNS)
    target = OUT_DIR / f"{variant}__{dataset}.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    print(f"[done] {variant} {dataset} -> {target.name} ({len(rows)} rows)", flush=True)
    return {"variant": variant, "dataset": dataset, "rows": rows}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fill the missing cells of the 2x2 ablation")
    parser.add_argument("--variant", default="original",
                        choices=("original", "degree_only", "bit_c10", "all"))
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--attacks", default="degree,random")
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--checkpoint", default="results/checkpoints/ppo_final_s0.pt")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    specs = json.loads((REPO_ROOT / "configs" / "graphs.json").read_text(encoding="utf-8"))["graphs"]
    if args.dataset.strip().lower() != "all":
        wanted = {token.strip() for token in args.dataset.split(",") if token.strip()}
        specs = [spec for spec in specs if spec["dataset"] in wanted]

    variants = ("original", "degree_only", "bit_c10") if args.variant == "all" else (args.variant,)
    attacks = tuple(token.strip() for token in args.attacks.split(",") if token.strip())
    if "-" in args.seeds:
        low, high = args.seeds.split("-")
        seeds = tuple(range(int(low), int(high) + 1))
    else:
        seeds = tuple(int(token) for token in args.seeds.split(",") if token.strip())

    tasks = [(spec, variant, attacks, seeds, args.checkpoint)
             for variant in variants for spec in specs]
    print(f"variants={variants} graphs={len(specs)} tasks={len(tasks)} workers={args.workers}")

    collected: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(_run_one, tasks, chunksize=1):
                collected.extend(result["rows"])
    else:
        for task in tasks:
            collected.extend(_run_one(task)["rows"])

    import pandas as pd
    cells = sorted(OUT_DIR.glob("*.csv"))
    if cells:
        merged = pd.concat([pd.read_csv(path) for path in cells], ignore_index=True)
        merged = merged.drop_duplicates(subset=["dataset", "policy", "attack", "seed"],
                                        keep="first")
        merged.to_csv(OUT_DIR / "ablation_all.csv", index=False)
        print(f"[merge] {len(merged)} rows from {len(cells)} cell files "
              f"-> {OUT_DIR / 'ablation_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
