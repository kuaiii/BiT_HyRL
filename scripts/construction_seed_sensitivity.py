"""Construction-seed sensitivity: is the reported gain a lucky realisation?

Every result so far builds the BiT graph with ``seed=0`` (the 2-switch decorrelation runs
``10M`` swaps from that seed).  The degree sequence is identical for every seed, so the only
thing that changes is *which* edges realise it.  A reviewer is entitled to ask whether the
reported gain depends on that choice -- especially now that we know the metric is sensitive
to another arbitrary choice (node labels, ARGUMENTS 5.5).

This script rebuilds each graph at several construction seeds, re-places controllers with the
same policy, and evaluates under the standard protocol.  The quantity of interest is not the
mean over seeds but the **spread**: if the worst seed still beats the un-reconstructed
original, the claim does not rest on a lucky draw.

Usage:
    python scripts/construction_seed_sensitivity.py
    python scripts/construction_seed_sensitivity.py --build-seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bit_hyrl.analysis.methods import build_method_graph  # noqa: E402
from bit_hyrl.evaluation.runner import run_evaluation  # noqa: E402
from bit_hyrl.io.graph_io import load_graph_spec  # noqa: E402
from bit_hyrl.rl.transfer import load_policy  # noqa: E402

#: single results root; Table 11's first row is read from the summary written next to it
OUT = REPO_ROOT / "results" / "family" / "construction_seed.csv"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重构随机种子的敏感性")
    parser.add_argument("--build-seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--c", default="1.6")
    parser.add_argument("--graphs", default="all")
    parser.add_argument("--checkpoint", default="results/checkpoints/ppo_final_s0.pt")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    registry = {g["dataset"]: g for g in
                json.loads((REPO_ROOT / "configs" / "graphs.json")
                           .read_text(encoding="utf-8"))["graphs"]}
    wanted = (list(registry) if args.graphs.strip().lower() == "all"
              else [t.strip() for t in args.graphs.split(",") if t.strip()])
    build_seeds = [int(t) for t in args.build_seeds.split(",") if t.strip()]
    policy_obj = load_policy(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(wanted, start=1):
        if dataset not in registry:
            continue
        spec = registry[dataset]
        original, _meta = load_graph_spec(spec)
        n_c = max(1, round(0.1 * original.number_of_nodes()))

        # baseline: the un-reconstructed graph, same placement and protocol
        base_controllers = [int(v) for v in policy_obj.select(original, n_c, greedy=True)]
        for attack in ("degree", "random"):
            for seed in range(args.eval_seeds):
                record = run_evaluation(
                    {"path": None, "dataset": dataset}, graph=original,
                    controllers=base_controllers, policy="probe", topology="original",
                    attack=attack, mode="structural", seed=seed, c=1.6, h=1,
                )
                rows.append({"dataset": dataset, "arm": "original", "build_seed": -1,
                             "attack": attack, "eval_seed": seed,
                             "r_res": round(float(record.r_res), 6)})

        for build_seed in build_seeds:
            graph = build_method_graph(original, "bit", seed=build_seed,
                                       c=float(args.c), h=1)
            controllers = [int(v) for v in policy_obj.select(graph, n_c, greedy=True)]
            for attack in ("degree", "random"):
                for seed in range(args.eval_seeds):
                    record = run_evaluation(
                        {"path": None, "dataset": dataset}, graph=graph,
                        controllers=controllers, policy="probe", topology="bit",
                        attack=attack, mode="structural", seed=seed, c=float(args.c), h=1,
                    )
                    rows.append({"dataset": dataset, "arm": "bit", "build_seed": build_seed,
                                 "attack": attack, "eval_seed": seed,
                                 "r_res": round(float(record.r_res), 6)})

        pd.DataFrame(rows).to_csv(out_path, index=False)   # incremental: timeout-safe
        print(f"[{index:2d}/{len(wanted)}] {dataset:16s} n={original.number_of_nodes():5d} "
              f"baseline={np.mean([r['r_res'] for r in rows if r['dataset'] == dataset and r['arm'] == 'original']):.4f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    print(f"\n[out] {out_path}  ({len(frame)} rows)")

    per = (frame.groupby(["dataset", "arm", "build_seed"])["r_res"].mean()
           .reset_index())
    print("\n=== per graph: baseline, seed range, worst-case gain ===")
    print(f"{'dataset':16s}{'baseline':>10s}{'seed mean':>11s}{'seed min':>10s}"
          f"{'seed max':>10s}{'worst gain':>12s}")
    summary = []
    for dataset, group in per.groupby("dataset"):
        base = group.loc[group["arm"] == "original", "r_res"]
        bit = group.loc[group["arm"] == "bit", "r_res"]
        if base.empty or bit.empty:
            continue
        base_value = float(base.iloc[0])
        worst = float(bit.min()) - base_value
        summary.append({"dataset": dataset, "baseline": base_value,
                        "bit_mean": float(bit.mean()), "bit_min": float(bit.min()),
                        "bit_max": float(bit.max()), "worst_gain": worst,
                        "seed_spread": float(bit.max() - bit.min())})
        print(f"{dataset:16s}{base_value:10.4f}{bit.mean():11.4f}{bit.min():10.4f}"
              f"{bit.max():10.4f}{worst:12.4f}")

    table = pd.DataFrame(summary).sort_values("worst_gain")
    table.to_csv(out_path.with_name("construction_seed_summary.csv"), index=False)
    print("\n=== aggregate ===")
    print(f"  build seeds tested      : {build_seeds}")
    print(f"  graphs                  : {len(table)}")
    print(f"  mean gain over baseline : {table['bit_mean'].mean() - table['baseline'].mean():+.4f}")
    print(f"  worst per-graph gain    : {table['worst_gain'].min():+.4f}")
    print(f"  graphs still positive on their worst seed: "
          f"{int((table['worst_gain'] > 0).sum())}/{len(table)}")
    print(f"  seed spread (max-min)   : median {table['seed_spread'].median():.4f}, "
          f"max {table['seed_spread'].max():.4f}")
    print("\n  worst five graphs on their worst construction seed:")
    print(table.head(5).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
