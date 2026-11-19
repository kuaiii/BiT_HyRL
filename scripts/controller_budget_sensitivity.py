"""Controller-budget sensitivity: does the gain survive at 5 %, 10 % and 20 %?"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bit_hyrl.analysis.methods import build_method_graph  # noqa: E402
from bit_hyrl.control.capacity import capacity_model  # noqa: E402
from bit_hyrl.control.cascade import run_cascade  # noqa: E402
from bit_hyrl.evaluation.curves import compute_robustness  # noqa: E402
from bit_hyrl.evaluation.runner import provisional_attack_sequence  # noqa: E402
from bit_hyrl.io.graph_io import load_graph_spec  # noqa: E402
from bit_hyrl.rl.transfer import load_policy  # noqa: E402

OUT = REPO_ROOT / "results" / "budget" / "budget_all.csv"


def evaluate(graph, controllers, n_c: int, ratio: float, attack: str, seed: int) -> dict[str, Any]:
    model = capacity_model(graph.number_of_nodes(),
                           SimpleNamespace(controller_ratio=ratio))
    sequence = provisional_attack_sequence(graph, attack, controllers=controllers, seed=seed)
    trace = run_cascade(graph, sequence, controllers, model,
                        SimpleNamespace(mode="structural", max_iter=50))
    metrics = compute_robustness(graph, trace, graph.number_of_nodes(), n_c)
    return {"r_res": round(float(metrics["r_res"]), 6),
            "auc_gcc": round(float(metrics["auc_gcc"]), 6),
            "auc_csa": round(float(metrics["auc_csa"]), 6),
            "auc_hc": round(float(metrics["auc_hc"]), 6)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sensitivity to the controller budget")
    parser.add_argument("--ratios", default="0.05,0.1,0.2")
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--c", default="1.6",
                        help="'auto' uses the per-graph deployment rule of Section 3")
    parser.add_argument("--graphs", default="all")
    parser.add_argument("--checkpoint", default="results/checkpoints/ppo_final_s0.pt")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    registry = {g["dataset"]: g for g in
                json.loads((REPO_ROOT / "configs" / "graphs.json")
                           .read_text(encoding="utf-8"))["graphs"]}
    wanted = (list(registry) if args.graphs.strip().lower() == "all"
              else [t.strip() for t in args.graphs.split(",") if t.strip()])
    ratios = [float(t) for t in args.ratios.split(",") if t.strip()]
    policy_obj = load_policy(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(wanted, start=1):
        if dataset not in registry:
            continue
        original, _meta = load_graph_spec(registry[dataset])
        if str(args.c).lower() == "auto":
            from bit_hyrl.topology.interval import deployed_c
            from family_bench import efficiency
            c_value = deployed_c(original.number_of_nodes(), original.number_of_edges(),
                                 efficiency(original))
        else:
            c_value = float(args.c)
        graphs = {"original": original,
                  "bit_c16": build_method_graph(original, "bit", seed=0,
                                                c=c_value, h=1)}
        for ratio in ratios:
            n_c = max(1, round(ratio * original.number_of_nodes()))
            for arm, graph in graphs.items():
                controllers = [int(v) for v in policy_obj.select(graph, n_c, greedy=True)]
                for attack in ("degree", "random"):
                    for seed in range(args.eval_seeds):
                        metrics = evaluate(graph, controllers, n_c, ratio, attack, seed)
                        rows.append({"dataset": dataset, "arm": arm, "ratio": ratio,
                                     "n_c": n_c, "attack": attack, "eval_seed": seed,
                                     **metrics})
        pd.DataFrame(rows).to_csv(out_path, index=False)
        best = [r["r_res"] for r in rows if r["dataset"] == dataset and r["arm"] == "bit_c16"]
        base = [r["r_res"] for r in rows if r["dataset"] == dataset and r["arm"] == "original"]
        print(f"[{index:2d}/{len(wanted)}] {dataset:16s} bit={sum(best) / max(1, len(best)):.4f} "
              f"orig={sum(base) / max(1, len(base)):.4f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    print(f"\n[out] {out_path}  ({len(frame)} rows)")

    table = frame.pivot_table(index="dataset", columns=["ratio", "arm"], values="r_res")
    print("\n=== gain over the un-reconstructed original, by budget ===")
    print(f"{'dataset':16s}" + "".join(f"{r:>10.0%}" for r in ratios))
    gains = {}
    for dataset, row in table.iterrows():
        line = f"{dataset:16s}"
        for ratio in ratios:
            gain = row[(ratio, "bit_c16")] - row[(ratio, "original")]
            gains[(dataset, ratio)] = gain
            line += f"{gain:+10.4f}"
        print(line)

    print("\n=== aggregate ===")
    for ratio in ratios:
        values = [gains[(d, ratio)] for d in table.index]
        positive = sum(1 for v in values if v > 0)
        print(f"  n_c = {ratio:>4.0%}  mean gain {sum(values) / len(values):+.4f}  "
              f"positive on {positive}/{len(values)} graphs  worst {min(values):+.4f}")
    print("\n  reading: the claim survives if the sign is stable across budgets,")
    print("  not if one budget happens to be best.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
