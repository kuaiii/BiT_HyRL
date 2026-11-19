"""Upper density bound for reconstruction: ``kbar`` = 20 and 30 at 20 seeds.

Compares the un-reconstructed graph against BiT at ``c`` = 1.0 and 1.6 under the
same placement and protocol as the main experiments.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bit_hyrl.analysis.methods import build_method_graph  # noqa: E402
from bit_hyrl.evaluation.runner import run_evaluation  # noqa: E402
from bit_hyrl.rl.transfer import load_policy  # noqa: E402

OUT = REPO_ROOT / "results" / "scale" / "dense_upper_bound.csv"

#: (label, N, target kbar)
CASES: tuple[tuple[str, int, float], ...] = (
    ("er-N100-k20", 100, 20.0),
    ("er-N300-k20", 300, 20.0),
    ("er-N100-k30", 100, 30.0),
    ("er-N300-k30", 300, 30.0),
    ("er-N500-k30", 500, 30.0),
)


def _build(n: int, kbar: float) -> nx.Graph:
    graph = nx.gnm_random_graph(n, int(round(kbar * n / 2.0)), seed=11)
    if not nx.is_connected(graph):
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        graph = nx.convert_node_labels_to_integers(graph)
    return graph


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="upper density bound at 20 seeds")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--checkpoint", default="results/checkpoints/ppo_final_s0.pt")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    policy_obj = load_policy(args.checkpoint)
    rows: list[dict[str, Any]] = []

    for label, n, kbar in CASES:
        original = _build(n, kbar)
        n_real = original.number_of_nodes()
        m_real = original.number_of_edges()
        kbar_real = 2.0 * m_real / n_real
        print(f"\n=== {label}  n={n_real} m={m_real} kbar={kbar_real:.1f} ===", flush=True)

        cases = [("original", original)]
        for c_value in (1.0, 1.6):
            cases.append((f"c={c_value}", build_method_graph(original, "bit", seed=0,
                                                             c=c_value, h=1)))

        for config, graph in cases:
            n_c = max(1, round(0.1 * graph.number_of_nodes()))
            controllers = [int(v) for v in policy_obj.select(graph, n_c, greedy=True)]
            scores: list[float] = []
            for seed in range(args.seeds):
                for attack in ("degree", "random"):
                    record = run_evaluation(
                        {"path": None, "dataset": label}, graph=graph,
                        controllers=controllers, policy="probe", topology="probe",
                        attack=attack, mode="structural", seed=seed, c=1.6, h=1,
                    )
                    scores.append(record.r_res)
            mean = sum(scores) / len(scores)
            variance = (sum((x - mean) ** 2 for x in scores) / max(1, len(scores) - 1))
            stderr = (variance / len(scores)) ** 0.5
            rows.append({
                "case": label, "config": config, "n": n_real,
                "kbar": round(kbar_real, 2), "seed_count": args.seeds,
                "r_res": round(mean, 4), "r_res_sd": round(variance ** 0.5, 4),
                "stderr": round(stderr, 4),
            })
            print(f"  {config:9s} R_res={mean:.4f} +/- {stderr:.4f} (sd {variance ** 0.5:.4f})",
                  flush=True)

    import pandas as pd
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(f"\n[out] {out}")

    print("\n=== reconstruction gain (BiT minus original) ===")
    pivot = frame.pivot_table(index="case", columns="config", values="r_res")
    for case in pivot.index:
        if {"c=1.6", "original"} <= set(pivot.columns):
            delta = pivot.loc[case, "c=1.6"] - pivot.loc[case, "original"]
            print(f"  {case:14s} c=1.6 vs original: {delta:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
