"""Refill the per-point curves that Fig. 4 and Fig. 6 need, and archive them losslessly.

``results`` kept only the scalar export of the method matrix
(``baselines/method_matrix/matrix_scalar.csv``), which drops ``f_grid`` / ``r_curve`` /
``csa_curve`` / ``hc_curve``.  The two curve figures (Fig. 4 ``Comparison_topology``,
Fig. 6 ``Bimodal network enhance resilience``) therefore cannot be regenerated from the
archive as it stands.

This script reruns **only the cells those two figures read**, at the same protocol as the
published tables (structural cascade, seeds 0--19, degree + random attack), and writes one
table that keeps every point:

    results/baselines/curves/curves_<dataset>.csv    one file per substrate
    results/baselines/curve_archive.csv              merged; what paper_figures.py reads

``BiT-HyRL`` is not rerun: its curves already exist at twenty seeds in
``results/family/ctrl/BiT-HyRL__<dataset>.csv`` and are copied in, so the figure keeps
reading the same deployment arm as the tables.  ``BiT-norecon`` (the un-reconstructed
substrate under the deployed policy, i.e. the ``Baseline`` curve) is rerun here because the
the ablation export is scalar-only too.

Usage (on the server, from the repository root)::

    python scripts/export_curves.py                      # BA-100 + Colt, seeds 0-19
    python scripts/export_curves.py --datasets BA-1000 --seeds 0-4
    python scripts/export_curves.py --check              # coverage report only

Every ``(dataset, seed)`` cell is written to disk as soon as it finishes, so an interrupted
run resumes where it stopped; ``--force`` recomputes cells that are already present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bit_hyrl.analysis.matrix import MATRIX_COLUMNS, _record_row  # noqa: E402
from bit_hyrl.analysis.method_matrix import run_method_matrix  # noqa: E402
from bit_hyrl.evaluation.runner import run_evaluation  # noqa: E402
from bit_hyrl.io.graph_io import load_graph_spec  # noqa: E402
from bit_hyrl.rl.transfer import load_policy  # noqa: E402

BASE = REPO_ROOT / "results" / "baselines"
CURVE_DIR = BASE / "curves"
CTRL_DIR = REPO_ROOT / "results" / "family" / "ctrl"
ARCHIVE = BASE / "curve_archive.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "ppo_final_s0.pt"

#: The six curves Fig. 4 and Fig. 6 plot, minus BiT-HyRL (read from family/ctrl) and
#: minus Baseline (built here from the un-reconstructed substrate + deployed policy).
EXTERNAL_METHODS: tuple[str, ...] = ("ONION", "UNITY", "FRED-ABL", "QDLM", "Eff-R-Net")
BASELINE_LABEL = "BiT-norecon"
BIT_LABEL = "BiT-HyRL"
KEY = ["dataset", "policy", "attack", "seed", "mode"]


def _parse_seeds(text: str) -> tuple[int, ...]:
    if "-" in text:
        low, high = text.split("-")
        return tuple(range(int(low), int(high) + 1))
    return tuple(int(token) for token in text.split(",") if token.strip())


def _specs(datasets: Sequence[str]) -> list[dict[str, Any]]:
    registry = json.loads(
        (REPO_ROOT / "configs" / "graphs.json").read_text(encoding="utf-8")
    )["graphs"]
    index = {spec["dataset"]: spec for spec in registry}
    missing = [name for name in datasets if name not in index]
    if missing:
        raise SystemExit(f"graphs.json 里没有这些数据集: {missing}")
    return [index[name] for name in datasets]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(MATRIX_COLUMNS))


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return _empty()
    frame = pd.read_csv(path)
    for column in MATRIX_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(MATRIX_COLUMNS)]


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.drop_duplicates(subset=KEY, keep="last")
    frame = frame.sort_values(["dataset", "policy", "attack", "seed"])
    frame.to_csv(path, index=False)


def _bit_arm(dataset: str) -> pd.DataFrame:
    """The deployed arm, copied from the twenty-seed family control export."""
    path = CTRL_DIR / f"{BIT_LABEL}__{dataset}.csv"
    if not path.exists():
        return _empty()
    frame = pd.read_csv(path)
    frame = frame[frame["policy"] == BIT_LABEL].copy()
    return frame[[column for column in MATRIX_COLUMNS if column in frame.columns]] \
        .reindex(columns=list(MATRIX_COLUMNS))


def _baseline_cell(spec: dict[str, Any], seed: int, attacks: Sequence[str],
                   checkpoint: Path) -> list[dict[str, Any]]:
    """Un-reconstructed substrate + deployed placement, the row the figures call Baseline."""
    original, _meta = load_graph_spec(spec)
    policy = load_policy(checkpoint)
    n_c = max(1, round(0.1 * original.number_of_nodes()))
    controllers = [int(v) for v in policy.select(original, n_c, greedy=True)]
    rows = []
    for attack in attacks:
        record = run_evaluation(
            spec, graph=original, controllers=controllers, policy=BASELINE_LABEL,
            topology="original", attack=attack, mode="structural", seed=seed,
            c=1.6, h=1,
        )
        rows.append(_record_row(record))
    return rows


def _have(frame: pd.DataFrame, dataset: str, policies: Sequence[str],
          n_attacks: int) -> set[tuple[str, int]]:
    """Cells already complete: a ``(policy, seed)`` counts only with every attack present."""
    subset = frame[(frame["dataset"] == dataset) & (frame["policy"].isin(policies))]
    if subset.empty:
        return set()
    counts = subset.groupby(["policy", "seed"]).size()
    return {(str(policy), int(seed)) for (policy, seed), count in counts.items()
            if int(count) >= n_attacks}


def run(datasets: Sequence[str], seeds: Sequence[int], attacks: Sequence[str],
        workers: int, checkpoint: Path, force: bool,
        methods: Sequence[str] = EXTERNAL_METHODS) -> int:
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint 不存在: {checkpoint}")
    if checkpoint != DEFAULT_CHECKPOINT:
        print(f"[ckpt] 用 {checkpoint.relative_to(REPO_ROOT)}")
    specs = _specs(datasets)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        dataset = spec["dataset"]
        target = CURVE_DIR / f"curves_{dataset}.csv"
        frame = _empty() if force else _read(target)

        # the deployed arm is copied in once; it is not recomputed here
        bit = _bit_arm(dataset)
        if not bit.empty:
            frame = pd.concat([frame[frame["policy"] != BIT_LABEL], bit], ignore_index=True)

        wanted = [(policy, seed) for policy in (*methods, BASELINE_LABEL)
                  for seed in seeds]
        already = _have(frame, dataset, (*methods, BASELINE_LABEL), len(attacks))
        todo = [(policy, seed) for policy, seed in wanted if (policy, seed) not in already]
        print(f"[{dataset}] {len(todo)}/{len(wanted)} 单元格待跑 "
              f"({len(seeds)} seeds × {len(methods) + 1} policies)", flush=True)

        for index, (policy, seed) in enumerate(todo, start=1):
            started = time.time()
            if policy == BASELINE_LABEL:
                rows = _baseline_cell(spec, seed, attacks, checkpoint)
            else:
                cell = run_method_matrix(
                    [spec], methods=(policy,), attacks=tuple(attacks),
                    mode="structural", seeds=(seed,), checkpoint=None,
                    workers=1,
                )
                rows = cell.to_dict("records")
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
            _write(frame, target)
            print(f"[{dataset}] {index}/{len(todo)} {policy} s{seed} "
                  f"{time.time() - started:.1f}s", flush=True)

    frames = [_read(CURVE_DIR / f"curves_{spec['dataset']}.csv") for spec in specs]
    merged = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    _write(merged, ARCHIVE)
    print(f"[merge] {len(merged)} 行 → {ARCHIVE}")
    return 0


def check(datasets: Sequence[str], seeds: Sequence[int]) -> int:
    """Print which curves the archive still lacks; 0 when the figures can be drawn."""
    frame = _read(ARCHIVE)
    policies = (*EXTERNAL_METHODS, BIT_LABEL, BASELINE_LABEL)
    gaps = 0
    for dataset in datasets:
        for policy in policies:
            subset = frame[(frame["dataset"] == dataset) & (frame["policy"] == policy)]
            missing = sorted(set(seeds) - set(int(seed) for seed in subset["seed"]))
            attacks = sorted(set(subset["attack"].astype(str)))
            if missing or len(attacks) < 2:
                gaps += 1
                print(f"  MISS {dataset:<10} {policy:<12} "
                      f"缺 seeds={missing or '-'} attacks={attacks}")
            else:
                print(f"  ok   {dataset:<10} {policy:<12} "
                      f"seeds={len(seeds)} attacks={attacks}")
    print("完整，可出图" if gaps == 0 else f"还缺 {gaps} 条曲线")
    return 0 if gaps == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="补齐图 4 / 图 6 的逐点曲线并归档")
    parser.add_argument("--datasets", default="BA-100,Colt",
                        help="逗号分隔的数据集名（默认图 4/6 用到的两张）")
    parser.add_argument("--seeds", default="0-19", help="0-19 或 0,1,2")
    parser.add_argument("--attacks", default="degree,random")
    parser.add_argument("--workers", type=int, default=1,
                        help="保留给批量调用；单格内部按方法串行，避免小 n 上进程开销")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--force", action="store_true", help="重算已存在的单元格")
    parser.add_argument("--check", action="store_true", help="只报告覆盖情况，不跑")
    parser.add_argument("--methods", default=",".join(EXTERNAL_METHODS),
                        help="逗号分隔的外部构造方法（默认图 4/6 的五种；加 ROMEN 很慢）")
    args = parser.parse_args(argv)

    datasets = tuple(token.strip() for token in args.datasets.split(",") if token.strip())
    seeds = _parse_seeds(args.seeds)
    attacks = tuple(token.strip() for token in args.attacks.split(",") if token.strip())

    if args.check:
        return check(datasets, seeds)
    if not attacks:
        raise SystemExit("--attacks 不能为空")
    methods = tuple(token.strip() for token in args.methods.split(",") if token.strip())
    return run(datasets, seeds, attacks, args.workers, Path(args.checkpoint), args.force,
               methods)


if __name__ == "__main__":
    raise SystemExit(main())
