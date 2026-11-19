"""Shard scheduler for :mod:`family_bench`.

Every ``(graph, seed)`` pair runs as one subprocess, largest graph first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="launch family_bench shards")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--label", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--max-n", type=int, default=0,
                        help="skip graphs above this node count (0 = no limit)")
    parser.add_argument("--out-dir", default="results/family/shards")
    parser.add_argument("--log-dir", default="results/family/logs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    specs = json.loads((REPO_ROOT / "configs" / "graphs.json")
                       .read_text(encoding="utf-8"))["graphs"]
    if args.datasets.strip().lower() != "all":
        wanted = {name.strip() for name in args.datasets.split(",") if name.strip()}
        specs = [spec for spec in specs if spec["dataset"] in wanted]
    if args.max_n:
        specs = [spec for spec in specs if spec["n"] <= args.max_n]
    specs.sort(key=lambda spec: -spec["n"])

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    out_dir = REPO_ROOT / args.out_dir
    log_dir = REPO_ROOT / args.log_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tag = f"__{args.label}" if args.label else ""
    tasks: list[tuple[str, int]] = []
    for spec in specs:
        for seed in seeds:
            path = out_dir / f"{spec['dataset']}__s{seed}{tag}.csv"
            if path.exists() and not args.force:
                continue
            tasks.append((spec["dataset"], seed))
    print(f"[plan] {len(tasks)} shards, {args.workers} lanes, label={args.label or '-'}",
          flush=True)
    if args.dry_run or not tasks:
        return 0

    def run(task: tuple[str, int]) -> tuple[str, int, float, int]:
        dataset, seed = task
        command = [sys.executable, str(REPO_ROOT / "scripts" / "family_bench.py"),
                   "--datasets", dataset, "--seeds", str(seed), "--methods", args.methods,
                   "--out-dir", str(out_dir)]
        if args.label:
            command += ["--label", args.label]
        if args.force:
            command += ["--force"]
        started = time.perf_counter()
        log_path = log_dir / f"{dataset}__s{seed}{tag}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=str(REPO_ROOT))
        return dataset, seed, time.perf_counter() - started, code

    started = time.perf_counter()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            dataset, seed, elapsed, code = future.result()
            failures += code != 0
            print(f"[{done}/{len(tasks)}] {dataset} s{seed} {elapsed:6.1f}s "
                  f"{'ok' if code == 0 else 'FAIL'}", flush=True)
    print(f"[done] {len(tasks)} shards in {(time.perf_counter() - started) / 60:.1f} min, "
          f"{failures} failures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
