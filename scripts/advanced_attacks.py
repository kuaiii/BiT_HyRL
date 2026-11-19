"""Advanced-attack sweep (Table 7): four attacks over eight arms."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bit_hyrl.analysis.methods import build_method_graph  # noqa: E402
from bit_hyrl.attack import (  # noqa: E402
    ACTION_TABLE,
    AdvancedAttackError,
    GdmUnavailableError,
    assert_permutation,
)
from bit_hyrl.control.capacity import capacity_model  # noqa: E402
from bit_hyrl.control.cascade import run_cascade  # noqa: E402
from bit_hyrl.evaluation.curves import compute_robustness  # noqa: E402
from bit_hyrl.io.graph_io import load_graph_spec  # noqa: E402
from bit_hyrl.rl.transfer import load_policy  # noqa: E402

#: single results root (see ``results/README.md``); Table 10 is built from here
OUT_DIR = REPO_ROOT / "results" / "advanced"
OUT_CSV = OUT_DIR / "advanced_attacks.csv"
OUT_SUMMARY = OUT_DIR / "advanced_attacks_summary.csv"
OUT_ERRORS = OUT_DIR / "advanced_attacks.errors.log"

#: The five graphs Table 7 averages over.
DEFAULT_GRAPHS: tuple[str, ...] = ("BA-100", "BA-200", "BA-300", "BA-400", "BA-500")

ARMS: dict[str, str | None] = {
    "Baseline": None,
    "BiT-HyRL": "bit",
    "FRED-ABL": "FRED-ABL",
    "Eff-R-Net": "Eff-R-Net",
    "ONION": "ONION",
    "QDLM": "QDLM",
    "ROMEN": "ROMEN",
    "UNITY": "UNITY",
}

#: BiT hub factor of the ``BiT-HyRL`` arm (the paper's operating point).
BIT_C: float = 1.6

#: Published cascade mode and iteration cap.
CASCADE_MODE: str = "structural"
CASCADE_MAX_ITER: int = 50

#: Hub factor / hub count used for every non-``BiT-HyRL`` construction.
CONSTRUCT_H: int = 1

CSV_COLUMNS: tuple[str, ...] = (
    "dataset", "arm", "attack", "seed", "r_res", "auc_gcc", "auc_csa", "auc_hc",
)


def _parse_strs(text: str) -> list[str]:
    """Split a comma-separated CLI argument into non-empty tokens."""
    return [token.strip() for token in text.split(",") if token.strip()]


def _parse_ints(text: str) -> list[int]:
    """Split a comma-separated CLI argument into ``int`` tokens."""
    return [int(token) for token in text.split(",") if token.strip()]


def _arm_graph(original, construct: str | None, *, seed: int):
    """Build one arm's evaluated graph via the shared method dispatcher."""
    if construct == "bit":
        from bit_hyrl.topology.interval import deployed_c
        from family_bench import efficiency
        n, m = original.number_of_nodes(), original.number_of_edges()
        return build_method_graph(original, "bit", seed=seed,
                                  c=deployed_c(n, m, efficiency(original)), h=CONSTRUCT_H)
    return build_method_graph(
        original, construct, seed=seed, c=BIT_C, h=CONSTRUCT_H
    )


def _evaluate_graph(
    dataset: str,
    spec: dict[str, Any],
    *,
    arms: Sequence[str],
    attacks: Sequence[str],
    seeds: Sequence[int],
    checkpoint: str,
    max_iter: int,
    construct_seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the full ``arm × attack × seed`` grid for one graph."""
    original, _meta = load_graph_spec(spec)
    policy = load_policy(checkpoint)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for arm in arms:
        construct = ARMS[arm]
        graph = _arm_graph(original, construct, seed=construct_seed)
        n_nodes = graph.number_of_nodes()
        model = capacity_model(n_nodes)
        n_controllers = model.n_controllers
        controllers = [
            int(node) for node in policy.select(graph, n_controllers, greedy=True)
        ]

        for attack in attacks:
            producer = ACTION_TABLE[attack]
            for seed in seeds:
                try:
                    sequence = producer(graph, seed=seed)
                    assert_permutation(sequence, graph, name=f"{arm}/{attack}/{seed}")
                    trace = run_cascade(
                        graph,
                        sequence,
                        controllers,
                        model,
                        SimpleNamespace(mode=CASCADE_MODE, max_iter=max_iter),
                    )
                    metrics = compute_robustness(
                        graph, trace, n_nodes, n_controllers
                    )
                    if not trace.audit_ok:  # pragma: no cover - defensive
                        raise RuntimeError("the cascade audit identity does not hold")
                    rows.append({
                        "dataset": dataset,
                        "arm": arm,
                        "attack": attack,
                        "seed": int(seed),
                        "r_res": float(metrics["r_res"]),
                        "auc_gcc": float(metrics["auc_gcc"]),
                        "auc_csa": float(metrics["auc_csa"]),
                        "auc_hc": float(metrics["auc_hc"]),
                    })
                except Exception as exc:
                    message = (
                        f"{dataset}/{arm}/{attack}/seed={seed}: "
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    )
                    errors.append(message)
                    print(f"[ERROR] {dataset} {arm} {attack} s={seed}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    rows.append({
                        "dataset": dataset,
                        "arm": arm,
                        "attack": attack,
                        "seed": int(seed),
                        "r_res": float("nan"),
                        "auc_gcc": float("nan"),
                        "auc_csa": float("nan"),
                        "auc_hc": float("nan"),
                    })
    return rows, errors


def _graph_task(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[str], float]:
    """Process-pool entry point for one graph (must be importable at module level)."""
    started = time.perf_counter()
    dataset = str(payload["dataset"])
    rows, errors = _evaluate_graph(
        dataset,
        payload["spec"],
        arms=payload["arms"],
        attacks=payload["attacks"],
        seeds=payload["seeds"],
        checkpoint=payload["checkpoint"],
        max_iter=int(payload["max_iter"]),
        construct_seed=int(payload["construct_seed"]),
    )
    return dataset, rows, errors, time.perf_counter() - started


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write ``rows`` to ``path`` (creating the directory), column order fixed."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows), columns=list(CSV_COLUMNS))
    frame.to_csv(path, index=False)


def _write_summary(csv_path: Path, path: Path) -> None:
    """Write the per-cell GCC mean / seed count table next to the raw CSV."""
    import pandas as pd

    frame = pd.read_csv(csv_path)
    ok = frame[frame["auc_gcc"].notna()]
    mean = ok.pivot_table(
        index="arm", columns="attack", values="auc_gcc", aggfunc="mean"
    )
    count = ok.pivot_table(
        index="arm", columns="attack", values="auc_gcc", aggfunc="count"
    )
    combined = mean.round(6).add_suffix("__mean").join(
        count.add_suffix("__n").astype("Int64")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path)


def _print_tables(csv_path: Path) -> None:
    """Print the two reported views: GCC mean and the seed counts behind it."""
    import pandas as pd

    frame = pd.read_csv(csv_path)
    ok = frame[frame["auc_gcc"].notna()]
    print("\n== mean AUC_GCC  (rows=arm, cols=attack; 5 BA graphs pooled) ==")
    pivot = ok.pivot_table(
        index="arm", columns="attack", values="auc_gcc", aggfunc="mean"
    )
    print(pivot.round(4).to_string())
    print("\n== evals per cell (rows=arm, cols=attack) ==")
    print(
        ok.pivot_table(
            index="arm", columns="attack", values="auc_gcc", aggfunc="count"
        ).to_string()
    )
    print("\n== mean r_res ==")
    print(
        ok.pivot_table(index="arm", columns="attack", values="r_res", aggfunc="mean")
        .round(4)
        .to_string()
    )
    failed = int(frame["auc_gcc"].isna().sum())
    print(f"\n== failed evaluations: {failed}/{len(frame)} ==")


def _selftest(
    graphs: Sequence[str],
    attacks: Sequence[str],
    seeds: Sequence[int],
) -> int:
    """Check permutation + determinism for every producer on real graphs."""
    registry = json.loads(
        (REPO_ROOT / "configs" / "graphs.json").read_text(encoding="utf-8")
    )["graphs"]
    specs = {entry["dataset"]: entry for entry in registry}
    failures: list[str] = []
    deterministic_only = {"adaptive", "correlated", "gdm"}

    for dataset in graphs:
        if dataset not in specs:
            failures.append(f"{dataset}: not in graphs.json")
            continue
        graph, _meta = load_graph_spec(specs[dataset])
        n_nodes = graph.number_of_nodes()
        for attack in attacks:
            producer = ACTION_TABLE[attack]
            baseline: list[int] | None = None
            for seed in seeds:
                try:
                    started = time.perf_counter()
                    first = producer(graph, seed=seed)
                    elapsed = time.perf_counter() - started
                    second = producer(graph, seed=seed)
                except GdmUnavailableError as exc:
                    failures.append(
                        f"{dataset}/{attack}/seed={seed}: GDM unavailable (hard blocker)\n{exc}"
                    )
                    print(f"[BLOCKED] {dataset} {attack} seed={seed}: {exc}",
                          flush=True)
                    continue
                except AdvancedAttackError as exc:
                    failures.append(f"{dataset}/{attack}/seed={seed}: {exc}")
                    continue
                try:
                    assert_permutation(first, graph, name=f"{dataset}/{attack}")
                except AdvancedAttackError as exc:
                    failures.append(f"{dataset}/{attack}/seed={seed}: {exc}")
                    continue
                if first != second:
                    failures.append(
                        f"{dataset}/{attack}/seed={seed}: two calls disagree (non-deterministic)"
                    )
                if attack in deterministic_only:
                    if baseline is None:
                        baseline = first
                    elif first != baseline:
                        failures.append(
                            f"{dataset}/{attack}/seed={seed}: the result depends on the seed, "
                            "but this attack should be seed-independent"
                        )
                print(
                    f"[ok] {dataset:<8} {attack:<11} seed={seed} "
                    f"n={n_nodes} len={len(first)} {elapsed * 1e3:6.1f}ms",
                    flush=True,
                )

    if failures:
        print("\n== SELF-CHECK FAILED ==")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\n== SELF-CHECK PASSED (permutation + determinism) ==")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced-attack sweep for Table tab:advanced_attacks."
    )
    parser.add_argument("--selftest", action="store_true",
                        help="only verify permutation + determinism of the producers")
    parser.add_argument("--graphs", default="", help="comma list; empty = the five BA graphs")
    parser.add_argument("--arms", default="", help="comma list; empty = all eight arms")
    parser.add_argument("--attacks", default="",
                        help=f"comma list; empty = {sorted(ACTION_TABLE)}")
    parser.add_argument("--seeds", default="0,1,2", help="attack seeds")
    parser.add_argument("--checkpoint", default="results/checkpoints/ppo_final_s0.pt",
                        help="PPO checkpoint for the unified placement")
    parser.add_argument("--max-iter", type=int, default=CASCADE_MAX_ITER)
    parser.add_argument("--construct-seed", type=int, default=0,
                        help="seed for the arm reconstructions")
    parser.add_argument("--workers", type=int, default=1,
                        help="graphs processed in parallel (1 = sequential)")
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args(argv)

    graphs = _parse_strs(args.graphs) or list(DEFAULT_GRAPHS)
    attacks = _parse_strs(args.attacks) or sorted(ACTION_TABLE)
    seeds = _parse_ints(args.seeds)
    arms = _parse_strs(args.arms) or list(ARMS)

    unknown_attacks = [name for name in attacks if name not in ACTION_TABLE]
    if unknown_attacks:
        parser.error(f"unknown attacks {unknown_attacks}; available: {sorted(ACTION_TABLE)}")
    unknown_arms = [name for name in arms if name not in ARMS]
    if unknown_arms:
        parser.error(f"unknown arms {unknown_arms}; available: {list(ARMS)}")

    if args.selftest:
        return _selftest(graphs, attacks, seeds)

    registry = json.loads(
        (REPO_ROOT / "configs" / "graphs.json").read_text(encoding="utf-8")
    )["graphs"]
    specs = {entry["dataset"]: entry for entry in registry}
    missing = [name for name in graphs if name not in specs]
    if missing:
        parser.error(f"graphs {missing} are not in configs/graphs.json")

    total = len(graphs) * len(arms) * len(attacks) * len(seeds)
    print(
        f"[start] {len(graphs)} graphs × {len(arms)} arms × {len(attacks)} attacks "
        f"× {len(seeds)} seeds = {total} evaluations "
        f"(mode={CASCADE_MODE}, max_iter={args.max_iter}, workers={args.workers})",
        flush=True,
    )

    payloads = [
        {
            "dataset": dataset,
            "spec": specs[dataset],
            "arms": list(arms),
            "attacks": list(attacks),
            "seeds": list(seeds),
            "checkpoint": args.checkpoint,
            "max_iter": args.max_iter,
            "construct_seed": args.construct_seed,
        }
        for dataset in graphs
    ]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    started = time.perf_counter()

    if args.workers <= 1:
        for payload in payloads:
            dataset, graph_rows, graph_errors, elapsed = _graph_task(payload)
            rows.extend(graph_rows)
            errors.extend(graph_errors)
            _flush(rows, errors, args.out)
            done = len(rows)
            eta = (time.perf_counter() - started) / done * (total - done) if done else 0.0
            print(
                f"[done] {dataset}  rows={done}/{total}  "
                f"graph={elapsed / 60:.1f}min  eta={eta / 60:.1f}min",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(args.workers),
                                 mp_context=context) as pool:
            futures = {pool.submit(_graph_task, payload): payload["dataset"]
                       for payload in payloads}
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    _, graph_rows, graph_errors, elapsed = future.result()
                except Exception as exc:
                    message = f"{dataset}: graph task failed {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    errors.append(message)
                    print(f"[ERROR] {message}", flush=True)
                    continue
                rows.extend(graph_rows)
                errors.extend(graph_errors)
                _flush(rows, errors, args.out)
                done = len(rows)
                eta = (time.perf_counter() - started) / done * (total - done) if done else 0.0
                print(
                    f"[done] {dataset}  rows={done}/{total}  "
                    f"graph={elapsed / 60:.1f}min  eta={eta / 60:.1f}min",
                    flush=True,
                )

    _flush(rows, errors, args.out)
    print(f"\n[out] {args.out}  ({len(rows)} rows, {len(errors)} error(s))")
    if rows:
        out_csv = Path(args.out)
        summary_path = (
            OUT_SUMMARY if out_csv == OUT_CSV else out_csv.with_name(
                out_csv.stem + "_summary.csv"
            )
        )
        _write_summary(out_csv, summary_path)
        print(f"[out] {summary_path}")
        _print_tables(out_csv)
    return 0


def _flush(rows: Sequence[dict[str, Any]], errors: Sequence[str], out: str) -> None:
    """Persist the accumulated rows and errors (called after every graph)."""
    if not rows:
        return
    _write_csv(rows, Path(out))
    if errors:
        OUT_ERRORS.parent.mkdir(parents=True, exist_ok=True)
        OUT_ERRORS.write_text("\n\n".join(errors) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
