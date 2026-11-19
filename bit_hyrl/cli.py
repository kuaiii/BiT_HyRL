"""Command-line entry point.

Sub-commands: ``fingerprint``, ``construct``, ``sweep-c``, ``verify``, ``train``,
``eval``, ``transfer``, ``ablate``, ``matrix``, ``method-matrix`` and
``method-stats``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import networkx as nx

from .io import paths as io_paths
from .io.graph_io import (
    GraphSpecMismatch,
    graph_fingerprint,
    load_graph,
    load_graph_spec,
)
from .io.paths import records_dir
from .io.jsonlog import dumps
from .topology import (
    C_NORM_GRID,
    DEFAULT_C_GRID,
    DegreeSequenceError,
    build_bimodal_graph,
    diagnose,
    plan_degree_sequence,
    sweep_dataset,
    sweep_dataset_c_norm,
    write_sweep_outputs,
)
from .control import (
    PROVISIONAL_ATTACK_METHODS,
    capacity_model,
    controllers_degree_ranked,
    feasibility_curve,
    format_feasibility_table,
    format_onset_table,
    onset_records,
    probe_max_utilisation,
    provisional_attack_sequence,
    run_cascade,
    write_feasibility_outputs,
    write_onset_outputs,
)
from .config.schema import CascadeConfig
from .evaluation import EvaluationRecord, build_eval_graph, run_evaluation
from .version import (
    FINGERPRINT_SCHEMA_VERSION,
    SWEEP_SCHEMA_VERSION,
    VERSION,
)

__all__ = ["main", "build_parser"]

LOGGER = logging.getLogger("bit_hyrl.cli")

_DEFERRED: dict[str, str] = {}

#: Paper reward weights ``(w_G, w_S, w_H, w_D)`` — distance disabled.
_PAPER_REWARD_WEIGHTS = {"structure": 0.6, "csa": 0.2, "h_load": 0.2, "distance": 0.0}


def _load_graph_registry(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load ``configs/graphs.json`` and resolve relative paths."""
    registry_path = Path(path) if path is not None else io_paths.GRAPHS_CONFIG_FILE
    if not registry_path.is_file():
        raise FileNotFoundError(f"graph registry not found: {registry_path}")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    specs: list[dict[str, Any]] = []
    for entry in payload.get("graphs", []):
        spec = dict(entry)
        raw_path = Path(str(spec["path"]))
        if not raw_path.is_absolute():
            raw_path = io_paths.PROJECT_ROOT / raw_path
        spec["path"] = raw_path
        specs.append(spec)
    return specs


def _find_spec(dataset: str) -> dict[str, Any]:
    """Return the registry entry for ``dataset``."""
    for spec in _load_graph_registry():
        if spec.get("dataset") == dataset:
            return spec
    known = [spec.get("dataset") for spec in _load_graph_registry()]
    raise KeyError(f"unknown dataset {dataset!r}; the registry has {known}")


def _parse_grid(value: str) -> list[float]:
    """Parse ``--grid``: either ``default`` or a comma-separated list."""
    if value.strip().lower() == "default":
        return [float(c) for c in DEFAULT_C_GRID]
    parts = [part for part in value.replace(";", ",").split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--grid cannot be empty")
    return [float(part) for part in parts]


def _split_csv(value: str) -> list[str]:
    """Split a comma/semicolon separated CLI list into non-empty stripped parts."""
    return [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]


def _resolve_specs(dataset: str) -> list[dict[str, Any]]:
    """Resolve ``--dataset`` (a name, a comma list, or ``all``) into specs."""
    parts = _split_csv(dataset)
    if not parts or parts == ["all"]:
        return _load_graph_registry()
    return [_find_spec(name) for name in parts]


def _print_json(payload: Any) -> None:
    """Print canonical JSON to stdout (the CLI's permitted output channel)."""
    sys.stdout.write(dumps(payload, indent=2))
    sys.stdout.write("\n")


def cmd_fingerprint(args: argparse.Namespace) -> int:
    """Load every pinned graph, fingerprint it and (optionally) pin the hashes."""
    specs = _load_graph_registry(args.graphs)
    report: dict[str, Any] = {}
    failures: list[str] = []
    for spec in specs:
        dataset = str(spec.get("dataset") or Path(str(spec["path"])).stem)
        try:
            if spec.get("n") is not None or spec.get("m") is not None:
                _, fingerprint = load_graph_spec(spec)
            else:
                graph = load_graph(spec["path"])
                fingerprint = graph_fingerprint(graph, path=spec["path"])
        except GraphSpecMismatch as exc:
            failures.append(str(exc))
            LOGGER.error("%s", exc)
            continue
        report[dataset] = fingerprint

    payload = {"schema_version": FINGERPRINT_SCHEMA_VERSION, "graphs": report}
    if args.write:
        target = io_paths.atomic_write_json(io_paths.fingerprints_file(), payload)
        LOGGER.info("fingerprints written to %s", target)
    if not args.quiet:
        _print_json(payload)
    if failures:
        LOGGER.error("%d graphs do not match the pinned fingerprint", len(failures))
        return 1
    return 0


def cmd_construct(args: argparse.Namespace) -> int:
    """Construct one BiT graph and print its plan, diagnostics and fingerprint."""
    spec = _find_spec(args.dataset)
    source, fingerprint_in = load_graph_spec(spec)
    plan = plan_degree_sequence(
        source.number_of_nodes(), source.number_of_edges(), c=args.c, h=args.h
    )
    result = build_bimodal_graph(
        plan,
        args.seed,
        realize=args.realize,
        rewire_swaps=args.rewire_swaps,
        enforce_connected=not args.allow_disconnected,
        max_repair_swaps=args.max_repair_swaps,
        dataset=args.dataset,
    )
    diagnostics = diagnose(
        result.graph,
        plan=result.plan,
        construction=result,
        include_betweenness=args.betweenness,
        c=args.c,
    )
    payload = {
        "dataset": args.dataset,
        "version": VERSION,
        "source_fingerprint": fingerprint_in,
        "construction": result.to_dict(),
        "diagnostics": diagnostics.to_dict(),
    }
    if args.write:
        target = io_paths.results_path("construct", f"{args.dataset}__c{args.c}.json")
        io_paths.atomic_write_json(target, payload)
        LOGGER.info("construction written to %s", target)
    if not args.quiet:
        _print_json(payload)
    return 0 if result.connected or args.allow_disconnected else 1


def _controllers_for(graph: nx.Graph, count: int) -> list[int]:
    """Degree-ranked controller placement with a node-id tie-break."""
    order = sorted(graph.nodes(), key=lambda node: (-graph.degree(node), node))
    return [int(node) for node in order[:count]]


def cmd_probe(args: argparse.Namespace) -> int:
    """Capacity model + ``probe_max_utilisation`` for one graph (no deletion)."""
    spec = _find_spec(args.dataset)
    source = load_graph(spec["path"])
    plan = plan_degree_sequence(
        source.number_of_nodes(), source.number_of_edges(), c=args.c, h=args.h
    )
    result = build_bimodal_graph(
        plan, args.seed, realize=args.realize, rewire_swaps=args.rewire_swaps,
        dataset=args.dataset,
    )
    model = capacity_model(result.graph.number_of_nodes())
    controllers = _controllers_for(result.graph, model.n_controllers)
    probe = probe_max_utilisation(result.graph, controllers, model)

    payload = {
        "dataset": args.dataset,
        "c": args.c,
        "seed": args.seed,
        "capacity": model.to_dict(),
        "controllers": list(controllers),
        "probe": probe.to_dict(),
        "verdict": {
            "u_nearest": probe.u_nearest,
            "u_after_spill": probe.u_after_spill,
            "feasible": probe.feasible,
            "ceiling_active": probe.ceiling_active,
            "n_at_ceiling": probe.n_at_ceiling,
            "capacity_binds": probe.capacity_binds,
            "n_spilled": probe.n_spilled,
            "spill_fraction": probe.spill_fraction,
            "n_unassigned": probe.n_unassigned,
        },
    }
    if args.write:
        target = io_paths.results_path("probe", f"{args.dataset}__c{args.c}.json")
        io_paths.atomic_write_json(target, payload)
        LOGGER.info("probe written to %s", target)
    if not args.quiet:
        _print_json(payload)
    return 0


def cmd_feasibility(args: argparse.Namespace) -> int:
    """Capacity feasibility along every attack sequence, per graph."""
    datasets = (
        [str(spec["dataset"]) for spec in _load_graph_registry()]
        if args.dataset == "all"
        else [args.dataset]
    )
    methods = (
        list(PROVISIONAL_ATTACK_METHODS) if args.attack == "all" else [args.attack]
    )

    structural_cfg = SimpleNamespace(mode="structural", max_iter=args.max_iter)
    capacity_cfg = SimpleNamespace(mode="capacity", max_iter=args.max_iter)
    curves: dict[str, list[Any]] = {"structural": [], "capacity": []}
    for dataset in datasets:
        spec = _find_spec(dataset)
        source = load_graph(spec["path"])
        plan = plan_degree_sequence(
            source.number_of_nodes(), source.number_of_edges(), c=args.c, h=args.h
        )
        result = build_bimodal_graph(
            plan, args.seed, realize=args.realize, rewire_swaps=args.rewire_swaps,
            dataset=dataset,
        )
        model = capacity_model(result.graph.number_of_nodes())
        controllers = controllers_degree_ranked(result.graph, model.n_controllers)
        for method in methods:
            sequence = provisional_attack_sequence(
                result.graph, method, controllers=controllers, seed=args.seed
            )
            for mode, cfg in (("structural", structural_cfg),
                              ("capacity", capacity_cfg)):
                curves[mode].append(
                    feasibility_curve(
                        result.graph, sequence, controllers, model, cfg,
                        dataset=dataset, attack=method,
                    )
                )
            if len(curves["structural"]) != len(curves["capacity"]):
                raise RuntimeError("the structural and capacity curve counts differ")

    records = onset_records(curves["structural"], curves["capacity"])
    table = format_onset_table(records)
    if args.write:
        written = write_onset_outputs(records, name="onset")
        LOGGER.info("onset records written to %s / %s", written["json"], written["txt"])
        for mode in ("structural", "capacity"):
            curves_written = write_feasibility_outputs(
                curves[mode], name=f"feasibility__{mode}"
            )
            LOGGER.info(
                "%s: feasibility curves written to %s / %s",
                mode, curves_written["json"], curves_written["txt"],
            )
    if not args.quiet:
        print(table)
        if args.mode != "structural":
            print(format_feasibility_table(curves[args.mode]))
    return 0


def cmd_sweep_c(args: argparse.Namespace) -> int:
    """Run the ``c`` sweep and optionally persist JSON + CSV."""
    if args.grid_kind == "c_norm":
        grid: Sequence[float] = (
            C_NORM_GRID
            if args.grid in (None, "default")
            else _parse_grid(args.grid)
        )
        sweeper = sweep_dataset_c_norm
    else:
        grid = DEFAULT_C_GRID if args.grid in (None, "default") else _parse_grid(args.grid)
        sweeper = sweep_dataset
    if args.dataset == "all":
        datasets = [str(spec["dataset"]) for spec in _load_graph_registry()]
    else:
        datasets = [args.dataset]

    everything: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        rows = sweeper(
            dataset, grid, h=args.h, seed=args.seed, build_graphs=not args.plan_only,
            realize=args.realize, rewire_swaps=args.rewire_swaps,
        )
        everything[dataset] = rows
        if args.write:
            written = write_sweep_outputs(
                rows, dataset,
                stem="sweep_c_norm" if args.grid_kind == "c_norm" else "sweep_c",
            )
            LOGGER.info("%s: written to %s / %s", dataset, written["json"], written["csv"])
        if not args.quiet and args.dataset != "all":
            _print_json(
                {
                    "schema_version": SWEEP_SCHEMA_VERSION,
                    "dataset": dataset,
                    "grid_kind": args.grid_kind,
                    "rows": rows,
                }
            )
    if args.write and args.dataset == "all":
        LOGGER.info("sweep written for %d datasets", len(everything))
    return 0


_GOLDEN_MONO = " (pinned values in configs/graphs.json)"

_GOLDEN_KAPPA_4DP: dict[str, float] = {
    "BA-100": 5.8660, "BA-200": 5.9340, "BA-300": 5.9607, "BA-400": 5.9706,
    "BA-500": 5.9765, "Cogentco": 2.5720, "Colt": 2.4124, "GtsCe": 2.6891,
    "UsCarrier": 2.4974,
}

_DISTANCE_BANDS: dict[str, tuple[int, float]] = {
    "BA-100": (10, 6.0), "BA-200": (10, 6.0), "BA-300": (10, 6.0),
    "BA-400": (10, 6.0), "BA-500": (10, 6.0),
    "Cogentco": (30, 12.0), "Colt": (30, 12.0), "GtsCe": (30, 12.0),
    "UsCarrier": (30, 12.0),
}

#: ``(n_c, C, n_switches, total_capacity, aggregate_slack)``
_GOLDEN_CAPACITY: dict[str, tuple[int, int, int, int, int]] = {
    "BA-100": (10, 15, 90, 150, 60),
    "BA-200": (20, 15, 180, 300, 120),
    "BA-300": (30, 15, 270, 450, 180),
    "BA-400": (40, 15, 360, 600, 240),
    "BA-500": (50, 15, 450, 750, 300),
    "Cogentco": (20, 15, 177, 300, 123),
    "Colt": (15, 16, 138, 240, 102),
    "GtsCe": (15, 15, 134, 225, 91),
    "UsCarrier": (16, 15, 142, 240, 98),
}


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a fixed-width ASCII table."""
    text_rows = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in text_rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _expand_attacks(attack: str) -> list[str]:
    """Resolve the ``--attack`` CLI value into a concrete method list."""
    if attack == "all":
        return list(PROVISIONAL_ATTACK_METHODS)
    return [attack]


def cmd_eval(args: argparse.Namespace) -> int:
    """End-to-end evaluation runner: original vs BiT dual curves."""
    if args.dataset == "all":
        specs = _load_graph_registry()
    else:
        specs = [_find_spec(args.dataset)]

    attacks = _expand_attacks(args.attack)
    topologies = (
        ("original", "bit") if args.topology == "both" else (args.topology,)
    )

    policy_obj = None
    if getattr(args, "policy", "degree_ranked") == "ppo":
        if not getattr(args, "checkpoint", None):
            LOGGER.error("--policy ppo requires --checkpoint")
            return 2
        from .rl.transfer import load_policy

        policy_obj = load_policy(args.checkpoint)

    records: list[EvaluationRecord] = []
    by_key: dict[tuple[str, str, str, str], EvaluationRecord] = {}
    for spec in specs:
        name = str(spec.get("dataset", ""))
        for topology in topologies:
            controllers = None
            policy_label = "degree_ranked"
            if policy_obj is not None:
                eval_graph = build_eval_graph(
                    spec, topology=topology, c=args.c, h=args.h, seed=args.seed,
                    realize=args.realize, rewire_swaps=args.rewire_swaps,
                )
                target_model = capacity_model(eval_graph.number_of_nodes())
                controllers = policy_obj.select(
                    eval_graph, target_model.n_controllers, greedy=True
                )
                policy_label = "ppo"
            for attack in attacks:
                record = run_evaluation(
                    spec,
                    topology=topology,
                    c=args.c,
                    h=args.h,
                    seed=args.seed,
                    attack=attack,
                    mode=args.mode,
                    realize=args.realize,
                    rewire_swaps=args.rewire_swaps,
                    controllers=controllers,
                    policy=policy_label,
                )
                records.append(record)
                by_key[(name, topology, attack, args.mode)] = record

    if args.topology == "both":
        header = f"{'dataset':<10} {'attack':<12} {'orig AUC':>9} {'bit AUC':>9}  differs"
        if not args.quiet:
            print(header)
            print("-" * len(header))
        differing = 0
        for spec in specs:
            name = str(spec.get("dataset", ""))
            for attack in attacks:
                original = by_key.get((name, "original", attack, args.mode))
                bit = by_key.get((name, "bit", attack, args.mode))
                if original is None or bit is None:
                    continue
                differ = abs(original.auc_r - bit.auc_r) > 1e-9
                differing += int(differ)
                if not args.quiet:
                    print(
                        f"{name:<10} {attack:<12} {original.auc_r:>9.4f} "
                        f"{bit.auc_r:>9.4f}  {'yes' if differ else 'no'}"
                    )
        if not args.quiet:
            print(
                f"\n{len(records)} records written to {records_dir()}; "
                f"dataset x attack cells where original and bit differ = {differing}"
            )

    LOGGER.info("eval finished: %d EvaluationRecords -> %s", len(records), records_dir())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a PPO controller-placement policy (paper Table-4 budget)."""
    import csv
    import time

    from .config.schema import TrainConfig
    from .io.paths import checkpoints_dir
    from .rl.trainer import train

    val_specs = None
    split_path = getattr(args, "split", None)
    if split_path:
        splits = json.loads(Path(split_path).read_text(encoding="utf-8"))
        train_paths = list(splits["train"])
        val_paths = list(splits["val"])
        if args.smoke:
            train_paths = train_paths[:4]
            val_paths = val_paths[:4]
        train_specs = [{"path": rel} for rel in train_paths]
        val_specs = [{"path": rel} for rel in val_paths]
        label = "splits"
    else:
        datasets = [part.strip() for part in str(args.dataset).split(",") if part.strip()]
        if not datasets or datasets == ["all"]:
            train_specs = _load_graph_registry()
        else:
            train_specs = [_find_spec(name) for name in datasets]
        label = "_".join(datasets) if datasets and datasets != ["all"] else "all"

    iterations = min(int(args.iterations), 2) if args.smoke else int(args.iterations)
    episodes = min(int(args.episodes), 2) if args.smoke else int(args.episodes)
    cfg = TrainConfig(
        total_iterations=iterations,
        episodes_per_iter=episodes,
        lr=float(args.lr),
        gamma=float(args.gamma),
        reward_weights=dict(_PAPER_REWARD_WEIGHTS),
    )
    out = Path(args.out) if args.out else checkpoints_dir() / f"ppo__{label}__s{args.seed}.pt"

    started = time.time()
    checkpoint = train(
        train_specs, cfg, out_path=out, seed=int(args.seed), verbose=not args.quiet,
        val_graphs=val_specs,
        patience=int(args.patience),
        eval_sample=int(args.eval_sample),
        value_coef=0.5, grad_clip=0.5, minibatch=32,
    )
    elapsed = time.time() - started

    history = list(checkpoint.get("history", []))
    val_history = list(checkpoint.get("val_history", []))
    hist_dir = io_paths.RESULTS_ROOT / "train"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_path = hist_dir / f"history_{label}__s{args.seed}.csv"
    with hist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "train_reward", "val_reward"])
        for index, train_reward in enumerate(history):
            val_reward = val_history[index] if index < len(val_history) else ""
            writer.writerow([index + 1, train_reward, val_reward])

    if not args.quiet:
        final = history[-1] if history else float("nan")
        print(f"training finished: {len(history)} iterations x {episodes} episodes, {elapsed:.2f}s")
        if val_history:
            print(f"val: {len(val_history)} points, best_val={checkpoint.get('best_val')}, "
                  f"stopped_iter={checkpoint.get('stopped_iteration')}")
        print(f"final training reward {final:.4f}; checkpoint -> {out}")
        print(f"history CSV → {hist_path}")
        print(
            "hyperparameters: lr=%g gamma=%g clip=%g gae=%g ent=%g epochs=%d rollouts/iter=%d "
            "minibatch=32 grad_clip=0.5 value_coef=0.5; reward weights 0.6/0.2/0.2/0"
            % (cfg.lr, cfg.gamma, cfg.clip, cfg.gae_lambda, cfg.entropy_coef, cfg.epochs, episodes)
        )
    LOGGER.info("train finished: %s -> %s (%.2fs)", label, out, elapsed)
    return 0


def cmd_transfer(args: argparse.Namespace) -> int:
    """Zero-shot transfer: place controllers on a new graph with a trained policy (T10)."""
    from .rl.transfer import apply as apply_policy, load_policy

    spec = _find_spec(args.dataset)
    graph, _meta = load_graph_spec(spec)
    model = capacity_model(graph.number_of_nodes())
    policy = load_policy(args.checkpoint)
    controllers = apply_policy(policy, graph, model.n_controllers)

    if not args.quiet:
        print(f"{args.dataset}: zero-shot placement of {len(controllers)}/{model.n_controllers} controllers")
        print(f"controllers = {controllers}")
    payload = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "n_controllers": model.n_controllers,
        "controllers": controllers,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("transfer result written to %s", out)
    LOGGER.info("transfer finished: %s <- %s", args.dataset, args.checkpoint)
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """2x2 ablation: topology (original/bit) x placement (degree_ranked/ppo) (T11)."""
    from .analysis.ablation import run_ablation, write_ablation_outputs

    specs = _resolve_specs(args.dataset)
    attacks = tuple(_split_csv(args.attacks)) or ("degree", "random")
    seeds = tuple(int(s) for s in _split_csv(args.seeds)) or (0,)
    if args.smoke:
        seeds = seeds[:1]
    result = run_ablation(
        specs, checkpoint=args.checkpoint, attacks=attacks, mode=args.mode,
        seeds=seeds, c=args.c, h=args.h,
    )
    out_dir = Path(args.out) if args.out else None
    paths = write_ablation_outputs(result, out_dir=out_dir)
    if not args.quiet:
        cells = result["overall_cell_means"]
        header = "topology/placement"
        print(f"{header:<16}{'degree_ranked':>15}{'ppo':>12}")
        for topology in ("original", "bit"):
            deg = cells.get(f"{topology}|degree_ranked", float("nan"))
            ppo = cells.get(f"{topology}|ppo", float("nan"))
            print(f"{topology:<16}{deg:>15.4f}{ppo:>12.4f}")
        effects = result["effects"]
        print(
            f"main effects: topology={effects['topology']:+.4f}  "
            f"policy={effects['policy']:+.4f}  interaction={effects['interaction']:+.6f}"
        )
        print(f"CSV → {paths['csv']}\nJSON → {paths['json']}")
    LOGGER.info("ablate finished: %d rows", len(result.get("rows", [])))
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Run the full experiment matrix and write the long-table CSV (T12)."""
    from .analysis.matrix import run_matrix, write_matrix_outputs

    specs = _resolve_specs(args.dataset)
    policies = tuple(_split_csv(args.policies)) or ("degree_ranked", "ppo")
    topologies = tuple(_split_csv(args.topologies)) or ("original", "bit")
    attacks = tuple(_split_csv(args.attacks)) or ("degree", "random")
    modes = tuple(_split_csv(args.modes)) or ("structural",)
    seeds = tuple(int(s) for s in _split_csv(args.seeds)) or (0,)
    if args.smoke:
        seeds = seeds[:1]
    frame = run_matrix(
        specs, policies=policies, topologies=topologies, attacks=attacks,
        modes=modes, seeds=seeds, checkpoint=args.checkpoint, c=args.c, h=args.h,
    )
    out_dir = Path(args.out) if args.out else None
    paths = write_matrix_outputs(frame, out_dir=out_dir)
    if not args.quiet:
        print(f"matrix: {len(frame)} rows -> {paths['csv']}")
    LOGGER.info("matrix finished: %d rows", len(frame))
    return 0


def cmd_method_matrix(args: argparse.Namespace) -> int:
    """Method-level matrix: 5 topology baselines + Eff-R-Net + 2 placement proxies + BiT-HyRL."""
    from .analysis.method_matrix import run_method_matrix, write_method_matrix_outputs
    from .analysis.methods import METHODS

    if args.smoke:
        specs = _resolve_specs("BA-100,ER-100")
        methods = ("BiT-HyRL", "ONION", "UNITY", "MWCL-sim", "MADQN-sim")
        attacks = ("degree",)
        seeds = (0,)
    else:
        if args.split:
            splits = json.loads(Path(args.split).read_text(encoding="utf-8"))
            specs = [{"path": rel, "dataset": Path(rel).stem} for rel in splits["test"]]
        else:
            specs = _resolve_specs(args.dataset)
        methods = tuple(_split_csv(args.methods)) or tuple(METHODS)
        attacks = tuple(_split_csv(args.attacks)) or ("degree", "random")
        seeds = tuple(int(s) for s in _split_csv(args.seeds)) or (0,)

    frame = run_method_matrix(
        specs, methods=methods, attacks=attacks, mode=args.mode, seeds=seeds,
        checkpoint=args.checkpoint, c=args.c, h=args.h, workers=args.workers,
    )
    out_dir = Path(args.out) if args.out else None
    paths = write_method_matrix_outputs(frame, out_dir=out_dir)
    if not args.quiet:
        print(f"method-matrix: {len(frame)} rows -> {paths['csv']}")
        for name in methods:
            subset = frame[frame["policy"] == name]
            if len(subset):
                row = subset.iloc[0]
                print(
                    f"  {name:<10} policy={row['policy']:<10} "
                    f"topology={row['topology']:<9} r_res={row['r_res']:.4f} "
                    f"auc_gcc={row['auc_gcc']:.4f}"
                )
    LOGGER.info("method-matrix finished: %d rows", len(frame))
    return 0


def cmd_method_stats(args: argparse.Namespace) -> int:
    """Method-vs-reference significance (exact Wilcoxon + Holm over GCC/CSA/H_C)."""
    import pandas as pd

    from .analysis.stats import method_significance

    frame = pd.read_csv(args.infile)
    if "policy" not in frame.columns:
        LOGGER.error("the input CSV has no policy column: %s", list(frame.columns))
        return 2
    sig = method_significance(frame, reference=args.reference)
    out = Path(args.out) if args.out else (io_paths.RESULTS_ROOT / "method_matrix" / "significance.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out, index=False)
    if not args.quiet:
        detail = sig[sig["metric"] != "__summary__"]
        for _, row in detail.iterrows():
            mean_txt = "n/a" if pd.isna(row["mean_diff"]) else f"{row['mean_diff']:+.4f}"
            print(
                f"  {row['method']:<10} {row['metric']:<8} n={row['n_pairs']} "
                f"Δ={mean_txt} p={row['wilcoxon_p']} holm={row['holm_p']}"
            )
        for _, row in sig[sig["metric"] == "__summary__"].iterrows():
            print(
                f"  [summary] {row['method']:<10} r_resΔ={row['mean_diff']} "
                f"r_res_p={row['wilcoxon_p']} {row['direction']}"
            )
        print(f"→ {out}")
    LOGGER.info("method-stats finished: %s (reference=%s)", out, args.reference)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Seed-aggregated mean/std/CI + Wilcoxon-vs-reference with Holm (T12 upgrade)."""
    import pandas as pd

    from .analysis.stats import seed_statistics, significance, write_metrics_outputs

    frame = pd.read_csv(args.infile)
    if args.metric not in frame.columns:
        LOGGER.error("metric %r is not in the columns %s", args.metric, list(frame.columns))
        return 2

    seed_df = seed_statistics(frame)
    family = ["auc_gcc", "auc_csa", "auc_hc"]
    if args.metric not in family:
        family.append(args.metric)
    sig_rows = (
        significance(frame, metrics=family, contrast=("bit", "original"))
        + significance(frame, metrics=family, contrast=("ppo", "degree_ranked"))
    )
    out_dir = Path(args.out) if args.out else None
    paths = write_metrics_outputs(seed_df, sig_rows, out_dir=out_dir)

    if not args.quiet:
        sub = seed_df[seed_df["metric"] == args.metric]
        print(f"== {args.metric}: per-seed aggregate mean/std/95%CI ==")
        for _, row in sub.iterrows():
            print(
                f"  {row['dataset']}/{row['topology']}/{row['policy']}/{row['attack']}: "
                f"mean={row['mean']:+.4f} std={row['std']:.4f} "
                f"CI95=[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
            )
        print("== significance: Wilcoxon signed-rank vs reference, Holm-corrected ==")
        for row in sig_rows:
            mean_diff = row.get("mean_diff")
            mean_txt = "n/a" if mean_diff is None else f"{mean_diff:+.4f}"
            print(
                f"  {row['contrast']:>16} {row['metric']:<9} n={row['n']} "
                f"Δ={mean_txt} p={row['wilcoxon_p']} holm={row['holm_p']:.4g} "
                f"sig={row['significant']}"
            )
        print(f"→ {paths['seed_csv']}\n→ {paths['significance_csv']}")
    LOGGER.info("stats finished")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    """Render figures (robustness / ablation / F10) from a matrix CSV (T12)."""
    import pandas as pd

    from .analysis.figures import make_all_figures

    frame = pd.read_csv(args.infile)
    out_dir = Path(args.out) if args.out else (io_paths.RESULTS_ROOT / "figures")
    figures = make_all_figures(frame, out_dir=out_dir, smoke=args.smoke)
    if not args.quiet:
        print(f"wrote {len(figures)} figures -> {out_dir}")
        for item in figures:
            print(f"  {item['name']}: png={item['size_png']}B  pdf={item['pdf']}")
    LOGGER.info("figures finished: %d", len(figures))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Run every hard numeric check in one pass and print the nine-graph table."""
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    specs = _load_graph_registry()
    grids = [float(c) for c in DEFAULT_C_GRID]

    loaded: dict[str, nx.Graph] = {}
    fp_ok = True
    for spec in specs:
        dataset = str(spec["dataset"])
        try:
            graph, _ = load_graph_spec(spec)
        except GraphSpecMismatch as exc:
            record(f"fingerprint:{dataset}", False, str(exc))
            fp_ok = False
            continue
        loaded[dataset] = graph
    record("fingerprint:all", fp_ok, f"{len(loaded)}/{len(specs)} graphs passed")

    idempotent = True
    for spec in specs:
        graph = load_graph(spec["path"])
        if graph_fingerprint(graph)["sha256"] != graph_fingerprint(load_graph(spec["path"]))["sha256"]:
            record(f"idempotent:{spec['dataset']}", False)
            idempotent = False
    record("idempotent:all", idempotent)

    conservation_ok = True
    combos = 0
    for spec in specs:
        graph = loaded.get(str(spec["dataset"]))
        if graph is None:
            continue
        for c in grids:
            combos += 1
            try:
                plan = plan_degree_sequence(
                    graph.number_of_nodes(), graph.number_of_edges(), c=c
                )
                if plan.sum_degree != 2 * graph.number_of_edges():
                    conservation_ok = False
            except DegreeSequenceError as exc:
                conservation_ok = False
                LOGGER.error("%s c=%s: %s", spec["dataset"], c, exc)
    record("conservation:Σk=2M", conservation_ok, f"{combos} combinations")

    c_value = args.c
    golden_header = (
        "dataset", "N", "M", "c*", "c_norm", "k_max", "k_low", "r", "degrees",
        "kappa", "D", "L", "repairs", "rw", "conn", "sha256",
    )
    golden_rows: list[list[Any]] = []
    golden_ok = True
    kappa_ok = True
    distance_ok = True
    for spec in specs:
        dataset = str(spec["dataset"])
        source = loaded.get(dataset)
        if source is None:
            continue
        plan = plan_degree_sequence(
            source.number_of_nodes(), source.number_of_edges(), c=c_value
        )
        result = build_bimodal_graph(plan, args.seed, dataset=dataset,
                                     realize=args.realize, rewire_swaps=args.rewire_swaps)
        diagnostics = diagnose(
            result.graph,
            plan=result.plan,
            construction=result,
            include_betweenness=False,
            c=c_value,
        )
        histogram = result.realized_degree_histogram
        strict = histogram == plan.degree_histogram()
        connected = result.connected
        edges_ok = result.final_edge_count == plan.m
        golden_ok = golden_ok and strict and connected and edges_ok
        if not (strict and connected and edges_ok):
            record(
                f"build:{dataset}", False,
                f"hist_match={strict} connected={connected} m_ok={edges_ok}",
            )

        expected_kappa = _GOLDEN_KAPPA_4DP.get(dataset)
        if expected_kappa is not None and round(diagnostics.kappa, 4) != expected_kappa:
            kappa_ok = False
            record(
                f"kappa:{dataset}", False,
                f"measured {diagnostics.kappa:.4f} != pinned {expected_kappa:.4f}",
            )

        band = _DISTANCE_BANDS.get(dataset)
        if band is not None:
            max_diameter, max_avg_path = band
            if (
                diagnostics.diameter is None
                or diagnostics.diameter > max_diameter
                or diagnostics.avg_path_length is None
                or diagnostics.avg_path_length > max_avg_path
            ):
                distance_ok = False
                record(
                    f"distance:{dataset}", False,
                    f"D={diagnostics.diameter} L={diagnostics.avg_path_length} "
                    f"band=D<={max_diameter},L<={max_avg_path}",
                )

        golden_rows.append(
            [
                dataset,
                plan.n,
                plan.m,
                f"{diagnostics.c_star:.2f}",
                f"{plan.c_norm():.3f}",
                plan.k_max,
                plan.k_low,
                plan.r,
                "{" + ", ".join(f"{d}:{n}" for d, n in histogram.items()) + "}",
                f"{diagnostics.kappa:.4f}",
                diagnostics.diameter,
                f"{diagnostics.avg_path_length:.4f}"
                if diagnostics.avg_path_length is not None else "-",
                result.repairs,
                result.rewire_swaps_used,
                connected,
                result.graph_sha256[:12],
            ]
        )
    record("golden:histogram+connected+edgecount", golden_ok)
    record("golden:kappa_4dp", kappa_ok)
    record("golden:distance_bands", distance_ok)

    determinism_ok = True
    for spec in specs[:2]:
        dataset = str(spec["dataset"])
        source = loaded.get(dataset)
        if source is None:
            continue
        plan = plan_degree_sequence(
            source.number_of_nodes(), source.number_of_edges(), c=c_value
        )
        first = build_bimodal_graph(plan, args.seed, dataset=dataset)
        second = build_bimodal_graph(plan, args.seed, dataset=dataset)
        if first.graph_sha256 != second.graph_sha256:
            determinism_ok = False
            record(f"determinism:{dataset}", False)
    record("determinism:same_seed_same_graph", determinism_ok)

    degenerate_ok = False
    detail = "no graph triggered a downgrade"
    for dataset, c_probe in (("Cogentco", 1.0), ("Colt", 1.0)):
        source = loaded.get(dataset)
        if source is None:
            continue
        plan = plan_degree_sequence(
            source.number_of_nodes(), source.number_of_edges(), c=c_probe
        )
        if plan.degenerate_fallback:
            degenerate_ok = plan.sum_degree == 2 * plan.m
            detail = f"{dataset} c={c_probe} -> {plan.degree_histogram()} Σ={plan.sum_degree}"
            break
    record("degenerate:rollback_conserves", degenerate_ok, detail)

    capped_ok = True
    for spec in specs:
        source = loaded.get(str(spec["dataset"]))
        if source is None:
            continue
        for c in grids:
            try:
                plan = plan_degree_sequence(
                    source.number_of_nodes(), source.number_of_edges(), c=c
                )
            except DegreeSequenceError as exc:
                LOGGER.warning("%s c=%s out of range, skipped: %s", spec["dataset"], c, exc)
                continue
            if plan.capped_at_n_minus_1 != (plan.k_max == plan.n - 1):
                capped_ok = False
                record(
                    f"capped_flag:{spec['dataset']}@c={c}", False,
                    f"k_max={plan.k_max} n-1={plan.n - 1} flag={plan.capped_at_n_minus_1}",
                )
    record("capped:flag_consistent_with_k_max", capped_ok)

    ba100 = loaded.get("BA-100")
    ba100_ok = ba100 is not None
    if ba100 is not None:
        for c in (20.0, 50.0):
            plan = plan_degree_sequence(ba100.number_of_nodes(), ba100.number_of_edges(), c=c)
            if not plan.capped_at_n_minus_1 or plan.k_max != plan.n - 1:
                ba100_ok = False
                record(f"capped:BA-100@c={c}", False, f"k_max={plan.k_max}")
    record("capped:BA-100@c>=20_touches_n_minus_1", ba100_ok)

    monotone_ok = True
    for spec in specs:
        source = loaded.get(str(spec["dataset"]))
        if source is None:
            continue
        previous = -1
        for c in grids:
            try:
                plan = plan_degree_sequence(
                    source.number_of_nodes(), source.number_of_edges(), c=c
                )
            except DegreeSequenceError as exc:
                LOGGER.warning("%s c=%s out of range, skipped: %s", spec["dataset"], c, exc)
                continue
            if plan.k_max < previous:
                monotone_ok = False
                record(f"monotone:{spec['dataset']}@c={c}", False)
            previous = plan.k_max
    record("monotone:k_max_nondecreasing_in_c", monotone_ok)

    capacity_ok = True
    for dataset, expected in _GOLDEN_CAPACITY.items():
        source = loaded.get(dataset)
        if source is None:
            continue
        model = capacity_model(source.number_of_nodes())
        observed = (
            model.n_controllers, model.capacity, model.n_switches,
            model.total_capacity, model.aggregate_slack,
        )
        if observed != expected:
            capacity_ok = False
            record(f"capacity:{dataset}", False, f"{observed} != {expected}")
    record("capacity:(n_c,C,total,slack)_per_graph", capacity_ok)

    probe_ok = True
    probe_summary: list[str] = []
    for dataset in _GOLDEN_CAPACITY:
        source = loaded.get(dataset)
        if source is None:
            continue
        plan = plan_degree_sequence(
            source.number_of_nodes(), source.number_of_edges(), c=c_value
        )
        result = build_bimodal_graph(plan, args.seed, dataset=dataset,
                                     realize=args.realize, rewire_swaps=args.rewire_swaps)
        model = capacity_model(result.graph.number_of_nodes())
        controllers = _controllers_for(result.graph, model.n_controllers)
        probe = probe_max_utilisation(result.graph, controllers, model)
        if sum(probe.loads_nearest) + probe.n_unassigned != model.n_switches:
            probe_ok = False
            record(f"probe:{dataset}", False, "sum L_j + unassigned != switch count")
        if probe.u_after_spill > 1.0 + 1e-9:
            probe_ok = False
            record(f"probe:{dataset}", False, "still overloaded after the spill")
        if probe.feasible and probe.n_unassigned:
            probe_ok = False
            record(f"probe:{dataset}", False, "aggregate feasible but nodes left unassigned")
        if probe.capacity_binds != (not probe.feasible):
            probe_ok = False
            record(f"probe:{dataset}", False, "capacity_binds disagrees with feasible")
        if not probe.ceiling_active:
            probe_ok = False
            record(f"probe:{dataset}", False, "no controller saturated after the spill, contradicts U_spill=1")
        if probe.ceiling_active == probe.capacity_binds:
            probe_ok = False
            record(f"probe:{dataset}", False,
                   "ceiling_active and capacity_binds are the same flag, two concepts conflated")
        if not probe.feasible:
            probe_ok = False
            record(f"probe:{dataset}", False,
                   f"infeasible at t=0: slack={model.aggregate_slack}")
        probe_summary.append(
            f"{dataset}=U_near {probe.u_nearest:.2f}/U_spill {probe.u_after_spill:.2f}"
            f"/{'feasible' if probe.feasible else 'infeasible'}"
            f"/saturated {probe.n_at_ceiling}"
        )
    record("probe:u_nearest_vs_u_after_spill_vs_feasible", probe_ok,
           "; ".join(probe_summary))

    cascade_ok = True
    for dataset in ("BA-100", "Cogentco", "Colt"):
        source = loaded.get(dataset)
        if source is None:
            continue
        plan = plan_degree_sequence(
            source.number_of_nodes(), source.number_of_edges(), c=c_value
        )
        result = build_bimodal_graph(plan, args.seed, dataset=dataset,
                                     realize=args.realize, rewire_swaps=args.rewire_swaps)
        model = capacity_model(result.graph.number_of_nodes())
        controllers = _controllers_for(result.graph, model.n_controllers)
        attack = sorted(result.graph.nodes(),
                        key=lambda v: (-result.graph.degree(v), v))
        for mode in ("structural", "capacity"):
            trace = run_cascade(result.graph, attack, controllers, model,
                                CascadeConfig(mode=mode))
            if not trace.audit_ok or len(trace.steps) != len(attack):
                cascade_ok = False
                record(f"cascade:{dataset}:{mode}", False)
            if abs(trace.final_fraction - 1.0) > 1e-12:
                cascade_ok = False
                record(f"cascade:{dataset}:{mode}", False, "the sequence never reaches f=1")
    record("cascade:audit_identity_every_step", cascade_ok)

    feasibility_ok = True
    feasibility_summary: list[str] = []
    for dataset in ("BA-100", "Colt"):
        source = loaded.get(dataset)
        if source is None:
            continue
        plan = plan_degree_sequence(
            source.number_of_nodes(), source.number_of_edges(), c=c_value
        )
        result = build_bimodal_graph(plan, args.seed, dataset=dataset,
                                     realize=args.realize, rewire_swaps=args.rewire_swaps)
        model = capacity_model(result.graph.number_of_nodes())
        controllers = controllers_degree_ranked(result.graph, model.n_controllers)
        for method in ("random", "degree", "controllers"):
            sequence = provisional_attack_sequence(
                result.graph, method, controllers=controllers, seed=args.seed
            )
            curve = feasibility_curve(
                result.graph, sequence, controllers, model,
                SimpleNamespace(mode="structural", max_iter=1),
                dataset=dataset, attack=method,
            )
            if not curve.points[0].feasible:
                feasibility_ok = False
                record(f"feasibility:{dataset}:{method}", False, "infeasible at t=0")
            ever_infeasible = any(
                not point.feasible for point in curve.points if point.non_trivial
            )
            if ever_infeasible and curve.first_persistent_infeasible_f is None:
                feasibility_ok = False
                record(
                    f"feasibility:{dataset}:{method}", False,
                    "infeasible states exist but no persistent onset is reported",
                )
            if method in ("degree", "controllers"):
                if curve.first_infeasible_f is None:
                    feasibility_ok = False
                    record(f"feasibility:{dataset}:{method}", False,
                           "targeted attack never becomes infeasible")
                elif curve.f_density_floor is not None and abs(
                    curve.first_infeasible_f - curve.f_density_floor
                ) > 1e-12:
                    feasibility_ok = False
                    record(
                        f"feasibility:{dataset}:{method}", False,
                        f"onset {curve.first_infeasible_f:.3f} disagrees with the density floor "
                        f"{curve.f_density_floor:.3f}",
                    )
            if method == "random" and curve.first_persistent_infeasible_f is not None:
                feasibility_ok = False
                record(
                    f"feasibility:{dataset}:{method}", False,
                    f"random attack is persistently infeasible from f={curve.first_persistent_infeasible_f:.3f}",
                )
            feasibility_summary.append(
                f"{dataset}/{method}:"
                f"{'—' if curve.first_persistent_infeasible_f is None else f'{curve.first_persistent_infeasible_f:.3f}'}"
            )
    record("feasibility:t0_feasible_and_onset_measurable", feasibility_ok,
           "first persistent infeasibility f: " + ", ".join(feasibility_summary))

    onset_ok = True
    onset_summary: list[str] = []
    structural_curves: list[Any] = []
    capacity_curves: list[Any] = []
    for dataset in _GOLDEN_CAPACITY:
        source = loaded.get(dataset)
        if source is None:
            continue
        plan = plan_degree_sequence(
            source.number_of_nodes(), source.number_of_edges(), c=c_value
        )
        result = build_bimodal_graph(plan, args.seed, dataset=dataset,
                                     realize=args.realize, rewire_swaps=args.rewire_swaps)
        model = capacity_model(result.graph.number_of_nodes())
        controllers = _controllers_for(result.graph, model.n_controllers)
        sequence = provisional_attack_sequence(
            result.graph, "degree", controllers=controllers, seed=args.seed
        )
        for mode, bucket in (("structural", structural_curves),
                             ("capacity", capacity_curves)):
            bucket.append(
                feasibility_curve(
                    result.graph, sequence, controllers, model,
                    SimpleNamespace(mode=mode, max_iter=50),
                    dataset=dataset, attack="degree",
                )
            )
    for onset in onset_records(structural_curves, capacity_curves):
        if not onset.feasible_at_t0:
            onset_ok = False
            record(f"onset:{onset.dataset}", False, "infeasible at t=0")
        if not (onset.onset_matches_floor and onset.onset_is_persistent):
            onset_ok = False
            record(f"onset:{onset.dataset}", False,
                   f"f*={onset.f_star} does not match the floor {onset.f_floor}, or is not persistent")
        if onset.extra_dead_at_onset <= 0:
            onset_ok = False
            record(f"onset:{onset.dataset}", False,
                   "the onset causes no extra death, so the capacity mechanism has no consequence")
        onset_summary.append(
            f"{onset.dataset}=f*{onset.f_star:.3f}"
            f"/+{onset.extra_dead_at_onset}→{onset.extra_dead_after_onset}"
        )
    record("onset:f_star_and_extra_death", onset_ok, "; ".join(onset_summary))


    sys.stdout.write(f"BiT-HyRL verify — c={c_value}, seed={args.seed}{_GOLDEN_MONO}\n\n")
    sys.stdout.write(_table(golden_header, golden_rows))
    sys.stdout.write("\n\n")
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        sys.stdout.write(
            f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}\n"
        )
    failures = [name for name, ok, _ in checks if not ok]
    sys.stdout.write("\n")
    if failures:
        sys.stdout.write(f"IS_PASS: NO - failed checks {failures}\n")
        return 1
    sys.stdout.write("IS_PASS: YES - all hard checks passed\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser for ``python -m bit_hyrl``."""
    parser = argparse.ArgumentParser(
        prog="bit_hyrl",
        description="BiT-HyRL command line entry point (BiT / Hy / RL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"bit_hyrl {VERSION}")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="log level",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    p_fp = subparsers.add_parser("fingerprint", help="read every pinned graph and print its fingerprint")
    p_fp.add_argument("--graphs", default=None, help="graph registry JSON (default configs/graphs.json)")
    p_fp.add_argument("--write", action="store_true", help="write to results/fingerprints/")
    p_fp.add_argument("--quiet", action="store_true", help="do not print JSON")
    p_fp.set_defaults(func=cmd_fingerprint)

    p_construct = subparsers.add_parser("construct", help="construct one BiT graph and print its diagnostics")
    p_construct.add_argument("--dataset", required=True, help="dataset name, e.g. BA-100")
    p_construct.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_construct.add_argument("--h", type=int, default=1, help="number of hubs")
    p_construct.add_argument("--seed", type=int, default=0, help="root seed")
    p_construct.add_argument(
        "--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
        help="realisation mode: hh+rewire (default) or hh (plain Havel-Hakimi)",
    )
    p_construct.add_argument(
        "--rewire-swaps", type=int, default=None,
        help="2-switch budget for decorrelation (default 10*m; 0 means --realize hh)",
    )
    p_construct.add_argument("--betweenness", action="store_true", help="also compute the betweenness diagnostic")
    p_construct.add_argument("--max-repair-swaps", type=int, default=20000, help="swap budget for connectivity repair")
    p_construct.add_argument("--allow-disconnected", action="store_true", help="allow a disconnected result")
    p_construct.add_argument("--write", action="store_true", help="write to results/construct/")
    p_construct.add_argument("--quiet", action="store_true", help="do not print JSON")
    p_construct.set_defaults(func=cmd_construct)

    p_probe = subparsers.add_parser(
        "probe", help="capacity model and utilisation probe (nothing is removed)"
    )
    p_probe.add_argument("--dataset", required=True, help="dataset name, e.g. BA-100")
    p_probe.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_probe.add_argument("--h", type=int, default=1, help="number of hubs")
    p_probe.add_argument("--seed", type=int, default=0, help="root seed")
    p_probe.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                         help="realisation mode")
    p_probe.add_argument("--rewire-swaps", type=int, default=None,
                         help="2-switch budget for decorrelation (default 10*m)")
    p_probe.add_argument("--write", action="store_true", help="write to results/probe/")
    p_probe.add_argument("--quiet", action="store_true", help="do not print JSON")
    p_probe.set_defaults(func=cmd_probe)

    p_sweep = subparsers.add_parser("sweep-c", help="sensitivity sweep over c")
    p_sweep.add_argument("--dataset", default="all", help="dataset name or all")
    p_sweep.add_argument("--grid", default=None,
                         help="None for the default grid, or a comma-separated list read according to --grid-kind")
    p_sweep.add_argument(
        "--grid-kind", default="c", choices=("c", "c_norm"),
        help="c: twelve values of the raw c; c_norm: eleven values normalised by c*",
    )
    p_sweep.add_argument("--h", type=int, default=1, help="number of hubs")
    p_sweep.add_argument("--seed", type=int, default=0, help="root seed")
    p_sweep.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                         help="realisation mode")
    p_sweep.add_argument("--rewire-swaps", type=int, default=None,
                         help="2-switch budget for decorrelation (default 10*m)")
    p_sweep.add_argument("--plan-only", action="store_true", help="solve the degree sequence only, no graph")
    p_sweep.add_argument("--write", action="store_true", help="write to results/sweep/<dataset>/")
    p_sweep.add_argument("--quiet", action="store_true", help="do not print JSON")
    p_sweep.set_defaults(func=cmd_sweep_c)

    p_feas = subparsers.add_parser(
        "feasibility", help="capacity feasibility along every attack sequence"
    )
    p_feas.add_argument("--dataset", default="all", help="dataset name or all")
    p_feas.add_argument("--attack", default="all",
                        help=f"attack method or all; one of {list(PROVISIONAL_ATTACK_METHODS)}")
    p_feas.add_argument("--mode", default="structural", choices=("structural", "capacity"),
                        help="cascade mode (default structural, as published)")
    p_feas.add_argument("--max-iter", type=int, default=50, help="fixed-point cap for capacity mode")
    p_feas.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_feas.add_argument("--h", type=int, default=1, help="number of hubs")
    p_feas.add_argument("--seed", type=int, default=0, help="root seed")
    p_feas.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                        help="realisation mode")
    p_feas.add_argument("--rewire-swaps", type=int, default=None,
                        help="2-switch budget for decorrelation (default 10*m)")
    p_feas.add_argument("--write", action="store_true", help="write to results/feasibility/")
    p_feas.add_argument("--quiet", action="store_true", help="do not print the tables")
    p_feas.set_defaults(func=cmd_feasibility)

    p_verify = subparsers.add_parser("verify", help="run every hard check in one pass (pinned values and invariants)")
    p_verify.add_argument("--all", action="store_true", help="kept for compatibility; verify always runs everything")
    p_verify.add_argument("--c", type=float, default=1.6, help="c used for the pinned comparison")
    p_verify.add_argument("--seed", type=int, default=0, help="root seed")
    p_verify.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                          help="realisation mode")
    p_verify.add_argument("--rewire-swaps", type=int, default=None,
                          help="2-switch budget for decorrelation (default 10*m)")
    p_verify.set_defaults(func=cmd_verify)

    p_eval = subparsers.add_parser(
        "eval", help="evaluate the original and reconstructed topologies"
    )
    p_eval.add_argument("--dataset", default="all", help="dataset name or all")
    p_eval.add_argument(
        "--topology", default="both", choices=("original", "bit", "both"),
        help="topology source: original (load), bit (rebuild from N, M) or both",
    )
    p_eval.add_argument("--c", type=float, default=1.6, help="hub factor c (bit topology only)")
    p_eval.add_argument("--h", type=int, default=1, help="number of hubs")
    p_eval.add_argument("--seed", type=int, default=0, help="root seed")
    p_eval.add_argument(
        "--attack", default="degree",
        help=f"attack method or all; one of {list(PROVISIONAL_ATTACK_METHODS)}",
    )
    p_eval.add_argument("--mode", default="structural", choices=("structural", "capacity"),
                        help="cascade mode (default structural, as published)")
    p_eval.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                        help="realisation mode of the bit topology")
    p_eval.add_argument("--rewire-swaps", type=int, default=None,
                        help="2-switch budget for decorrelation (default 10*m)")
    p_eval.add_argument("--write", action="store_true", help="write to results/records/")
    p_eval.add_argument(
        "--policy", default="degree_ranked", choices=("degree_ranked", "ppo"),
        help="placement policy: degree_ranked (baseline) or ppo (needs --checkpoint)",
    )
    p_eval.add_argument("--checkpoint", default=None, help="PPO checkpoint path (required with --policy ppo)")
    p_eval.add_argument("--quiet", action="store_true", help="do not print the reconciliation lines")
    p_eval.set_defaults(func=cmd_eval)

    p_train = subparsers.add_parser(
        "train", help="train the PPO placement policy"
    )
    p_train.add_argument(
        "--dataset", default="all", help="dataset name, comma-separated list, or all"
    )
    p_train.add_argument("--split", default=None,
                         help="train on the train split and early-stop on val (e.g. configs/splits.json)")
    p_train.add_argument("--iterations", type=int, default=200, help="outer iterations")
    p_train.add_argument("--episodes", type=int, default=8, help="rollout episodes per iteration")
    p_train.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    p_train.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    p_train.add_argument("--patience", type=int, default=20, help="early-stop patience on val (0 disables)")
    p_train.add_argument("--eval-sample", type=int, default=20, help="size of the val evaluation subset")
    p_train.add_argument("--seed", type=int, default=0, help="root seed")
    p_train.add_argument("--out", default=None, help="checkpoint output path (default results/checkpoints/)")
    p_train.add_argument(
        "--smoke", action="store_true", help="smoke run: at most 2 iterations, 2 episodes, 4 graphs per split"
    )
    p_train.add_argument("--quiet", action="store_true", help="do not print the reward per iteration")
    p_train.set_defaults(func=cmd_train)

    p_transfer = subparsers.add_parser(
        "transfer", help="zero-shot transfer: place controllers on a new graph"
    )
    p_transfer.add_argument("--dataset", required=True, help="target dataset name, e.g. ER-100")
    p_transfer.add_argument("--checkpoint", required=True, help="trained checkpoint path")
    p_transfer.add_argument("--seed", type=int, default=0, help="root seed (kept for compatibility)")
    p_transfer.add_argument("--out", default=None, help="write the result JSON to this path")
    p_transfer.add_argument("--quiet", action="store_true", help="do not print the controller set")
    p_transfer.set_defaults(func=cmd_transfer)

    p_ablate = subparsers.add_parser("ablate", help="2x2 ablation: topology x placement")
    p_ablate.add_argument("--dataset", default="all", help="dataset name, comma-separated list, or all")
    p_ablate.add_argument("--checkpoint", default=None, help="PPO checkpoint (needed for ppo placement)")
    p_ablate.add_argument("--attacks", default="degree,random", help="comma-separated attack methods")
    p_ablate.add_argument("--mode", default="structural", choices=("structural", "capacity"))
    p_ablate.add_argument("--seeds", default="0", help="comma-separated seeds")
    p_ablate.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_ablate.add_argument("--h", type=int, default=1, help="number of hubs")
    p_ablate.add_argument("--out", default=None, help="output directory (default results/ablation)")
    p_ablate.add_argument("--smoke", action="store_true", help="smoke run: one seed only")
    p_ablate.add_argument("--quiet", action="store_true", help="do not print the 2x2 table")
    p_ablate.set_defaults(func=cmd_ablate)

    p_matrix = subparsers.add_parser("matrix", help="long-form experiment matrix")
    p_matrix.add_argument("--dataset", default="all", help="dataset name, comma-separated list, or all")
    p_matrix.add_argument("--policies", default="degree_ranked,ppo", help="comma-separated policies")
    p_matrix.add_argument("--topologies", default="original,bit", help="comma-separated topologies")
    p_matrix.add_argument("--attacks", default="degree,random", help="comma-separated attacks")
    p_matrix.add_argument("--modes", default="structural", help="comma-separated cascade modes")
    p_matrix.add_argument("--seeds", default="0", help="comma-separated seeds")
    p_matrix.add_argument("--checkpoint", default=None, help="PPO checkpoint")
    p_matrix.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_matrix.add_argument("--h", type=int, default=1, help="number of hubs")
    p_matrix.add_argument("--out", default=None, help="output directory (default results/matrix)")
    p_matrix.add_argument("--smoke", action="store_true", help="smoke run: one seed only")
    p_matrix.add_argument("--quiet", action="store_true", help="do not print the row count")
    p_matrix.set_defaults(func=cmd_matrix)

    p_mm = subparsers.add_parser(
        "method-matrix", help="method-level matrix: five construction baselines, Eff-R-Net, two placement proxies and BiT-HyRL"
    )
    p_mm.add_argument("--dataset", default="all", help="dataset name, comma-separated list, or all")
    p_mm.add_argument("--split", default=None, help="use the test list of splits.json")
    p_mm.add_argument("--methods", default="", help="comma-separated method names; empty means all")
    p_mm.add_argument("--attacks", default="degree,random", help="comma-separated attacks")
    p_mm.add_argument("--mode", default="structural", choices=("structural", "capacity"))
    p_mm.add_argument("--seeds", default="0", help="comma-separated seeds")
    p_mm.add_argument("--checkpoint", default=None, help="PPO checkpoint (needed for the BiT-HyRL placement)")
    p_mm.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_mm.add_argument("--h", type=int, default=1, help="number of hubs")
    p_mm.add_argument(
        "--workers", type=int, default=1,
        help="worker processes; 1 runs serially",
    )
    p_mm.add_argument("--out", default=None, help="output directory (default results/method_matrix)")
    p_mm.add_argument(
        "--smoke", action="store_true",
        help="smoke run: two graphs, five methods, one attack, one seed",
    )
    p_mm.add_argument("--quiet", action="store_true", help="do not print the row count")
    p_mm.set_defaults(func=cmd_method_matrix)

    p_ms = subparsers.add_parser(
        "method-stats", help="significance between methods: exact Wilcoxon with Holm correction"
    )
    p_ms.add_argument("--in", dest="infile", required=True, help="path to the method_matrix CSV")
    p_ms.add_argument("--reference", default="BiT-HyRL", help="reference method (default BiT-HyRL)")
    p_ms.add_argument("--out", default=None, help="output CSV (default results/method_matrix/significance.csv)")
    p_ms.add_argument("--quiet", action="store_true", help="do not print")
    p_ms.set_defaults(func=cmd_method_stats)

    p_stats = subparsers.add_parser("stats", help="paired statistics: Wilcoxon and bootstrap CI")
    p_stats.add_argument("--in", dest="infile", required=True, help="path to the matrix CSV")
    p_stats.add_argument("--metric", default="auc_r", help="metric column to compare")
    p_stats.add_argument("--out", default=None, help="output directory (default results/stats)")
    p_stats.add_argument("--quiet", action="store_true", help="do not print")
    p_stats.set_defaults(func=cmd_stats)

    p_figures = subparsers.add_parser("figures", help="draw the robustness, ablation and metric panels")
    p_figures.add_argument("--in", dest="infile", required=True, help="path to the matrix CSV")
    p_figures.add_argument("--out", default=None, help="output directory (default results/figures)")
    p_figures.add_argument("--smoke", action="store_true", help="kept for compatibility")
    p_figures.add_argument("--quiet", action="store_true", help="do not print the file list")
    p_figures.set_defaults(func=cmd_figures)

    for name, owner in _DEFERRED.items():
        stub = subparsers.add_parser(name, help=f"not implemented yet ({owner})")
        stub.set_defaults(func=_make_stub(name, owner))

    return parser


def _make_stub(name: str, owner: str):
    """Create a deferred-command handler that fails loudly."""

    def _stub(_args: argparse.Namespace) -> int:
        raise NotImplementedError(f"sub-command {name!r} is not implemented yet ({owner})")

    _stub.__name__ = f"cmd_{name}_deferred"
    return _stub


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python -m bit_hyrl``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except NotImplementedError as exc:
        LOGGER.error("%s", exc)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
