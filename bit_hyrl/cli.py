"""Single command-line entry point.

Implemented in this batch:

* ``fingerprint`` — tolerant load + fingerprint of every pinned graph;
* ``construct``   — one BiT construction with its plan and diagnostics;
* ``sweep-c``     — the ``c`` sensitivity sweep;
* ``verify``      — every hard numeric check, in one pass.

The remaining sub-commands exist so that ``--help`` is the full contract, and
raise :class:`NotImplementedError` naming the task that will implement them.  A
stub that fails loudly is honest; a stub that silently returns zeros is not.
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

#: Sub-commands deferred to later tasks.  ``--help`` shows them all; calling one
#: raises immediately with the owning task id.
#: Sub-commands still deferred.  Empty: T11 (ablate) and T12 (matrix/stats/figures)
#: are implemented by :mod:`bit_hyrl.analysis` and wired below.
_DEFERRED: dict[str, str] = {}

#: Paper reward weights ``(w_G, w_S, w_H, w_D)`` — distance disabled.
_PAPER_REWARD_WEIGHTS = {"structure": 0.6, "csa": 0.2, "h_load": 0.2, "distance": 0.0}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_graph_registry(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load ``configs/graphs.json`` and resolve relative paths.

    Args:
        path: Override for the default registry location.

    Returns:
        List of graph specs, each with an absolute ``path``.

    Raises:
        FileNotFoundError: If the registry is missing.
    """
    registry_path = Path(path) if path is not None else io_paths.GRAPHS_CONFIG_FILE
    if not registry_path.is_file():
        raise FileNotFoundError(f"找不到图清单 {registry_path}")
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
    """Return the registry entry for ``dataset``.

    Raises:
        KeyError: If the dataset is not in the registry.
    """
    for spec in _load_graph_registry():
        if spec.get("dataset") == dataset:
            return spec
    known = [spec.get("dataset") for spec in _load_graph_registry()]
    raise KeyError(f"未知数据集 {dataset!r}，清单里有 {known}")


def _parse_grid(value: str) -> list[float]:
    """Parse ``--grid``: either ``default`` or a comma-separated list."""
    if value.strip().lower() == "default":
        return [float(c) for c in DEFAULT_C_GRID]
    parts = [part for part in value.replace(";", ",").split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--grid 不能为空")
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


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #
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
        LOGGER.info("指纹已写入 %s", target)
    if not args.quiet:
        _print_json(payload)
    if failures:
        LOGGER.error("共 %d 张图与金标准不符", len(failures))
        return 1
    return 0


# --------------------------------------------------------------------------- #
# construct
# --------------------------------------------------------------------------- #
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
        LOGGER.info("构造结果已写入 %s", target)
    if not args.quiet:
        _print_json(payload)
    return 0 if result.connected or args.allow_disconnected else 1


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def _controllers_for(graph: nx.Graph, count: int) -> list[int]:
    """Degree-ranked controller placement with a node-id tie-break.

    The real placement policies arrive with the RL layer; ``probe`` needs a
    deterministic baseline so that "does θ bind?" can be answered before any
    policy exists.
    """
    order = sorted(graph.nodes(), key=lambda node: (-graph.degree(node), node))
    return [int(node) for node in order[:count]]


def cmd_probe(args: argparse.Namespace) -> int:
    """Capacity model + ``probe_max_utilisation`` for one graph (no deletion).

    The probe reports three separate quantities because they answer three
    different questions:

    * ``u_nearest`` — how lopsided the degree-ranked placement is if capacity is
      ignored.  Measured 1.13–2.73 on the nine graphs;
    * ``u_after_spill`` — the utilisation that actually occurs once the
      capacity-enforced spill is applied.  Measured **1.000** everywhere, with no
      unassigned switch;
    * ``feasible`` — ``Σ demand <= Σ capacity`` per component.  **True** on all
      nine, which is the only sound basis for "capacity does not bind at t = 0".

    ``ceiling_active`` and ``capacity_binds`` answer two different questions and
    must not be merged: ``ceiling_active`` is ``True`` on all nine graphs (some
    controller is exactly at ``C`` — the ceiling is *live*), while
    ``capacity_binds`` is ``False`` on all nine (the demand is *satisfiable*).
    A known aggregate slack of 60…300 can never make ``capacity_binds`` true at
    ``t = 0``.
    """
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
        LOGGER.info("探针结果已写入 %s", target)
    if not args.quiet:
        _print_json(payload)
    return 0


# --------------------------------------------------------------------------- #
# feasibility
# --------------------------------------------------------------------------- #
def cmd_feasibility(args: argparse.Namespace) -> int:
    """Capacity feasibility along every attack sequence, per graph.

    This is the evidence: at ``t = 0`` the constraint is slack on all nine
    graphs, so the publishable statement is *when* it stops being slack — ``f*``
    — and how much extra failure it then causes.  Both are produced here:

    * the per-mode curves (``f / k / Σdemand / Σcapacity / feasible`` per step);
    * the **onset table**, which pairs each ``structural`` curve with the
      ``capacity`` curve over the identical attack sequence and reports
      ``f*`` plus ``extra_dead_after_onset`` (nodes the overload cascade kills
      that ``structural`` mode would not have killed).

    ``--mode`` still selects which curve the *primary* artefact is written for,
    but both modes are always run, because ``f*`` without the extra-death column
    would be a threshold with no consequence attached to it.
    """
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
            # Structural and capacity share the sequence *instance* above, so the
            # step counts necessarily agree and the pairing cannot silently break.
            if len(curves["structural"]) != len(curves["capacity"]):
                raise RuntimeError("structural 与 capacity 曲线数量不一致")

    records = onset_records(curves["structural"], curves["capacity"])
    table = format_onset_table(records)
    if args.write:
        written = write_onset_outputs(records, name="onset")
        LOGGER.info("首次不可行统计已写盘 %s / %s", written["json"], written["txt"])
        for mode in ("structural", "capacity"):
            curves_written = write_feasibility_outputs(
                curves[mode], name=f"feasibility__{mode}"
            )
            LOGGER.info(
                "%s 可行性曲线已写盘 %s / %s",
                mode, curves_written["json"], curves_written["txt"],
            )
    if not args.quiet:
        print(table)
        if args.mode != "structural":
            print(format_feasibility_table(curves[args.mode]))
    return 0


# --------------------------------------------------------------------------- #
# sweep-c
# --------------------------------------------------------------------------- #
def cmd_sweep_c(args: argparse.Namespace) -> int:
    """Run the ``c`` sweep and optionally persist JSON + CSV.

    ``--grid-kind c`` (default) sweeps raw ``c`` on the design's 12 points;
    ``--grid-kind c_norm`` sweeps ``c_norm`` on the 11 normalised points, which is
    the controlled comparison — the same ``c`` means a different shape on every
    graph, so only the normalised grid puts all nine at the same position on one
    shape spectrum.
    """
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
            LOGGER.info("%s: 写盘 %s / %s", dataset, written["json"], written["csv"])
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
        LOGGER.info("已写盘 %d 个数据集的 sweep", len(everything))
    return 0


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
_GOLDEN_MONO = "（金标准见 configs/graphs.json）"

#: κ at four decimals for the nine constructed graphs (c = 1.6, h = 1).  κ is a
#: function of the degree sequence alone, so it is invariant under ``realize``.
_GOLDEN_KAPPA_4DP: dict[str, float] = {
    "BA-100": 5.8660, "BA-200": 5.9340, "BA-300": 5.9607, "BA-400": 5.9706,
    "BA-500": 5.9765, "Cogentco": 2.5720, "Colt": 2.4124, "GtsCe": 2.6891,
    "UsCarrier": 2.4974,
}

#: ``(max diameter, max mean path)`` per graph — the *ensemble* band, i.e. the only
#: distance claim the paper may make.  A single realisation scatters by ±1 on the
#: diameter and by up to 5 % on the mean path, so a point value here would be a
#: false precision.
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


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def _expand_attacks(attack: str) -> list[str]:
    """Resolve the ``--attack`` CLI value into a concrete method list."""
    if attack == "all":
        return list(PROVISIONAL_ATTACK_METHODS)
    return [attack]


def cmd_eval(args: argparse.Namespace) -> int:
    """End-to-end evaluation runner: original vs BiT dual curves.

    For every selected graph it runs the pipeline under each requested topology and
    persists one :class:`~bit_hyrl.evaluation.record.EvaluationRecord` per
    ``(topology, attack, mode)`` into ``results/records/``.  With
    ``--topology both`` it also prints a one-line comparison of the robustness area
    (AUC of the ``r_curve``) between the original substrate and the BiT
    reconstruction — the core evidence for interpretation A.
    """
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
            LOGGER.error("--policy ppo 需要 --checkpoint")
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
        header = f"{'图':<10} {'攻击':<12} {'orig AUC':>9} {'bit AUC':>9}  不同"
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
                        f"{bit.auc_r:>9.4f}  {'是' if differ else '否'}"
                    )
        if not args.quiet:
            print(
                f"\n{len(records)} 条记录已写盘 {records_dir()}；"
                f"original 与 bit 的 AUC 不同的图×攻击 = {differing}"
            )

    LOGGER.info("eval 完成：%d 条 EvaluationRecord → %s", len(records), records_dir())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a PPO controller-placement policy (paper Table-4 budget).

    Two modes: ``--split configs/splits.json`` trains on the split's ``train``
    list and early-stops on its ``val`` list; otherwise ``--dataset`` names
    registry graphs (legacy).  All hyper-parameters use the paper's Table-4 values.
    """
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
        lr=float(args.lr),          # paper Table 4: 1e-4
        gamma=float(args.gamma),    # paper Table 4: 0.99
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

    # History CSV (train + val reward per iteration).
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
        print(f"训练完成：{len(history)} 迭代 × {episodes} episode/迭代，耗时 {elapsed:.2f}s")
        if val_history:
            print(f"val：{len(val_history)} 点，best_val={checkpoint.get('best_val')}，"
                  f"stopped_iter={checkpoint.get('stopped_iteration')}")
        print(f"最终 train 奖励 {final:.4f}；checkpoint → {out}")
        print(f"history CSV → {hist_path}")
        print(
            "论文表4超参：lr=%g γ=%g clip=%g gae=%g ent=%g epochs=%d rollouts/iter=%d "
            "minibatch=32 grad_clip=0.5 value_coef=0.5；奖励权重 0.6/0.2/0.2/0"
            % (cfg.lr, cfg.gamma, cfg.clip, cfg.gae_lambda, cfg.entropy_coef, cfg.epochs, episodes)
        )
    LOGGER.info("train 完成：%s → %s（%.2fs）", label, out, elapsed)
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
        print(f"{args.dataset}: 零样本放置 {len(controllers)}/{model.n_controllers} 个控制器")
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
        LOGGER.info("transfer 结果已写 %s", out)
    LOGGER.info("transfer 完成：%s ← %s", args.dataset, args.checkpoint)
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
        header = "拓扑/放置"
        print(f"{header:<16}{'degree_ranked':>15}{'ppo':>12}")
        for topology in ("original", "bit"):
            deg = cells.get(f"{topology}|degree_ranked", float("nan"))
            ppo = cells.get(f"{topology}|ppo", float("nan"))
            print(f"{topology:<16}{deg:>15.4f}{ppo:>12.4f}")
        effects = result["effects"]
        print(
            f"主效应: topology={effects['topology']:+.4f}  "
            f"policy={effects['policy']:+.4f}  interaction={effects['interaction']:+.6f}"
        )
        print(f"CSV → {paths['csv']}\nJSON → {paths['json']}")
    LOGGER.info("ablate 完成：%d 行", len(result.get("rows", [])))
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
        print(f"matrix: {len(frame)} 行 → {paths['csv']}")
    LOGGER.info("matrix 完成：%d 行", len(frame))
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
        print(f"method-matrix: {len(frame)} 行 → {paths['csv']}")
        for name in methods:
            subset = frame[frame["policy"] == name]
            if len(subset):
                row = subset.iloc[0]
                print(
                    f"  {name:<10} policy={row['policy']:<10} "
                    f"topology={row['topology']:<9} r_res={row['r_res']:.4f} "
                    f"auc_gcc={row['auc_gcc']:.4f}"
                )
    LOGGER.info("method-matrix 完成：%d 行", len(frame))
    return 0


def cmd_method_stats(args: argparse.Namespace) -> int:
    """Method-vs-reference significance (exact Wilcoxon + Holm over GCC/CSA/H_C)."""
    import pandas as pd

    from .analysis.stats import method_significance

    frame = pd.read_csv(args.infile)
    if "policy" not in frame.columns:
        LOGGER.error("输入 CSV 缺 policy 列：%s", list(frame.columns))
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
    LOGGER.info("method-stats 完成：%s（reference=%s）", out, args.reference)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Seed-aggregated mean/std/CI + Wilcoxon-vs-reference with Holm (T12 upgrade)."""
    import pandas as pd

    from .analysis.stats import seed_statistics, significance, write_metrics_outputs

    frame = pd.read_csv(args.infile)
    if args.metric not in frame.columns:
        LOGGER.error("metric %r 不在列 %s", args.metric, list(frame.columns))
        return 2

    seed_df = seed_statistics(frame)
    # Holm family: the three core robustness metrics, plus the requested metric if
    # it is not one of them (so `--metric r_res` still reports a Holm-adjusted p).
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
        print(f"== {args.metric}: seed 聚合 mean/std/95%CI ==")
        for _, row in sub.iterrows():
            print(
                f"  {row['dataset']}/{row['topology']}/{row['policy']}/{row['attack']}: "
                f"mean={row['mean']:+.4f} std={row['std']:.4f} "
                f"CI95=[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
            )
        print("== 显著性：Wilcoxon signed-rank vs 参照 + Holm 校正 ==")
        for row in sig_rows:
            mean_diff = row.get("mean_diff")
            mean_txt = "n/a" if mean_diff is None else f"{mean_diff:+.4f}"
            print(
                f"  {row['contrast']:>16} {row['metric']:<9} n={row['n']} "
                f"Δ={mean_txt} p={row['wilcoxon_p']} holm={row['holm_p']:.4g} "
                f"sig={row['significant']}"
            )
        print(f"→ {paths['seed_csv']}\n→ {paths['significance_csv']}")
    LOGGER.info("stats 完成")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    """Render figures (robustness / ablation / F10) from a matrix CSV (T12)."""
    import pandas as pd

    from .analysis.figures import make_all_figures

    frame = pd.read_csv(args.infile)
    out_dir = Path(args.out) if args.out else (io_paths.RESULTS_ROOT / "figures")
    figures = make_all_figures(frame, out_dir=out_dir, smoke=args.smoke)
    if not args.quiet:
        print(f"生成 {len(figures)} 张图 → {out_dir}")
        for item in figures:
            print(f"  {item['name']}: png={item['size_png']}B  pdf={item['pdf']}")
    LOGGER.info("figures 完成：%d 张", len(figures))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Run every hard numeric check in one pass and print the nine-graph table."""
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    specs = _load_graph_registry()
    grids = [float(c) for c in DEFAULT_C_GRID]

    # ---- 1. fingerprints vs the pinned (N, M, duplicates) ------------------ #
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
    record("fingerprint:all", fp_ok, f"{len(loaded)}/{len(specs)} 张图通过")

    # ---- 2. idempotent normalisation -------------------------------------- #
    idempotent = True
    for spec in specs:
        graph = load_graph(spec["path"])
        if graph_fingerprint(graph)["sha256"] != graph_fingerprint(load_graph(spec["path"]))["sha256"]:
            record(f"idempotent:{spec['dataset']}", False)
            idempotent = False
    record("idempotent:all", idempotent)

    # ---- 3. Σk == 2M over 9 graphs × 12 c values --------------------------- #
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
    record("conservation:Σk=2M", conservation_ok, f"{combos} 组组合")

    # ---- 4. nine-graph golden table at c = 1.6 ---------------------------- #
    c_value = args.c
    golden_header = (
        "图", "N", "M", "c*", "c_norm", "k_max", "k_low", "r", "度值集合",
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
                f"实测 {diagnostics.kappa:.4f} != 金标准 {expected_kappa:.4f}",
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

    # ---- 5. determinism: rebuild with the same seed ------------------------ #
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

    # ---- 6. degenerate rollback ------------------------------------------- #
    degenerate_ok = False
    detail = "未找到可触发退化的图"
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

    # ---- 7. the n-1 ceiling is *flagged*, never silently truncated --------- #
    # The flag must agree with the arithmetic for every (graph, c) pair.  Note
    # that "c >= 20 implies capped" is measurable only on BA-100 (k_bar = 5.82,
    # so 20 * 5.82 = 116 > 99); on the other eight graphs c = 50 still stays
    # below n - 1 (e.g. BA-300: 50 * 5.94 = 297 < 299).  Asserting the
    # implication globally would be wrong, so the check is (a) flag/arith
    # consistency everywhere and (b) BA-100 capped at c in {20, 50}.
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
                # An inadmissible (graph, c) pair (e.g. the N=100 degenerate
                # boundary) must not abort the whole verification sweep.
                LOGGER.warning("%s c=%s 越界跳过：%s", spec["dataset"], c, exc)
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

    # ---- 8. c monotonicity ------------------------------------------------- #
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
                LOGGER.warning("%s c=%s 越界跳过：%s", spec["dataset"], c, exc)
                continue
            if plan.k_max < previous:
                monotone_ok = False
                record(f"monotone:{spec['dataset']}@c={c}", False)
            previous = plan.k_max
    record("monotone:k_max_nondecreasing_in_c", monotone_ok)

    # ---- 9. capacity contract: per graph, never just BA-100 ---------- #
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

    # ---- 10. the utilisation probe: three quantities, none conflated ------ #
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
            record(f"probe:{dataset}", False, "Σ L_j + 未指派 != 交换节点数")
        # 硬要求：指派后的利用率必须 <= 1；聚合可行时不允许出现顺延失败。
        if probe.u_after_spill > 1.0 + 1e-9:
            probe_ok = False
            record(f"probe:{dataset}", False, "顺延后仍超载")
        if probe.feasible and probe.n_unassigned:
            probe_ok = False
            record(f"probe:{dataset}", False, "聚合可行却有节点未指派")
        if probe.capacity_binds != (not probe.feasible):
            probe_ok = False
            record(f"probe:{dataset}", False, "capacity_binds 与 feasible 不一致")
        # 两个概念必须分开：天花板活跃（有控制器正好满载）在九图全为 True，
        # 而"容量绑定"（不可满足）在九图全为 False。合并二者正是口径错误的根源。
        if not probe.ceiling_active:
            probe_ok = False
            record(f"probe:{dataset}", False, "顺延后无控制器满载，与 U_spill=1 矛盾")
        if probe.ceiling_active == probe.capacity_binds:
            probe_ok = False
            record(f"probe:{dataset}", False,
                   "ceiling_active 与 capacity_binds 同值，两个概念被混为一谈")
        # 九图在 t=0 必须聚合可行（slack >= 60，这是可证的，不是观测巧合）。
        if not probe.feasible:
            probe_ok = False
            record(f"probe:{dataset}", False,
                   f"t=0 却不可行：slack={model.aggregate_slack}")
        probe_summary.append(
            f"{dataset}=U_near {probe.u_nearest:.2f}/U_spill {probe.u_after_spill:.2f}"
            f"/{'可行' if probe.feasible else '不可行'}"
            f"/满载{probe.n_at_ceiling}"
        )
    record("probe:u_nearest_vs_u_after_spill_vs_feasible", probe_ok,
           "; ".join(probe_summary))

    # ---- 11. cascade audit identity --------------------------------- #
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
                record(f"cascade:{dataset}:{mode}", False, "全序列未打到 f=1")
    record("cascade:audit_identity_每步成立", cascade_ok)

    # ---- 12. capacity feasibility over the attack ------------------ #
    # t=0 必须聚合可行；否则"容量是否绑定"这个问题本身就被问错了。
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
                record(f"feasibility:{dataset}:{method}", False, "t=0 却不可行")
            # 有不可行状态 ⇒ 必须能报到"首次持续不可行"，否则统计口径有洞。
            ever_infeasible = any(
                not point.feasible for point in curve.points if point.non_trivial
            )
            if ever_infeasible and curve.first_persistent_infeasible_f is None:
                feasibility_ok = False
                record(
                    f"feasibility:{dataset}:{method}", False,
                    "存在不可行状态却报不出首次持续不可行",
                )
            # 定向攻击：密度地板与实测起跳点必须落在同一档（理论即机制）。
            if method in ("degree", "controllers"):
                if curve.first_infeasible_f is None:
                    feasibility_ok = False
                    record(f"feasibility:{dataset}:{method}", False,
                           "定向攻击却从未不可行")
                elif curve.f_density_floor is not None and abs(
                    curve.first_infeasible_f - curve.f_density_floor
                ) > 1e-12:
                    feasibility_ok = False
                    record(
                        f"feasibility:{dataset}:{method}", False,
                        f"起跳点 {curve.first_infeasible_f:.3f} 与密度地板 "
                        f"{curve.f_density_floor:.3f} 不一致",
                    )
            # 随机攻击：控制器与被攻击节点同比例消失，密度守在 1/(C+1) 之上，
            # 因此不允许出现"持续不可行"。
            if method == "random" and curve.first_persistent_infeasible_f is not None:
                feasibility_ok = False
                record(
                    f"feasibility:{dataset}:{method}", False,
                    f"随机攻击竟在 f={curve.first_persistent_infeasible_f:.3f} 持续不可行",
                )
            feasibility_summary.append(
                f"{dataset}/{method}:"
                f"{'—' if curve.first_persistent_infeasible_f is None else f'{curve.first_persistent_infeasible_f:.3f}'}"
            )
    record("feasibility:t0_可行且首次不可行可测", feasibility_ok,
           "首次持续不可行 f: " + ", ".join(feasibility_summary))

    # ---- 13. the onset table: f* and the extra death it causes ----- #
    # f* alone is a threshold with no consequence attached; the record pairs it
    # with the extra death measured against `structural` mode on the same
    # sequence.  Both are produced by `onset_records`, the same function the
    # `feasibility` subcommand writes to `results/onset.{json,txt}`.
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
            record(f"onset:{onset.dataset}", False, "t=0 却不可行")
        if not (onset.onset_matches_floor and onset.onset_is_persistent):
            onset_ok = False
            record(f"onset:{onset.dataset}", False,
                   f"f*={onset.f_star} 与地板 {onset.f_floor} 不一致或不持续")
        if onset.extra_dead_at_onset <= 0:
            onset_ok = False
            record(f"onset:{onset.dataset}", False,
                   "起跳点没有产生任何额外死亡，容量机制没有后果")
        onset_summary.append(
            f"{onset.dataset}=f*{onset.f_star:.3f}"
            f"/+{onset.extra_dead_at_onset}→{onset.extra_dead_after_onset}"
        )
    record("onset:f_星与过载级联额外死亡", onset_ok, "; ".join(onset_summary))


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
        sys.stdout.write(f"IS_PASS: NO — 失败项 {failures}\n")
        return 1
    sys.stdout.write("IS_PASS: YES — 所有硬校验通过\n")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser for ``python -m bit_hyrl``."""
    parser = argparse.ArgumentParser(
        prog="bit_hyrl",
        description="BiT-HyRL 单一命令行入口（BiT / Hy / RL 三层）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"bit_hyrl {VERSION}")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="日志级别",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    p_fp = subparsers.add_parser("fingerprint", help="容错读取并输出九图指纹")
    p_fp.add_argument("--graphs", default=None, help="图清单 JSON（默认 configs/graphs.json）")
    p_fp.add_argument("--write", action="store_true", help="写入 results/fingerprints/")
    p_fp.add_argument("--quiet", action="store_true", help="不打印 JSON")
    p_fp.set_defaults(func=cmd_fingerprint)

    p_construct = subparsers.add_parser("construct", help="构造一张 BiT 图并输出诊断")
    p_construct.add_argument("--dataset", required=True, help="数据集名，如 BA-100")
    p_construct.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_construct.add_argument("--h", type=int, default=1, help="hub 个数")
    p_construct.add_argument("--seed", type=int, default=0, help="根种子")
    p_construct.add_argument(
        "--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
        help="实现模式：hh+rewire（默认，去相关）或 hh（纯 HH，对照用）",
    )
    p_construct.add_argument(
        "--rewire-swaps", type=int, default=None,
        help="去相关 2-switch 预算（默认 10*m；0 等价于 --realize hh）",
    )
    p_construct.add_argument("--betweenness", action="store_true", help="额外计算介数 BC 诊断")
    p_construct.add_argument("--max-repair-swaps", type=int, default=20000, help="连通修复 swap 预算")
    p_construct.add_argument("--allow-disconnected", action="store_true", help="允许不连通（不修复）")
    p_construct.add_argument("--write", action="store_true", help="写入 results/construct/")
    p_construct.add_argument("--quiet", action="store_true", help="不打印 JSON")
    p_construct.set_defaults(func=cmd_construct)

    p_probe = subparsers.add_parser(
        "probe", help="容量模型 + 利用率探针（只算不删）"
    )
    p_probe.add_argument("--dataset", required=True, help="数据集名，如 BA-100")
    p_probe.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_probe.add_argument("--h", type=int, default=1, help="hub 个数")
    p_probe.add_argument("--seed", type=int, default=0, help="根种子")
    p_probe.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                         help="实现模式")
    p_probe.add_argument("--rewire-swaps", type=int, default=None,
                         help="去相关 2-switch 预算（默认 10*m，与库默认一致）")
    p_probe.add_argument("--write", action="store_true", help="写入 results/probe/")
    p_probe.add_argument("--quiet", action="store_true", help="不打印 JSON")
    p_probe.set_defaults(func=cmd_probe)

    p_sweep = subparsers.add_parser("sweep-c", help="c 敏感性扫描")
    p_sweep.add_argument("--dataset", default="all", help="数据集名或 all")
    p_sweep.add_argument("--grid", default=None,
                         help="None=默认网格；或逗号分隔的数值列表（按 --grid-kind 解释）")
    p_sweep.add_argument(
        "--grid-kind", default="c", choices=("c", "c_norm"),
        help="c=原始 c 的 12 档（设计文档口径）；c_norm=归一化 11 档（受控对照，推荐）",
    )
    p_sweep.add_argument("--h", type=int, default=1, help="hub 个数")
    p_sweep.add_argument("--seed", type=int, default=0, help="根种子")
    p_sweep.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                         help="实现模式")
    p_sweep.add_argument("--rewire-swaps", type=int, default=None,
                         help="去相关 2-switch 预算（默认 10*m）")
    p_sweep.add_argument("--plan-only", action="store_true", help="只解度序列，不建图（快）")
    p_sweep.add_argument("--write", action="store_true", help="写入 results/sweep/<dataset>/")
    p_sweep.add_argument("--quiet", action="store_true", help="不打印 JSON")
    p_sweep.set_defaults(func=cmd_sweep_c)

    p_feas = subparsers.add_parser
    p_feas.add_argument("--dataset", default="all", help="数据集名或 all")
    p_feas.add_argument("--attack", default="all",
                        help=f"攻击方法或 all；可选 {list(PROVISIONAL_ATTACK_METHODS)}")
    p_feas.add_argument("--mode", default="structural", choices=("structural", "capacity"),
                        help="级联模式（默认 structural，与已发表口径一致）")
    p_feas.add_argument("--max-iter", type=int, default=50, help="capacity 模式的不动点上限")
    p_feas.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_feas.add_argument("--h", type=int, default=1, help="hub 个数")
    p_feas.add_argument("--seed", type=int, default=0, help="根种子")
    p_feas.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                        help="实现模式")
    p_feas.add_argument("--rewire-swaps", type=int, default=None,
                        help="去相关 2-switch 预算（默认 10*m）")
    p_feas.add_argument("--write", action="store_true", help="写入 results/feasibility/")
    p_feas.add_argument("--quiet", action="store_true", help="不打印表")
    p_feas.set_defaults(func=cmd_feasibility)

    p_verify = subparsers.add_parser("verify", help="一次性跑完所有硬校验（金标准 + 不变量）")
    p_verify.add_argument("--all", action="store_true", help="保留参数，verify 恒跑全部")
    p_verify.add_argument("--c", type=float, default=1.6, help="金标准对照所用的 c")
    p_verify.add_argument("--seed", type=int, default=0, help="根种子")
    p_verify.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                          help="实现模式")
    p_verify.add_argument("--rewire-swaps", type=int, default=None,
                          help="去相关 2-switch 预算（默认 10*m，与库默认一致）")
    p_verify.set_defaults(func=cmd_verify)

    p_eval = subparsers.add_parser
    p_eval.add_argument("--dataset", default="all", help="数据集名或 all")
    p_eval.add_argument(
        "--topology", default="both", choices=("original", "bit", "both"),
        help="拓扑来源：original=载入原图；bit=由 (N,M) 重建；both=两者都跑并对账",
    )
    p_eval.add_argument("--c", type=float, default=1.6, help="hub factor c（仅 bit 拓扑使用）")
    p_eval.add_argument("--h", type=int, default=1, help="hub 个数")
    p_eval.add_argument("--seed", type=int, default=0, help="根种子")
    p_eval.add_argument(
        "--attack", default="degree",
        help=f"攻击方法或 all；可选 {list(PROVISIONAL_ATTACK_METHODS)}",
    )
    p_eval.add_argument("--mode", default="structural", choices=("structural", "capacity"),
                        help="级联模式（默认 structural，与已发表口径一致）")
    p_eval.add_argument("--realize", default="hh+rewire", choices=("hh", "hh+rewire"),
                        help="bit 拓扑的实现模式")
    p_eval.add_argument("--rewire-swaps", type=int, default=None,
                        help="去相关 2-switch 预算（默认 10*m）")
    p_eval.add_argument("--write", action="store_true", help="写入 results/records/")
    p_eval.add_argument(
        "--policy", default="degree_ranked", choices=("degree_ranked", "ppo"),
        help="控制器放置策略：degree_ranked（基线）或 ppo（需 --checkpoint）",
    )
    p_eval.add_argument("--checkpoint", default=None, help="PPO 检查点路径（--policy ppo 时必填）")
    p_eval.add_argument("--quiet", action="store_true", help="不打印对账行")
    p_eval.set_defaults(func=cmd_eval)

    p_train = subparsers.add_parser(
        "train", help="PPO 训练控制器放置策略（T10）"
    )
    p_train.add_argument(
        "--dataset", default="all", help="数据集名，逗号分隔多个，或 all"
    )
    p_train.add_argument("--split", default=None,
                         help="在 splits.json 的 train 上训练、val 上早停（如 configs/splits.json）")
    p_train.add_argument("--iterations", type=int, default=200, help="外层次数（论文表4）")
    p_train.add_argument("--episodes", type=int, default=8, help="每迭代 rollout episode 数（论文表4）")
    p_train.add_argument("--lr", type=float, default=1e-4, help="Adam 学习率（论文表4=1e-4）")
    p_train.add_argument("--gamma", type=float, default=0.99, help="折扣因子（论文表4=0.99）")
    p_train.add_argument("--patience", type=int, default=20, help="val 早停 patience（0=关）")
    p_train.add_argument("--eval-sample", type=int, default=20, help="val 评估子集大小")
    p_train.add_argument("--seed", type=int, default=0, help="根种子")
    p_train.add_argument("--out", default=None, help="检查点输出路径（默认 results/checkpoints/）")
    p_train.add_argument(
        "--smoke", action="store_true", help="冒烟：iterations≤2、episodes≤2、train/val 各取前 4 张"
    )
    p_train.add_argument("--quiet", action="store_true", help="不打印每迭代奖励")
    p_train.set_defaults(func=cmd_train)

    p_transfer = subparsers.add_parser(
        "transfer", help="零样本迁移：用已训练策略在新图上放置控制器（T10）"
    )
    p_transfer.add_argument("--dataset", required=True, help="目标数据集名，如 ER-100")
    p_transfer.add_argument("--checkpoint", required=True, help="已训练检查点路径")
    p_transfer.add_argument("--seed", type=int, default=0, help="根种子（保留参数）")
    p_transfer.add_argument("--out", default=None, help="把结果 JSON 写到该路径")
    p_transfer.add_argument("--quiet", action="store_true", help="不打印控制器集合")
    p_transfer.set_defaults(func=cmd_transfer)

    p_ablate = subparsers.add_parser("ablate", help="2x2 消融：拓扑×放置（T11）")
    p_ablate.add_argument("--dataset", default="all", help="数据集名或逗号列表或 all")
    p_ablate.add_argument("--checkpoint", default=None, help="PPO 检查点（ppo 放置所需）")
    p_ablate.add_argument("--attacks", default="degree,random", help="逗号分隔攻击方法")
    p_ablate.add_argument("--mode", default="structural", choices=("structural", "capacity"))
    p_ablate.add_argument("--seeds", default="0", help="逗号分隔种子")
    p_ablate.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_ablate.add_argument("--h", type=int, default=1, help="hub 个数")
    p_ablate.add_argument("--out", default=None, help="输出目录（默认 results/ablation）")
    p_ablate.add_argument("--smoke", action="store_true", help="冒烟：只取 1 个 seed")
    p_ablate.add_argument("--quiet", action="store_true", help="不打印 2×2 表")
    p_ablate.set_defaults(func=cmd_ablate)

    p_matrix = subparsers.add_parser("matrix", help="实验矩阵长表（T12）")
    p_matrix.add_argument("--dataset", default="all", help="数据集名或逗号列表或 all")
    p_matrix.add_argument("--policies", default="degree_ranked,ppo", help="逗号分隔策略")
    p_matrix.add_argument("--topologies", default="original,bit", help="逗号分隔拓扑")
    p_matrix.add_argument("--attacks", default="degree,random", help="逗号分隔攻击")
    p_matrix.add_argument("--modes", default="structural", help="逗号分隔级联模式")
    p_matrix.add_argument("--seeds", default="0", help="逗号分隔种子")
    p_matrix.add_argument("--checkpoint", default=None, help="PPO 检查点")
    p_matrix.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_matrix.add_argument("--h", type=int, default=1, help="hub 个数")
    p_matrix.add_argument("--out", default=None, help="输出目录（默认 results/matrix）")
    p_matrix.add_argument("--smoke", action="store_true", help="冒烟：只取 1 个 seed")
    p_matrix.add_argument("--quiet", action="store_true", help="不打印行数")
    p_matrix.set_defaults(func=cmd_matrix)

    p_mm = subparsers.add_parser(
        "method-matrix", help="方法级矩阵：5 拓扑 baseline + Eff-R-Net + 2 放置代理 + BiT-HyRL（Task#4）"
    )
    p_mm.add_argument("--dataset", default="all", help="数据集名或逗号列表或 all")
    p_mm.add_argument("--split", default=None, help="用 splits.json 的 test 列表")
    p_mm.add_argument("--methods", default="", help="逗号分隔方法名；空=全部 METHODS")
    p_mm.add_argument("--attacks", default="degree,random", help="逗号分隔攻击")
    p_mm.add_argument("--mode", default="structural", choices=("structural", "capacity"))
    p_mm.add_argument("--seeds", default="0", help="逗号分隔种子")
    p_mm.add_argument("--checkpoint", default=None, help="PPO 检查点（BiT-HyRL 放置需要）")
    p_mm.add_argument("--c", type=float, default=1.6, help="hub factor c")
    p_mm.add_argument("--h", type=int, default=1, help="hub 个数")
    p_mm.add_argument(
        "--workers", type=int, default=1,
        help="并行进程数；1=串行，建议 ≤ 物理核数",
    )
    p_mm.add_argument("--out", default=None, help="输出目录（默认 results/method_matrix）")
    p_mm.add_argument(
        "--smoke", action="store_true",
        help="冒烟：BA-100,ER-100 × {BiT-HyRL,ONION,UNITY,MWCL-sim,MADQN-sim} × 1 攻击 × 1 种子",
    )
    p_mm.add_argument("--quiet", action="store_true", help="不打印行数")
    p_mm.set_defaults(func=cmd_method_matrix)

    p_ms = subparsers.add_parser(
        "method-stats", help="方法间显著性：exact Wilcoxon + Holm（GCC/CSA/H_C），参照可切"
    )
    p_ms.add_argument("--in", dest="infile", required=True, help="method_matrix CSV 路径")
    p_ms.add_argument("--reference", default="BiT-HyRL", help="参照方法（默认 BiT-HyRL）")
    p_ms.add_argument("--out", default=None, help="输出 CSV（默认 results/method_matrix/significance.csv）")
    p_ms.add_argument("--quiet", action="store_true", help="不打印")
    p_ms.set_defaults(func=cmd_method_stats)

    p_stats = subparsers.add_parser("stats", help="配对统计：Wilcoxon + bootstrap CI（T12）")
    p_stats.add_argument("--in", dest="infile", required=True, help="matrix CSV 路径")
    p_stats.add_argument("--metric", default="auc_r", help="比较的指标列")
    p_stats.add_argument("--out", default=None, help="输出目录（默认 results/stats）")
    p_stats.add_argument("--quiet", action="store_true", help="不打印")
    p_stats.set_defaults(func=cmd_stats)

    p_figures = subparsers.add_parser("figures", help="出图：鲁棒曲线/消融/F10（T12）")
    p_figures.add_argument("--in", dest="infile", required=True, help="matrix CSV 路径")
    p_figures.add_argument("--out", default=None, help="输出目录（默认 results/figures）")
    p_figures.add_argument("--smoke", action="store_true", help="保留参数")
    p_figures.add_argument("--quiet", action="store_true", help="不打印文件清单")
    p_figures.set_defaults(func=cmd_figures)

    for name, owner in _DEFERRED.items():
        stub = subparsers.add_parser(name, help=f"待实现（{owner}）")
        stub.set_defaults(func=_make_stub(name, owner))

    return parser


def _make_stub(name: str, owner: str):
    """Create a deferred-command handler that fails loudly."""

    def _stub(_args: argparse.Namespace) -> int:
        raise NotImplementedError(f"子命令 {name!r} 尚未实现，归属 {owner}")

    _stub.__name__ = f"cmd_{name}_deferred"
    return _stub


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python -m bit_hyrl``.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
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
