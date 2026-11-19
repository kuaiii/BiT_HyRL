"""End-to-end evaluation of a single graph."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import networkx as nx

from ..control.cascade import run_cascade
from ..control.capacity import capacity_model
from ..control.feasibility import (
    controllers_degree_ranked,
    provisional_attack_sequence,
)
from ..io.graph_io import load_graph_spec
from ..topology.builder import build_bimodal_graph
from ..topology.degree_sequence import plan_degree_sequence
from .curves import compute_robustness
from .metrics import calculate_csa, calculate_h_c, calculate_h_c_norm
from .record import EvaluationRecord

#: Accepted topology selectors.
TOPOLOGIES: tuple[str, ...] = ("original", "bit")


def _trapezoidal(x: Sequence[float], y: Sequence[float]) -> float:
    """Composite trapezoidal area under the ``(x, y)`` curve."""
    if len(x) < 2:
        return 0.0
    area = 0.0
    for index in range(1, len(x)):
        step = x[index] - x[index - 1]
        area += 0.5 * (y[index] + y[index - 1]) * step
    return area


def build_eval_graph(
    spec: Mapping[str, Any],
    *,
    topology: str,
    c: float = 1.6,
    h: int = 1,
    seed: int = 0,
    realize: str = "hh+rewire",
    rewire_swaps: int | None = None,
) -> nx.Graph:
    """Materialise the graph ``run_evaluation`` will evaluate (T10 helper)."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, got {topology!r}")
    graph, _meta = load_graph_spec(spec)
    if topology == "bit":
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        plan = plan_degree_sequence(n_nodes, n_edges, c=c, h=h)
        result = build_bimodal_graph(
            plan,
            seed,
            realize=realize,
            rewire_swaps=rewire_swaps,
            dataset=str(spec.get("dataset", "")),
        )
        graph = result.graph
    return graph


def run_evaluation(
    spec: Mapping[str, Any],
    *,
    topology: str,
    c: float = 1.6,
    h: int = 1,
    seed: int = 0,
    attack: str = "degree",
    mode: str = "structural",
    realize: str = "hh+rewire",
    rewire_swaps: int | None = None,
    cfg: Any = None,
    controllers: Sequence[int] | None = None,
    policy: str = "degree_ranked",
    placement_seconds: float = 0.0,
    graph: nx.Graph | None = None,
) -> EvaluationRecord:
    """Run one ``(topology, attack, mode)`` evaluation and persist its record."""
    t_start = time.perf_counter()
    if graph is None:
        graph = build_eval_graph(
            spec, topology=topology, c=c, h=h, seed=seed, realize=realize,
            rewire_swaps=rewire_swaps,
        )
    n_nodes = graph.number_of_nodes()

    model = capacity_model(n_nodes)
    if controllers is None:
        fitted_controllers: list[int] = controllers_degree_ranked(
            graph, model.n_controllers
        )
    else:
        fitted_controllers = [int(node) for node in controllers]

    sequence = provisional_attack_sequence(
        graph, attack, controllers=fitted_controllers, seed=seed
    )

    cascade_cfg = (
        cfg if cfg is not None else SimpleNamespace(mode=mode, max_iter=50)
    )
    trace = run_cascade(graph, sequence, fitted_controllers, model, cascade_cfg)

    f_grid: list[float] = list(trace.f_grid)
    r_curve: list[float] = list(trace.r_curve)
    auc_r = _trapezoidal(f_grid, r_curve)
    collapse_f = trace.final_fraction

    robustness = compute_robustness(graph, trace, n_nodes, len(fitted_controllers))

    final_step = trace.steps[-1] if trace.steps else None
    if final_step is not None:
        alive_nodes = set(final_step.alive_nodes)
        controllers_alive = list(final_step.controllers_alive)
        loads = dict(final_step.loads)
        subgraph: nx.Graph = graph.subgraph(alive_nodes)
        csa_final = calculate_csa(
            subgraph, controllers_alive, initial_num_switches=n_nodes
        )
        h_c_final = calculate_h_c(loads)
        h_c_norm_final = calculate_h_c_norm(loads, n_c0=len(fitted_controllers))
    else:  # pragma: no cover - a non-empty attack always yields >= 1 step
        csa_final = h_c_final = h_c_norm_final = 0.0

    eval_seconds = time.perf_counter() - t_start

    record = EvaluationRecord(
        dataset=str(spec.get("dataset", "")),
        topology=topology,
        c=float(c),
        h=int(h),
        seed=int(seed),
        n_controllers=model.n_controllers,
        attack=attack,
        mode=mode,
        policy=policy,
        f_grid=tuple(f_grid),
        r_curve=tuple(r_curve),
        csa_final=float(csa_final),
        h_c_final=float(h_c_final),
        h_c_norm_final=float(h_c_norm_final),
        auc_r=float(auc_r),
        collapse_f=float(collapse_f),
        audit_ok=bool(trace.audit_ok),
        auc_gcc=float(robustness["auc_gcc"]),
        auc_csa=float(robustness["auc_csa"]),
        auc_hc=float(robustness["auc_hc"]),
        r_res=float(robustness["r_res"]),
        collapse_f_gcc02=float(robustness["collapse_f_gcc02"]),
        r_res_half=float(robustness["r_res_half"]),
        eval_seconds=float(eval_seconds),
        placement_seconds=float(placement_seconds),
        csa_curve=tuple(float(v) for v in robustness["csa_curve"]),
        hc_curve=tuple(float(v) for v in robustness["hc_curve"]),
    )
    record.save()
    return record
