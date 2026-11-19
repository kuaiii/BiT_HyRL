"""End-to-end evaluation runner.

:func:`run_evaluation` is the single entry point that turns one graph spec into one
:class:`~bit_hyrl.evaluation.record.EvaluationRecord`.  It is deliberately
graph-agnostic: the *same* pipeline serves the **original** substrate (load the
input graph) and the **BiT-reconstructed** proposal (rebuild from ``(N, M)``), which
is exactly what makes interpretation A — "original-substrate baseline vs
BiT-reconstructed proposal" — observable.  The downstream control/cascade code is
already graph-agnostic (it takes any :class:`networkx.Graph` and never calls BiT),
so the only switch is the ``topology`` selector feeding the common pipeline.

One call evaluates **one** ``(attack, mode)`` and returns one record; the CLI loops
over graphs, topologies and attacks and persists every record.  The record's
``attack``/``mode`` fields are singular, so there is no hidden batching inside this
function — call it once per combination.

The final-state metrics are computed on the **last cascade step**: its surviving
node set induces the subgraph for CSA, and its controller ``loads`` map drives the
H_C entropy.  This is the "final surviving state" the task specifies.
"""

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
    """Composite trapezoidal area under the ``(x, y)`` curve.

    ``x`` is assumed non-decreasing (it is ``f_direct``, which is monotone by
    construction).  A single point or empty curve has zero area.
    """
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
    """Materialise the graph ``run_evaluation`` will evaluate (T10 helper).

    Pulled out of :func:`run_evaluation` so the PPO CLI can place controllers on
    the **exact** graph the runner builds (identical ``(spec, seed, c, h)`` ⇒
    identical graph), instead of duplicating the BiT reconstruction and risking
    drift.

    Args:
        spec: A ``graphs.json``-style spec.
        topology: ``"original"`` (load the input graph) or ``"bit"`` (reconstruct).
        c: Hub factor for the BiT reconstruction.
        h: Number of hubs.
        seed: Root seed for the reconstruction.
        realize: BiT realisation mode.
        rewire_swaps: 2-switch budget for ``"hh+rewire"``.

    Returns:
        The evaluated :class:`networkx.Graph`.
    """
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology 必须是 {TOPOLOGIES}，收到 {topology!r}")
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
    """Run one ``(topology, attack, mode)`` evaluation and persist its record.

    Pipeline (identical for both topologies):

    1. ``graph, meta = load_graph_spec(spec)`` — load + pin ``(N, M, sha256)``.
    2. if ``topology == "bit"``: rebuild from ``(N, M)`` via
       :func:`plan_degree_sequence` + :func:`build_bimodal_graph`; otherwise keep
       the loaded graph.
    3. ``model = capacity_model(N)``; ``controllers = controllers_degree_ranked(...)``.
    4. ``seq = provisional_attack_sequence(...)``; ``trace = run_cascade(...)``.
    5. Extract ``f_grid``/``r_curve``/``auc_r``/``collapse_f`` and the final-state
       metrics (CSA / H_C / H_C_norm), build the record and :meth:`save` it.

    Args:
        spec: A ``configs/graphs.json``-style spec (at least ``path`` and
            ``dataset``).
        topology: ``"original"`` or ``"bit"``.
        c: Hub factor for the BiT reconstruction (ignored when ``topology`` is
            ``"original"`` but always recorded).
        h: Number of hubs for the BiT reconstruction.
        seed: Root seed (passed through; no global ``random.seed``).
        attack: Attack method name.
        mode: Cascade mode (``"structural"`` or ``"capacity"``).
        realize: BiT realisation mode (``"hh"`` / ``"hh+rewire"``).
        rewire_swaps: 2-switch budget for the ``"hh+rewire"`` realisation.
        cfg: Optional cascade config; defaults to
            ``SimpleNamespace(mode=mode, max_iter=50)``.
        controllers: Optional explicit controller ids.  When ``None`` (default) the
            degree-ranked policy is used — the historical behaviour, unchanged.
        policy: Placement-policy label recorded on the record (``"degree_ranked"``
            by default, ``"ppo"`` when the caller supplies PPO controllers).
        graph: Optional pre-built graph.  When given, the runner evaluates *this*
            graph directly (skips load/reconstruct) — used by the method matrix to
            evaluate a baseline's constructed graph.  ``None`` keeps the original
            ``topology`` behaviour.
        placement_seconds: Wall-clock time the caller spent selecting the
            controllers (PPO ``.select``); ``0.0`` for the degree-ranked policy.
            Stored on the record so the placement cost is measurable.

    Returns:
        The persisted :class:`EvaluationRecord`.

    Raises:
        ValueError: For an unknown ``topology``.
    """
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

    # ---- per-step three-term decomposition (GCC / CSA / H_C) -------------- #
    robustness = compute_robustness(graph, trace, n_nodes, len(fitted_controllers))

    # ---- final surviving state (legacy scalar fields, unchanged) ---------- #
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
        # Same fixed ln(n_c^0) denominator as the per-step ``hc_curve`` so that
        # ``h_c_norm_final == hc_curve[-1]``.
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
