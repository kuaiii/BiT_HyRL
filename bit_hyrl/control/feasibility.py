"""Capacity feasibility as a function of the removal fraction.

Reports the fraction at which the surviving demand first exceeds the surviving
controller capacity, together with the closed-form estimate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx

from ..io import paths as io_paths
from ..rng import derive_permutation
from .capacity import (
    CapacityModel,
    ComponentCapacity,
    component_capacity_audit,
    probe_max_utilisation,
)
from .cascade import CascadeTrace, run_cascade

__all__ = [
    "PROVISIONAL_ATTACK_METHODS",
    "FeasibilityPoint",
    "FeasibilityCurve",
    "ModeGap",
    "OnsetRecord",
    "controllers_degree_ranked",
    "density_floor",
    "feasibility_curve",
    "format_feasibility_table",
    "format_mode_gap_table",
    "format_onset_table",
    "fstar_asymptotic",
    "fstar_closed_form",
    "mode_gap",
    "onset_records",
    "provisional_attack_sequence",
    "write_feasibility_outputs",
    "write_onset_outputs",
]

PROVISIONAL_ATTACK_METHODS: tuple[str, ...] = (
    "random",
    "degree",
    "betweenness",
    "closeness",
    "coreness",
    "eigenvector",
    "controllers",
)


def controllers_degree_ranked(graph: nx.Graph, count: int) -> list[int]:
    """Return ``count`` controllers, ranked by ``(-degree, node_id)``."""
    order = sorted(graph.nodes(), key=lambda node: (-graph.degree(node), int(node)))
    return sorted(int(node) for node in order[: max(0, int(count))])


def density_floor(capacity: int) -> float:
    """Return ``1 / (C + 1)``, the controller density below which load cannot fit."""
    if capacity < 0:
        raise ValueError(f"capacity cannot be negative, got {capacity}")
    return 1.0 / float(int(capacity) + 1)


def fstar_closed_form(n_nodes: int, n_controllers: int,
                      capacity: int) -> tuple[float, float]:
    """Return ``(t_c, f*)`` for the controller-first attack, in closed form."""
    n = int(n_nodes)
    k = int(n_controllers)
    c = int(capacity)
    if n <= 0 or c <= 0:
        raise ValueError(f"n_nodes and capacity must be positive, got {n_nodes} / {capacity}")
    if k < 0 or k > n:
        raise ValueError(f"n_controllers={k} is outside [0, {n}]")
    t_c = (k * (c + 1) - n) / float(c)
    steps = math.floor(t_c) + 1
    return t_c, min(steps, n) / float(n)


def fstar_asymptotic(controller_density: float, theta: float) -> float:
    """Return the asymptotic onset ``rho·(theta + rho) / (1 + theta)``."""
    rho = float(controller_density)
    th = float(theta)
    if rho <= 0.0:
        raise ValueError(f"controller_density must be positive, got {controller_density}")
    if th <= 0.0:
        raise ValueError(f"theta must be positive, got {theta}")
    return rho * (th + rho) / (1.0 + th)


_STATIC_CENTRALITIES: frozenset[str] = frozenset(
    {"degree", "betweenness", "closeness", "coreness", "eigenvector"}
)

_STATIC_CACHE: dict[tuple[str, int], tuple[nx.Graph, tuple[int, ...]]] = {}


def _centrality_scores(graph: nx.Graph, method: str) -> dict[int, float]:
    """Return the raw centrality the attack ranks on. Higher is removed earlier."""
    if method == "degree":
        return {int(node): float(graph.degree(node)) for node in graph.nodes()}
    if method == "betweenness":
        return {int(k): float(v) for k, v in nx.betweenness_centrality(graph).items()}
    if method == "closeness":
        return {int(k): float(v) for k, v in nx.closeness_centrality(graph).items()}
    if method == "coreness":
        return {int(k): float(v) for k, v in nx.core_number(graph).items()}
    if method == "eigenvector":
        try:
            raw = nx.eigenvector_centrality_numpy(graph)
        except Exception:
            raw = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
        return {int(k): float(v) for k, v in raw.items()}
    raise ValueError(f"unknown static centrality {method!r}")


def _static_sequence(graph: nx.Graph, method: str) -> tuple[int, ...]:
    """Ranked removal order for a static centrality, memoised on the graph object."""
    key = (method, id(graph))
    hit = _STATIC_CACHE.get(key)
    if hit is not None and hit[0] is graph:
        return hit[1]
    sequence = tuple(_total_order(graph, _centrality_scores(graph, method)))
    _STATIC_CACHE[key] = (graph, sequence)
    return sequence


def clear_static_cache() -> None:
    """Drop memoised attack orders (tests, or when graphs are rebuilt in place)."""
    _STATIC_CACHE.clear()

def _total_order(graph: nx.Graph, scores: dict[int, float]) -> list[int]:
    """Sort nodes by ``(-score, node_id)`` — the shared tie-break of"""
    return sorted(graph.nodes(), key=lambda node: (-scores.get(int(node), 0.0), int(node)))


def provisional_attack_sequence(graph: nx.Graph, method: str, *, controllers:
                                Sequence[int] = (), seed: int = 0) -> tuple[int, ...]:
    """Return a deterministic attack sequence for ``method``."""
    nodes = sorted(int(node) for node in graph.nodes())
    if method == "random":
        return tuple(int(node) for node in derive_permutation(seed, len(nodes), "attack", "random"))
    if method == "degree":
        scores = {int(node): float(graph.degree(node)) for node in nodes}
        return tuple(_total_order(graph, scores))
    if method in _STATIC_CENTRALITIES:
        return _static_sequence(graph, method)
    if method == "controllers":
        head = [int(node) for node in _total_order(
            graph, {int(node): float(graph.degree(node)) for node in nodes}
        ) if int(node) in set(int(c) for c in controllers)]
        head_set = set(head)
        tail = [int(node) for node in _total_order(
            graph, {int(node): float(graph.degree(node)) for node in nodes}
        ) if int(node) not in head_set]
        return tuple(head + tail)
    raise ValueError(
        f"unknown attack method {method!r}, expected one of {list(PROVISIONAL_ATTACK_METHODS)}"
    )


@dataclass(frozen=True)
class FeasibilityPoint:
    """One sampled state of the feasibility curve."""

    f: float
    f_direct: float
    n_alive: int
    n_controllers: int
    controller_density: float
    demand: int
    capacity: int
    slack: int
    feasible: bool
    capacity_pressure: int
    n_overloaded: int
    u_blind: float
    controllerless_components: int
    gcc_size: int
    components: tuple[ComponentCapacity, ...] = ()

    @property
    def non_trivial(self) -> bool:
        """``True`` when the state still has something to serve or protect."""
        return self.n_alive > 0 and self.demand > 0

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view (components summarised, not expanded)."""
        return {
            "f": self.f,
            "f_direct": self.f_direct,
            "n_alive": self.n_alive,
            "n_controllers": self.n_controllers,
            "controller_density": self.controller_density,
            "demand": self.demand,
            "capacity": self.capacity,
            "slack": self.slack,
            "feasible": self.feasible,
            "capacity_pressure": self.capacity_pressure,
            "n_overloaded": self.n_overloaded,
            "u_blind": self.u_blind,
            "controllerless_components": self.controllerless_components,
            "gcc_size": self.gcc_size,
            "n_components": len(self.components),
        }


@dataclass(frozen=True)
class FeasibilityCurve:
    """The feasibility history of one ``(graph, attack, mode)`` triple."""

    dataset: str
    attack: str
    mode: str
    capacity: int
    n_nodes: int
    n_controllers_initial: int
    density_floor: float
    points: tuple[FeasibilityPoint, ...]
    first_infeasible_f: float | None
    first_persistent_infeasible_f: float | None
    f_density_floor: float | None
    infeasible_fraction: float
    collapse_f: float | None
    collapse_f_direct: float | None
    n_non_trivial: int
    max_capacity_pressure: int
    peak_pressure_f: float | None
    final_fraction: float
    trace: CascadeTrace

    def to_dict(self, *, with_points: bool = True) -> dict[str, Any]:
        """Plain-``dict`` view; ``trace`` is embedded in its own summary form."""
        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "attack": self.attack,
            "mode": self.mode,
            "capacity": self.capacity,
            "n_nodes": self.n_nodes,
            "n_controllers_initial": self.n_controllers_initial,
            "density_floor": self.density_floor,
            "first_infeasible_f": self.first_infeasible_f,
            "first_persistent_infeasible_f": self.first_persistent_infeasible_f,
            "f_density_floor": self.f_density_floor,
            "infeasible_fraction": self.infeasible_fraction,
            "collapse_f": self.collapse_f,
            "collapse_f_direct": self.collapse_f_direct,
            "n_non_trivial": self.n_non_trivial,
            "max_capacity_pressure": self.max_capacity_pressure,
            "peak_pressure_f": self.peak_pressure_f,
            "final_fraction": self.final_fraction,
            "trace_audit_ok": self.trace.audit_ok,
        }
        if with_points:
            payload["points"] = [point.to_dict() for point in self.points]
        return payload


def _feasibility_of(trace_step_alive: Iterable[int], controllers: Iterable[int],
                    graph: nx.Graph, model: CapacityModel
                    ) -> tuple[tuple[ComponentCapacity, ...], bool]:
    """Return ``(components, feasible)`` for one state."""
    components = component_capacity_audit(
        graph, controllers, model=model, alive=trace_step_alive
    )
    return components, all(component.feasible for component in components)


def feasibility_curve(graph: nx.Graph, attack_sequence: Sequence[int],
                      controllers: Sequence[int], model: CapacityModel,
                      cfg: Any = None, *, dataset: str = "",
                      attack: str = "") -> FeasibilityCurve:
    """Run the cascade and record the capacity verdict at every step."""
    trace = run_cascade(graph, attack_sequence, controllers, model, cfg)
    n = graph.number_of_nodes() or 1
    floor = density_floor(model.capacity)

    points: list[FeasibilityPoint] = []

    initial_alive = sorted(int(node) for node in graph.nodes())
    initial_controllers = sorted({int(node) for node in controllers})
    initial_components, initial_feasible = _feasibility_of(
        initial_alive, initial_controllers, graph, model
    )
    initial_probe = probe_max_utilisation(graph, initial_controllers, model)
    points.append(
        FeasibilityPoint(
            f=0.0,
            f_direct=0.0,
            n_alive=len(initial_alive),
            n_controllers=len(initial_controllers),
            controller_density=(len(initial_controllers) / len(initial_alive))
            if initial_alive else 0.0,
            demand=sum(component.demand for component in initial_components),
            capacity=sum(component.capacity for component in initial_components),
            slack=sum(
                component.capacity - component.demand for component in initial_components
            ),
            feasible=initial_feasible,
            capacity_pressure=sum(
                max(0, component.demand - component.capacity)
                for component in initial_components
            ),
            n_overloaded=initial_probe.n_overloaded,
            u_blind=initial_probe.u_nearest,
            controllerless_components=sum(
                1 for component in initial_components if component.controllerless
            ),
            gcc_size=max(
                (component.n_nodes for component in initial_components), default=0
            ),
            components=initial_components,
        )
    )

    for step in trace.steps:
        components, feasible = _feasibility_of(
            step.alive_nodes, step.controllers_alive, graph, model
        )
        demand = sum(component.demand for component in components)
        capacity_total = sum(component.capacity for component in components)
        pressure = sum(
            max(0, component.demand - component.capacity) for component in components
        )
        n_alive = len(step.alive_nodes)
        k = len(step.controllers_alive)
        points.append(
            FeasibilityPoint(
                f=step.f_total,
                f_direct=step.f_direct,
                n_alive=n_alive,
                n_controllers=k,
                controller_density=(k / n_alive) if n_alive else 0.0,
                demand=demand,
                capacity=capacity_total,
                slack=capacity_total - demand,
                feasible=feasible,
                capacity_pressure=pressure,
                n_overloaded=sum(
                    1 for load in step.loads.values() if load > model.capacity
                ),
                u_blind=step.max_utilisation,
                controllerless_components=sum(
                    1 for component in components if component.controllerless
                ),
                gcc_size=step.gcc_size,
                components=components,
            )
        )

    first_infeasible = next((point.f for point in points if not point.feasible), None)
    non_trivial = [point for point in points if point.non_trivial]
    infeasible_fraction = (
        sum(1 for point in non_trivial if not point.feasible) / len(non_trivial)
        if non_trivial else 0.0
    )
    f_floor = next(
        (point.f for point in points if point.controller_density < floor), None
    )

    first_persistent: float | None = None
    for index in range(len(non_trivial) - 1, -1, -1):
        if non_trivial[index].feasible:
            break
        first_persistent = non_trivial[index].f
    max_pressure = max((point.capacity_pressure for point in points), default=0)
    peak_pressure_f = next(
        (point.f for point in points if point.capacity_pressure == max_pressure),
        None,
    ) if max_pressure > 0 else None

    collapse_point = next(
        (point for point in points if point.n_alive == 0), None
    )
    return FeasibilityCurve(
        dataset=dataset,
        attack=attack,
        mode=trace.mode,
        capacity=int(model.capacity),
        n_nodes=n,
        n_controllers_initial=len({int(node) for node in controllers}),
        density_floor=floor,
        points=tuple(points),
        first_infeasible_f=first_infeasible,
        first_persistent_infeasible_f=first_persistent,
        f_density_floor=f_floor,
        infeasible_fraction=infeasible_fraction,
        collapse_f=collapse_point.f if collapse_point is not None else None,
        collapse_f_direct=(
            collapse_point.f_direct if collapse_point is not None else None
        ),
        n_non_trivial=len(non_trivial),
        max_capacity_pressure=max_pressure,
        peak_pressure_f=peak_pressure_f,
        final_fraction=trace.final_fraction,
        trace=trace,
    )


def _format_optional(value: float | None) -> str:
    """Render ``None`` as ``—`` and a float with three decimals."""
    return "—" if value is None else f"{value:.3f}"


@dataclass(frozen=True)
class ModeGap:
    """How much extra failure ``capacity`` mode adds over ``structural``."""

    dataset: str
    attack: str
    max_extra_dead: int
    max_extra_dead_f_direct: float | None
    final_extra_dead: int
    mean_extra_dead: float
    n_nodes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view."""
        return {
            "dataset": self.dataset,
            "attack": self.attack,
            "max_extra_dead": self.max_extra_dead,
            "max_extra_dead_f_direct": self.max_extra_dead_f_direct,
            "final_extra_dead": self.final_extra_dead,
            "mean_extra_dead": self.mean_extra_dead,
            "n_nodes": self.n_nodes,
        }


def mode_gap(structural: FeasibilityCurve, capacity: FeasibilityCurve) -> ModeGap:
    """Compare a ``structural`` and a ``capacity`` curve on the same attack."""
    if len(structural.points) != len(capacity.points):
        raise ValueError(
            "the structural and capacity curves differ in length ("
            f"{len(structural.points)} vs {len(capacity.points)}), cannot pair them"
        )
    differences = [
        (cap.f_direct, struct.n_alive - cap.n_alive)
        for struct, cap in zip(structural.points, capacity.points)
    ]
    peak_f, peak_value = max(differences, key=lambda item: (item[1], -item[0]))
    return ModeGap(
        dataset=capacity.dataset or structural.dataset,
        attack=capacity.attack or structural.attack,
        max_extra_dead=max(0, peak_value),
        max_extra_dead_f_direct=peak_f if peak_value > 0 else None,
        final_extra_dead=differences[-1][1] if differences else 0,
        mean_extra_dead=(
            sum(value for _, value in differences) / len(differences)
            if differences else 0.0
        ),
        n_nodes=capacity.n_nodes,
    )


def format_mode_gap_table(gaps: Sequence[ModeGap]) -> str:
    """Render the ``structural`` vs ``capacity`` gap as a plain-text table."""
    header = (
        f"{'dataset':<10} {'attack':<12} {'peak extra dead':>16} {'f_direct@peak':>14} "
        f"{'final':>6} {'mean':>8} {'share':>7}"
    )
    lines = [header, "-" * len(header)]
    for gap in gaps:
        share = (
            f"{100.0 * gap.max_extra_dead / gap.n_nodes:.1f}%"
            if gap.n_nodes else "—"
        )
        lines.append(
            f"{gap.dataset:<10} {gap.attack:<12} {gap.max_extra_dead:>16} "
            f"{_format_optional(gap.max_extra_dead_f_direct):>14} "
            f"{gap.final_extra_dead:>6} "
            f"{gap.mean_extra_dead:>8.2f} {share:>7}"
        )
    return "\n".join(lines) + "\n"


def format_feasibility_table(curves: Sequence[FeasibilityCurve]) -> str:
    """Render the ``first infeasible f`` summary as a plain-text table."""
    header = (
        f"{'dataset':<10} {'attack':<12} {'mode':<11} {'C':>3} {'k0':>3} "
        f"{'1/(C+1)':>8} {'f_density':>10} {'f_first':>8} {'f_persist':>10} "
        f"{'infeas%':>8} {'peak load':>9} {'t=0':>5}"
    )
    lines = [header, "-" * len(header)]
    for curve in curves:
        first_point_feasible = curve.points[0].feasible if curve.points else True
        lines.append(
            f"{curve.dataset:<10} {curve.attack:<12} {curve.mode:<11} "
            f"{curve.capacity:>3} {curve.n_controllers_initial:>3} "
            f"{curve.density_floor:>8.4f} "
            f"{_format_optional(curve.f_density_floor):>10} "
            f"{_format_optional(curve.first_infeasible_f):>8} "
            f"{_format_optional(curve.first_persistent_infeasible_f):>10} "
            f"{100.0 * curve.infeasible_fraction:>7.1f}% "
            f"{curve.max_capacity_pressure:>8} "
            f"{str(first_point_feasible):>5}"
        )
    return "\n".join(lines) + "\n"


def write_feasibility_outputs(curves: Sequence[FeasibilityCurve],
                             out_dir: Path | None = None,
                             *, name: str = "feasibility") -> dict[str, Path]:
    """Write the JSON record and the plain-text table."""
    if not curves:
        raise ValueError("empty feasibility result, refusing to write")
    target = Path(out_dir) if out_dir is not None else io_paths.results_path(
        "feasibility", ""
    ).parent
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{name}.json"
    txt_path = target / f"{name}.txt"

    payload = {
        "name": name,
        "n_curves": len(curves),
        "density_floor_note": "1/(C+1); the overload cascade can only start below this controller density",
        "curves": [curve.to_dict() for curve in curves],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    txt_path.write_text(format_feasibility_table(curves), encoding="utf-8")
    return {"json": json_path, "txt": txt_path}


@dataclass(frozen=True)
class OnsetRecord:
    """First infeasibility and how much extra failure it unleashes."""

    dataset: str
    attack: str
    n_nodes: int
    capacity: int
    density_floor: float
    n_controllers_initial: int
    feasible_at_t0: bool
    f_floor: float | None
    f_star: float | None
    f_direct_star: float | None
    f_star_persistent: float | None
    k_at_f_star: int | None
    density_at_f_star: float | None
    demand_at_f_star: int | None
    capacity_at_f_star: int | None
    controllerless_at_f_star: int
    infeasible_fraction: float
    max_capacity_pressure: int
    extra_dead_at_onset: int
    extra_dead_fraction_at_onset: float
    extra_dead_after_onset: int
    extra_dead_fraction: float
    final_extra_dead: int
    mode_compared: bool

    @property
    def onset_matches_floor(self) -> bool:
        """``f*`` coincides exactly with the density-floor crossing."""
        return (
            self.f_star is not None
            and self.f_floor is not None
            and self.f_star == self.f_floor
        )

    @property
    def onset_is_persistent(self) -> bool:
        """``f*`` is also the first *persistent* infeasibility."""
        return (
            self.f_star is not None
            and self.f_star_persistent is not None
            and self.f_star == self.f_star_persistent
        )

    @property
    def onset_kind(self) -> str | None:
        """Why ``f*`` is infeasible: ``"capacity"``, ``"partition"`` or ``None``."""
        if self.f_star is None:
            return None
        return "partition" if self.controllerless_at_f_star > 0 else "capacity"

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view."""
        return {
            "dataset": self.dataset,
            "attack": self.attack,
            "n_nodes": self.n_nodes,
            "capacity": self.capacity,
            "density_floor": self.density_floor,
            "n_controllers_initial": self.n_controllers_initial,
            "feasible_at_t0": self.feasible_at_t0,
            "f_floor": self.f_floor,
            "f_star": self.f_star,
            "f_direct_star": self.f_direct_star,
            "f_star_persistent": self.f_star_persistent,
            "onset_kind": self.onset_kind,
            "onset_matches_floor": self.onset_matches_floor,
            "onset_is_persistent": self.onset_is_persistent,
            "k_at_f_star": self.k_at_f_star,
            "density_at_f_star": self.density_at_f_star,
            "demand_at_f_star": self.demand_at_f_star,
            "capacity_at_f_star": self.capacity_at_f_star,
            "controllerless_at_f_star": self.controllerless_at_f_star,
            "infeasible_fraction": self.infeasible_fraction,
            "max_capacity_pressure": self.max_capacity_pressure,
            "extra_dead_at_onset": self.extra_dead_at_onset,
            "extra_dead_fraction_at_onset": self.extra_dead_fraction_at_onset,
            "extra_dead_after_onset": self.extra_dead_after_onset,
            "extra_dead_fraction": self.extra_dead_fraction,
            "final_extra_dead": self.final_extra_dead,
            "mode_compared": self.mode_compared,
        }


def _one_onset(structural: FeasibilityCurve,
               capacity: FeasibilityCurve | None) -> OnsetRecord:
    """Build one :class:`OnsetRecord` from a (structural, capacity) pair."""
    onset_index = next(
        (index for index, point in enumerate(structural.points) if not point.feasible),
        None,
    )
    onset_point = (
        structural.points[onset_index] if onset_index is not None else None
    )
    f_star = structural.first_infeasible_f
    if onset_point is not None and f_star != onset_point.f:  # pragma: no cover
        raise ValueError(
            f"first_infeasible_f={f_star} disagrees with the index-based onset {onset_point.f}"
        )

    extra_after = 0
    extra_at_onset = 0
    final_extra = 0
    if capacity is not None:
        if len(capacity.points) != len(structural.points):
            raise ValueError(
                "the structural and capacity curves differ in length ("
                f"{len(structural.points)} vs {len(capacity.points)}), cannot pair them"
            )
        differences = [
            struct.n_alive - cap.n_alive
            for struct, cap in zip(structural.points, capacity.points)
        ]
        if onset_index is not None:
            extra_at_onset = differences[onset_index]
            extra_after = max(differences[onset_index:], default=0)
        final_extra = differences[-1] if differences else 0

    n_nodes = structural.n_nodes or 1
    return OnsetRecord(
        dataset=structural.dataset,
        attack=structural.attack,
        n_nodes=structural.n_nodes,
        capacity=structural.capacity,
        density_floor=structural.density_floor,
        n_controllers_initial=structural.n_controllers_initial,
        feasible_at_t0=structural.points[0].feasible if structural.points else True,
        f_floor=structural.f_density_floor,
        f_star=f_star,
        f_direct_star=onset_point.f_direct if onset_point is not None else None,
        f_star_persistent=structural.first_persistent_infeasible_f,
        k_at_f_star=onset_point.n_controllers if onset_point is not None else None,
        density_at_f_star=(
            onset_point.controller_density if onset_point is not None else None
        ),
        demand_at_f_star=onset_point.demand if onset_point is not None else None,
        capacity_at_f_star=onset_point.capacity if onset_point is not None else None,
        controllerless_at_f_star=(
            onset_point.controllerless_components if onset_point is not None else 0
        ),
        infeasible_fraction=structural.infeasible_fraction,
        max_capacity_pressure=structural.max_capacity_pressure,
        extra_dead_at_onset=max(0, extra_at_onset),
        extra_dead_fraction_at_onset=float(max(0, extra_at_onset)) / float(n_nodes),
        extra_dead_after_onset=max(0, extra_after),
        extra_dead_fraction=float(max(0, extra_after)) / float(n_nodes),
        final_extra_dead=final_extra,
        mode_compared=capacity is not None,
    )


def onset_records(structural: Sequence[FeasibilityCurve],
                  capacity: Sequence[FeasibilityCurve] | None = None
                  ) -> tuple[OnsetRecord, ...]:
    """Summarise every ``(dataset, attack)`` pair into an :class:`OnsetRecord`."""
    structural_keys = {(curve.dataset, curve.attack) for curve in structural}
    orphaned = [
        f"{curve.dataset}/{curve.attack}"
        for curve in (capacity or ())
        if (curve.dataset, curve.attack) not in structural_keys
    ]
    if orphaned:
        raise ValueError(f"capacity curves without a structural counterpart: {orphaned}")

    by_key = {
        (curve.dataset, curve.attack): curve for curve in (capacity or ())
    }
    return tuple(
        _one_onset(curve, by_key.get((curve.dataset, curve.attack)))
        for curve in structural
    )


def format_onset_table(records: Sequence[OnsetRecord]) -> str:
    """Render the feasibility-evidence table (one row per ``(dataset, attack)``)."""
    header = (
        f"{'dataset':<10} {'attack':<12} {'N':>4} {'C':>3} {'nc0':>4} {'1/(C+1)':>8} "
        f"{'f_floor':>8} {'f*':>7} {'onset':>7} {'persist':>7} {'k(f*)':>6} {'rho(f*)':>8} "
        f"{'SD/SC(f*)':>11} {'infeas%':>8} {'peak':>6} "
        f"{'extra@f*':>11} {'after f*':>9} {'share':>7}"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        demand_capacity = (
            f"{record.demand_at_f_star}/{record.capacity_at_f_star}"
            if record.demand_at_f_star is not None
            else "—"
        )
        kind = {"capacity": "capacity", "partition": "partition", None: "—"}[record.onset_kind]
        lines.append(
            f"{record.dataset:<10} {record.attack:<12} {record.n_nodes:>4} "
            f"{record.capacity:>3} {record.n_controllers_initial:>4} "
            f"{record.density_floor:>8.4f} "
            f"{_format_optional(record.f_floor):>8} "
            f"{_format_optional(record.f_star):>7} "
            f"{kind:>7} "
            f"{('yes' if record.onset_is_persistent else 'no'):>7} "
            f"{(record.k_at_f_star if record.k_at_f_star is not None else '—'):>6} "
            f"{(f'{record.density_at_f_star:.4f}' if record.density_at_f_star is not None else '—'):>8} "
            f"{demand_capacity:>11} "
            f"{100.0 * record.infeasible_fraction:>7.1f}% "
            f"{record.max_capacity_pressure:>6} "
            f"{record.extra_dead_at_onset:>11} "
            f"{record.extra_dead_after_onset:>9} "
            f"{100.0 * record.extra_dead_fraction:>6.2f}%"
        )
    return "\n".join(lines) + "\n"


def write_onset_outputs(records: Sequence[OnsetRecord],
                        out_dir: Path | None = None,
                        *, name: str = "onset") -> dict[str, Path]:
    """Write the onset JSON record and the plain-text table."""
    if not records:
        raise ValueError("empty onset result, refusing to write")
    target = Path(out_dir) if out_dir is not None else io_paths.results_path(
        "feasibility", ""
    ).parent
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{name}.json"
    txt_path = target / f"{name}.txt"

    payload = {
        "name": name,
        "n_records": len(records),
        "definition": {
            "f_star": "removal fraction at which some component first has demand > capacity",
            "f_floor": "removal fraction at which the controller density k/n_alive first drops below 1/(C+1)",
            "extra_dead_after_onset": (
                "nodes lost in capacity mode beyond structural mode for f >= f*, on the same attack sequence"
            ),
            "capacity_binds": "true only when feasible == False or some switch cannot be assigned",
        },
        "records": [record.to_dict() for record in records],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    txt_path.write_text(format_onset_table(records), encoding="utf-8")
    return {"json": json_path, "txt": txt_path}
