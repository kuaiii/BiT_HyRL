"""Controller capacity model and the utilisation probe.

One switch carries one unit of demand; a controller of capacity ``C`` serves
``C`` switches.  The probe reports where demand exceeds capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx

from ..topology.degree_sequence import round_half_up

__all__ = [
    "DEFAULT_CONTROLLER_RATIO",
    "DEFAULT_DEMAND",
    "DEFAULT_THETA",
    "GOLDEN_CAPACITY",
    "CapacityError",
    "CapacityModel",
    "ComponentCapacity",
    "UtilisationProbe",
    "capacity_model",
    "capacity_per_controller",
    "component_capacity_audit",
    "n_controllers_for",
    "probe_max_utilisation",
    "GOLDEN_N",
]

#: Default capacity-relaxation factor θ.
DEFAULT_THETA = 0.5

#: Default controller-to-node ratio (``n_c = round(0.1 · N)``).
DEFAULT_CONTROLLER_RATIO = 0.1

#: Default per-switch demand (homogeneous, unit).
DEFAULT_DEMAND = 1

GOLDEN_N: tuple[tuple[str, int], ...] = (
    ("BA-100", 100),
    ("BA-200", 200),
    ("BA-300", 300),
    ("BA-400", 400),
    ("BA-500", 500),
    ("Cogentco", 197),
    ("Colt", 153),
    ("GtsCe", 149),
    ("UsCarrier", 158),
)

GOLDEN_CAPACITY: dict[str, tuple[int, int, int, int, int]] = {
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


class CapacityError(ValueError):
    """Raised when a capacity specification is internally inconsistent."""


def n_controllers_for(n: int, ratio: float = DEFAULT_CONTROLLER_RATIO,
                      override: int | None = None) -> int:
    """Return ``n_c``, the number of controllers."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise CapacityError(f"n must be an integer, got {n!r}")
    if n < 1:
        raise CapacityError(f"n must be >= 1, got {n}")
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, int):
            raise CapacityError(f"n_controllers must be an integer, got {override!r}")
        if not 1 <= override <= n:
            raise CapacityError(f"n_controllers={override} out of range, expected 1 <= n_c <= n={n}")
        return int(override)
    if not math.isfinite(float(ratio)) or float(ratio) <= 0.0:
        raise CapacityError(f"controller_ratio must be a finite positive number, got {ratio!r}")
    return max(1, round_half_up(float(ratio) * n))


def capacity_per_controller(n: int, n_c: int, theta: float = DEFAULT_THETA) -> int:
    """Return ``C = ceil((1 + theta) · N / n_c)``."""
    if isinstance(n_c, bool) or not isinstance(n_c, int) or n_c < 1:
        raise CapacityError(f"n_c must be an integer >= 1, got {n_c!r}")
    if not math.isfinite(float(theta)) or float(theta) < 0.0:
        raise CapacityError(f"theta must be a finite non-negative number, got {theta!r}")
    return max(1, int(math.ceil((1.0 + float(theta)) * n / n_c)))


@dataclass(frozen=True)
class CapacityModel:
    """The solved capacity budget of one graph."""

    theta: float
    n_controllers: int
    capacity: int
    n_switches: int
    total_capacity: int
    aggregate_slack: int
    demand: int = DEFAULT_DEMAND
    ratio: float = DEFAULT_CONTROLLER_RATIO
    capacity_mode: str = "formula"

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view (goes straight into ``EvaluationRecord.capacity``)."""
        return {
            "theta": self.theta,
            "n_controllers": self.n_controllers,
            "capacity": self.capacity,
            "n_switches": self.n_switches,
            "total_capacity": self.total_capacity,
            "aggregate_slack": self.aggregate_slack,
            "demand": self.demand,
            "ratio": self.ratio,
            "capacity_mode": self.capacity_mode,
        }


def capacity_model(n: int, cfg: Any = None) -> CapacityModel:
    """Build the :class:`CapacityModel` for a graph with ``n`` nodes."""
    theta = float(getattr(cfg, "theta", DEFAULT_THETA)) if cfg is not None else DEFAULT_THETA
    ratio = (
        float(getattr(cfg, "controller_ratio", DEFAULT_CONTROLLER_RATIO))
        if cfg is not None else DEFAULT_CONTROLLER_RATIO
    )
    override = getattr(cfg, "n_controllers", None) if cfg is not None else None
    demand = int(getattr(cfg, "demand", DEFAULT_DEMAND)) if cfg is not None else DEFAULT_DEMAND
    mode = str(getattr(cfg, "capacity_mode", "formula")) if cfg is not None else "formula"
    explicit = getattr(cfg, "capacity_explicit", None) if cfg is not None else None

    if demand < 1:
        raise CapacityError(f"demand must be >= 1, got {demand}")

    n_c = n_controllers_for(n, ratio=ratio, override=override)
    n_switches = n - n_c

    if mode == "formula":
        capacity = capacity_per_controller(n, n_c, theta=theta)
    elif mode == "explicit":
        if explicit is None:
            raise CapacityError("capacity_mode='explicit' requires capacity_explicit")
        capacity = int(explicit)
        if capacity < 1:
            raise CapacityError(f"capacity_explicit must be >= 1, got {capacity}")
    else:
        raise CapacityError(f"unknown capacity_mode {mode!r} (expected 'formula' or 'explicit')")

    total = n_c * capacity
    return CapacityModel(
        theta=theta,
        n_controllers=n_c,
        capacity=capacity,
        n_switches=n_switches,
        total_capacity=total,
        aggregate_slack=total - n_switches * demand,
        demand=demand,
        ratio=ratio,
        capacity_mode=mode,
    )


@dataclass(frozen=True)
class ComponentCapacity:
    """Aggregate capacity audit of one connected component."""

    n_nodes: int
    n_controllers: int
    demand: int
    capacity: int
    feasible: bool
    controllerless: bool

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view."""
        return {
            "n_nodes": self.n_nodes,
            "n_controllers": self.n_controllers,
            "demand": self.demand,
            "capacity": self.capacity,
            "feasible": self.feasible,
            "controllerless": self.controllerless,
        }


def component_capacity_audit(graph: nx.Graph, controllers: Sequence[int], *,
                             model: CapacityModel,
                             alive: Iterable[int] | None = None
                             ) -> tuple[ComponentCapacity, ...]:
    """Return the per-component capacity audit of the ``alive`` subgraph."""
    alive_set = (
        {int(node) for node in graph.nodes()}
        if alive is None else {int(node) for node in alive}
    )
    controller_set = {int(node) for node in controllers} & alive_set
    subgraph = graph.subgraph(alive_set)
    components = [set(component) for component in nx.connected_components(subgraph)]
    components.sort(key=lambda component: min(component) if component else -1)

    audit: list[ComponentCapacity] = []
    for component in components:
        n_controllers = len(component & controller_set)
        n_nodes = len(component)
        demand = (n_nodes - n_controllers) * int(model.demand)
        capacity = n_controllers * int(model.capacity)
        audit.append(
            ComponentCapacity(
                n_nodes=n_nodes,
                n_controllers=n_controllers,
                demand=demand,
                capacity=capacity,
                feasible=demand <= capacity,
                controllerless=n_controllers == 0,
            )
        )
    return tuple(audit)


@dataclass(frozen=True)
class UtilisationProbe:
    """Result of :func:`probe_max_utilisation` — loads only, nothing deleted."""

    controllers: tuple[int, ...]
    loads_nearest: tuple[int, ...]
    utilisation_nearest: tuple[float, ...]
    u_nearest: float
    loads_after_spill: tuple[int, ...]
    utilisation_after_spill: tuple[float, ...]
    u_after_spill: float
    feasible: bool
    capacity_binds: bool
    ceiling_active: bool
    feasible_but_spill_failed: bool
    n_overloaded: int
    n_unassigned: int
    n_spilled: int
    n_controllerless_components: int
    capacity: int
    n_switches: int
    n_at_ceiling: int = 0
    controller: int | None = None
    components: tuple[ComponentCapacity, ...] = ()

    @property
    def max_utilisation(self) -> float:
        """Alias of :attr:`u_nearest`."""
        return self.u_nearest

    @property
    def max_utilisation_feasible(self) -> float:
        """Alias of :attr:`u_after_spill`."""
        return self.u_after_spill

    @property
    def loads(self) -> tuple[int, ...]:
        """Alias of :attr:`loads_nearest`."""
        return self.loads_nearest

    @property
    def utilisation(self) -> tuple[float, ...]:
        """Alias of :attr:`utilisation_nearest`."""
        return self.utilisation_nearest

    @property
    def binds(self) -> bool:
        """Alias of :attr:`capacity_binds` — ``not feasible`` or unplaced switches."""
        return self.capacity_binds

    @property
    def spill_fraction(self) -> float:
        """Share of switches that leave their nearest controller (``0.0`` if none)."""
        return (float(self.n_spilled) / float(self.n_switches)) if self.n_switches else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view, with a one-line verdict for tables."""
        return {
            "controllers": list(self.controllers),
            "loads_nearest": list(self.loads_nearest),
            "utilisation_nearest": list(self.utilisation_nearest),
            "u_nearest": self.u_nearest,
            "loads_after_spill": list(self.loads_after_spill),
            "utilisation_after_spill": list(self.utilisation_after_spill),
            "u_after_spill": self.u_after_spill,
            "feasible": self.feasible,
            "capacity_binds": self.capacity_binds,
            "ceiling_active": self.ceiling_active,
            "n_at_ceiling": self.n_at_ceiling,
            "feasible_but_spill_failed": self.feasible_but_spill_failed,
            "n_overloaded": self.n_overloaded,
            "n_unassigned": self.n_unassigned,
            "n_spilled": self.n_spilled,
            "spill_fraction": self.spill_fraction,
            "n_controllerless_components": self.n_controllerless_components,
            "capacity": self.capacity,
            "n_switches": self.n_switches,
            "controller": self.controller,
            "components": [component.to_dict() for component in self.components],
        }


def probe_max_utilisation(graph: nx.Graph, controllers: Sequence[int],
                          model: CapacityModel, *,
                          eps: float = 1e-9) -> UtilisationProbe:
    """Measure the load profile of both assignments, then judge feasibility."""
    from .assignment import assign_single_controller

    ordered = tuple(sorted({int(node) for node in controllers}))
    missing = [node for node in ordered if node not in graph]
    if missing:
        raise CapacityError(f"controller {missing} is not in the graph (n={graph.number_of_nodes()})")

    blind = assign_single_controller(graph, ordered, model, enforce_capacity=False)
    enforced = assign_single_controller(graph, ordered, model, enforce_capacity=True)

    capacity = int(model.capacity)
    loads_nearest = tuple(int(blind.loads.get(node, 0)) for node in ordered)
    utilisation_nearest = tuple(
        (float(load) / float(capacity)) if capacity > 0 else 0.0
        for load in loads_nearest
    )
    if utilisation_nearest:
        max_index = max(
            range(len(utilisation_nearest)),
            key=lambda i: (utilisation_nearest[i], -ordered[i]),
        )
        u_nearest = utilisation_nearest[max_index]
        worst = ordered[max_index]
    else:
        u_nearest = 0.0
        worst = None
    n_overloaded = sum(1 for value in utilisation_nearest if value > 1.0 + eps)

    loads_after_spill = tuple(int(enforced.loads.get(node, 0)) for node in ordered)
    utilisation_after_spill = tuple(
        (float(load) / float(capacity)) if capacity > 0 else 0.0
        for load in loads_after_spill
    )
    u_after_spill = max(utilisation_after_spill) if utilisation_after_spill else 0.0
    n_at_ceiling = sum(1 for load in loads_after_spill if load == capacity)

    components = component_capacity_audit(graph, ordered, model=model)
    feasible = all(component.feasible for component in components)
    n_controllerless = sum(1 for component in components if component.controllerless)
    n_unassigned = len(enforced.unassigned)
    spilled = sum(
        1
        for node, controller in blind.owner.items()
        if enforced.owner.get(node) != controller
    ) + n_unassigned

    defect = feasible and (u_after_spill > 1.0 + eps or n_unassigned > 0)
    if defect:  # pragma: no cover - guarded by the greedy argument in the module doc
        raise CapacityError(
            "aggregate capacity is feasible but the spill failed: "
            f"u_after_spill={u_after_spill:.6f}，n_unassigned={n_unassigned}。"
            "this is a defect in the assignment, not a property of the data."
        )

    return UtilisationProbe(
        controllers=ordered,
        loads_nearest=loads_nearest,
        utilisation_nearest=utilisation_nearest,
        u_nearest=u_nearest,
        loads_after_spill=loads_after_spill,
        utilisation_after_spill=utilisation_after_spill,
        u_after_spill=u_after_spill,
        feasible=feasible,
        capacity_binds=(not feasible) or n_unassigned > 0,
        ceiling_active=n_at_ceiling > 0,
        feasible_but_spill_failed=defect,
        n_overloaded=n_overloaded,
        n_unassigned=n_unassigned,
        n_spilled=spilled,
        n_controllerless_components=n_controllerless,
        capacity=capacity,
        n_switches=model.n_switches,
        n_at_ceiling=n_at_ceiling,
        controller=worst,
        components=components,
    )
