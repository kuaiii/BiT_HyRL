"""Cascading-failure engine with ``structural`` and ``capacity`` modes.

Both modes share one loop; they differ only in what happens after a node is
removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx

from .assignment import Assignment, assign_single_controller, reassign_excess
from .capacity import CapacityModel

__all__ = [
    "CASCADE_MODES",
    "CascadeError",
    "CascadeStep",
    "CascadeTrace",
    "recompute_loads",
    "run_cascade",
    "surviving_components",
]

#: Supported cascade modes, in the design's order of preference.
CASCADE_MODES: tuple[str, ...] = ("structural", "capacity")


class CascadeError(RuntimeError):
    """Raised when a cascade trace violates its own audit identity."""


def surviving_components(graph: nx.Graph, alive: Iterable[int]) -> list[set[int]]:
    """Return the connected components of the subgraph induced by ``alive``."""
    alive_set = {int(node) for node in alive}
    subgraph = graph.subgraph(alive_set)
    components = [set(component) for component in nx.connected_components(subgraph)]
    components.sort(key=lambda component: min(component) if component else -1)
    return components


def recompute_loads(owner: dict[int, int], controllers_alive: Iterable[int],
                    alive: Iterable[int], demand: int = 1) -> dict[int, int]:
    """Return ``{controller: L_j}`` for the surviving assignment."""
    alive_set = {int(node) for node in alive}
    survivors = sorted({int(c) for c in controllers_alive})
    loads = {controller: 0 for controller in survivors}
    survivor_set = set(survivors)
    for node, controller in owner.items():
        if node in alive_set and controller in survivor_set:
            loads[controller] += demand
    return loads


@dataclass(frozen=True)
class CascadeStep:
    """One emitted state of a cascade."""

    step: int
    f_direct: float
    f_total: float
    alive_nodes: tuple[int, ...]
    controllers_alive: tuple[int, ...]
    loads: dict[int, int]
    max_utilisation: float
    reassigned: int
    removed_components: tuple[tuple[int, ...], ...]
    gcc_size: int
    converged: bool = True
    iterations: int = 0

    def to_dict(self, *, with_nodes: bool = False) -> dict[str, Any]:
        """Plain-``dict`` view."""
        payload: dict[str, Any] = {
            "step": self.step,
            "f_direct": self.f_direct,
            "f_total": self.f_total,
            "n_alive": len(self.alive_nodes),
            "controllers_alive": list(self.controllers_alive),
            "loads": {str(k): v for k, v in sorted(self.loads.items())},
            "max_utilisation": self.max_utilisation,
            "reassigned": self.reassigned,
            "removed_components": [list(component) for component in self.removed_components],
            "gcc_size": self.gcc_size,
            "converged": self.converged,
            "iterations": self.iterations,
        }
        if with_nodes:
            payload["alive_nodes"] = list(self.alive_nodes)
        return payload


@dataclass(frozen=True)
class CascadeTrace:
    """A full cascade run."""

    mode: str
    direct_sequence: tuple[int, ...]
    steps: tuple[CascadeStep, ...]
    iterations_per_step: tuple[int, ...]
    final_fraction: float
    n_nodes: int
    n_controllers: int
    audit_ok: bool = True

    def to_dict(self, *, with_nodes: bool = False) -> dict[str, Any]:
        """Plain-``dict`` view of the whole trace."""
        return {
            "mode": self.mode,
            "n_nodes": self.n_nodes,
            "n_controllers": self.n_controllers,
            "length": len(self.direct_sequence),
            "final_fraction": self.final_fraction,
            "iterations_per_step": list(self.iterations_per_step),
            "audit_ok": self.audit_ok,
            "steps": [step.to_dict(with_nodes=with_nodes) for step in self.steps],
        }

    @property
    def f_grid(self) -> tuple[float, ...]:
        """``f_direct`` of every step — the x-axis of the ``r_curve``."""
        return tuple(step.f_direct for step in self.steps)

    @property
    def r_curve(self) -> tuple[float, ...]:
        """``gcc_size / N`` of every step — the y-axis of the ``r_curve``."""
        return tuple(
            (step.gcc_size / self.n_nodes) if self.n_nodes else 0.0 for step in self.steps
        )


def run_cascade(graph: nx.Graph, attack_sequence: Sequence[int],
                controllers: Sequence[int], model: CapacityModel,
                cfg: Any = None) -> CascadeTrace:
    """Run the attack sequence and return the full trace."""
    mode = str(getattr(cfg, "mode", "structural")) if cfg is not None else "structural"
    max_iter = int(getattr(cfg, "max_iter", 50)) if cfg is not None else 50
    policy = (
        str(getattr(cfg, "reassign_policy", "nearest_with_capacity"))
        if cfg is not None else "nearest_with_capacity"
    )
    if mode not in CASCADE_MODES:
        raise CascadeError(f"unknown cascade mode {mode!r}, expected one of {list(CASCADE_MODES)}")
    if policy != "nearest_with_capacity":
        raise CascadeError(
            f"unknown reassign_policy {policy!r} (only 'nearest_with_capacity' is supported)"
        )
    if max_iter < 1:
        raise CascadeError(f"max_iter must be >= 1, got {max_iter}")

    nodes = sorted(int(node) for node in graph.nodes())
    node_set = set(nodes)
    n = len(nodes)
    initial_controllers = sorted({int(c) for c in controllers})
    unknown = [node for node in initial_controllers if node not in node_set]
    if unknown:
        raise CascadeError(f"controller {unknown} is not in the graph (n={n})")

    alive: set[int] = set(nodes)
    controllers_alive: set[int] = set(initial_controllers)

    if n - len(initial_controllers) > 0:
        owner: dict[int, int] = assign_single_controller(
            graph, initial_controllers, model, enforce_capacity=False
        ).owner
    else:
        owner = {}

    steps: list[CascadeStep] = []
    iterations_per_step: list[int] = []
    direct_removed = 0
    cascade_removed = 0
    audit_ok = True

    for index, raw_node in enumerate(attack_sequence):
        node = int(raw_node)
        if node not in node_set:
            raise CascadeError(f"attack sequence contains node {node}, outside the graph (n={n})")

        reassigned_this_step = 0
        removed_components: list[tuple[int, ...]] = []
        iterations = 0
        converged = True

        if node in alive:
            alive.discard(node)
            direct_removed += 1
            controllers_alive.discard(node)
            owner.pop(node, None)

            if mode == "structural":
                for component in surviving_components(graph, alive):
                    if component.isdisjoint(controllers_alive):
                        alive -= component
                        cascade_removed += len(component)
                        removed_components.append(tuple(sorted(component)))
            else:
                assignment = Assignment(
                    owner=dict(owner),
                    loads={},
                    unassigned=(),
                    order_used=(),
                    is_single_valued=True,
                    controllers=tuple(sorted(controllers_alive)),
                    capacity=model.capacity,
                    demand=model.demand,
                )
                while True:
                    iterations += 1
                    if not controllers_alive:
                        cascade_removed += len(alive)
                        removed_components.append(tuple(sorted(alive)))
                        alive.clear()
                        converged = True
                        break
                    loads = recompute_loads(
                        owner, controllers_alive, alive, demand=model.demand
                    )
                    overdrawn = [
                        controller for controller in sorted(loads)
                        if loads[controller] > model.capacity
                    ]
                    orphaned = tuple(
                        sorted(
                            node for node in alive
                            if node not in controllers_alive
                            and (node not in owner or owner[node] not in controllers_alive)
                        )
                    )
                    if not overdrawn and not orphaned:
                        converged = True
                        break
                    if iterations >= max_iter:
                        converged = False
                        break
                    assignment = Assignment(
                        owner=dict(owner),
                        loads=dict(loads),
                        unassigned=orphaned,
                        order_used=(),
                        is_single_valued=True,
                        controllers=tuple(sorted(controllers_alive)),
                        capacity=model.capacity,
                        demand=model.demand,
                    )
                    assignment, moved = reassign_excess(
                        graph, assignment, model,
                        alive_nodes=alive,
                        controllers_alive=controllers_alive,
                    )
                    reassigned_this_step += moved
                    owner = dict(assignment.owner)
                    failed = set(assignment.unassigned)
                    if failed:
                        alive -= failed
                        cascade_removed += len(failed)
                    for component in surviving_components(graph, alive):
                        if component.isdisjoint(controllers_alive):
                            alive -= component
                            cascade_removed += len(component)
                            removed_components.append(tuple(sorted(component)))
                    if not assignment.owner and not overdrawn:
                        converged = True
                        break

        loads = recompute_loads(owner, controllers_alive, alive, demand=model.demand)
        max_utilisation = (
            max((load / model.capacity) for load in loads.values())
            if loads and model.capacity > 0 else 0.0
        )
        components = surviving_components(graph, alive)
        gcc_size = max((len(component) for component in components), default=0)

        if direct_removed + cascade_removed + len(alive) != n:
            audit_ok = False
            raise CascadeError(
                "audit identity violated: "
                f"direct={direct_removed} + cascade={cascade_removed} + alive={len(alive)} "
                f"!= N={n}（step {index}）"
            )

        steps.append(
            CascadeStep(
                step=index,
                f_direct=direct_removed / n if n else 0.0,
                f_total=1.0 - len(alive) / n if n else 0.0,
                alive_nodes=tuple(sorted(alive)),
                controllers_alive=tuple(sorted(controllers_alive)),
                loads=loads,
                max_utilisation=max_utilisation,
                reassigned=reassigned_this_step,
                removed_components=tuple(removed_components),
                gcc_size=gcc_size,
                converged=converged,
                iterations=iterations,
            )
        )
        iterations_per_step.append(iterations)

    final_fraction = steps[-1].f_total if steps else 0.0
    if direct_removed + cascade_removed + len(alive) != n:  # pragma: no cover
        audit_ok = False

    return CascadeTrace(
        mode=mode,
        direct_sequence=tuple(int(node) for node in attack_sequence),
        steps=tuple(steps),
        iterations_per_step=tuple(iterations_per_step),
        final_fraction=final_fraction,
        n_nodes=n,
        n_controllers=len(initial_controllers),
        audit_ok=audit_ok,
    )
