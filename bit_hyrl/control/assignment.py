"""Single-valued controller assignment with a deterministic tie-break order.

Every switch is served by exactly one controller, and the assignment is a pure
function of the graph, the controller set and the tie-break rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx

from .capacity import CapacityError, CapacityModel

__all__ = [
    "Assignment",
    "assign_single_controller",
    "bfs_rows",
    "reassign_excess",
]

#: Sentinel distance used when a controller cannot see a node at all.
_UNREACHABLE = 10**9


@dataclass(frozen=True)
class Assignment:
    """A single-valued switch → controller map."""

    owner: dict[int, int]
    loads: dict[int, int]
    unassigned: tuple[int, ...]
    order_used: tuple[tuple[int, int, int], ...]
    is_single_valued: bool
    controllers: tuple[int, ...] = ()
    capacity: int = 0
    demand: int = 1
    enforce_capacity: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view; ``order_used`` is summarised by its length."""
        return {
            "owner": {str(node): controller for node, controller in sorted(self.owner.items())},
            "loads": {str(controller): load for controller, load in sorted(self.loads.items())},
            "unassigned": list(self.unassigned),
            "n_owner": len(self.owner),
            "n_unassigned": len(self.unassigned),
            "order_used_length": len(self.order_used),
            "is_single_valued": self.is_single_valued,
            "controllers": list(self.controllers),
            "capacity": self.capacity,
            "demand": self.demand,
            "enforce_capacity": self.enforce_capacity,
        }

    @property
    def n_assigned(self) -> int:
        """Number of switches with an owner."""
        return len(self.owner)


def bfs_rows(graph: nx.Graph, controllers: Sequence[int]) -> dict[int, dict[int, int]]:
    """Return ``{controller: {node: distance}}`` for every controller in one sweep."""
    rows: dict[int, dict[int, int]] = {}
    for controller in controllers:
        if controller not in graph:
            raise CapacityError(f"controller {controller} is not in the graph (n={graph.number_of_nodes()})")
        rows[int(controller)] = dict(nx.single_source_shortest_path_length(graph, controller))
    return rows


def assign_single_controller(graph: nx.Graph, controllers: Sequence[int],
                             model: CapacityModel, *,
                             enforce_capacity: bool = True) -> Assignment:
    """Assign every switch to one controller, nearest first with spill-over."""
    ordered = tuple(sorted({int(node) for node in controllers}))
    if not ordered:
        raise CapacityError("controllers is empty, nothing to assign")
    missing = [node for node in ordered if node not in graph]
    if missing:
        raise CapacityError(f"controller {missing} is not in the graph (n={graph.number_of_nodes()})")

    node_list = sorted(int(node) for node in graph.nodes())
    controller_set = set(ordered)
    switch_set = [node for node in node_list if node not in controller_set]
    capacity = int(model.capacity)
    demand = int(model.demand)

    dist = bfs_rows(graph, ordered)

    triples: list[tuple[int, int, int]] = []
    for node in switch_set:
        for controller in ordered:
            reachable = dist[controller].get(node)
            if reachable is not None:
                triples.append((reachable, node, controller))
    triples.sort()

    owner: dict[int, int] = {}
    loads: dict[int, int] = {controller: 0 for controller in ordered}
    for distance, node, controller in triples:
        if node in owner:
            continue
        if enforce_capacity and loads[controller] + demand > capacity:
            continue
        owner[node] = controller
        loads[controller] += demand

    unassigned = tuple(node for node in switch_set if node not in owner)
    single_valued = (
        len(owner) + len(unassigned) == len(switch_set)
        and set(owner).isdisjoint(unassigned)
        and set(owner) <= set(switch_set)
        and set(owner.values()) <= set(ordered)
    )
    if not single_valued:  # pragma: no cover - structural guarantee
        raise CapacityError("assignment is not single-valued: internal consistency check failed")

    return Assignment(
        owner=owner,
        loads=loads,
        unassigned=unassigned,
        order_used=tuple(triples),
        is_single_valued=single_valued,
        controllers=ordered,
        capacity=capacity,
        demand=demand,
        enforce_capacity=enforce_capacity,
    )


def reassign_excess(graph: nx.Graph, assignment: Assignment, model: CapacityModel, *,
                    alive_nodes: Iterable[int] | None = None,
                    controllers_alive: Iterable[int] | None = None
                    ) -> tuple[Assignment, int]:
    """Relocate the surplus of overdrawn controllers."""
    alive = set(int(node) for node in graph.nodes()) if alive_nodes is None else {
        int(node) for node in alive_nodes
    }
    survivors = (
        tuple(sorted(set(int(c) for c in assignment.controllers if int(c) in alive)))
        if controllers_alive is None
        else tuple(sorted({int(c) for c in controllers_alive if int(c) in alive}))
    )
    if not survivors:
        raise CapacityError("every controller is down, reassign_excess has nowhere to assign")

    capacity = int(model.capacity)
    demand = int(model.demand)
    dist = bfs_rows(graph, survivors)

    owner: dict[int, int] = {}
    loads: dict[int, int] = {controller: 0 for controller in survivors}
    pending: list[int] = [int(node) for node in assignment.unassigned if int(node) in alive]

    for node in sorted(assignment.owner):
        controller = assignment.owner[node]
        if node not in alive or controller not in survivors or node not in dist[controller]:
            if node in alive:
                pending.append(node)
            continue
        owner[node] = controller
        loads[controller] += demand

    for controller in survivors:
        while loads[controller] > capacity:
            candidates = [node for node, held_by in owner.items() if held_by == controller]
            if not candidates:
                break  # pragma: no cover - loads is derived from owner
            worst = max(
                candidates,
                key=lambda node: (dist[controller].get(node, _UNREACHABLE), -node),
            )
            owner.pop(worst)
            loads[controller] -= demand
            pending.append(worst)

    moved = 0
    unassigned: list[int] = []
    for node in sorted(set(pending)):
        best: tuple[int, int] | None = None
        for controller in survivors:
            reachable = dist[controller].get(node)
            if reachable is None or loads[controller] + demand > capacity:
                continue
            key = (reachable, controller)
            if best is None or key < best:
                best = key
        if best is None:
            unassigned.append(node)
            continue
        owner[node] = best[1]
        loads[best[1]] += demand
        moved += 1

    single_valued = (
        len(owner) + len(unassigned) == len(alive - set(survivors))
        and set(owner).isdisjoint(unassigned)
    )
    result = Assignment(
        owner=owner,
        loads=loads,
        unassigned=tuple(sorted(unassigned)),
        order_used=(),
        is_single_valued=single_valued,
        controllers=survivors,
        capacity=capacity,
        demand=demand,
    )
    return result, moved
