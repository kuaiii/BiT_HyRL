"""Single-valued controller assignment with a strict total order.

Five properties, all of them asserted rather than assumed:

**A. Single-valued (mutually exclusive).**
Every switch node has exactly one owner.  ``len(owner) + len(unassigned) ==
n_switches`` and the two sets are disjoint, so a node can be either owned once or
unowned, never owned twice.  ``Assignment.is_single_valued`` is ``True`` by
construction; the test-suite recomputes it independently.

**B. Deterministic total order.**
Candidates are the triples ``(dist, node_id, controller_id)`` consumed in
ascending order — distance first, then node id, then controller id.  No
implementation here iterates a ``dict`` or a ``set`` in a way that can influence a
decision; the input node order therefore cannot change the output, which is
exactly what ``test_assignment_determinism.py`` shuffles to check.

**C. Complexity.**
One BFS per controller (``O(n_c · (N + M))``) plus ``O(N · n_c log(n_c))`` for the
sort.  Milliseconds on BA-500.

**D. Overflow spill and excess relocation.**
A switch that cannot fit its nearest controller **spills to the next-nearest one
with room** (this is the "capacity overflow" behaviour, not an error).  When a
controller is overdrawn, :func:`reassign_excess` moves its surplus nodes to the
nearest controller that still has room; nodes with nowhere to go become
unassigned and the caller treats them as failed.

**E. Never load-balance — the H_C metric depends on it.**
This is a hard constraint, not a default.  ``H_C`` measures the
**evenness of the load shares** ``L_j / ΣL``; a balancing assignment would drive
it to ``1`` by construction and destroy the metric's discriminating power — every
topology, every ``c`` and every degree of attack severity would score the same.
The assignment is therefore **minimal-deviation**: nearest first, and a switch
moves only when capacity forces it to.  Concretely:

* a switch is never moved to a farther controller while a nearer one has room;
* an overdrawn controller never sheds load "for fairness" — only its *surplus*
  beyond ``C`` is shed, and it is shed farthest-served-first;
* no global objective (max-min load, sum of latencies, variance) is ever solved.

If a future change makes ``H_C`` collapse to a constant, the cause is a violation
of this paragraph, not a property of the network.

Two deliberate reading choices are documented where they occur, because the
specification leaves them open:

* which nodes an overdrawn controller gives up — :func:`reassign_excess` evicts
  the **farthest-served first** (``-dist`` then ``node_id``); evicting the
  best-served node to a *farther* controller would be irrational;
* :attr:`Assignment.order_used` records the whole total order, but
  :meth:`Assignment.to_dict` reports only its length so that a 500-node record
  does not carry 22 500 triples.
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
    """A single-valued switch → controller map.

    Attributes:
        owner: ``switch_id -> controller_id``.
        loads: ``controller_id -> L_j`` (in units of ``demand``).
        unassigned: Switches with no controller (every candidate was full).
        order_used: The full ``(dist, node_id, controller_id)`` total order, as
            consumed.  ``to_dict`` reports only the length.
        is_single_valued: Always ``True``; kept as a field so callers can assert.
        controllers: Controller ids, ascending.
        capacity: The ``C`` this assignment was built against.
        demand: Per-switch demand.
        enforce_capacity: Whether the capacity ceiling was applied.
    """

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
    """Return ``{controller: {node: distance}}`` for every controller in one sweep.

    Args:
        graph: Undirected graph.
        controllers: Controller node ids (already validated to exist).

    Returns:
        A dict of BFS distance maps.  Nodes a controller cannot reach are simply
        absent.

    Raises:
        CapacityError: If a controller is not a node of ``graph``.
    """
    rows: dict[int, dict[int, int]] = {}
    for controller in controllers:
        if controller not in graph:
            raise CapacityError(f"控制器 {controller} 不在图里（n={graph.number_of_nodes()}）")
        rows[int(controller)] = dict(nx.single_source_shortest_path_length(graph, controller))
    return rows


def assign_single_controller(graph: nx.Graph, controllers: Sequence[int],
                             model: CapacityModel, *,
                             enforce_capacity: bool = True) -> Assignment:
    """Assign every switch to one controller, nearest first with spill-over.

    Args:
        graph: Evaluated graph; controllers must be nodes of it.
        controllers: Controller node ids (duplicates collapse, order irrelevant).
        model: Solved :class:`~bit_hyrl.control.capacity.CapacityModel`.
        enforce_capacity: When ``True`` (default) a switch spills to the next
            nearest controller with room and never exceeds ``C``.  When
            ``False`` every switch simply takes its nearest controller, which can
            overload — that is the *diagnostic* profile
            :func:`~bit_hyrl.control.capacity.probe_max_utilisation` reports,
            because a capacity-feasible profile has ``U_j <= 1`` by construction
            and would answer "does θ bind?" with a tautology.

    Returns:
        A single-valued :class:`Assignment`.

    Raises:
        CapacityError: If a controller is not a node of ``graph``, or if the
            controller set covers every node (no switches left to assign).
    """
    ordered = tuple(sorted({int(node) for node in controllers}))
    if not ordered:
        raise CapacityError("controllers 为空，无法指派")
    missing = [node for node in ordered if node not in graph]
    if missing:
        raise CapacityError(f"控制器 {missing} 不在图里（n={graph.number_of_nodes()}）")

    node_list = sorted(int(node) for node in graph.nodes())
    controller_set = set(ordered)
    switch_set = [node for node in node_list if node not in controller_set]
    capacity = int(model.capacity)
    demand = int(model.demand)

    dist = bfs_rows(graph, ordered)

    # The deterministic total order: distance, then node id, then controller id.
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
        raise CapacityError("指派不是单值的：内部一致性检查失败")

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
    """Relocate the surplus of overdrawn controllers.

    Nodes whose owner is dead, or which can no longer reach their owner, are
    treated as pending too, so a single call covers both "the controller is
    overdrawn" and "the network moved underneath the assignment".

    Eviction order is **farthest-served first** then node id: a controller gives
    up the nodes it serves worst, because moving a well-served node to a farther
    controller would raise latency for no gain.

    Args:
        graph: Full graph (edges are used for distances).
        assignment: The current assignment.
        model: Solved capacity model (``capacity`` and ``demand`` are used).
        alive_nodes: Surviving nodes; defaults to every node of ``graph``.
        controllers_alive: Surviving controllers; defaults to
            ``assignment.controllers``.

    Returns:
        ``(new_assignment, moved)`` where ``moved`` counts nodes that found a new
        controller.  Nodes that found none end up in ``new_assignment.unassigned``.

    Raises:
        CapacityError: If no controller survives.
    """
    alive = set(int(node) for node in graph.nodes()) if alive_nodes is None else {
        int(node) for node in alive_nodes
    }
    survivors = (
        tuple(sorted(set(int(c) for c in assignment.controllers if int(c) in alive)))
        if controllers_alive is None
        else tuple(sorted({int(c) for c in controllers_alive if int(c) in alive}))
    )
    if not survivors:
        raise CapacityError("全部控制器失效，reassign_excess 无目标可指派")

    capacity = int(model.capacity)
    demand = int(model.demand)
    dist = bfs_rows(graph, survivors)

    owner: dict[int, int] = {}
    loads: dict[int, int] = {controller: 0 for controller in survivors}
    # Nodes already known to have no owner (e.g. their controller died before this
    # call) must be retried, not ignored: they carry demand that no ``loads``
    # entry accounts for, so skipping them would let an over-capacity component
    # look converged.
    pending: list[int] = [int(node) for node in assignment.unassigned if int(node) in alive]

    for node in sorted(assignment.owner):
        controller = assignment.owner[node]
        if node not in alive or controller not in survivors or node not in dist[controller]:
            if node in alive:
                pending.append(node)
            continue
        owner[node] = controller
        loads[controller] += demand

    # -- evict the surplus of every overdrawn controller -------------------- #
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

    # -- place the pending nodes, node id ascending for determinism --------- #
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
