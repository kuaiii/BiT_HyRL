"""Cascading-failure engine: ``structural`` (default) and ``capacity``.

Two modes, one loop, and the difference is *only* what happens after a node dies:

``structural`` — the published protocol
    Delete the attacked node.  Then delete, **as a whole block**, every connected
    component of the surviving subgraph that contains no surviving controller.
    Nothing is re-assigned and capacity is never consulted.  This is the semantics
    the already-published numbers were produced under, so it stays the default and
    is not allowed to drift.

``capacity`` — opt-in, off by default
    Delete the attacked node, then iterate to a fixed point:

    1. recompute ``L_j`` from the current assignment;
    2. find the overdrawn controllers (``L_j > C``);
    3. hand their surplus to :func:`~bit_hyrl.control.assignment.reassign_excess`,
       which moves each surplus node to the nearest controller with room;
    4. nodes with nowhere to go fail (they leave the alive set);
    5. delete controller-less components again;
    6. repeat, up to ``max_iter``.

    Because step 1 uses the *capacity-blind* nearest assignment as the starting
    point, overload is a real event rather than a tautology; and because the loop
    is bounded, ``max_iter = 1`` leaves the trace visibly non-converged
    (``CascadeStep.converged = False``), which
    ``tests/test_cascade_capacity.py`` asserts directly.

The audit identity
------------------
Every step carries counts that must satisfy

.. code-block:: text

    direct_removed + cascade_removed + alive == N

and :func:`run_cascade` re-checks it *after every step*, not just at the end.
That is what makes "``f_total`` went up" attributable: a rise in ``f_total`` that
is not explained by ``f_direct`` must come from ``cascade_removed``, and the two
can never disagree silently.
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
    """Return the connected components of the subgraph induced by ``alive``.

    Args:
        graph: The full graph.
        alive: Surviving node ids.

    Returns:
        Components as sets, ordered by their smallest node id so the result is
        independent of iteration order.
    """
    alive_set = {int(node) for node in alive}
    subgraph = graph.subgraph(alive_set)
    components = [set(component) for component in nx.connected_components(subgraph)]
    components.sort(key=lambda component: min(component) if component else -1)
    return components


def recompute_loads(owner: dict[int, int], controllers_alive: Iterable[int],
                    alive: Iterable[int], demand: int = 1) -> dict[int, int]:
    """Return ``{controller: L_j}`` for the surviving assignment.

    A switch contributes to ``L_j`` only if it is alive **and** its owner is
    alive; a switch whose owner died is simply not counted (it is picked up by
    the re-assignment step of the capacity mode, or by the controller-less
    component removal of the structural mode).

    Args:
        owner: ``switch -> controller`` map.
        controllers_alive: Surviving controller ids.
        alive: Surviving node ids.
        demand: Per-switch demand.

    Returns:
        A dict with an entry for **every** surviving controller, zeros included,
        so callers never have to guard against missing keys.
    """
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
    """One emitted state of a cascade.

    Attributes:
        step: Zero-based index into :attr:`CascadeTrace.direct_sequence`.  A step
            is emitted for **every** entry of the attack sequence, including the
            ones that turn out to be already dead, so ``len(steps)`` always equals
            ``len(direct_sequence)`` and ``r_curve``/``f_grid`` can be interpolated
            without guessing.
        f_direct: Cumulative directly-attacked fraction, ``direct_removed / N``.
        f_total: Cumulative failed fraction, ``1 - |alive| / N``.
        alive_nodes: Surviving node ids, ascending.
        controllers_alive: Surviving controller ids, ascending.
        loads: ``{controller: L_j}`` for the surviving controllers.
        max_utilisation: ``max_j L_j / C`` (``0.0`` when no controller survives).
        reassigned: Nodes moved by :func:`reassign_excess` during this step.
        removed_components: Components deleted as a block during this step.
        gcc_size: Size of the largest surviving component.
        converged: ``capacity`` mode only — ``True`` when the fixed-point loop
            exited because nothing was overloaded, ``False`` when it hit
            ``max_iter``.  Always ``True`` in ``structural`` mode.
        iterations: Fixed-point iterations consumed by this step (``0`` in
            ``structural`` mode).
    """

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
        """Plain-``dict`` view.

        Args:
            with_nodes: Include the full surviving node list.  Off by default so
                a 500-step trace stays readable.

        Returns:
            A JSON-ready dict.
        """
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
    """A full cascade run.

    Attributes:
        mode: ``"structural"`` or ``"capacity"``.
        direct_sequence: The attack sequence actually consumed (as given).
        steps: One :class:`CascadeStep` per attack-sequence entry.
        iterations_per_step: Fixed-point iterations per step (all zero in
            ``structural`` mode).
        final_fraction: ``f_total`` of the last step (``0.0`` for an empty
            sequence).
        n_nodes: ``N`` of the graph the cascade ran on.
        n_controllers: Initial controller count.
        audit_ok: Whether ``direct + cascade + alive == N`` held at every step.
    """

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
    """Run the attack sequence and return the full trace.

    Args:
        graph: The evaluated graph (node ids ``0 .. n-1``).
        attack_sequence: Node ids in attack order.  Entries that are already dead
            are consumed but change nothing (``f`` stays monotone by
            construction).
        controllers: Initial controller ids.
        model: Solved :class:`CapacityModel`; ``capacity`` is used *only* in
            ``capacity`` mode.
        cfg: A ``CascadeConfig``-like object with ``mode`` (default
            ``"structural"``), ``max_iter`` (default 50) and ``reassign_policy``
            (default ``"nearest_with_capacity"``), or ``None`` for defaults.

    Returns:
        A :class:`CascadeTrace`.

    Raises:
        CascadeError: If ``mode`` is unknown, if a controller or attack node is
            not a node of ``graph``, or if the audit identity fails at any step.
    """
    mode = str(getattr(cfg, "mode", "structural")) if cfg is not None else "structural"
    max_iter = int(getattr(cfg, "max_iter", 50)) if cfg is not None else 50
    policy = (
        str(getattr(cfg, "reassign_policy", "nearest_with_capacity"))
        if cfg is not None else "nearest_with_capacity"
    )
    if mode not in CASCADE_MODES:
        raise CascadeError(f"未知级联模式 {mode!r}，支持 {list(CASCADE_MODES)}")
    if policy != "nearest_with_capacity":
        raise CascadeError(
            f"未知 reassign_policy {policy!r}（当前仅支持 'nearest_with_capacity'）"
        )
    if max_iter < 1:
        raise CascadeError(f"max_iter 必须 >= 1，收到 {max_iter}")

    nodes = sorted(int(node) for node in graph.nodes())
    node_set = set(nodes)
    n = len(nodes)
    initial_controllers = sorted({int(c) for c in controllers})
    unknown = [node for node in initial_controllers if node not in node_set]
    if unknown:
        raise CascadeError(f"控制器 {unknown} 不在图里（n={n}）")

    alive: set[int] = set(nodes)
    controllers_alive: set[int] = set(initial_controllers)

    # Capacity mode starts from the capacity-blind nearest assignment, so that an
    # overload is a measurable event; structural mode keeps a fixed assignment
    # purely for the load bookkeeping reported in each step.
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
            raise CascadeError(f"攻击序列含图外节点 {node}（n={n}）")

        reassigned_this_step = 0
        removed_components: list[tuple[int, ...]] = []
        iterations = 0
        converged = True

        if node in alive:
            alive.discard(node)
            direct_removed += 1
            controllers_alive.discard(node)
            owner.pop(node, None)

            # ---- structural mode: remove controller-less components as blocks - #
            if mode == "structural":
                for component in surviving_components(graph, alive):
                    if component.isdisjoint(controllers_alive):
                        alive -= component
                        cascade_removed += len(component)
                        removed_components.append(tuple(sorted(component)))
            # ---- capacity mode: overload -> reassign -> remove -> repeat ------ #
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
                        # Every controller is gone: everything left fails at once.
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
                    # Orphaned switches matter as much as overdrawn controllers.
                    # A node whose controller died is *not* counted in ``loads`` at
                    # all, so an overload test alone declares "converged" while the
                    # component is genuinely over capacity: e.g. 5 surviving
                    # controllers with C = 15 serve 75 switches, but 90 switches are
                    # alive because 45 of them still point at dead controllers.  A
                    # stale ``owner`` entry is as orphaned as a missing one, so both
                    # are fed back into the assignment.
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

        # ---- bookkeeping and the audit identity --------------------------- #
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
                "审计恒等式被破坏："
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
