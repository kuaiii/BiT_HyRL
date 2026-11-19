"""Controller-plane capacity model and the utilisation probe.

The model is deliberately trivial — one switch, one unit of demand, every
controller identical — because the *interesting* question is not how much
capacity exists but where it sits:

.. code-block:: text

    l_i(t) = 1                                    # unit demand, homogeneous
    n_c    = max(1, round_half_up(ratio · N))     # ratio = 0.1 by default
    C      = ceil((1 + theta) · N / n_c)          # theta = 0.5 by default

``round_half_up`` is :func:`math.floor(x + 0.5)`, **not** the builtin ``round``:
banker's rounding would map ``15.3 → 15`` and ``N/n_c = 10.2 → 1.5·10.2 = 15.3 →
16`` into something else for adjacent N, and the whole ``C`` table would stop
being reproducible.

⚠️ ``C`` is **not** always 15
------------------------------
The design document used to claim "homogeneous capacity ``C = 15``".  That is
only true when ``N / n_c`` happens to be exactly 10, which requires ``n_c`` to
divide ``N``.  The precise table for the nine evaluation graphs is:

=========== ==== ==== ======== ==== ============ ========= =============
graph        N    n_c  N/n_c    C    switches     total_cap aggregate_slack
=========== ==== ==== ======== ==== ============ ========= =============
BA-100      100   10   10.000   15   90           150       60
BA-200      200   20   10.000   15   180          300       120
BA-300      300   30   10.000   15   270          450       180
BA-400      400   40   10.000   15   360          600       240
BA-500      500   50   10.000   15   450          750       300
Cogentco    197   20    9.850   15   177          300       123
**Colt**    153   15   10.200   **16**  138        240       102
GtsCe       149   15    9.933   15   134          225       91
UsCarrier   158   16    9.875   15   142          240       98
=========== ==== ==== ======== ==== ============ ========= =============

**Colt is the only graph with ``C = 16``** (``round_half_up(15.3) = 15``,
``N/n_c = 10.2``, ``1.5 × 10.2 = 15.3``, ``ceil = 16``).  Any sentence in the
paper that says "isomorphic capacity ``C = 15``" must become "``C = 15``
(16 for Colt)", and the overload test in ``capacity`` cascading mode must use
**that graph's own ``C``**.

Why the probe exists
--------------------------------------------
The *smallest* aggregate slack over the nine graphs is BA-100's ``60`` — total
capacity exceeds total demand by 66 %.  **Aggregate capacity is therefore never
the binding constraint at** ``t = 0``, and any number that says otherwise is
measuring the wrong thing.  The probe consequently reports three quantities and
never conflates them:

``u_nearest``
    ``max_j L_j / C`` under the capacity-blind nearest assignment.  Measured:
    1.60 (BA-100) … 2.73 (BA-500) on the BA series, 1.13–1.69 on the real
    networks.  This says the degree-ranked placement is **spatially lopsided**;
    it says nothing about whether the budget is big enough, because on a
    connected graph every switch reaches every controller.
``u_after_spill``
    ``max_j L_j / C`` after the total-order assignment with capacity-enforced
    spill.  Measured: **1.000 on all nine** graphs, with **0** unassigned
    switches.  This is the utilisation that actually occurs.
``feasible``
    ``Σ demand <= Σ capacity`` per connected component.  **True on all nine**,
    which is the honest verdict at ``t = 0``: ``theta = 0.5`` does not bind
    before anything is attacked.

So the right sentence is not "capacity binds / does not bind" but "capacity is
slack at ``t = 0`` and becomes active only after the controller density is
driven below ``1/(C+1)``" — which is what
:func:`~bit_hyrl.control.feasibility.feasibility_curve` measures as a
function of the removal fraction.

⚠️ ``ceiling_active`` is **not** ``capacity_binds``
---------------------------------------------------
Two different questions were being answered with one boolean, and that is what
made the first headline wrong.  They are kept apart now:

``ceiling_active`` (:attr:`UtilisationProbe.ceiling_active`, with the count
:attr:`UtilisationProbe.n_at_ceiling`)
    "Is some controller exactly at ``C``?"  This is about the **shape** of the
    load: the degree-ranked placement is lopsided, so the total-order assignment
    trims every over-full catchment to ``C`` and pushes the surplus to the
    next-nearest controller that still has room.  Measured: **True on all nine
    graphs** (``u_after_spill == 1.000`` on all nine), with ``n_at_ceiling`` in
    ``3 … 14`` — e.g. ``3`` of ``10`` controllers on BA-100, ``14`` of ``50`` on
    BA-500 — and ``2.2 % … 18.9 %`` of the switches pushed away from their
    nearest controller (``n_spilled`` 3 … 68).  The ceiling is therefore
    *reached somewhere*, not *everywhere*: it says the constraint is **live** —
    it is doing work — and it refutes "the capacity model has no effect".
``capacity_binds`` (:attr:`UtilisationProbe.capacity_binds`)
    "Is the demand **unsatisfiable**?"  Strictly ``not feasible``, where
    ``feasible`` is the per-component aggregate test ``Σ demand <= Σ capacity``,
    **or** ``n_unassigned > 0`` — a switch that could not be placed at all.
    This is the *only* quantity the paper may call "capacity binds".  Measured:
    **False on all nine** at ``t = 0``.

A ``u_after_spill`` above 1 while ``feasible`` holds is an implementation defect
(not a data property) and :func:`probe_max_utilisation` raises on it.
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

#: Node counts of the nine evaluation graphs, in the design's table order.  Kept
#: here so the ``(n_c, C)`` contract can be asserted without loading any file.
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

#: ``{dataset: (n_c, C, n_switches, total_capacity, aggregate_slack)}`` — the
#: table, digit for digit.
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
    """Return ``n_c``, the number of controllers.

    Args:
        n: Node count.
        ratio: Controller ratio, default ``0.1``.
        override: Explicit ``n_c``; wins over ``ratio`` when not ``None``.

    Returns:
        ``max(1, round_half_up(ratio · n))``, or ``override``.

    Raises:
        CapacityError: On a non-positive ``n``, a non-finite/negative ``ratio``,
            or an ``override`` outside ``[1, n]``.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise CapacityError(f"n 必须是整数，收到 {n!r}")
    if n < 1:
        raise CapacityError(f"n 必须 >= 1，收到 {n}")
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, int):
            raise CapacityError(f"n_controllers 必须是整数，收到 {override!r}")
        if not 1 <= override <= n:
            raise CapacityError(f"n_controllers={override} 越界，应满足 1 <= n_c <= n={n}")
        return int(override)
    if not math.isfinite(float(ratio)) or float(ratio) <= 0.0:
        raise CapacityError(f"controller_ratio 必须是有限正数，收到 {ratio!r}")
    return max(1, round_half_up(float(ratio) * n))


def capacity_per_controller(n: int, n_c: int, theta: float = DEFAULT_THETA) -> int:
    """Return ``C = ceil((1 + theta) · N / n_c)``.

    Args:
        n: Node count.
        n_c: Number of controllers.
        theta: Capacity-relaxation factor.

    Returns:
        The per-controller capacity, at least 1.

    Raises:
        CapacityError: On ``n_c < 1`` or a non-finite/negative ``theta``.
    """
    if isinstance(n_c, bool) or not isinstance(n_c, int) or n_c < 1:
        raise CapacityError(f"n_c 必须是 >= 1 的整数，收到 {n_c!r}")
    if not math.isfinite(float(theta)) or float(theta) < 0.0:
        raise CapacityError(f"theta 必须是有限非负数，收到 {theta!r}")
    return max(1, int(math.ceil((1.0 + float(theta)) * n / n_c)))


@dataclass(frozen=True)
class CapacityModel:
    """The solved capacity budget of one graph.

    Attributes:
        theta: Capacity-relaxation factor used.
        n_controllers: ``n_c``.
        capacity: Per-controller capacity ``C``.
        n_switches: ``N - n_c``, the number of demand-carrying nodes.
        total_capacity: ``n_c · C``.
        aggregate_slack: ``total_capacity - n_switches · demand``; negative would
            mean the *aggregate* budget is short.  It never is on the nine
            graphs (minimum 60), which is exactly why the probe is needed.
        demand: Per-switch demand (default 1).
        ratio: Controller ratio that produced ``n_c``.
        capacity_mode: ``"formula"`` or ``"explicit"``.
    """

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
    """Build the :class:`CapacityModel` for a graph with ``n`` nodes.

    Args:
        n: Node count.
        cfg: A ``CapacityConfig``-like object (attributes ``theta``,
            ``controller_ratio``, ``n_controllers``, ``demand``,
            ``capacity_mode``, ``capacity_explicit``) or ``None`` for defaults.

    Returns:
        The solved model.

    Raises:
        CapacityError: If ``capacity_mode`` is unknown, if ``capacity_explicit``
            is missing for ``"explicit"`` mode, or if the explicit capacity is
            smaller than the largest load any single assignment could produce.
    """
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
        raise CapacityError(f"demand 必须 >= 1，收到 {demand}")

    n_c = n_controllers_for(n, ratio=ratio, override=override)
    n_switches = n - n_c

    if mode == "formula":
        capacity = capacity_per_controller(n, n_c, theta=theta)
    elif mode == "explicit":
        if explicit is None:
            raise CapacityError("capacity_mode='explicit' 但 capacity_explicit 未给出")
        capacity = int(explicit)
        if capacity < 1:
            raise CapacityError(f"capacity_explicit 必须 >= 1，收到 {capacity}")
    else:
        raise CapacityError(f"未知 capacity_mode {mode!r}（支持 'formula' / 'explicit'）")

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
    """Aggregate capacity audit of one connected component.

    The feasibility test is *aggregate per component*, and that is exact rather
    than heuristic: inside a component every switch can reach every controller,
    so with unit demands and identical capacities the load fits iff
    ``demand <= capacity``.

    Attributes:
        n_nodes: Nodes in the component.
        n_controllers: Controllers in the component.
        demand: ``(n_nodes - n_controllers) · demand_per_switch``.
        capacity: ``n_controllers · C``.
        feasible: ``demand <= capacity``.
        controllerless: ``n_controllers == 0`` — the structural-cascade trigger.
    """

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
    """Return the per-component capacity audit of the ``alive`` subgraph.

    Args:
        graph: The full graph (only edges matter).
        controllers: Controller node ids; those outside ``alive`` are ignored.
        model: Solved capacity model (``capacity`` and ``demand`` are used).
        alive: Surviving nodes; defaults to every node of ``graph``.

    Returns:
        One :class:`ComponentCapacity` per component, ordered by smallest node id
        so the tuple is independent of iteration order.
    """
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
    """Result of :func:`probe_max_utilisation` — loads only, nothing deleted.

    The probe reports **three** quantities, not one, because collapsing them is
    how a capacity model ends up with a meaningless headline:

    * :attr:`u_nearest` — ``max_j L_j / C`` under the **capacity-blind**
      nearest-first assignment.  This is the "what if we ignored capacity" figure
      and it is a *shape* statement about the placement, not a feasibility one:
      it is ``1.60`` on BA-100 whose aggregate slack is ``60`` (total capacity
      150 versus total demand 90).  When the graph is connected every switch can
      reach every controller, so ``u_nearest > 1`` can never mean "the budget is
      too small" — only "the load is unevenly distributed";
    * :attr:`u_after_spill` — ``max_j L_j / C`` after the
      ``(dist, node_id, controller_id)`` total-order assignment with
      capacity-enforced spill to the next-nearest controller that still has
      room.  This is the utilísation that actually happens.  It is ``<= 1`` by
      construction, and the probe **asserts** that rather than trusting it;
    * :attr:`feasible` — the aggregate test per connected component:
      ``Σ demand <= Σ capacity``.  This is the only predicate that may be
      reported as "capacity binds", i.e. :attr:`capacity_binds`
      ``= not feasible``.

    ``feasible == True`` together with ``u_after_spill > 1`` (or with
    unassigned switches) would be an implementation defect, so it is surfaced
    separately as :attr:`feasible_but_spill_failed` and raised on.

    **Do not read** :attr:`u_nearest` **or** :attr:`ceiling_active` as
    "capacity binds": they answer "how lopsided is the load" and "is the ceiling
    being touched" respectively.  ``capacity_binds`` is the unsatisfiability
    predicate and nothing else.

    Attributes:
        controllers: Controller node ids, ascending.
        loads_nearest: ``L_j`` under the blind nearest assignment.
        utilisation_nearest: ``U_j = L_j / C``, same order.
        u_nearest: ``max_j U_j`` for the blind assignment.
        loads_after_spill: ``L_j`` under the capacity-enforced assignment.
        utilisation_after_spill: ``U_j`` for the enforced assignment.
        u_after_spill: ``max_j U_j`` for the enforced assignment, ``<= 1``.
        feasible: Every component satisfies ``Σ demand <= Σ capacity``.
        capacity_binds: ``not feasible`` **or** ``n_unassigned > 0`` — the
            unsatisfiability predicate, and the only thing the paper may call
            "capacity binds".
        ceiling_active: Some controller is exactly at ``C`` under the enforced
            assignment.  Measured ``True`` on all nine graphs; it says the
            constraint is *live*, not that it is *violated*.
        n_at_ceiling: How many controllers satisfy ``L_j == C`` exactly.
        feasible_but_spill_failed: ``feasible`` yet the spill could not place
            everything (implementation defect; asserted against).
        n_overloaded: Controllers with ``U_j > 1 + eps`` under the blind profile.
        n_unassigned: Switches the enforced assignment could not place.
        n_spilled: Switches the enforced assignment moved away from their
            nearest controller (or could not place).
        n_controllerless_components: Components with no controller at all.
        capacity: The ``C`` used.
        n_switches: ``N - n_c``.
        controller: The controller with the largest blind ``U_j``, or ``None``.
        components: The per-component aggregate audit.
    """

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

    # -- backwards-compatible views --
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
        """Alias of :attr:`capacity_binds` — ``not feasible`` or unplaced switches.

        Mirrors :attr:`capacity_binds` exactly (it is *not* derived from
        ``max_utilisation``); see the class docstring for why the two must not be
        conflated.
        """
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
    """Measure the load profile of both assignments, then judge feasibility.

    **Nothing is deleted** — this is a pure measurement, run *before* any
    cascade, so that "does ``theta = 0.5`` bind at ``t = 0``?" is answered by
    data rather than assumed either way.

    The three outputs are kept apart on purpose (see
    :class:`UtilisationProbe`): ``u_nearest`` describes the *shape* of the
    degree-ranked placement, ``u_after_spill`` describes the utilisation that
    actually occurs, and ``feasible`` is the only thing that may be called
    "capacity binds".  Reporting ``u_nearest`` as the headline would say a
    graph with 66 % aggregate slack is over capacity, which is wrong; reporting
    only ``u_after_spill`` would hide that the nearest assignment is lopsided.

    Two further flags are returned so that no caller has to re-derive them:

    * ``ceiling_active`` / ``n_at_ceiling`` — some controller is exactly at
      ``C``.  True on all nine intact graphs; it means the ceiling is *live*;
    * ``capacity_binds`` — ``not feasible or n_unassigned > 0``.  False on all
      nine intact graphs; it means the demand is *unsatisfiable*.

    Args:
        graph: The evaluated graph (node ids ``0 .. n-1``).
        controllers: Controller node ids; duplicates are collapsed.
        model: The solved :class:`CapacityModel`.
        eps: Relative tolerance.

    Returns:
        A :class:`UtilisationProbe`.

    Raises:
        CapacityError: If a controller is not a node of ``graph``, or if
            ``feasible`` holds but the enforced assignment still overloads a
            controller (an implementation defect, never a data property).
    """
    from .assignment import assign_single_controller  # local: avoids a cycle

    ordered = tuple(sorted({int(node) for node in controllers}))
    missing = [node for node in ordered if node not in graph]
    if missing:
        raise CapacityError(f"控制器 {missing} 不在图里（n={graph.number_of_nodes()}）")

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
    # "Reaching the ceiling" is an integer event, so it is tested as one: a load
    # of exactly C leaves no room, and there is no float noise to tolerance away.
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
            "聚合可行但顺延失败："
            f"u_after_spill={u_after_spill:.6f}，n_unassigned={n_unassigned}。"
            "这是指派实现的缺陷，不是数据性质。"
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
        # Unsatisfiable = the aggregate test fails, *or* a switch could not be
        # placed at all.  Never "some controller is full" — that is
        # `ceiling_active` below and it is true on every intact graph.
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
