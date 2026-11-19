"""Capacity feasibility as a function of the removal fraction.

The question this module answers
--------------------------------
"Does ``theta = 0.5`` bind?" has two different answers at two different times,
and conflating them is what produced the wrong headline in the first place:

* at ``t = 0`` the answer is **no**.  On all nine graphs ``Σ demand <= Σ capacity``
  per connected component (aggregate slack 60…300), ``u_after_spill = 1.000`` and
  no switch is unassigned.  Reporting ``u_nearest = 1.60…2.73`` as "the capacity
  binds" would say a graph with 66 % slack is over budget, which is false;
* after the controllers start dying the answer changes, and **when** it changes is
  the publishable sentence.

The exact condition
-------------------
Inside one connected component every switch can reach every controller, so with
unit demands and identical capacities the load fits iff

.. code-block:: text

    n_comp − k  <=  k · C        ⇔        n_comp <= k · (C + 1)

where ``n_comp`` is the surviving node count of the component and ``k`` the
surviving controller count inside it.  With ``C = 15`` that is ``n_comp <= 16k``,
i.e. a **controller density floor of** ``k / n_comp >= 1/(C+1) = 6.25 %``.

At ``t = 0`` the density is ``n_c / N ≈ 10 %``, comfortably above the floor — so
the constraint is slack.  It becomes active only when a targeted attack kills the
controllers *faster* than it kills switches, which is precisely what a
degree-ranked placement invites: the controllers are the hubs, and a
degree-descending attack removes them first.  :func:`feasibility_curve` measures
that race step by step and reports

* ``first_infeasible_f`` — the first ``f`` at which some component violates the
  condition (the transient event);
* ``first_persistent_infeasible_f`` — the first ``f`` after which it never
  recovers (the state change; the number that belongs in the main text);
* ``f_density_floor`` — the first ``f`` at which the measured controller density
  ``k / n_alive`` drops below ``1/(C+1)``, i.e. the theoretical prediction that
  the two measured numbers must bracket.

The measured table (seed 0, ``c = 1.6``, ``hh+rewire``, ``—`` = never)
---------------------------------------------------------------------
.. code-block:: text

    attack                    f* band      persists      extra deaths at f*
    degree / controllers      0.042–0.047  9/9           14–23
    betweenness               0.107–0.713  1/7 (BA-100)  0–19
    random                    —            —             —

``degree`` and ``controllers`` are the same attack (the controllers *are* the
top-degree nodes), so they produce byte-identical rows.  ``betweenness`` gives
an onset on seven of nine graphs that recovers on six of them — shattering the
graph also spreads the survivors over several controllers, so the density comes
back even though the topology does not.

``f*`` on its own would be a threshold with no consequence attached, so
:func:`onset_records` pairs it with ``extra_dead_after_onset``: on a single
component the extra death at the onset is exactly ``Σdemand − Σcapacity`` (15 on
all four BA graphs), and the overload cascade then saturates the bound
``n_comp = k·(C+1)`` — measured peak 70–88 % of ``N`` killed beyond what
``structural`` mode kills.  That saturation is the strongest evidence that
``n_comp <= k(C+1)`` is not merely a test but the attractor of the dynamics.

Attacks: a minimal built-in set
-------------------------------
The full attack *protocols* live in :mod:`bit_hyrl.attack`.  The small
generator here exists so that the feasibility question can be answered without
touching the evaluation layer, and it uses the same tie-break rule
(``sorted(nodes, key=lambda v: (-score[v], v))``), so the sequences are
interchangeable with the ones produced there.  The consumer,
:func:`feasibility_curve`, takes an explicit sequence and therefore never
needs to change when a different producer is substituted.
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

#: Minimal attack set used by the feasibility probe.  ``random`` / ``degree`` /
#: ``betweenness`` are the three the paper reports; ``controllers`` is the
#: adversarial variant that targets the control plane itself and is what makes
#: the capacity constraint fire.
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
    """Return ``count`` controllers, ranked by ``(-degree, node_id)``.

    The same placement the probe and the CLI use, kept in one place so the
    feasibility scan and the ``t = 0`` probe are measuring the same system.

    Args:
        graph: Evaluated graph.
        count: How many controllers to place.

    Returns:
        Controller node ids, ascending.
    """
    order = sorted(graph.nodes(), key=lambda node: (-graph.degree(node), int(node)))
    return sorted(int(node) for node in order[: max(0, int(count))])


def density_floor(capacity: int) -> float:
    """Return ``1 / (C + 1)``, the controller density below which load cannot fit.

    Args:
        capacity: Per-controller capacity ``C``.

    Returns:
        The floor as a fraction of the surviving node count.
    """
    if capacity < 0:
        raise ValueError(f"capacity 不能为负，收到 {capacity}")
    return 1.0 / float(int(capacity) + 1)


def fstar_closed_form(n_nodes: int, n_controllers: int,
                      capacity: int) -> tuple[float, float]:
    """Return ``(t_c, f*)`` for the controller-first attack, in closed form.

    Under a controller-first attack (which the ``degree`` sequence *is* whenever
    the controllers are the top-degree nodes) the first ``t`` removals are all
    controllers, and inside one connected component the aggregate condition is

    .. code-block:: text

        N − t  <=  (n_c − t) · (C + 1)
        ⇔   t  >=  (n_c·(C+1) − N) / C  =:  t_c

    so the first *integer* removal count that breaks feasibility is
    ``floor(t_c) + 1`` and

    .. code-block:: text

        f* = (floor(t_c) + 1) / N

    Both quantities are in the **attack-budget** unit (``f_direct``): the
    derivation assumes the cascade removes nothing before the onset.  That
    assumption is checked, not assumed — see
    :func:`~bit_hyrl.control.feasibility.feasibility_curve`'s ``f_direct_star``
    and ``collapse_f_direct``, and the note on the two units below.

    ⚠️ Two different ``f`` units live in this module and they must not be mixed:

    * ``f_direct`` — attacked nodes over ``N``.  This is the *budget* the operator
      spends, and the one ``f*`` is derived in;
    * ``f_total`` — failed nodes over ``N``, cascade included.  This is the state
      the network is in.

    They coincide only while the cascade removes nothing, which under
    ``degree`` attack holds exactly for ``t <= n_c`` (there is nothing to orphan
    yet).  On a graph where fragments start dying earlier, ``f_total > f_direct``
    at the same step, so a claim phrased in ``f`` needs the unit named.

    Args:
        n_nodes: ``N``.
        n_controllers: ``n_c``.
        capacity: ``C``.

    Returns:
        ``(t_c, f*)`` — the real-valued threshold and the sampled onset.

    Raises:
        ValueError: If ``N`` or ``C`` is non-positive, or ``n_c > N``.
    """
    n = int(n_nodes)
    k = int(n_controllers)
    c = int(capacity)
    if n <= 0 or c <= 0:
        raise ValueError(f"n_nodes / capacity 必须为正，收到 {n_nodes} / {capacity}")
    if k < 0 or k > n:
        raise ValueError(f"n_controllers={k} 不在 [0, {n}] 内")
    t_c = (k * (c + 1) - n) / float(c)
    steps = math.floor(t_c) + 1
    return t_c, min(steps, n) / float(n)


def fstar_asymptotic(controller_density: float, theta: float) -> float:
    """Return the asymptotic onset ``rho·(theta + rho) / (1 + theta)``.

    Substituting ``C = (1 + theta)/rho`` (i.e. ``C = (1+theta)·N/n_c``) into
    ``t_c / N = (n_c·(C+1) − N) / (C·N)`` gives, in the limit of large ``N``,

    .. code-block:: text

        f* ≈ rho·(theta + rho) / (1 + theta)

    which with the defaults ``rho = 0.1``, ``theta = 0.5`` is ``0.04``.  **The
    onset is therefore fixed by the two design knobs alone** — the topology only
    enters through the ``floor(·)`` discretisation, which is why the nine graphs
    land in ``0.042 … 0.050`` instead of all on one number.  That is the sentence
    to quote: it says the capacity constraint activates at a point the designer
    chooses, not one the graph chooses.

    Args:
        controller_density: ``rho = n_c / N``.
        theta: The capacity relaxation ``C = ceil((1+theta)·N/n_c)``.

    Returns:
        The asymptotic onset fraction.
    """
    rho = float(controller_density)
    th = float(theta)
    if rho <= 0.0:
        raise ValueError(f"controller_density 必须为正，收到 {controller_density}")
    if th <= 0.0:
        raise ValueError(f"theta 必须为正，收到 {theta}")
    return rho * (th + rho) / (1.0 + th)



#: Centralities that depend only on the graph, never on the seed or the controller set.
#: Their removal order is therefore computed once per graph and reused for every seed
#: and every method that shares the substrate; betweenness on a 1 500-node graph costs
#: seconds, and paying it once per seed would dominate the whole sweep.
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
            # ARPACK does not converge on some disconnected substrates; the power
            # iteration variant is slower but defined there.
            raw = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
        return {int(k): float(v) for k, v in raw.items()}
    raise ValueError(f"未知静态中心性 {method!r}")


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
    """Return a deterministic attack sequence for ``method``.

    Minimal stand-in for the full attack registry — see the module
    docstring.  Every method shares the ``(-score, node_id)`` total order, so the
    sequences are byte-identical to what the registry will produce for the same
    method parameters.

    Args:
        graph: The graph being attacked.
        method: One of :data:`PROVISIONAL_ATTACK_METHODS`.
        controllers: Controller ids (used by ``"controllers"``, which attacks
            them first and then falls back to degree order).
        seed: Root seed; only ``"random"`` consumes it, via
            ``derive_permutation(seed, N, "attack", "random")``.

    Returns:
        The node ids in attack order (a permutation of ``0 .. N-1``).

    Raises:
        ValueError: For an unknown ``method``.
    """
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
        f"未知攻击方法 {method!r}，支持 {list(PROVISIONAL_ATTACK_METHODS)}"
    )


@dataclass(frozen=True)
class FeasibilityPoint:
    """One sampled state of the feasibility curve.

    Attributes:
        f: ``f_total`` at this step — the removal fraction, cascade included.
        f_direct: ``f_direct`` — the directly attacked fraction.
        n_alive: Surviving node count.
        n_controllers: Surviving controller count ``k``.
        controller_density: ``k / n_alive`` (``0.0`` when nothing is alive).
        demand: ``Σ demand`` over the surviving switches.
        capacity: ``Σ capacity`` over the surviving controllers.
        slack: ``capacity − demand``; negative means the aggregate is short.
        feasible: Every surviving component satisfies ``n_comp <= k·(C+1)``.
        capacity_pressure: ``Σ_components max(0, demand − capacity)`` — how many
            switches would have to fail for the aggregate to fit.  ``0`` means
            capacity mode and structural mode agree at this state; a positive
            value is the *volume* of the extra failure capacity mode would cause,
            which is a far more informative quantity than a boolean.
        n_overloaded: Controllers with ``L_j > C`` under the blind assignment.
        u_blind: ``max_j L_j / C`` of the blind assignment (``0.0`` if no
            controller survives).
        controllerless_components: Components with no controller.
        gcc_size: Largest surviving component.
        components: The per-component audit.
    """

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
        """``True`` when the state still has something to serve or protect.

        The all-dead state is *vacuously* feasible (no component, so no
        violation), and counting it would make every "first persistent
        infeasibility" look like it recovers at ``f = 1``.
        """
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
    """The feasibility history of one ``(graph, attack, mode)`` triple.

    Attributes:
        dataset: Dataset label.
        attack: Attack method label.
        mode: ``"structural"`` or ``"capacity"``.
        capacity: Per-controller ``C``.
        n_nodes: ``N``.
        n_controllers_initial: ``n_c``.
        density_floor: ``1/(C+1)`` — the theoretical threshold.
        points: The ``t = 0`` state followed by one :class:`FeasibilityPoint` per
            attack-sequence entry, i.e. ``len(points) == len(sequence) + 1``.
        first_infeasible_f: First ``f`` with ``feasible == False``, or ``None``.
        first_persistent_infeasible_f: First ``f`` after which every non-trivial
            state is infeasible, or ``None`` (capacity never becomes the dominant
            mechanism).  The all-dead state is excluded — it is vacuously feasible
            and would otherwise mask a genuine persistent infeasibility.
        f_density_floor: First ``f`` with ``controller_density < density_floor``,
            or ``None``.
        infeasible_fraction: Fraction of **non-trivial** sampled states that are
            infeasible.  This, not a boolean, is the severity measure: it says for
            how much of the removal budget the constraint is active.

            ⚠️ The denominator is the **non-trivial window**, not the whole curve:
            ``n_non_trivial`` states, ending at ``collapse_f``.  Under
            ``structural`` mode the window is short by construction — see
            :attr:`collapse_f` — so ``0.50`` on BA-100 means "5 of the 10 states
            that had anything left to serve", not "half the attack".
        collapse_f: First ``f`` (``f_total``) with ``n_alive == 0``, or ``None``.
            This is ``1.000`` on the nine graphs under ``degree`` attack: losing
            the last controller orphans everything at once, so the whole network
            fails in a single step and ``f_total`` jumps straight to 1.  That is
            the quantified form of the "controller loss = total loss" assumption
            and it is why the non-trivial window ends where it does.
        collapse_f_direct: ``f_direct`` at that same state — the *budget* spent to
            reach total collapse.  Measured ``= n_c / N`` (0.100 on the BA series,
            0.1015 on Cogentco) under ``structural`` mode with a controller-first
            attack, because the t-th removal *is* the t-th controller until they
            run out.  This is the number to quote for "how much attack the network
            survives"; ``collapse_f`` alone would say "100 % of the nodes died",
            which is true and useless.
        n_non_trivial: Number of sampled states with something left to serve
            (``n_alive > 0`` and ``demand > 0``); the denominator of
            ``infeasible_fraction``.
        max_capacity_pressure: Largest ``capacity_pressure`` over the whole run.
        peak_pressure_f: The ``f`` at which that maximum occurs, or ``None``.
        final_fraction: ``f`` of the last sampled state.
        trace: The underlying :class:`~.cascade.CascadeTrace`.
    """

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
    """Run the cascade and record the capacity verdict at every step.

    The verdict is recomputed from the **surviving** node and controller sets,
    never inferred from ``f``: an attack can remove switches without touching a
    single controller (feasibility improves) or controllers without removing
    switches (feasibility collapses), so the only honest source is the state.

    Args:
        graph: The evaluated graph.
        attack_sequence: Node ids in attack order.
        controllers: Initial controller ids.
        model: Solved capacity model; ``capacity`` is used **only** to audit
            feasibility here, never to delete a node — the removals come from the
            cascade alone, so this measurement cannot perturb the dynamics it is
            measuring.
        cfg: A ``CascadeConfig``-like object (``mode``, ``max_iter``), or ``None``.
        dataset: Dataset label for the output record.
        attack: Attack-method label for the output record.

    Returns:
        A :class:`FeasibilityCurve`.
    """
    trace = run_cascade(graph, attack_sequence, controllers, model, cfg)
    n = graph.number_of_nodes() or 1
    floor = density_floor(model.capacity)

    points: list[FeasibilityPoint] = []

    # ---- the t = 0 state, *before* any removal ---------------------------- #
    # `run_cascade` emits one step per attack entry, so its first step is already
    # at f > 0.  The question is specifically about the untouched network, so
    # the initial state is prepended explicitly rather than approximated by the
    # first post-attack step (which on a 100-node graph is 1 % of the budget away).
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

    # The first f after which *every* non-trivial state is infeasible.  The
    # all-dead tail is skipped: it is vacuously feasible and would otherwise make
    # every curve look like it recovers.
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

    # Total collapse, in both units.  `f` alone is useless here — under a
    # controller-first attack the last controller takes every remaining node with
    # it, so `f` jumps 0.09 -> 1.00 in one step; the informative number is the
    # budget that bought the collapse (`f_direct`).
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
    """How much extra failure ``capacity`` mode adds over ``structural``.

    The two modes consume the *same* attack sequence, so step ``i`` of each curve
    is the same ``f_direct`` and the difference in surviving node count is
    directly attributable to the capacity mechanism.  This is the quantity that
    says whether the opt-in mode changes anything, and by how much.

    Attributes:
        dataset: Dataset label.
        attack: Attack-method label.
        max_extra_dead: Largest ``n_alive(structural) − n_alive(capacity)``.
        max_extra_dead_f_direct: ``f_direct`` where that maximum occurs.
        final_extra_dead: The difference at the last step.
        mean_extra_dead: Mean difference over the whole sequence.
        n_nodes: ``N``, so the peak can be quoted as a share of the network.
    """

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
    """Compare a ``structural`` and a ``capacity`` curve on the same attack.

    Args:
        structural: The ``structural``-mode curve.
        capacity: The ``capacity``-mode curve, over the **same** attack sequence.

    Returns:
        A :class:`ModeGap`.

    Raises:
        ValueError: If the two curves were not produced from the same sequence
            (different lengths), which would make the comparison meaningless.
    """
    if len(structural.points) != len(capacity.points):
        raise ValueError(
            "structural 与 capacity 曲线的步数不同（"
            f"{len(structural.points)} vs {len(capacity.points)}），无法配对比较"
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
    """Render the ``structural`` vs ``capacity`` gap as a plain-text table.

    One row per ``(dataset, attack)``.  The columns are the four numbers that
    matter and nothing else: the peak extra death, the ``f_direct`` at which it
    happens (a peak without its location is not quotable), the end-of-sequence
    difference and the mean over the sequence.  A negative mean is possible and
    meaningful: ``capacity`` mode can kill *fewer* nodes at a given step, because
    removing the excess early keeps the surviving fragments serviceable longer.

    Args:
        gaps: The comparisons to summarise.

    Returns:
        A newline-terminated ASCII table.
    """
    header = (
        f"{'图':<10} {'攻击':<12} {'峰值超额死亡':>12} {'f_direct@峰值':>13} "
        f"{'末端':>6} {'均值':>8} {'占 N':>7}"
    )
    lines = [header, "-" * len(header)]
    for gap in gaps:
        share = (
            f"{100.0 * gap.max_extra_dead / gap.n_nodes:.1f}%"
            if gap.n_nodes else "—"
        )
        lines.append(
            f"{gap.dataset:<10} {gap.attack:<12} {gap.max_extra_dead:>12} "
            f"{_format_optional(gap.max_extra_dead_f_direct):>13} "
            f"{gap.final_extra_dead:>6} "
            f"{gap.mean_extra_dead:>8.2f} {share:>7}"
        )
    return "\n".join(lines) + "\n"


def format_feasibility_table(curves: Sequence[FeasibilityCurve]) -> str:
    """Render the ``first infeasible f`` summary as a plain-text table.

    One row per ``(dataset, attack)``.  Columns: the theoretical density-floor
    crossing, the first transient infeasibility, the first *persistent* one, and
    the ``t = 0`` verdict — so a reader can see at a glance that the constraint is
    slack at ``t = 0`` and when it stops being slack.

    Args:
        curves: The curves to summarise.

    Returns:
        A newline-terminated ASCII table.
    """
    header = (
        f"{'图':<10} {'攻击':<12} {'模式':<11} {'C':>3} {'k0':>3} "
        f"{'1/(C+1)':>8} {'f_density':>10} {'f_first':>8} {'f_persist':>10} "
        f"{'不可行%':>8} {'压力峰值':>8} {'t=0':>5}"
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
    """Write the JSON record and the plain-text table.

    Args:
        curves: The curves to persist.
        out_dir: Target directory; defaults to ``results/feasibility``.
        name: File stem.

    Returns:
        ``{"json": path, "txt": path}``.

    Raises:
        ValueError: If ``curves`` is empty — an empty artefact is a bug.
    """
    if not curves:
        raise ValueError("feasibility 结果为空，拒绝写空文件")
    target = Path(out_dir) if out_dir is not None else io_paths.results_path(
        "feasibility", ""
    ).parent
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{name}.json"
    txt_path = target / f"{name}.txt"

    payload = {
        "name": name,
        "n_curves": len(curves),
        "density_floor_note": "1/(C+1)；控制器密度低于此值时过载级联才可能触发",
        "curves": [curve.to_dict() for curve in curves],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    txt_path.write_text(format_feasibility_table(curves), encoding="utf-8")
    return {"json": json_path, "txt": txt_path}


# --------------------------------------------------------------------------- #
# The onset table — the one number the analysis actually needs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OnsetRecord:
    """First infeasibility and how much extra failure it unleashes.

    ``f*`` alone ("capacity becomes unsatisfiable at 4.4 % of removals") does not
    say whether the mechanism *matters*: on a graph where the overload cascade
    then kills nobody, ``f*`` is a threshold nobody crosses.  The record
    therefore pairs the onset with the **extra** death it causes, measured
    against the ``structural`` run on the identical attack sequence:

    .. code-block:: text

        extra_dead(f) = n_alive(structural, f) − n_alive(capacity, f)

    This is the only clean attribution available, because both modes consume the
    same sequence: ``structural`` removes exactly the attacked nodes plus the
    components that lost their controller, so any *additional* death in
    ``capacity`` mode is caused by the overload mechanism and nothing else.

    Attributes:
        dataset: Dataset label.
        attack: Attack-method label.
        n_nodes: ``N``.
        capacity: Per-controller ``C``.
        density_floor: ``1/(C+1)``.
        n_controllers_initial: ``n_c``.
        feasible_at_t0: Verdict on the untouched network (``True`` on all nine).
        f_floor: First ``f`` with ``k/n_alive < 1/(C+1)``, or ``None``.
        f_star: First ``f`` (``f_total``) at which some component is unsatisfiable,
            or ``None``.
        f_direct_star: ``f_direct`` at that same state — the **attack budget**
            spent to reach the onset, and the unit the closed form
            :func:`fstar_closed_form` is derived in.  On a graph whose cascade
            removes nothing before the onset the two are equal; when fragments
            start dying earlier, ``f_direct_star < f_star`` and only this one is
            comparable with the closed form.
        f_star_persistent: First ``f`` after which it never recovers, or ``None``.
        k_at_f_star: Surviving controllers at ``f*``, or ``None``.
        density_at_f_star: ``k / n_alive`` at ``f*``, or ``None``.
        demand_at_f_star: ``Σ demand`` at ``f*``, or ``None``.
        capacity_at_f_star: ``Σ capacity`` at ``f*``, or ``None``.
        controllerless_at_f_star: Components with **no controller** at ``f*``.

            ``feasible`` is false for two different reasons, and only one of them
            is a capacity statement: ``demand > capacity`` inside a component (a
            genuine shortfall) versus a component that has *no controller at all*
            (a control-plane partition, where ``capacity == 0`` by definition).
            In ``structural`` mode the cascade deletes controller-less components
            as blocks, so this is ``0`` on every measured row — which is exactly
            why it is recorded: the zero is a *degenerate check on the cascade*,
            not a coincidence.  A non-zero value would mean the structural pass
            stopped removing orphaned fragments, and every ``f*`` after that point
            would be mis-attributed.

        infeasible_fraction: Share of non-trivial states that are infeasible.
        max_capacity_pressure: Peak ``Σ max(0, demand − capacity)``.
        extra_dead_at_onset: ``extra_dead(f*)`` — the extra death **at** the
            onset state itself.  This is the conservative number.
        extra_dead_fraction_at_onset: ``extra_dead_at_onset / N``.
        extra_dead_after_onset: ``max_{f >= f*} extra_dead(f)``; ``0`` when the
            onset removes nobody extra and when no onset was found.
        extra_dead_fraction: ``extra_dead_after_onset / N``.
        final_extra_dead: ``extra_dead`` at the last sampled state.
        mode_compared: ``True`` when a ``capacity``-mode curve was paired, so the
            extra-death columns are measurements rather than placeholders.
    """

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
        """``f*`` coincides exactly with the density-floor crossing.

        On every graph and on the targeted attacks this is ``True`` to the last
        bit — the two quantities are read off the same sampled state, so the
        comparison is exact rather than approximate.
        """
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
        """Why ``f*`` is infeasible: ``"capacity"``, ``"partition"`` or ``None``.

        ``"capacity"`` — every component has a controller, and at least one has
        ``demand > capacity``.  This is the mechanism;
        ``"partition"`` — some component has no controller at all (``capacity``
        is ``0`` there by definition).  The topology is what failed, not the
        budget, so this must never be quoted as "capacity binds".
        """
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
    # Index the curves rather than looking states up by `f`: the attack sequence
    # can contain already-dead nodes, so several consecutive steps share the same
    # `f` and a `f`-keyed search would silently return the wrong (earlier,
    # feasible) state — which is how a partition onset would be mislabelled as a
    # capacity one.
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
            f"first_infeasible_f={f_star} 与按索引找到的起跳点 {onset_point.f} 不一致"
        )

    extra_after = 0
    extra_at_onset = 0
    final_extra = 0
    if capacity is not None:
        if len(capacity.points) != len(structural.points):
            raise ValueError(
                "structural 与 capacity 曲线的步数不同（"
                f"{len(structural.points)} vs {len(capacity.points)}），无法配对"
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
    """Summarise every ``(dataset, attack)`` pair into an :class:`OnsetRecord`.

    Args:
        structural: The ``structural``-mode curves.
        capacity: Optional ``capacity``-mode curves over the *same* attack
            sequences.  When given, the extra-death columns become measurements;
            when omitted they are ``0`` and ``mode_compared`` is ``False``.

    Returns:
        One record per structural curve, in input order.

    Raises:
        ValueError: If a capacity curve is present for an attack whose step count
            differs from the structural one (the pairing would be meaningless).
    """
    structural_keys = {(curve.dataset, curve.attack) for curve in structural}
    orphaned = [
        f"{curve.dataset}/{curve.attack}"
        for curve in (capacity or ())
        if (curve.dataset, curve.attack) not in structural_keys
    ]
    if orphaned:
        raise ValueError(f"capacity 曲线没有对应的 structural 曲线：{orphaned}")

    by_key = {
        (curve.dataset, curve.attack): curve for curve in (capacity or ())
    }
    return tuple(
        _one_onset(curve, by_key.get((curve.dataset, curve.attack)))
        for curve in structural
    )


def format_onset_table(records: Sequence[OnsetRecord]) -> str:
    """Render the feasibility-evidence table (one row per ``(dataset, attack)``).

    Columns are chosen so the sentence can be written straight off the table:
    the theoretical floor, the measured onset, whether the two coincide, and how
    many additional nodes the overload cascade kills after the onset.

    Args:
        records: The records to summarise.

    Returns:
        A newline-terminated ASCII table.
    """
    header = (
        f"{'图':<10} {'攻击':<12} {'N':>4} {'C':>3} {'nc0':>4} {'1/(C+1)':>8} "
        f"{'f_floor':>8} {'f*':>7} {'成因':>6} {'持续':>5} {'k(f*)':>6} {'rho(f*)':>8} "
        f"{'ΣD/ΣC(f*)':>11} {'不可行%':>8} {'压峰':>5} "
        f"{'超额死亡@f*':>11} {'f*后峰值':>9} {'占比':>7}"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        demand_capacity = (
            f"{record.demand_at_f_star}/{record.capacity_at_f_star}"
            if record.demand_at_f_star is not None
            else "—"
        )
        kind = {"capacity": "容量", "partition": "割裂", None: "—"}[record.onset_kind]
        lines.append(
            f"{record.dataset:<10} {record.attack:<12} {record.n_nodes:>4} "
            f"{record.capacity:>3} {record.n_controllers_initial:>4} "
            f"{record.density_floor:>8.4f} "
            f"{_format_optional(record.f_floor):>8} "
            f"{_format_optional(record.f_star):>7} "
            f"{kind:>6} "
            f"{('是' if record.onset_is_persistent else '否'):>5} "
            f"{(record.k_at_f_star if record.k_at_f_star is not None else '—'):>6} "
            f"{(f'{record.density_at_f_star:.4f}' if record.density_at_f_star is not None else '—'):>8} "
            f"{demand_capacity:>11} "
            f"{100.0 * record.infeasible_fraction:>7.1f}% "
            f"{record.max_capacity_pressure:>5} "
            f"{record.extra_dead_at_onset:>11} "
            f"{record.extra_dead_after_onset:>9} "
            f"{100.0 * record.extra_dead_fraction:>6.2f}%"
        )
    return "\n".join(lines) + "\n"


def write_onset_outputs(records: Sequence[OnsetRecord],
                        out_dir: Path | None = None,
                        *, name: str = "onset") -> dict[str, Path]:
    """Write the onset JSON record and the plain-text table.

    Args:
        records: The records to persist.
        out_dir: Target directory; defaults to ``results/feasibility``.
        name: File stem.

    Returns:
        ``{"json": path, "txt": path}``.

    Raises:
        ValueError: If ``records`` is empty — an empty artefact is a bug.
    """
    if not records:
        raise ValueError("onset 结果为空，拒绝写空文件")
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
            "f_star": "首次出现某分量 Σ需求 > Σ容量 的移除比例 f",
            "f_floor": "首次出现控制器密度 k/n_alive < 1/(C+1) 的 f",
            "extra_dead_after_onset": (
                "f >= f* 之后 capacity 模式相对 structural 模式多死的节点数（同一条攻击序列）"
            ),
            "capacity_binds": "仅当 feasible == False（或存在无法指派的交换机）时为真",
        },
        "records": [record.to_dict() for record in records],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    txt_path.write_text(format_onset_table(records), encoding="utf-8")
    return {"json": json_path, "txt": txt_path}
