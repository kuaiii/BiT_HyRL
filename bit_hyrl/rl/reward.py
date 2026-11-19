"""Placement reward following the ``(w_G, w_S, w_H)`` decomposition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import networkx as nx

from ..control.capacity import capacity_model
from ..control.cascade import run_cascade
from ..control.feasibility import provisional_attack_sequence
from ..evaluation.metrics import calculate_csa, calculate_h_c_norm

__all__ = ["DEFAULT_REWARD_WEIGHTS", "placement_reward"]

#: Paper weights ``(w_G, w_S, w_H, w_D)`` — structure dominates, distance off.
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "structure": 0.6,
    "csa": 0.2,
    "h_load": 0.2,
    "distance": 0.0,
}

#: The two attacks the reward averages over (targeted + blind).
DEFAULT_REWARD_ATTACKS: tuple[str, ...] = ("degree", "random")


def _weights_from(cfg) -> dict[str, float]:
    """Return the four reward weights, falling back to the paper defaults."""
    weights = dict(DEFAULT_REWARD_WEIGHTS)
    if cfg is not None:
        raw = getattr(cfg, "reward_weights", None)
        if raw:
            for key in DEFAULT_REWARD_WEIGHTS:
                if key in raw:
                    weights[key] = float(raw[key])
    return weights


def _distance_term(graph: nx.Graph, controllers: list[int]) -> float:
    """Mean ``1 / (1 + hops)`` from each switch to its nearest controller."""
    if not controllers:
        return 0.0
    lengths = nx.multi_source_dijkstra_path_length(graph, controllers)
    switches = [node for node in graph.nodes() if node not in set(controllers)]
    if not switches:
        return 1.0
    total = 0.0
    for node in switches:
        hops = lengths.get(node, len(graph))
        total += 1.0 / (1.0 + hops)
    return total / len(switches)


def placement_reward(
    graph: nx.Graph,
    controllers: Sequence[int],
    cfg=None,
    *,
    seed: int | None = None,
    attacks: Sequence[str] = DEFAULT_REWARD_ATTACKS,
) -> dict[str, float]:
    """Score a controller placement; return the four terms and the weighted sum."""
    weights = _weights_from(cfg)
    frac = float(getattr(cfg, "attack_prefix_fraction", 0.3) if cfg is not None else 0.3)
    if seed is None:
        seed = int(getattr(cfg, "seed", 0) or 0) if cfg is not None else 0

    controllers = sorted({int(node) for node in controllers})
    n = graph.number_of_nodes()
    zero = {"structure": 0.0, "csa": 0.0, "h_load": 0.0, "distance": 0.0, "weighted": 0.0}
    if n == 0 or not controllers:
        return zero

    model = capacity_model(n)
    cascade_cfg = SimpleNamespace(mode="structural", max_iter=50)

    structure_terms: list[float] = []
    csa_terms: list[float] = []
    h_terms: list[float] = []

    for attack in attacks:
        sequence = provisional_attack_sequence(
            graph, attack, controllers=controllers, seed=seed
        )
        prefix = max(1, int(round(len(sequence) * frac)))
        prefix_sequence = sequence[:prefix]
        trace = run_cascade(graph, prefix_sequence, controllers, model, cascade_cfg)

        gcc = [1.0] + list(trace.r_curve)
        structure_terms.append(sum(gcc) / len(gcc))

        if trace.steps:
            final = trace.steps[-1]
            alive = set(final.alive_nodes)
            controllers_alive = list(final.controllers_alive)
            subgraph = graph.subgraph(alive)
            csa_terms.append(
                calculate_csa(subgraph, controllers_alive, initial_num_switches=n)
            )
            h_terms.append(calculate_h_c_norm(final.loads, n_c0=len(controllers)))
        else:  # pragma: no cover - a non-empty prefix always yields a step
            csa_terms.append(0.0)
            h_terms.append(0.0)

    structure = sum(structure_terms) / len(structure_terms)
    csa = sum(csa_terms) / len(csa_terms)
    h_load = sum(h_terms) / len(h_terms)
    distance = _distance_term(graph, controllers) if weights["distance"] > 0.0 else 0.0

    weighted = (
        weights["structure"] * structure
        + weights["csa"] * csa
        + weights["h_load"] * h_load
        + weights["distance"] * distance
    )
    return {
        "structure": float(structure),
        "csa": float(csa),
        "h_load": float(h_load),
        "distance": float(distance),
        "weighted": float(weighted),
    }
