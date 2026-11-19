"""Control-plane robustness metrics: GCC, CSA and ``H_C``.

The runner builds the surviving subgraph and the final load map from the last
cascade step and hands them to these functions.
"""

from __future__ import annotations

import math
from typing import Mapping

import networkx as nx


def calculate_csa(
    G: nx.Graph,
    centers: list[int],
    alpha: float = 0.2,
    initial_num_switches: int | None = None,
) -> float:
    """Control Supply Availability (CSA) over an evaluated subgraph."""
    if G.number_of_nodes() == 0:
        return 0.0
    centers_set = set(centers)
    switch_nodes = [node for node in G.nodes() if node not in centers_set]
    if not switch_nodes:
        return 0.0

    node_to_k: dict[int, int] = {}
    for component in nx.connected_components(G):
        component_centers = len(component & centers_set)
        for node in component:
            node_to_k[node] = component_centers

    total = 0.0
    for node in switch_nodes:
        k_i = node_to_k.get(node, 0)
        total += 1.0 - math.exp(-alpha * k_i)

    denominator = (
        initial_num_switches if initial_num_switches is not None else len(switch_nodes)
    )
    if denominator <= 0:
        return 0.0
    return total / denominator


def calculate_h_c(loads: Mapping[int, int]) -> float:
    """Load-share control entropy ``H_C`` over a ``{controller: L_j}`` mapping."""
    total_load = sum(loads.values())
    if total_load <= 0:
        return 0.0
    entropy = 0.0
    for load in loads.values():
        if load > 0:
            p_j = load / total_load
            entropy += p_j * math.log(p_j)
    return -entropy


def calculate_h_c_norm(loads: Mapping[int, int], n_c0: int | None = None) -> float:
    """Control-capacity **retention** ``H_C / ln(n_c^0)``, clamped to ``[0, 1]``."""
    total_load = sum(loads.values())
    if total_load <= 0:
        return 0.0
    if n_c0 is None:
        n_c0 = len(loads)
    if n_c0 <= 1:
        return 0.0
    h_c = calculate_h_c(loads)
    max_entropy = math.log(n_c0)
    if max_entropy <= 0:  # pragma: no cover - guarded by n_c0 > 1
        return 0.0
    return min(1.0, max(0.0, h_c / max_entropy))


R_RES_NOTE = (
    "R_res = mean of three [0,1] curve areas (GCC, CSA, H_C); a curve-area statement, "
    "not a survival fraction. H_C is the control-capacity RETENTION H_C/ln(n_c^0) with a "
    "FIXED denominator n_c^0 (initial controllers); the old ln(surviving) self-normalisation "
    "was insensitive to collapse and inflated R_res. Report the three AUCs separately, "
    "collapse_f_gcc02, and r_res_half."
)
