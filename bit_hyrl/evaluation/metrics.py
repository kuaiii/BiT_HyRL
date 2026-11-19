"""Control-plane robustness metrics for the BiT-HyRL evaluation layer.

These three functions used to live in the legacy ``network_metrics/functional/
controller.py`` tree.  The package keeps **zero** imports into that tree, so the implementations are re-done here, compactly and
self-contained.

All three metrics are graph-agnostic: the runner builds the surviving induced
subgraph (and the final controller load map) from the last cascade step and hands
it to these functions, so the same code path serves the *original* substrate and
the *BiT-reconstructed* proposal.

NOTE on H_C — the paper defines control entropy as the entropy of the *load
share* across controllers, ``p_j = L_j / Σ_j L_j`` (main.tex §3).  The legacy
implementation instead used the spatial share (switches per controller
component); we follow the paper here.  When exactly one controller survives the
normaliser ``ln(n_controllers)`` vanishes, so we follow the paper's convention
(main.tex:320) of reporting full concentration = ``1.0`` rather than ``0``/NaN.
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
    """Control Supply Availability (CSA) over an evaluated subgraph.

    .. math::

        \\text{CSA} = \\frac{1}{|V_S|} \\sum_i \\left(1 - e^{-\\alpha k_i}\\right)

    where ``k_i`` is the number of controllers inside the connected component of
    switch ``i`` and ``V_S`` is the set of surviving switches.  Passing
    ``initial_num_switches`` fixes the denominator at the **original** switch
    count (the paper fixes it at ``N``) so that CSA values at different removal
    fractions stay comparable on a common scale.

    Args:
        G: The evaluated graph (typically the subgraph induced by the surviving
            nodes).  Must be node-indexed with the same ids as ``centers``.
        centers: Surviving controller node ids.
        alpha: Decay factor (default ``0.2``, per main.tex:300).
        initial_num_switches: Denominator override; defaults to the live switch
            count of ``G``.

    Returns:
        The CSA value in ``[0, 1]``.  Empty graph or no surviving switches -> ``0.0``.
    """
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
    """Load-share control entropy ``H_C`` over a ``{controller: L_j}`` mapping.

    .. math::

        H_C = -\\sum_j p_j \\ln p_j, \\qquad p_j = L_j / \\sum_j L_j

    A uniform load share yields the maximum entropy; a single controller carrying
    all load yields ``H_C = 0`` (no spread).  This is the raw entropy in nats;
    :func:`calculate_h_c_norm` normalises it for the CCE proxy.

    Args:
        loads: ``{controller_id: load}`` for the surviving controllers.  Every
            surviving controller is expected as a key (zero-load entries included);
            zero-load controllers contribute a zero probability and therefore
            nothing to the sum.

    Returns:
        The entropy in nats.  No load at all -> ``0.0``.
    """
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
    """Control-capacity **retention** ``H_C / ln(n_c^0)``, clamped to ``[0, 1]``.

    .. math::

        H_{C,\\text{norm}} = \\operatorname{clamp}\\
            \\left(\\frac{-\\sum_j p_j \\ln p_j}{\\ln n_c^0}, 0, 1\\right)

    The denominator is the **fixed** initial controller count ``n_c^0`` (NOT the
    number of surviving controllers).  This is the audit-corrected definition: the
    old ``ln(surviving)`` self-normalisation stayed near ``1`` and inflated ``R_res``;
    a fixed denominator instead measures how much of the initial control capacity is
    **retained**, so a concentrated/decayed control plane scores low.

    Conventions (paper Eq.): ``n_c^0 <= 1`` -> ``0``; no load -> ``0``; **a single
    surviving controller -> ``0``** (control fully concentrated — no spread, and
    definitely not the old ``1``); uniform load over ``n_c^0`` controllers -> ``1``.

    Args:
        loads: ``{controller_id: load}`` for the surviving controllers (zero-load
            entries may be present).
        n_c0: Initial controller count ``n_c^0`` — the fixed normaliser.  When
            ``None`` (legacy callers), falls back to ``len(loads)`` (the surviving
            controllers present); all callers pass it explicitly.

    Returns:
        The retained control-capacity fraction in ``[0, 1]``.
    """
    total_load = sum(loads.values())
    if total_load <= 0:
        return 0.0
    if n_c0 is None:
        n_c0 = len(loads)
    if n_c0 <= 1:
        # 0 or 1 controller: control fully concentrated (no spread) -> 0, not 1.
        return 0.0
    h_c = calculate_h_c(loads)
    max_entropy = math.log(n_c0)
    if max_entropy <= 0:  # pragma: no cover - guarded by n_c0 > 1
        return 0.0
    return min(1.0, max(0.0, h_c / max_entropy))


#: Audit note for the ``R_res`` composite (see :mod:`bit_hyrl.evaluation.curves`).
#: ``R_res`` is a **curve-area** statement, NOT a survival fraction — ``> 0.5`` does
#: not mean "50 % of nodes survive".  It is the mean of three ``[0, 1]`` trapezoidal
#: AUCs ``(AUC_GCC + AUC_CSA + AUC_HC) / 3``.  H_C is normalised by the **fixed**
#: ``ln(n_controllers^0)`` (initial count), not by ``ln(surviving controllers)`` —
#: the self-normalised form sat near 1 and inflated the headline (audit finding).
#: Always report the three terms separately, plus ``collapse_f_gcc02`` (first ``f``
#: with GCC < 0.2) and the half-window ``r_res_half`` (f <= 0.5).
R_RES_NOTE = (
    "R_res = mean of three [0,1] curve areas (GCC, CSA, H_C); a curve-area statement, "
    "not a survival fraction. H_C is the control-capacity RETENTION H_C/ln(n_c^0) with a "
    "FIXED denominator n_c^0 (initial controllers); the old ln(surviving) self-normalisation "
    "was insensitive to collapse and inflated R_res. Report the three AUCs separately, "
    "collapse_f_gcc02, and r_res_half."
)
