"""Method-level baselines: construction + placement dispatch (paper §Experiment).

main.tex L604 protocol: *placement* baselines are applied on the **original** graph;
*topology* baselines use their own reconstruction + a **unified** placement (MADQN-sim).
``METHODS`` is the single table the method matrix reads, and
:func:`build_method_graph` / :func:`method_controllers` dispatch a method's two stages
without importing the old tree.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from ..baselines.placement.registry import PLACEMENT_REGISTRY
from ..baselines.topology.registry import CONSTRUCT_REGISTRY
from ..control.capacity import capacity_model
from ..rl.transfer import load_policy
from ..topology.builder import build_bimodal_graph
from ..topology.degree_sequence import plan_degree_sequence

__all__ = ["METHODS", "build_method_graph", "method_controllers"]

#: ``method -> {"construct": <name|"bit"|None>, "place": <"ppo"|"MADQN-sim"|"MWCL-sim">}``.
METHODS: dict[str, dict[str, Any]] = {
    "BiT-HyRL": {"construct": "bit", "place": "ppo"},
    "ONION": {"construct": "ONION", "place": "MADQN-sim"},
    "ROMEN": {"construct": "ROMEN", "place": "MADQN-sim"},
    "UNITY": {"construct": "UNITY", "place": "MADQN-sim"},
    "FRED-ABL": {"construct": "FRED-ABL", "place": "MADQN-sim"},
    "QDLM": {"construct": "QDLM", "place": "MADQN-sim"},
    "Eff-R-Net": {"construct": "Eff-R-Net", "place": "MADQN-sim"},
    "GRG": {"construct": "GRG", "place": "MADQN-sim"},
    "GRG-proxy": {"construct": "GRG-proxy", "place": "MADQN-sim"},
    # rewiring family: same N/M/degree sequence, only the correlations change
    "R-GREEDY": {"construct": "R-GREEDY", "place": "MADQN-sim"},
    "SA-R": {"construct": "SA-R", "place": "MADQN-sim"},
    "ASSORT-SWAP": {"construct": "ASSORT-SWAP", "place": "MADQN-sim"},
    "RAND-SWAP": {"construct": "RAND-SWAP", "place": "MADQN-sim"},
    "MWCL-sim": {"construct": None, "place": "MWCL-sim"},
    "MADQN-sim": {"construct": None, "place": "MADQN-sim"},
    # placement-only baselines: blind (lower bound) and textbook k-median
    "Random-placement": {"construct": None, "place": "Random-placement"},
    "KMedian": {"construct": None, "place": "KMedian"},
    # original substrate + PPO placement: the un-reconstructed reference row
    "Original": {"construct": None, "place": "ppo"},
}


def build_method_graph(
    original_G: nx.Graph,
    construct: str | None,
    *,
    seed: int = 0,
    c: float = 1.6,
    h: int = 1,
) -> nx.Graph:
    """Return the graph a method evaluates on.

    ``None`` -> the original graph (placement baseline); ``"bit"`` -> BiT bimodal
    reconstruction; any other name -> that topology baseline from the registry.
    """
    if not construct:
        return original_G
    if construct == "bit":
        n = original_G.number_of_nodes()
        m = original_G.number_of_edges()
        plan = plan_degree_sequence(n, m, c=c, h=h)
        return build_bimodal_graph(plan, seed).graph
    if construct not in CONSTRUCT_REGISTRY:
        raise KeyError(f"未知构造基线 {construct!r}；可用 {list(CONSTRUCT_REGISTRY)}")
    return CONSTRUCT_REGISTRY[construct](original_G, seed=seed)


def method_controllers(
    graph: nx.Graph,
    place: str,
    *,
    checkpoint: str | None = None,
    seed: int = 0,
    policy_obj: Any = None,
) -> list[int]:
    """Return the method's controller placement on ``graph`` (``n_c`` fixed by capacity)."""
    n_c = capacity_model(graph.number_of_nodes()).n_controllers
    if place == "ppo":
        policy = policy_obj if policy_obj is not None else load_policy(checkpoint)
        return [int(node) for node in policy.select(graph, n_c, greedy=True)]
    if place not in PLACEMENT_REGISTRY:
        raise KeyError(f"未知放置策略 {place!r}；可用 {list(PLACEMENT_REGISTRY)}")
    selector = PLACEMENT_REGISTRY[place]  # a ``select(graph, budget, seed)`` callable
    return [int(node) for node in selector(graph, n_c, seed=seed)]
