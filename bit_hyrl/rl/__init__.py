"""The RL / PPO controller-placement layer of ``bit_hyrl`` (T10).

Public surface:

* :class:`PPOPolicy` / :func:`degree_ranked` — placement policies;
* :func:`placement_reward` — the paper-weighted reward;
* :func:`train` — the PPO trainer;
* :func:`load_policy` / :func:`apply` — zero-shot transfer;
* :class:`ActorCritic` / :func:`node_features` — the network and its inputs.

The heavy ``torch`` import is confined to this package; the CLI imports it lazily
so non-RL sub-commands stay fast.
"""

from __future__ import annotations

from .features import FEATURE_NAMES, feature_config, feature_dim, node_features
from .gat import GATEncoder, GATLayer, edge_index_from_graph
from .placement import PPOPolicy, degree_ranked
from .ppo import ActorCritic, compute_returns, masked_distribution, surrogate_loss
from .reward import DEFAULT_REWARD_WEIGHTS, placement_reward
from .trainer import build_model, make_policy, resolve_graphs, train
from .transfer import apply, load_checkpoint, load_policy

__all__ = [
    "FEATURE_NAMES",
    "feature_config",
    "feature_dim",
    "node_features",
    "GATEncoder",
    "GATLayer",
    "edge_index_from_graph",
    "PPOPolicy",
    "degree_ranked",
    "ActorCritic",
    "compute_returns",
    "masked_distribution",
    "surrogate_loss",
    "DEFAULT_REWARD_WEIGHTS",
    "placement_reward",
    "build_model",
    "make_policy",
    "resolve_graphs",
    "train",
    "apply",
    "load_checkpoint",
    "load_policy",
]
