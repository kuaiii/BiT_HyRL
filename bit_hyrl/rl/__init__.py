"""Controller placement: PPO policy, reward and training."""

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
