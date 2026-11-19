"""Zero-shot transfer of a trained PPO placement policy.

``load_policy`` rebuilds the actor-critic described by a checkpoint and
``apply`` places controllers on a *new* graph without any retraining — the
zero-shot setting the paper studies.  The feature schema travels with the checkpoint,
so the same policy can be applied to graphs of a different size / family.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch

from ..config.schema import TrainConfig
from .placement import DEFAULT_DECODE, PPOPolicy
from .ppo import ActorCritic

__all__ = ["load_checkpoint", "load_policy", "apply"]


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint dict written by :func:`bit_hyrl.rl.trainer.train`."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def load_policy(path: str | Path, device: str = "cpu") -> PPOPolicy:
    """Rebuild a :class:`PPOPolicy` from a checkpoint path."""
    checkpoint = load_checkpoint(path)
    cfg = TrainConfig.from_dict(checkpoint["cfg"])
    decode = {**DEFAULT_DECODE, **checkpoint.get("decode", {})}
    model = ActorCritic(
        int(checkpoint["feature_dim"]),
        cfg.hidden_dim,
        cfg.gcn_layers,
        heads=int(decode.get("heads", 2)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return PPOPolicy(model, decode, device=device)


def apply(policy: PPOPolicy, graph: nx.Graph, n_c: int) -> list[int]:
    """Zero-shot greedy placement of ``n_c`` controllers on ``graph``."""
    return policy.select(graph, n_c, greedy=True)
