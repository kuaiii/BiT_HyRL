"""Actor-critic network and the PPO objective.

The policy places controllers sequentially without replacement: one logit per
node, masked at every step.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat import GATEncoder

__all__ = [
    "ActorCritic",
    "masked_distribution",
    "compute_returns",
    "surrogate_loss",
]

_MASK_VALUE = -1.0e9


class ActorCritic(nn.Module):
    """GAT encoder + linear actor head + MLP critic head."""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, heads: int = 2) -> None:
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, layers, heads=heads)
        self.actor = nn.Linear(hidden_dim, 1)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings ``H`` (``N x hidden_dim``)."""
        return self.encoder(x, edge_index)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Return one raw action score per node (``N``)."""
        return self.actor(h).squeeze(-1)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        """Return the scalar state value from the mean-pooled embedding."""
        return self.critic(h.mean(dim=0)).squeeze(-1)


def masked_distribution(
    logits: torch.Tensor, used: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask ``used`` node logits and return ``(probs, log_probs, entropy)``."""
    used = list(used)
    masked = logits
    if used:
        masked = logits.clone()
        idx = torch.as_tensor(used, dtype=torch.long, device=logits.device)
        masked = masked.index_fill(0, idx, _MASK_VALUE)
    log_probs = torch.log_softmax(masked, dim=0)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum()
    return probs, log_probs, entropy


def compute_returns(rewards: Sequence[float], gamma: float = 1.0) -> list[float]:
    """Discounted returns ``G_t = sum_{k>=t} gamma^{k-t} r_k`` (backward pass)."""
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def surrogate_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    *,
    clip: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, float, float, float]:
    """PPO clipped-surrogate loss with a value baseline and an entropy bonus."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(value, returns)
    entropy_mean = entropy.mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    return total, float(policy_loss), float(value_loss), float(entropy_mean)
