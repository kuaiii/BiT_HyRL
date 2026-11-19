"""Actor-critic network and the PPO objective (T10, paper Eq. PPO).

The policy is a *sequential, without-replacement* controller placer: at each step
it scores every not-yet-selected node, masks the already-selected ones, and picks
one.  The actor head therefore just emits one logit per node; the masking +
softmax happens in :func:`masked_distribution`, shared by rollout (sampling) and
the PPO update (re-scoring).

The critic regresses the terminal placement reward from the mean-pooled graph
embedding.  With ``gamma = 1`` and a single terminal reward, the discounted return
of every step in an episode equals that terminal reward — see
:func:`compute_returns`.
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

#: Small constant used to forbid already-selected nodes (``-inf`` breaks softmax
#: gradients when a whole row is masked; ``-1e9`` is finite and safe).
_MASK_VALUE = -1.0e9


class ActorCritic(nn.Module):
    """GAT encoder + linear actor head + MLP critic head.

    Args:
        in_dim: Node-feature width.
        hidden_dim: GAT embedding width.
        layers: Number of GAT layers (``gcn_layers``).
        heads: Attention heads per layer.
    """

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
    """Mask ``used`` node logits and return ``(probs, log_probs, entropy)``.

    Args:
        logits: Raw per-node scores (``N``).
        used: Node indices already selected in this episode.

    Returns:
        ``probs`` and ``log_probs`` of shape ``(N,)`` and the scalar entropy
        ``-\\sum_i p_i \\log p_i`` of the masked categorical.
    """
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
    """Discounted returns ``G_t = sum_{k>=t} gamma^{k-t} r_k`` (backward pass).

    With a single terminal reward and ``gamma = 1`` the result is the terminal
    reward replicated to every step — which is exactly the credit assignment the
    placement episode needs.
    """
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
    """PPO clipped-surrogate loss with a value baseline and an entropy bonus.

    Args:
        new_log_probs: Re-scored log-probs of the taken actions (``T``).
        old_log_probs: Log-probs recorded during the rollout (``T``).
        advantages: ``G_t - V(s)`` per step (``T``).
        value: Current scalar critic estimate.
        returns: Target value (the terminal reward).
        entropy: Mean policy entropy (scalar).
        clip: PPO clip range.
        value_coef: Weight of the value loss.
        entropy_coef: Weight of the entropy bonus.

    Returns:
        ``(total_loss, policy_loss, value_loss, entropy)`` — the last three are
        floats for logging, ``total_loss`` is a differentiable scalar.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(value, returns)
    entropy_mean = entropy.mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    return total, float(policy_loss), float(value_loss), float(entropy_mean)
