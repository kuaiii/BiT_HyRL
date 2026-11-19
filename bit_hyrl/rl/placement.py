"""Sequential controller placement policies (T10).

:class:`PPOPolicy` wraps a trained :class:`~bit_hyrl.rl.ppo.ActorCritic` and
places ``n_c`` controllers one at a time, never picking the same node twice.
:func:`degree_ranked` is the deterministic baseline kept in the same module so the
``eval --policy`` command can compare the two on identical graphs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .. import rng as _rng
from .gat import edge_index_from_graph
from .features import feature_config, node_features
from .ppo import ActorCritic, masked_distribution

__all__ = ["PPOPolicy", "degree_ranked", "DEFAULT_DECODE"]

#: Default decode/feature configuration recorded inside a checkpoint.
DEFAULT_DECODE: dict[str, Any] = {
    "include_betweenness": True,
    "betweenness_k": 64,
    "heads": 2,
}


class PPOPolicy:
    """Greedy or sampled sequential placement with a trained actor-critic.

    Args:
        model: A trained :class:`ActorCritic`.
        config: Decode/feature schema (``include_betweenness``, ``betweenness_k``,
            ``heads``).  Missing keys fall back to :data:`DEFAULT_DECODE`.
        device: Torch device (CPU by default).
    """

    def __init__(
        self, model: ActorCritic, config: dict[str, Any] | None = None, device: str = "cpu"
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device
        self.config: dict[str, Any] = {**DEFAULT_DECODE, **(config or {})}

    def _features(self, graph, seed: int) -> torch.Tensor:
        arr = node_features(
            graph,
            include_betweenness=bool(self.config["include_betweenness"]),
            betweenness_k=int(self.config["betweenness_k"]),
            seed=seed,
        )
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def select(
        self, graph, n_c: int, greedy: bool = True, *, seed: int = 0
    ) -> list[int]:
        """Select ``n_c`` distinct controllers, returns ascending node ids.

        Args:
            graph: The graph to place controllers on.
            n_c: How many controllers to place (clipped to ``[0, N]``).
            greedy: If True take the arg-max each step; otherwise sample from the
                masked categorical (deterministically, via the derived seed).
            seed: Root seed for the sampling path (ignored when ``greedy=True``).

        Returns:
            A sorted list of controller node ids.
        """
        n = graph.number_of_nodes()
        n_c = max(0, min(int(n_c), int(n)))
        if n_c == 0:
            return []
        x = self._features(graph, seed)
        edge_index = edge_index_from_graph(graph).to(self.device)
        h = self.model.encode(x, edge_index)
        base_logits = self.model.logits(h)

        generator = _rng.derive(seed, "ppo", "select") if not greedy else None
        used: set[int] = set()
        selected: list[int] = []
        for _ in range(n_c):
            probs, _log_probs, _entropy = masked_distribution(base_logits, used)
            probs_np = probs.detach().cpu().numpy()
            if greedy:
                pick = int(np.argmax(probs_np))
            else:
                assert generator is not None
                cum = np.cumsum(probs_np)
                draw = float(generator.random()) * float(cum[-1])
                pick = int(np.searchsorted(cum, draw, side="right"))
                pick = min(pick, n - 1)
            used.add(pick)
            selected.append(pick)
        return sorted(int(node) for node in selected)

    # ``apply`` is an alias used by the CLI/transfer helpers.
    def apply(self, graph, n_c: int) -> list[int]:
        """Greedy zero-shot placement (see :meth:`select`)."""
        return self.select(graph, n_c, greedy=True)


def degree_ranked(graph, n_c: int) -> list[int]:
    """Deterministic ``(-degree, node_id)`` placement baseline."""
    order = sorted(graph.nodes(), key=lambda node: (-graph.degree(node), int(node)))
    return sorted(int(node) for node in order[: max(0, int(n_c))])
