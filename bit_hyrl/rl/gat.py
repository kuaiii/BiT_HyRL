"""A compact, dependency-light graph-attention encoder (T10).

This is a real multi-head graph-attention layer implemented with **pure
``torch`` tensor ops** (``index_add`` / ``scatter_reduce``), so the RL layer runs
on any CPU ``torch`` install without depending on the exact ``torch_geometric``
build being importable at runtime.  ``torch_geometric`` is still an allowed
dependency (it is installed in the environment) but is not required here.

The encoder maps an ``N x d`` node-feature matrix plus an ``edge_index``
(``2 x E``, self-loops included) to ``N x hidden_dim`` node embeddings ``H`` that
the actor/critic heads consume.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["edge_index_from_graph", "GATLayer", "GATEncoder"]


def edge_index_from_graph(graph: nx.Graph, *, add_self_loops: bool = True) -> torch.Tensor:
    """Build a bidirectional ``2 x E`` ``edge_index`` (``int64``) from a graph.

    Node ids are mapped to their position in ``sorted(graph.nodes())`` so the
    tensor rows line up with :func:`bit_hyrl.rl.features.node_features`.

    Args:
        graph: The evaluated graph.
        add_self_loops: If True, append an ``i -> i`` edge for every node so that a
            node with no other neighbour still aggregates its own state.

    Returns:
        An ``int64`` tensor of shape ``(2, E)``.
    """
    nodes = sorted(int(node) for node in graph.nodes())
    index = {node: pos for pos, node in enumerate(nodes)}
    src: list[int] = []
    dst: list[int] = []
    for u, v in graph.edges():
        a = index[int(u)]
        b = index[int(v)]
        src.extend((a, b))
        dst.extend((b, a))
    if add_self_loops:
        for pos in range(len(nodes)):
            src.append(pos)
            dst.append(pos)
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _scatter_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax ``scores`` (``E x heads``) over the neighbours of each destination.

    Args:
        scores: Raw attention logits, shape ``(E, heads)``.
        dst: Destination node index per edge, shape ``(E,)``.
        num_nodes: ``N``.

    Returns:
        Attention weights of the same shape ``(E, heads)`` summing to 1 per
        ``(destination, head)``.
    """
    num_edges, heads = scores.shape
    col = dst.unsqueeze(1).expand(num_edges, heads)
    neg_inf = torch.full(
        (num_nodes, heads), float("-inf"), dtype=scores.dtype, device=scores.device
    )
    # Detached max is only a numerical-stability shift; gradients still flow via exp().
    max_per_node = neg_inf.scatter_reduce(
        0, col, scores.detach(), reduce="amax", include_self=True
    )
    # Gather the per-(destination, head) max back onto every edge (row-wise gather
    # along dim 0 — advanced indexing ``max_per_node[col]`` would add a trailing
    # axis and produce the wrong shape).
    shifted = scores - max_per_node.gather(0, col)
    exp_scores = torch.exp(shifted)
    denom = torch.zeros(
        num_nodes, heads, dtype=scores.dtype, device=scores.device
    ).index_add_(0, dst, exp_scores)
    return exp_scores / (denom[dst] + 1e-16)


class GATLayer(nn.Module):
    """One multi-head graph-attention layer with a residual + LayerNorm.

    Args:
        dim: Feature width (preserved by the residual connection).
        heads: Number of attention heads; silently downgraded to 1 if ``dim`` is
            not divisible by ``heads``.
        dropout: Attention-weight dropout probability (training only).
    """

    def __init__(self, dim: int, heads: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = dim // self.heads
        if self.head_dim * self.heads != dim:
            self.heads = 1
            self.head_dim = dim
        self.lin = nn.Linear(dim, self.heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.out = nn.Linear(self.heads * self.head_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply attention to ``h`` (``N x dim``) over ``edge_index`` (``2 x E``)."""
        num_nodes = h.size(0)
        projected = self.lin(h).view(num_nodes, self.heads, self.head_dim)
        src, dst = edge_index[0], edge_index[1]
        # Per-edge, per-head attention logit (GAT symmetric score).
        score = (projected[src] * self.att_src).sum(-1) + (
            projected[dst] * self.att_dst
        ).sum(-1)
        score = F.leaky_relu(score, negative_slope=0.2)
        alpha = _scatter_softmax(score, dst, num_nodes)
        if self.training and self.dropout > 0.0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        messages = alpha.unsqueeze(-1) * projected[src]
        aggregated = torch.zeros(
            num_nodes, self.heads, self.head_dim, dtype=h.dtype, device=h.device
        ).index_add(0, dst, messages)
        out = self.out(aggregated.reshape(num_nodes, self.heads * self.head_dim))
        return self.norm(out + h)


class GATEncoder(nn.Module):
    """Stack of :class:`GATLayer` mapping node features to node embeddings.

    Args:
        in_dim: Input feature width ``d``.
        hidden_dim: Embedding width (kept constant across layers for the residual).
        layers: Number of attention layers (``gcn_layers``).
        heads: Attention heads per layer.
        dropout: Attention-weight dropout.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GATLayer(hidden_dim, heads=heads, dropout=dropout)
            for _ in range(max(1, int(layers)))
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode features ``x`` (``N x d``) to embeddings (``N x hidden_dim``)."""
        h = self.input(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h
