"""Graph-attention encoder built from plain ``torch`` operations.

Kept independent of ``torch_geometric`` so the RL layer runs on any CPU torch
installation.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["edge_index_from_graph", "GATLayer", "GATEncoder"]


def edge_index_from_graph(graph: nx.Graph, *, add_self_loops: bool = True) -> torch.Tensor:
    """Build a bidirectional ``2 x E`` ``edge_index`` (``int64``) from a graph."""
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
    """Softmax ``scores`` (``E x heads``) over the neighbours of each destination."""
    num_edges, heads = scores.shape
    col = dst.unsqueeze(1).expand(num_edges, heads)
    neg_inf = torch.full(
        (num_nodes, heads), float("-inf"), dtype=scores.dtype, device=scores.device
    )
    max_per_node = neg_inf.scatter_reduce(
        0, col, scores.detach(), reduce="amax", include_self=True
    )
    shifted = scores - max_per_node.gather(0, col)
    exp_scores = torch.exp(shifted)
    denom = torch.zeros(
        num_nodes, heads, dtype=scores.dtype, device=scores.device
    ).index_add_(0, dst, exp_scores)
    return exp_scores / (denom[dst] + 1e-16)


class GATLayer(nn.Module):
    """One multi-head graph-attention layer with a residual + LayerNorm."""

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
    """Stack of :class:`GATLayer` mapping node features to node embeddings."""

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
