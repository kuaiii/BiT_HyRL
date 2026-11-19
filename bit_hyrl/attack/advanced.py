"""Advanced attack sequences: GDM, adaptive and correlated orderings (Table 7).

Each producer returns a full permutation of the node set; the evaluation layer
feeds it to the cascade exactly like the basic removal orders.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable, Iterable, Sequence

import networkx as nx
import numpy as np

__all__ = [
    "ACTION_TABLE",
    "AdvancedAttackError",
    "GdmUnavailableError",
    "GDM_WEIGHTS_PATH",
    "adaptive_sequence",
    "assert_permutation",
    "correlated_sequence",
    "gdm_sequence",
    "multistage_sequence",
]

GDM_WEIGHTS_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "network_dismantling"
    / "GDM"
    / "models_newpg"
    / "gdm_nx_best.pth"
)


class AdvancedAttackError(RuntimeError):
    """Raised when an advanced attack cannot produce a valid permutation."""


class GdmUnavailableError(AdvancedAttackError):
    """Raised when the lazily-bridged GDM dismantler cannot be loaded or run."""


class _Residual:
    """Mutable view of the residual graph: an alive set plus incremental degrees."""

    __slots__ = ("graph", "nodes", "alive", "adj", "degree")

    def __init__(self, graph: nx.Graph) -> None:
        self.graph: nx.Graph = graph
        self.nodes: tuple[int, ...] = tuple(sorted(int(node) for node in graph.nodes()))
        self.alive: set[int] = set(self.nodes)
        self.adj: dict[int, tuple[int, ...]] = {
            node: tuple(int(other) for other in graph.neighbors(node))
            for node in self.nodes
        }
        self.degree: dict[int, int] = {
            node: len(self.adj[node]) for node in self.nodes
        }

    def remove(self, node: int) -> None:
        """Delete ``node`` from the residual graph and update its neighbours."""
        if node not in self.alive:
            raise AdvancedAttackError(f"node {node} was already removed, the sequence repeats it")
        self.alive.discard(node)
        for other in self.adj[node]:
            if other in self.alive:
                self.degree[other] -= 1

    def max_degree(self, pool: Iterable[int] | None = None) -> int:
        """Return the maximum-residual-degree node of ``pool`` (default: all alive)."""
        candidates = self.alive if pool is None else (set(pool) & self.alive)
        if not candidates:
            raise AdvancedAttackError("empty candidate set, cannot pick another node")
        return min(candidates, key=lambda node: (-self.degree[node], node))


def assert_permutation(
    sequence: Sequence[int],
    graph: nx.Graph,
    *,
    name: str = "attack",
) -> list[int]:
    """Verify that ``sequence`` is a permutation of ``graph``'s node set."""
    ordered = [int(node) for node in sequence]
    expected = {int(node) for node in graph.nodes()}
    if len(ordered) != len(expected):
        raise AdvancedAttackError(
            f"{name}: sequence length {len(ordered)} != node count {len(expected)}"
        )
    if len(set(ordered)) != len(ordered):
        duplicates = sorted({node for node in ordered if ordered.count(node) > 1})
        raise AdvancedAttackError(f"{name}: duplicated nodes in the sequence {duplicates[:10]}")
    if set(ordered) != expected:
        missing = sorted(expected - set(ordered))[:10]
        extra = sorted(set(ordered) - expected)[:10]
        raise AdvancedAttackError(f"{name}: sequence and node set disagree (missing {missing}, extra {extra})")
    return ordered


def adaptive_sequence(graph: nx.Graph, *, seed: int = 0) -> list[int]:
    """Remove the maximum-**residual**-degree node at every step."""
    del seed
    residual = _Residual(graph)
    order: list[int] = []
    while residual.alive:
        node = residual.max_degree()
        order.append(node)
        residual.remove(node)
    return assert_permutation(order, graph, name="adaptive")


def correlated_sequence(graph: nx.Graph, *, seed: int = 0) -> list[int]:
    """Attack the frontier first, so removals stay spatially correlated."""
    del seed
    residual = _Residual(graph)
    order: list[int] = []
    frontier: set[int] = set()

    while residual.alive:
        candidates = frontier & residual.alive
        node = residual.max_degree(candidates) if candidates else residual.max_degree()
        order.append(node)
        touched = residual.adj[node]
        residual.remove(node)
        frontier.discard(node)
        frontier.update(touched)
    return assert_permutation(order, graph, name="correlated")


def _largest_component_after_removal(subgraph: nx.Graph, node: int) -> int:
    """Return the size of the largest connected component of ``subgraph - node``."""
    trimmed = subgraph.copy()
    trimmed.remove_node(node)
    if trimmed.number_of_nodes() == 0:
        return 0
    return max(len(component) for component in nx.connected_components(trimmed))


def _articulation_pick(residual: _Residual) -> int:
    """Pick the articulation point whose removal leaves the smallest GCC."""
    if not residual.alive:
        raise AdvancedAttackError("residual graph is empty, no further articulation point")
    subgraph = residual.graph.subgraph(residual.alive)
    articulation_points = sorted(set(nx.articulation_points(subgraph)))
    if not articulation_points:
        return residual.max_degree()
    best_node = articulation_points[0]
    best_size = _largest_component_after_removal(subgraph, best_node)
    for node in articulation_points[1:]:
        size = _largest_component_after_removal(subgraph, node)
        if size < best_size:
            best_node, best_size = node, size
    return best_node


def multistage_sequence(graph: nx.Graph, *, seed: int = 0) -> list[int]:
    """Chain three stages over one removal budget."""
    residual = _Residual(graph)
    total = len(residual.nodes)
    cut_adaptive = total // 3
    cut_articulation = (2 * total) // 3

    order: list[int] = []

    for _ in range(cut_adaptive):
        node = residual.max_degree()
        order.append(node)
        residual.remove(node)

    while len(order) < cut_articulation and residual.alive:
        node = _articulation_pick(residual)
        order.append(node)
        residual.remove(node)

    remaining = sorted(residual.alive)
    if remaining:
        generator = np.random.default_rng(int(seed))
        permutation = generator.permutation(len(remaining))
        order.extend(remaining[int(index)] for index in permutation)

    return assert_permutation(order, graph, name="multistage")


def _to_canonical_indexing(graph: nx.Graph, nodes: Sequence[int]) -> nx.Graph:
    """Return a copy of ``graph`` whose node ids are ``0 .. N-1`` in sorted order."""
    canonical = list(range(len(nodes)))
    if list(nodes) == canonical:
        return graph.copy()
    return nx.relabel_nodes(
        graph, {node: index for index, node in enumerate(nodes)}, copy=True
    )


def gdm_sequence(graph: nx.Graph, *, seed: int = 0) -> list[int]:
    """Order nodes by the GDM graph-attention dismantling score, descending."""
    del seed
    nodes = tuple(sorted(int(node) for node in graph.nodes()))
    if not nodes:
        raise GdmUnavailableError("GDM cannot run on an empty graph")

    weights = GDM_WEIGHTS_PATH
    if not weights.is_file():
        raise GdmUnavailableError(
            f"GDM weights not found: {weights} (expected at network_dismantling/GDM/models_newpg/gdm_nx_best.pth)"
        )

    try:  # pragma: no cover - depends on the optional GNN stack
        import torch  # noqa: F401  (imported for the device probe only)
        from network_dismantling.GDM.predictors_nx import load_model, predict_scores
    except Exception as exc:
        raise GdmUnavailableError(
            "GDM bridge failed: cannot import network_dismantling.GDM.predictors_nx"
            " (requires torch / torch_geometric / torch_sparse / torch_scatter)"
            f"original error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        ) from exc

    try:  # pragma: no cover - depends on the optional GNN stack
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, features, _config = load_model(str(weights), device=device)
        scores = predict_scores(
            _to_canonical_indexing(graph, nodes), model, features, device=device
        )
    except Exception as exc:
        raise GdmUnavailableError(
            "GDM inference failed: "
            f"original error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        ) from exc

    score = np.asarray(scores, dtype=float).reshape(-1)
    if score.shape[0] != len(nodes):
        raise GdmUnavailableError(
            f"GDM score length {score.shape[0]} != node count {len(nodes)}"
        )

    ranked = sorted(range(len(nodes)), key=lambda index: (-score[index], nodes[index]))
    order = [nodes[index] for index in ranked]
    return assert_permutation(order, graph, name="gdm")


ACTION_TABLE: dict[str, Callable[..., list[int]]] = {
    "adaptive": adaptive_sequence,
    "correlated": correlated_sequence,
    "multistage": multistage_sequence,
    "gdm": gdm_sequence,
}
