"""CPU-cheap node features for the PPO controller-placement policy (T10).

Every feature is **deterministic** for a given graph (and seed, for the sampled
betweenness) and normalised to roughly ``[0, 1]`` so the encoder sees a
well-conditioned input.  No process-global RNG is touched: the betweenness
sampler gets a local integer seed derived through
:func:`bit_hyrl.rng.derive_int`, preserving the package's single-source-of-
randomness contract (see :mod:`bit_hyrl.rng`).

Feature order is fixed by :data:`FEATURE_NAMES` so a checkpoint can be loaded on
a *different* graph than it was trained on.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .. import rng as _rng

__all__ = ["FEATURE_NAMES", "feature_config", "feature_dim", "node_features"]

#: Canonical feature order (degree-based, then topology-centrality).  ``betweenness``
#: is the only optional member; the rest are always present.
FEATURE_NAMES: tuple[str, ...] = ("degree", "coreness", "clustering", "betweenness")


def feature_dim(include_betweenness: bool = True) -> int:
    """Return the feature-vector width for the given configuration."""
    return len(FEATURE_NAMES) if include_betweenness else len(FEATURE_NAMES) - 1


def feature_config(
    *, include_betweenness: bool = True, betweenness_k: int = 64
) -> dict[str, object]:
    """Return the serialisable feature schema recorded inside a checkpoint.

    Args:
        include_betweenness: Whether the sampled betweenness column is present.
        betweenness_k: Number of source samples for the approximate betweenness.

    Returns:
        A JSON-friendly dict describing the feature extractor.
    """
    return {
        "names": list(FEATURE_NAMES[: feature_dim(include_betweenness)]),
        "include_betweenness": bool(include_betweenness),
        "betweenness_k": int(betweenness_k),
        "dim": feature_dim(include_betweenness),
    }


def _normalise(column: np.ndarray) -> np.ndarray:
    """Scale a non-negative column into ``[0, 1]`` by its peak (zeros stay zeros)."""
    if column.size == 0:
        return column
    peak = float(np.max(np.abs(column)))
    if peak > 0.0:
        return column / peak
    return column


def node_features(
    graph: nx.Graph,
    *,
    include_betweenness: bool = True,
    betweenness_k: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Return an ``N x d`` float32 feature matrix, rows ordered by ascending node id.

    Args:
        graph: The evaluated graph (node ids ``0 .. N-1`` after loading).
        include_betweenness: If True, append a sampled-betweenness column.
        betweenness_k: Source-sample budget for the approximate betweenness.
        seed: Root seed; the betweenness sampler derives a local integer from it.

    Returns:
        A ``(N, feature_dim(include_betweenness))`` ``float32`` array.
    """
    nodes = sorted(int(node) for node in graph.nodes())
    dim = feature_dim(include_betweenness)
    if not nodes:
        return np.zeros((0, dim), dtype=np.float32)
    n = len(nodes)

    degree = np.array([graph.degree(node) for node in nodes], dtype=np.float64)
    core_number = nx.core_number(graph)
    coreness = np.array([core_number.get(node, 0) for node in nodes], dtype=np.float64)
    clustering = nx.clustering(graph)
    clust = np.array([clustering.get(node, 0.0) for node in nodes], dtype=np.float64)

    columns = [_normalise(degree), _normalise(coreness), _normalise(clust)]

    if include_betweenness:
        k = min(n, max(1, int(betweenness_k)))
        local_seed = _rng.derive_int(
            seed, "ppo", "features", "betweenness", low=0, high=2**31 - 1
        )
        betweenness = nx.betweenness_centrality(graph, k=k, seed=local_seed, normalized=True)
        between = np.array(
            [betweenness.get(node, 0.0) for node in nodes], dtype=np.float64
        )
        columns.append(_normalise(between))

    return np.stack(columns, axis=1).astype(np.float32)
