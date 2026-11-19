"""Shared helpers for the ported baselines."""

from __future__ import annotations

import contextlib
import logging
import random

import networkx as nx
import numpy as np

LOGGER = logging.getLogger("bit_hyrl.baselines")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

__all__ = [
    "finalize_construct",
    "positions_or_seeded",
    "normalized_env_graph",
    "seeded_legacy",
]


@contextlib.contextmanager
def seeded_legacy(seed: int):
    """Run a block under a **seed-derived** global RNG, then restore it exactly."""
    children = np.random.SeedSequence(int(seed)).spawn(3)
    r_py = int(children[0].generate_state(1, dtype=np.uint32)[0])
    r_np = int(children[1].generate_state(1, dtype=np.uint32)[0])
    r_torch = int(children[2].generate_state(1, dtype=np.uint32)[0])

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = None
    cuda_states = None
    if torch is not None:
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()

    random.seed(r_py)
    np.random.seed(r_np)
    if torch is not None:
        torch.manual_seed(r_torch)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def _ensure_connected(graph: nx.Graph) -> int:
    """Join components by adding the **minimum** ``k-1`` edges (no node dropped)."""
    components = [set(component) for component in nx.connected_components(graph)]
    if len(components) <= 1:
        return 0
    components.sort(key=lambda comp: (-len(comp), min(comp)))
    main = components[0]
    added = 0
    for other in components[1:]:
        u = max(main, key=lambda node: (graph.degree(node), -node))
        v = max(other, key=lambda node: (graph.degree(node), -node))
        graph.add_edge(u, v)
        main |= other
        added += 1
    return added


def finalize_construct(graph, name: str = "") -> nx.Graph:
    """Coerce to a **simple, connected, ``0..n-1``** graph that PRESERVES ``N``."""
    g = nx.Graph()
    g.add_nodes_from(graph.nodes())
    g.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    n_in = graph.number_of_nodes()
    n = g.number_of_nodes()
    m_in = g.number_of_edges()
    if n == 0:
        return nx.convert_node_labels_to_integers(g, ordering="sorted")

    k = nx.number_connected_components(g)
    if k > 1:
        added = _ensure_connected(g)
        LOGGER.info(
            "construct=%s n=%d m_in=%d m_out=%d components=%d (+%d edges)",
            name or "?", n, m_in, g.number_of_edges(), k, added,
        )

    out = nx.convert_node_labels_to_integers(g, ordering="sorted")
    if out.number_of_nodes() != n_in:
        raise ValueError(
            f"finalize_construct lost nodes: before={n_in} "
            f"after={out.number_of_nodes()} components={k}"
        )
    if not nx.is_connected(out):
        raise ValueError(
            f"finalize_construct produced disconnected graph: n={out.number_of_nodes()} "
            f"components={nx.number_connected_components(out)}"
        )
    return out


def positions_or_seeded(graph: nx.Graph, seed: int, *, scale: float = 1.0) -> dict:
    """Return ``{node: (x, y)}`` — node ``pos`` attrs if complete, else seeded uniform."""
    pos: dict = {}
    for node in graph.nodes():
        value = graph.nodes[node].get("pos")
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            pos[node] = (float(value[0]), float(value[1]))
    if len(pos) != graph.number_of_nodes():
        rng = np.random.default_rng(seed)
        pos = {node: (float(rng.random()) * scale, float(rng.random()) * scale)
               for node in graph.nodes()}
    return pos


def normalized_env_graph(graph: nx.Graph, seed: int) -> tuple[nx.Graph, dict]:
    """Return ``(G_norm, norm_pos)`` with nodes ``0..n-1`` and ``pos`` in ``[0, 1]``."""
    pos = positions_or_seeded(graph, seed, scale=1.0)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-12)
    lo_x, lo_y = min(xs), min(ys)
    norm_pos: dict = {}
    g_norm = nx.Graph()
    g_norm.add_nodes_from(range(graph.number_of_nodes()))
    node_map = {}
    for index, node in enumerate(graph.nodes()):
        node_map[node] = index
        norm_pos[index] = ((pos[node][0] - lo_x) / span, (pos[node][1] - lo_y) / span)
        g_norm.nodes[index]["pos"] = norm_pos[index]
    for u, v in graph.edges():
        g_norm.add_edge(node_map[u], node_map[v])
    return g_norm, norm_pos
