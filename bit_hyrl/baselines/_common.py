"""Shared helpers for the ported baselines.

Every topology baseline returns a graph with the **same N** as its input and the
``0 .. n-1`` label convention used across the package.  Connectivity is repaired by
**adding the minimum number of edges** to join the connected components (never by
dropping nodes), so ``N`` is preserved exactly and baseline comparisons stay fair —
``M`` may grow by ``k-1`` where ``k`` is the number of components.  Positions for the
position-dependent baselines come from node ``pos`` attributes when present, else
from a **locally seeded** ``numpy`` generator (``default_rng``) — never the
process-global RNG.
"""

from __future__ import annotations

import contextlib
import logging
import random

import networkx as nx
import numpy as np

LOGGER = logging.getLogger("bit_hyrl.baselines")

try:  # torch is required by ROMEN; keep the import optional for the lighter baselines
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
    """Run a block under a **seed-derived** global RNG, then restore it exactly.

    The ported baselines are third-party research code that draw from the
    process-global ``random`` / ``numpy.random`` / ``torch`` generators.  To make a
    given ``seed`` reproducible *without polluting* the core pipeline, this
    context manager:

    1. derives three independent seeds from ``numpy.random.SeedSequence(seed)``;
    2. saves the global ``random`` / ``numpy.random`` / ``torch`` (and CUDA) state;
    3. seeds all three from the derived values inside the block;
    4. restores every saved state on exit (``finally``) — even on exception.

    So ``construct(G, seed=s)`` is bit-reproducible for a fixed ``s``, and the
    calling core (which uses :mod:`bit_hyrl.rng` SeedSequence derivation) is left
    untouched.  ``baselines/`` is excluded from the static ``test_no_global_seed``
    scan precisely because this vendored code needs the module-level RNG.
    """
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
    """Join components by adding the **minimum** ``k-1`` edges (no node dropped).

    Repeatedly connects the current **largest** component's highest-degree node to the
    highest-degree node of the next component (degree-then-smallest-id tie-break, so
    deterministic), merging as it goes.  Returns the number of added edges.
    """
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
    """Coerce to a **simple, connected, ``0..n-1``** graph that PRESERVES ``N``.

    If the constructed graph is disconnected it is repaired by adding the minimum
    ``k-1`` edges (one per extra connected component) between high-degree endpoints
    — **    no node is dropped**, so ``N`` stays exact and baselines are not weakened.
    ``M`` may increase by up to ``k-1`` edges (small for most methods — +0/+1 — but up
    to tens when a method shatters a large graph into many components, e.g. FRED-ABL /
    QDLM).  Self-loops and parallel
    edges are removed (simple graph).

    Fail-fast: raises ``ValueError`` if the result loses a node or is disconnected,
    so a future regression explodes on the first run instead of silently shrinking.
    """
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
