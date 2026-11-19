"""Advanced (Tier-B/Tier-C) attack sequences — the producers behind Table 7.

``main.tex`` Table ``tab:advanced_attacks`` reports four attackers that are *not*
static centrality rankings, because a static ranking is evaluated once on the
original graph and then replayed on a topology it does not recognise.  The four
are defined as follows and this module implements exactly that, no more:

``adaptive`` — dynamic maximum residual degree
    At every step recompute the degree **inside the residual graph** and remove
    the largest one.  This is the classic "recalculate after each removal"
    attack; it is strictly stronger than ``degree`` on a hub-and-spoke topology
    because deleting the first hub promotes the second, which a static ranking
    has already committed to a later position.  Ties are broken by ascending
    node id so the sequence is reproducible.

``correlated`` — frontier-limited, regionally chained failure
    Start from the global maximum-degree node.  After that, restrict the choice
    to the **attack frontier** — the still-alive neighbours of already-removed
    nodes — and take the maximum-degree node there.  When the frontier empties
    (the attack has eaten an entire connected region) it falls back to the
    global maximum-degree node and starts a new region.  The point is that
    removals are *spatially correlated*, which is what a cascading
    geographically-localised outage looks like; an independent per-node ranking
    cannot express that.

``multistage`` — three chained stages over one budget
    ① ``adaptive`` for the first third; ② **articulation-point-first** for the
    middle third, where the articulation-point set is ranked by "largest
    connected component remaining after removal" (smallest wins, ties by node
    id) and non-separable graphs fall back to maximum residual degree;
    ③ ``random`` over the remaining third.  The three stages together cover
    **every** node exactly once, so the produced list is still a permutation.

``gdm`` — Graph Dismantling Machine (Grassia et al., 2021)
    A GAT trained to regress a node's value *as part of a removal set*.  The
    implementation is the repository's vendored copy under
    ``network_dismantling/GDM`` and it is reached through a **lazy import** so
    that the core package keeps its "zero imports from the legacy tree" promise
    (:mod:`bit_hyrl` docstring): nothing is imported until the function is
    called, and a missing dependency raises
    :class:`GdmUnavailableError` carrying the original traceback instead of
    silently substituting a different algorithm.

Why every function returns a full permutation
---------------------------------------------
:func:`~bit_hyrl.control.cascade.run_cascade` emits one step per entry of the
sequence, so ``f_grid`` and ``r_curve`` are only comparable across attacks when
every sequence has the same length ``N``.  A producer that returned a truncated
sequence (e.g. GDM's own dismantler stops once the LCC is small enough) would
shift the AUC integration window and make the comparison meaningless.  All four
functions therefore return ``N`` distinct ids covering the whole node set, and
:func:`assert_permutation` states that contract as a run-time check rather than a
comment.

Determinism
-----------
``adaptive``, ``correlated`` and ``multistage`` are deterministic given the graph
(the first two do not consume ``seed`` at all; ``multistage`` uses it only for its
random third stage, through ``numpy.random.default_rng(seed)``).  ``gdm`` is
deterministic too — the predictor runs in ``eval`` mode and the tie-break is by
node id — so its ``seed`` argument is accepted for signature uniformity and
ignored.  No function touches process-global random state, which is the
:mod:`bit_hyrl.rng` contract.
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

#: Weights of the GDM graph-attention dismantler shipped with the repository.
#: Resolved from this file (``<repo>/bit_hyrl/attack/advanced.py``) so the
#: module works from any working directory.
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
    """Raised when the lazily-bridged GDM dismantler cannot be loaded or run.

    Carries the original exception text so the caller can report the *real*
    blocker (a missing wheel, a missing checkpoint, a shape mismatch) instead of
    a generic "GDM failed".
    """


# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #
class _Residual:
    """Mutable view of the residual graph: an alive set plus incremental degrees.

    Recomputing ``graph.degree(v)`` after every removal is ``O(N·(N+M))`` and
    dominates the run time on a 500-node graph, so the degree of every node is
    cached here and decremented by one for each surviving neighbour of the node
    being removed.  The cache is the only state; the input graph is never
    mutated, which is what makes the four producers side-effect free.

    Attributes:
        graph: The evaluated graph (read-only).
        nodes: All node ids, ascending.
        alive: The currently surviving node ids.
        adj: ``{node: tuple(neighbours)}`` — a plain-tuple adjacency snapshot so
            that lookups do not re-enter networkx per step.
        degree: Residual degree ``deg(node)`` for every surviving node.
    """

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
        """Delete ``node`` from the residual graph and update its neighbours.

        Args:
            node: A surviving node id.

        Raises:
            AdvancedAttackError: If ``node`` is already gone (a producer bug —
                a duplicate in the sequence would break the permutation
                contract).
        """
        if node not in self.alive:
            raise AdvancedAttackError(f"节点 {node} 已被移除，攻击序列出现重复")
        self.alive.discard(node)
        for other in self.adj[node]:
            if other in self.alive:
                self.degree[other] -= 1

    def max_degree(self, pool: Iterable[int] | None = None) -> int:
        """Return the maximum-residual-degree node of ``pool`` (default: all alive).

        Args:
            pool: Candidate ids; the ones no longer alive are ignored.

        Returns:
            The node maximising ``(-degree, node_id)``, i.e. highest residual
            degree with ties broken by the **smallest** node id.

        Raises:
            AdvancedAttackError: If the candidate pool is empty.
        """
        candidates = self.alive if pool is None else (set(pool) & self.alive)
        if not candidates:
            raise AdvancedAttackError("候选集合为空，无法继续选择攻击节点")
        return min(candidates, key=lambda node: (-self.degree[node], node))


def assert_permutation(
    sequence: Sequence[int],
    graph: nx.Graph,
    *,
    name: str = "attack",
) -> list[int]:
    """Verify that ``sequence`` is a permutation of ``graph``'s node set.

    This is the machine-checked form of the module-level contract.  It raises
    instead of using a bare ``assert`` because assertions disappear under
    ``python -O``, and a truncated attack sequence silently shortens the AUC
    integration window — an error that would corrupt a published table rather
    than crash a run.

    Args:
        sequence: The candidate removal order.
        graph: The graph the order is supposed to cover.
        name: Producer name, used in error messages.

    Returns:
        The sequence as a list of plain ``int`` (a convenience for callers that
        want the validated value).

    Raises:
        AdvancedAttackError: If the length differs from ``N``, if any id repeats,
            or if the id set differs from the node set.
    """
    ordered = [int(node) for node in sequence]
    expected = {int(node) for node in graph.nodes()}
    if len(ordered) != len(expected):
        raise AdvancedAttackError(
            f"{name}: 序列长度 {len(ordered)} != 节点数 {len(expected)}"
        )
    if len(set(ordered)) != len(ordered):
        duplicates = sorted({node for node in ordered if ordered.count(node) > 1})
        raise AdvancedAttackError(f"{name}: 序列含重复节点 {duplicates[:10]}")
    if set(ordered) != expected:
        missing = sorted(expected - set(ordered))[:10]
        extra = sorted(set(ordered) - expected)[:10]
        raise AdvancedAttackError(f"{name}: 序列与节点集合不一致（缺 {missing}，多 {extra}）")
    return ordered


# --------------------------------------------------------------------------- #
# 1. adaptive — dynamic maximum residual degree
# --------------------------------------------------------------------------- #
def adaptive_sequence(graph: nx.Graph, *, seed: int = 0) -> list[int]:
    """Remove the maximum-**residual**-degree node at every step.

    The degree is recomputed on the graph *as it stands after* the previous
    removals, which is what distinguishes this attack from the static
    ``degree`` ranking in
    :func:`~bit_hyrl.control.feasibility.provisional_attack_sequence`.  On a
    bimodal (hub + mesh) topology the difference is large: the static ranking
    cannot promote the second hub while the first is still standing, so it
    spends removals on mid-degree mesh nodes that the dynamic version would
    leave for last.

    Ties are broken by ascending node id, so the output depends only on the
    graph.

    Args:
        graph: The graph to order (never mutated).
        seed: Accepted for signature uniformity; the attack is deterministic and
            never draws a number.

    Returns:
        A list of ``N`` distinct node ids — the removal order.

    Raises:
        AdvancedAttackError: If the produced order is not a permutation (a bug
            in this function, never in the input).
    """
    del seed  # deterministic by construction; kept for a uniform signature
    residual = _Residual(graph)
    order: list[int] = []
    while residual.alive:
        node = residual.max_degree()
        order.append(node)
        residual.remove(node)
    return assert_permutation(order, graph, name="adaptive")


# --------------------------------------------------------------------------- #
# 2. correlated — frontier-limited regional chain
# --------------------------------------------------------------------------- #
def correlated_sequence(graph: nx.Graph, *, seed: int = 0) -> list[int]:
    """Attack the frontier first, so removals stay spatially correlated.

    Step 0 is the global maximum-degree node.  Afterwards the candidate set is
    the **attack frontier** — alive nodes adjacent to at least one removed node —
    and the maximum-residual-degree node inside it is removed.  The effect is a
    failure that eats one region and then spills into the neighbouring region,
    rather than one that jumps between the far-away hubs a global ranking would
    pick.

    When the frontier is exhausted (every remaining node is disconnected from
    everything removed so far, which happens once a whole connected component has
    been consumed) the attack restarts from the global maximum-degree node.  That
    fallback is required for the sequence to remain a permutation, and it is also
    the semantically honest choice: a regional outage that has already isolated a
    region has to begin somewhere new.

    Args:
        graph: The graph to order (never mutated).
        seed: Accepted for signature uniformity; the attack is deterministic.

    Returns:
        A list of ``N`` distinct node ids — the removal order.

    Raises:
        AdvancedAttackError: If the produced order is not a permutation.
    """
    del seed  # deterministic by construction; kept for a uniform signature
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


# --------------------------------------------------------------------------- #
# 3. multistage — adaptive -> articulation-first -> random
# --------------------------------------------------------------------------- #
def _largest_component_after_removal(subgraph: nx.Graph, node: int) -> int:
    """Return the size of the largest connected component of ``subgraph - node``.

    Args:
        subgraph: A non-empty graph view.
        node: A node of ``subgraph`` to exclude.

    Returns:
        The surviving largest-component size (``0`` when nothing is left).
    """
    trimmed = subgraph.copy()
    trimmed.remove_node(node)
    if trimmed.number_of_nodes() == 0:
        return 0
    return max(len(component) for component in nx.connected_components(trimmed))


def _articulation_pick(residual: _Residual) -> int:
    """Pick the articulation point whose removal leaves the smallest GCC.

    "Smallest largest-component" is the operational reading of *cut the graph
    apart with the fewest removals*: a node that splits the residual graph into
    two halves is worth removing before a degree-2 leaf.  Ties are broken by
    ascending node id.

    When the residual graph has no articulation point (2-connected, or reduced to
    a single node) the function falls back to the maximum residual degree, which
    is the same rule stage ① uses.

    Args:
        residual: The residual graph state.

    Returns:
        The node to remove next.

    Raises:
        AdvancedAttackError: If there is nothing left to remove.
    """
    if not residual.alive:
        raise AdvancedAttackError("残差图已空，无法继续选择割点")
    subgraph = residual.graph.subgraph(residual.alive)
    articulation_points = sorted(set(nx.articulation_points(subgraph)))
    if not articulation_points:
        return residual.max_degree()
    best_node = articulation_points[0]
    best_size = _largest_component_after_removal(subgraph, best_node)
    for node in articulation_points[1:]:
        size = _largest_component_after_removal(subgraph, node)
        # ``articulation_points`` is iterated in ascending id order, so a strict
        # ``<`` keeps the smallest id among ties without an explicit key tuple.
        if size < best_size:
            best_node, best_size = node, size
    return best_node


def multistage_sequence(graph: nx.Graph, *, seed: int = 0) -> list[int]:
    """Chain three stages over one removal budget.

    The budget is partitioned by integer cuts ``N // 3`` and ``2 * N // 3``, so
    the three stages consume ``⌊N/3⌋``, ``⌊2N/3⌋ − ⌊N/3⌋`` and the remainder —
    together exactly ``N`` nodes, with no overlap and no gap:

    =============== =====================================================
    stage           rule
    =============== =====================================================
    ① ``adaptive``  maximum residual degree until ``N // 3`` nodes are gone
    ② articulation   articulation point minimising the largest remaining
                    component, else maximum residual degree
    ③ ``random``     uniform permutation of whatever is left
    =============== =====================================================

    The intent is to separate *what* the first wave does from *how much* of the
    budget a smart rule is allowed to spend: stage ② gets a slice the adaptive
    rule would have wasted on mesh nodes, and stage ③ is the control that keeps
    the last third of the curve comparable with the ``random`` attack.

    Args:
        graph: The graph to order (never mutated).
        seed: Seed for stage ③ only, via ``numpy.random.default_rng(seed)``.

    Returns:
        A list of ``N`` distinct node ids — the removal order.

    Raises:
        AdvancedAttackError: If the produced order is not a permutation.
    """
    residual = _Residual(graph)
    total = len(residual.nodes)
    cut_adaptive = total // 3
    cut_articulation = (2 * total) // 3

    order: list[int] = []

    # ---- stage 1: adaptive ------------------------------------------------ #
    for _ in range(cut_adaptive):
        node = residual.max_degree()
        order.append(node)
        residual.remove(node)

    # ---- stage 2: articulation-point first -------------------------------- #
    while len(order) < cut_articulation and residual.alive:
        node = _articulation_pick(residual)
        order.append(node)
        residual.remove(node)

    # ---- stage 3: random over the remainder -------------------------------- #
    remaining = sorted(residual.alive)
    if remaining:
        generator = np.random.default_rng(int(seed))
        permutation = generator.permutation(len(remaining))
        order.extend(remaining[int(index)] for index in permutation)

    return assert_permutation(order, graph, name="multistage")


# --------------------------------------------------------------------------- #
# 4. GDM — lazily bridged learned dismantler
# --------------------------------------------------------------------------- #
def _to_canonical_indexing(graph: nx.Graph, nodes: Sequence[int]) -> nx.Graph:
    """Return a copy of ``graph`` whose node ids are ``0 .. N-1`` in sorted order.

    The vendored GDM predictor builds its ``edge_index`` from the raw node labels
    (``network_dismantling/GDM/dataset_providers_nx.py``), and returns scores
    aligned with ``sorted(G.nodes())``.  That is only correct when the labels
    *are* ``0 .. N-1``, so any other labelling silently permutes the edge list.
    Relabelling here is what makes the bridge safe for a graph whose ids are not
    already canonical.

    Args:
        graph: The graph to relabel.
        nodes: Node ids in ascending order (the score order GDM assumes).

    Returns:
        A relabelled copy; a plain copy when the labelling is already canonical.
    """
    canonical = list(range(len(nodes)))
    if list(nodes) == canonical:
        return graph.copy()
    return nx.relabel_nodes(
        graph, {node: index for index, node in enumerate(nodes)}, copy=True
    )


def gdm_sequence(graph: nx.Graph, *, seed: int = 0) -> list[int]:
    """Order nodes by the GDM graph-attention dismantling score, descending.

    This is a **lazy bridge**, not a re-implementation: ``torch``,
    ``torch_geometric``, ``torch_sparse`` and the vendored
    ``network_dismantling.GDM`` tree are imported inside the function, so
    ``import bit_hyrl`` keeps working on a machine without a GNN stack (the
    package-level "zero imports from the legacy tree" promise).  A failure —
    missing wheel, missing checkpoint, shape mismatch — is re-raised as
    :class:`GdmUnavailableError` **with the original traceback attached**.  The
    function never falls back to a surrogate heuristic: a silently substituted
    attack would be reported in Table 7 as "GDM" and that would be a fabricated
    number.

    Semantics.  ``gdm_dismantle`` in the vendored bridge removes the
    highest-scoring node, recomputes the LCC and stops once the LCC is at or
    below ``stop_condition``.  Because the score is **static** (predicted once,
    never re-predicted), the order it produces is the descending-score order
    truncated at the stopping index.  Continuing that same rule to the end of the
    node set therefore yields the removal order GDM implies for a full attack
    budget, which is what :func:`~bit_hyrl.control.cascade.run_cascade` needs;
    it does not change the model, the features or the ranking.

    Args:
        graph: The graph to order (never mutated).
        seed: Accepted for signature uniformity; GDM is deterministic (fixed
            weights, ``model.eval()``, ascending-id tie-break), so it is unused.

    Returns:
        A list of ``N`` distinct node ids — the removal order.

    Raises:
        GdmUnavailableError: If the GNN stack or the checkpoint cannot be used,
            or if the predictor returns a score vector of the wrong length.
        AdvancedAttackError: If the produced order is not a permutation.
    """
    del seed  # GDM is deterministic; kept for a uniform signature
    nodes = tuple(sorted(int(node) for node in graph.nodes()))
    if not nodes:
        raise GdmUnavailableError("GDM 无法在空图上运行")

    weights = GDM_WEIGHTS_PATH
    if not weights.is_file():
        raise GdmUnavailableError(
            f"GDM 权重缺失：{weights}（仓库内应为 network_dismantling/GDM/models_newpg/gdm_nx_best.pth）"
        )

    try:  # pragma: no cover - depends on the optional GNN stack
        import torch  # noqa: F401  (imported for the device probe only)
        from network_dismantling.GDM.predictors_nx import load_model, predict_scores
    except Exception as exc:
        raise GdmUnavailableError(
            "GDM 惰性桥接失败：无法导入 network_dismantling.GDM.predictors_nx"
            "（需要 torch / torch_geometric / torch_sparse / torch_scatter）。"
            f"原始报错：{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        ) from exc

    try:  # pragma: no cover - depends on the optional GNN stack
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, features, _config = load_model(str(weights), device=device)
        scores = predict_scores(
            _to_canonical_indexing(graph, nodes), model, features, device=device
        )
    except Exception as exc:
        raise GdmUnavailableError(
            "GDM 推理失败："
            f"原始报错：{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        ) from exc

    score = np.asarray(scores, dtype=float).reshape(-1)
    if score.shape[0] != len(nodes):
        raise GdmUnavailableError(
            f"GDM 打分长度 {score.shape[0]} != 节点数 {len(nodes)}"
        )

    # ``scores[i]`` belongs to ``sorted(graph.nodes())[i]``; descending score,
    # ties by ascending node id — the same ``(-score, node_id)`` total order the
    # rest of the package uses.
    ranked = sorted(range(len(nodes)), key=lambda index: (-score[index], nodes[index]))
    order = [nodes[index] for index in ranked]
    return assert_permutation(order, graph, name="gdm")


#: ``attack name -> producer``.  The sweep script dispatches through this table so
#: that adding an attack is a one-line change with no ``if/elif`` chain to keep in
#: sync across files.
ACTION_TABLE: dict[str, Callable[..., list[int]]] = {
    "adaptive": adaptive_sequence,
    "correlated": correlated_sequence,
    "multistage": multistage_sequence,
    "gdm": gdm_sequence,
}
