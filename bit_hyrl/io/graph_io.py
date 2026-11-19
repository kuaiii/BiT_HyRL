"""Tolerant graph reading, normalisation and fingerprinting.

Why this module exists
----------------------
``dataset/testdata/Cogentco.gml`` and ``Colt.gml`` contain **duplicate edge
records**.  ``networkx.read_gml`` (>= 3.x) validates that and dies with
``NetworkXError: edge #70 (42--143) is duplicated``, so the whole pipeline
cannot even load two of the nine evaluation graphs.  Worse, a loader that
silently lets ``nx.Graph`` deduplicate would hide *how many* records were
dropped, and that count is needed to reconcile the adjacency matrices against
Table 1 of the paper.

This module therefore parses the GML blocks itself and normalises in a fixed,
documented order.  The normalisation order is **not** an implementation detail —
the golden ``(N, M)`` pairs depend on it, so it is frozen here and guarded by a
synthetic self-loop + parallel-edge fixture in ``tests/test_graph_io.py``:

1. directed → undirected (fold each arc into an unordered pair);
2. remove parallel edges, keeping the first occurrence;
3. remove self loops;
4. drop unused labels and relabel ``0 .. n-1`` in ascending original-label order;
5. only then compute ``N`` and ``M``.

Step 2 is what turns Cogentco's 245 edge records into ``M = 243`` and Colt's
191 into ``M = 177``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import networkx as nx

from .jsonlog import sha256_text

__all__ = [
    "GraphLoadError",
    "GraphSpecMismatch",
    "GMLNode",
    "GMLEdge",
    "RawGraph",
    "read_gml_records",
    "normalize_edges",
    "load_graph_with_meta",
    "load_graph",
    "graph_fingerprint",
    "load_graph_spec",
    "edge_list_sha256",
]

_SCALAR_INT = re.compile(r"^[+-]?\d+$")
_SCALAR_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")

#: Key under which the loader stashes its provenance inside ``G.graph``.
META_KEY = "_bit_hyrl_meta"


class GraphLoadError(RuntimeError):
    """Raised when a graph file cannot be parsed into a simple undirected graph."""


class GraphSpecMismatch(AssertionError):
    """Raised when a loaded graph disagrees with its pinned ``(N, M)`` contract.

    This is deliberately *not* swallowed: the numeric contract of the paper is
    pinned to these values, so a mismatch must stop the run and print the diff
    rather than let a wrong ``M`` propagate into every downstream table.
    """


# --------------------------------------------------------------------------- #
# GML scanning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GMLNode:
    """One parsed ``node [ ... ]`` block."""

    id: int
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GMLEdge:
    """One parsed ``edge [ ... ]`` block."""

    source: int
    target: int
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawGraph:
    """Result of a raw parse, before any normalisation."""

    nodes: tuple[GMLNode, ...]
    edges: tuple[GMLEdge, ...]
    graph_attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def raw_edge_records(self) -> int:
        """Number of edge blocks as they appear in the file."""
        return len(self.edges)


def _match_bracket(text: str, start: int) -> int:
    """Return the index of the ``]`` matching the ``[`` at ``start``.

    Quoted strings are skipped, and a backslash escapes the next character, so a
    label such as ``"]"`` cannot unbalance the scan.

    Raises:
        GraphLoadError: If the bracket is never closed.
    """
    depth = 0
    index = start
    length = len(text)
    while index < length:
        char = text[index]
        if char == '"':
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == '"':
                    break
                index += 1
            index += 1
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise GraphLoadError("GML 括号不配对：缺少 ']'")


def _iter_blocks(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(keyword, body)`` for every top-level ``keyword [ ... ]`` block.

    ``graph [ ... ]`` blocks are entered recursively, so the ``node`` / ``edge``
    blocks nested inside them are yielded too.
    """
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == '"':
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == '"':
                    break
                index += 1
            index += 1
            continue
        if char.isalpha() or char == "_":
            end = index
            while end < length and (text[end].isalnum() or text[end] == "_"):
                end += 1
            keyword = text[index:end]
            probe = end
            while probe < length and text[probe] in " \t\r\n":
                probe += 1
            if probe < length and text[probe] == "[":
                closing = _match_bracket(text, probe)
                body = text[probe + 1: closing]
                if keyword == "graph":
                    yield from _iter_blocks(body)
                elif keyword in ("node", "edge"):
                    yield keyword, body
                index = closing + 1
                continue
            index = end
            continue
        index += 1


def _scalar(token: str) -> Any:
    """Convert a bare GML token into ``int`` / ``float`` / ``str``."""
    token = token.strip()
    if _SCALAR_INT.match(token):
        return int(token)
    if _SCALAR_FLOAT.match(token):
        return float(token)
    lowered = token.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return token


def _unescape(raw: str) -> str:
    """Undo the minimal GML string escaping (``\\"`` and ``\\\\``)."""
    return raw.replace('\\"', '"').replace("\\\\", "\\")


def _tokenize_body(body: str) -> list[tuple[str, str]]:
    """Split a block body into ``("bare"|"str", value)`` tokens.

    Nested ``[ ... ]`` sub-blocks are skipped wholesale (they are consumed but
    produce no tokens), so a third-party file that nests attributes cannot be
    mistaken for a flat key/value list.
    """
    tokens: list[tuple[str, str]] = []
    index = 0
    length = len(body)
    while index < length:
        char = body[index]
        if char in " \t\r\n":
            index += 1
            continue
        if char == "[":
            index = _match_bracket(body, index) + 1
            continue
        if char == "]":
            index += 1
            continue
        if char == '"':
            cursor = index + 1
            while cursor < length:
                if body[cursor] == "\\":
                    cursor += 2
                    continue
                if body[cursor] == '"':
                    break
                cursor += 1
            tokens.append(("str", _unescape(body[index + 1: cursor])))
            index = cursor + 1
            continue
        cursor = index
        while cursor < length and body[cursor] not in ' \t\r\n[]"':
            cursor += 1
        tokens.append(("bare", body[index:cursor]))
        index = cursor
    return tokens


def _parse_attributes(body: str) -> dict[str, Any]:
    """Parse the ``key value`` attribute pairs of a block body.

    Works on a token stream rather than on physical lines, so a one-line block
    (``node [ id 0 label "a" ]``) parses exactly like a multi-line one, and a
    quoted value containing spaces or brackets cannot be split.
    """
    tokens = _tokenize_body(body)
    attributes: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        kind, key = tokens[index]
        if kind != "bare":  # pragma: no cover - malformed block, ignore gracefully
            index += 1
            continue
        if index + 1 >= len(tokens):
            attributes[key] = True  # bare flag, e.g. ``Label``
            break
        value_kind, value = tokens[index + 1]
        attributes[key] = value if value_kind == "str" else _scalar(value)
        index += 2
    return attributes


def read_gml_records(path: str | Path) -> RawGraph:
    """Parse a GML file into raw blocks without any normalisation.

    Args:
        path: Path to the ``.gml`` file.

    Returns:
        A :class:`RawGraph` holding every node and edge record verbatim,
        including duplicates and self loops.

    Raises:
        GraphLoadError: If the file cannot be read, has no node block, or an
            edge block lacks ``source`` / ``target``.
    """
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - filesystem failure
        raise GraphLoadError(f"无法读取 {source_path}: {exc}") from exc

    nodes: list[GMLNode] = []
    edges: list[GMLEdge] = []
    # Graph-level attributes are not consumed by any code path (only node ids
    # and edges matter, and the fingerprints are taken over the normalised edge
    # list), so the field is kept for API completeness and left empty.
    graph_attributes: dict[str, Any] = {}

    for keyword, body in _iter_blocks(text):
        attributes = _parse_attributes(body)
        if keyword == "node":
            identifier = attributes.get("id")
            if not isinstance(identifier, int):
                raise GraphLoadError(f"{source_path}: node 块缺少整数 id（{attributes!r}）")
            nodes.append(GMLNode(id=identifier, attributes=attributes))
        else:
            source = attributes.get("source")
            target = attributes.get("target")
            if not isinstance(source, int) or not isinstance(target, int):
                raise GraphLoadError(
                    f"{source_path}: edge 块缺少整数 source/target（{attributes!r}）"
                )
            edges.append(GMLEdge(source=source, target=target, attributes=attributes))

    if not nodes:
        raise GraphLoadError(f"{source_path}: 未解析到任何 node 块")

    return RawGraph(nodes=tuple(nodes), edges=tuple(edges), graph_attributes=graph_attributes)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NormalizationReport:
    """Counts produced by the normalisation pipeline (for reconciliation)."""

    raw_edge_records: int
    duplicates_removed: int
    self_loops_removed: int
    n: int
    m: int

    def to_dict(self) -> dict[str, int]:
        """Plain-``dict`` view."""
        return {
            "raw_edge_records": self.raw_edge_records,
            "duplicates_removed": self.duplicates_removed,
            "self_loops_removed": self.self_loops_removed,
            "n": self.n,
            "m": self.m,
        }


def normalize_edges(pairs: Any, *, dedupe: bool = True,
                    drop_self_loops: bool = True) -> tuple[list[tuple[int, int]], int, int]:
    """Apply steps 1–3 of the normalisation order to raw ``(u, v)`` pairs.

    Order matters and is fixed: fold to undirected → drop parallel edges
    (keeping the first occurrence) → drop self loops.

    Args:
        pairs: Iterable of ``(u, v)`` integer pairs, possibly directed, possibly
            containing duplicates and self loops.
        dedupe: Whether step 2 runs (``True`` is the contract).
        drop_self_loops: Whether step 3 runs.

    Returns:
        ``(edges, duplicates_removed, self_loops_removed)`` where ``edges`` is a
        list of canonical ``(min, max)`` pairs in first-seen order.

    Raises:
        GraphLoadError: If a pair is not a 2-tuple of integers.
    """
    seen: set[tuple[int, int]] = set()
    kept: list[tuple[int, int]] = []
    duplicates = 0
    loops = 0
    for pair in pairs:
        try:
            u, v = pair
        except (TypeError, ValueError) as exc:
            raise GraphLoadError(f"边记录必须是 (u, v) 二元组，收到 {pair!r}") from exc
        if isinstance(u, bool) or isinstance(v, bool) or not isinstance(u, int) or not isinstance(v, int):
            raise GraphLoadError(f"边端点必须是整数，收到 {(u, v)!r}")
        key = (u, v) if u <= v else (v, u)
        if dedupe:
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
        if drop_self_loops and key[0] == key[1]:
            loops += 1
            if dedupe:
                seen.discard(key)
            continue
        kept.append(key)
    return kept, duplicates, loops


def load_graph_with_meta(path: str | Path, *, dedupe: bool = True,
                         drop_self_loops: bool = True,
                         relabel: bool = True) -> tuple[nx.Graph, dict[str, Any]]:
    """Load a graph and return it together with its provenance metadata.

    Args:
        path: Input file. ``.gml`` uses the tolerant reader; ``.graphml`` and
            ``.edgelist`` / ``.txt`` fall back to networkx readers, then go
            through the same normalisation.
        dedupe: Drop parallel edges (must stay ``True`` for the pinned values).
        drop_self_loops: Drop self loops.
        relabel: Relabel to ``0 .. n-1`` in ascending original-label order.

    Returns:
        ``(graph, meta)``. ``meta`` carries ``raw_edge_records``,
        ``duplicates_removed``, ``self_loops_removed``, ``n``, ``m`` and the
        source ``path``.

    Raises:
        GraphLoadError: On unreadable files or malformed records.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise GraphLoadError(f"图文件不存在: {source_path}")
    suffix = source_path.suffix.lower()

    if suffix == ".gml":
        raw = read_gml_records(source_path)
        node_ids = [node.id for node in raw.nodes]
        pairs = [(edge.source, edge.target) for edge in raw.edges]
        raw_edge_records = raw.raw_edge_records
    elif suffix == ".graphml":
        try:
            loaded = nx.read_graphml(source_path)
        except Exception as exc:
            raise GraphLoadError(f"无法解析 {source_path}: {exc}") from exc
        node_ids = list(loaded.nodes())
        pairs = list(loaded.edges())
        raw_edge_records = len(pairs)
    elif suffix in (".edgelist", ".edges", ".txt"):
        try:
            handle = source_path.open("r", encoding="utf-8")
        except OSError as exc:  # pragma: no cover
            raise GraphLoadError(f"无法读取 {source_path}: {exc}") from exc
        with handle:
            node_ids = []
            pairs = []
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise GraphLoadError(f"{source_path}: 边行至少需要两个端点，收到 {line!r}")
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError as exc:
                    raise GraphLoadError(
                        f"{source_path}: 边端点必须是整数，收到 {line!r}"
                    ) from exc
                node_ids.extend((u, v))
                pairs.append((u, v))
        raw_edge_records = len(pairs)
    else:
        raise GraphLoadError(f"不支持的图格式 {suffix!r}（支持 .gml/.graphml/.edgelist）")

    edges, duplicates, loops = normalize_edges(
        pairs, dedupe=dedupe, drop_self_loops=drop_self_loops
    )

    # Step 4: stable relabelling in ascending original-label order.
    all_ids = set(int(node) for node in node_ids)
    for u, v in edges:
        all_ids.add(u)
        all_ids.add(v)
    ordered = sorted(all_ids) if relabel else list(all_ids)
    mapping = {old: new for new, old in enumerate(ordered)}

    graph = nx.Graph()
    if relabel:
        graph.add_nodes_from(range(len(ordered)))
        graph.add_edges_from((mapping[u], mapping[v]) for u, v in edges)
    else:
        graph.add_nodes_from(ordered)
        graph.add_edges_from(edges)

    meta = {
        "path": source_path.as_posix(),
        "name": source_path.stem,
        "raw_edge_records": raw_edge_records,
        "duplicates_removed": duplicates,
        "self_loops_removed": loops,
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "relabeled": relabel,
        "deduped": dedupe,
    }
    graph.graph[META_KEY] = meta
    return graph, meta


def load_graph(path: str | Path, *, dedupe: bool = True, drop_self_loops: bool = True,
               relabel: bool = True) -> nx.Graph:
    """Load a graph with the tolerant normalisation pipeline.

    See :func:`load_graph_with_meta` for the semantics of the flags.
    """
    graph, _ = load_graph_with_meta(
        path, dedupe=dedupe, drop_self_loops=drop_self_loops, relabel=relabel
    )
    return graph


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def edge_list_sha256(graph: nx.Graph) -> str:
    """SHA-256 over the sorted canonical edge list of ``graph``.

    ``N`` is part of the payload so that isolated nodes are visible in the hash;
    everything else about the graph (labels, attributes) is deliberately not,
    which makes the fingerprint invariant under relabelling and idempotent under
    re-normalisation.
    """
    canonical = sorted(
        (u, v) if u <= v else (v, u) for u, v in graph.edges()
    )
    payload = f"n={graph.number_of_nodes()}\n" + "\n".join(f"{u}-{v}" for u, v in canonical)
    return sha256_text(payload)


def graph_fingerprint(graph: nx.Graph, *, path: str | Path | None = None,
                      duplicates_removed: int | None = None,
                      raw_edge_records: int | None = None,
                      self_loops_removed: int | None = None) -> dict[str, Any]:
    """Build the fingerprint record of ``graph``.

    The fingerprint is stable and idempotent: loading the same file twice, or
    fingerprinting the same in-memory graph twice, yields the same ``sha256``.

    Args:
        graph: Simple undirected graph.
        path: Optional source path; defaults to the metadata stashed by the
            loader.
        duplicates_removed: Override for the loader-provided count.
        raw_edge_records: Override for the loader-provided count.
        self_loops_removed: Override for the loader-provided count.

    Returns:
        ``{"n", "m", "sha256", "path", "duplicates_removed",
        "raw_edge_records", "self_loops_removed"}``.
    """
    meta = graph.graph.get(META_KEY, {}) if hasattr(graph, "graph") else {}
    resolved_path = path if path is not None else meta.get("path")
    return {
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "sha256": edge_list_sha256(graph),
        "path": Path(resolved_path).as_posix() if resolved_path is not None else None,
        "duplicates_removed": (
            duplicates_removed if duplicates_removed is not None
            else int(meta.get("duplicates_removed", 0))
        ),
        "raw_edge_records": (
            raw_edge_records if raw_edge_records is not None
            else int(meta.get("raw_edge_records", graph.number_of_edges()))
        ),
        "self_loops_removed": (
            self_loops_removed if self_loops_removed is not None
            else int(meta.get("self_loops_removed", 0))
        ),
    }


def load_graph_spec(spec: Mapping[str, Any]) -> tuple[nx.Graph, dict[str, Any]]:
    """Load a graph from a ``configs/graphs.json``-style spec and verify it.

    Args:
        spec: Mapping with at least ``path``; optionally ``dataset``, ``n``,
            ``m``, ``duplicates_removed`` and ``sha256`` as pinned expectations.

    Returns:
        ``(graph, fingerprint)``.

    Raises:
        GraphSpecMismatch: If any pinned expectation is violated. The message
            spells out the expected and observed values so the diff is visible
            without a debugger.
        GraphLoadError: If the file cannot be parsed.
    """
    if "path" not in spec:
        raise GraphLoadError(f"graph spec 缺少 'path': {spec!r}")
    graph = load_graph(spec["path"])
    fingerprint = graph_fingerprint(graph, path=spec["path"])
    name = spec.get("dataset") or spec.get("name") or Path(str(spec["path"])).stem

    mismatches: list[str] = []
    for key in ("n", "m", "duplicates_removed", "raw_edge_records"):
        if key in spec and spec[key] is not None and int(spec[key]) != int(fingerprint[key]):
            mismatches.append(
                f"{key}: 期望 {spec[key]}，实测 {fingerprint[key]}（差 {int(fingerprint[key]) - int(spec[key])}）"
            )
    if "sha256" in spec and spec["sha256"] and str(spec["sha256"]) != fingerprint["sha256"]:
        mismatches.append(
            f"sha256: 期望 {spec['sha256']}，实测 {fingerprint['sha256']}"
        )
    if mismatches:
        raise GraphSpecMismatch(
            f"{name} 与金标准不符：" + "；".join(mismatches)
        )
    return graph, fingerprint
