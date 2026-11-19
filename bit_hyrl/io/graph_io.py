"""Reading, normalising and fingerprinting the pinned graphs.

The loader accepts the two GML dialects in ``dataset/testdata`` plus plain edge
lists, and refuses anything whose fingerprint disagrees with ``configs/graphs.json``.
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
    """Raised when a loaded graph disagrees with its pinned ``(N, M)`` contract."""


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
    """Return the index of the ``]`` matching the ``[`` at ``start``."""
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
    raise GraphLoadError("unbalanced GML block: missing ']'")


def _iter_blocks(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(keyword, body)`` for every top-level ``keyword [ ... ]`` block."""
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
    """Split a block body into ``("bare"|"str", value)`` tokens."""
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
    """Parse the ``key value`` attribute pairs of a block body."""
    tokens = _tokenize_body(body)
    attributes: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        kind, key = tokens[index]
        if kind != "bare":  # pragma: no cover - malformed block, ignore gracefully
            index += 1
            continue
        if index + 1 >= len(tokens):
            attributes[key] = True
            break
        value_kind, value = tokens[index + 1]
        attributes[key] = value if value_kind == "str" else _scalar(value)
        index += 2
    return attributes


def read_gml_records(path: str | Path) -> RawGraph:
    """Parse a GML file into raw blocks without any normalisation."""
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - filesystem failure
        raise GraphLoadError(f"cannot read {source_path}: {exc}") from exc

    nodes: list[GMLNode] = []
    edges: list[GMLEdge] = []
    graph_attributes: dict[str, Any] = {}

    for keyword, body in _iter_blocks(text):
        attributes = _parse_attributes(body)
        if keyword == "node":
            identifier = attributes.get("id")
            if not isinstance(identifier, int):
                raise GraphLoadError(f"{source_path}: node block without an integer id ({attributes!r})")
            nodes.append(GMLNode(id=identifier, attributes=attributes))
        else:
            source = attributes.get("source")
            target = attributes.get("target")
            if not isinstance(source, int) or not isinstance(target, int):
                raise GraphLoadError(
                    f"{source_path}: edge block without integer source/target ({attributes!r})"
                )
            edges.append(GMLEdge(source=source, target=target, attributes=attributes))

    if not nodes:
        raise GraphLoadError(f"{source_path}: no node block was parsed")

    return RawGraph(nodes=tuple(nodes), edges=tuple(edges), graph_attributes=graph_attributes)


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
    """Apply steps 1–3 of the normalisation order to raw ``(u, v)`` pairs."""
    seen: set[tuple[int, int]] = set()
    kept: list[tuple[int, int]] = []
    duplicates = 0
    loops = 0
    for pair in pairs:
        try:
            u, v = pair
        except (TypeError, ValueError) as exc:
            raise GraphLoadError(f"an edge must be a (u, v) pair, got {pair!r}") from exc
        if isinstance(u, bool) or isinstance(v, bool) or not isinstance(u, int) or not isinstance(v, int):
            raise GraphLoadError(f"edge endpoints must be integers, got {(u, v)!r}")
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
    """Load a graph and return it together with its provenance metadata."""
    source_path = Path(path)
    if not source_path.is_file():
        raise GraphLoadError(f"graph file not found: {source_path}")
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
            raise GraphLoadError(f"cannot parse {source_path}: {exc}") from exc
        node_ids = list(loaded.nodes())
        pairs = list(loaded.edges())
        raw_edge_records = len(pairs)
    elif suffix in (".edgelist", ".edges", ".txt"):
        try:
            handle = source_path.open("r", encoding="utf-8")
        except OSError as exc:  # pragma: no cover
            raise GraphLoadError(f"cannot read {source_path}: {exc}") from exc
        with handle:
            node_ids = []
            pairs = []
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise GraphLoadError(f"{source_path}: an edge line needs two endpoints, got {line!r}")
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError as exc:
                    raise GraphLoadError(
                        f"{source_path}: edge endpoints must be integers, got {line!r}"
                    ) from exc
                node_ids.extend((u, v))
                pairs.append((u, v))
        raw_edge_records = len(pairs)
    else:
        raise GraphLoadError(f"unsupported graph format {suffix!r} (expected .gml, .graphml or .edgelist)")

    edges, duplicates, loops = normalize_edges(
        pairs, dedupe=dedupe, drop_self_loops=drop_self_loops
    )

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
    """Load a graph with the tolerant normalisation pipeline."""
    graph, _ = load_graph_with_meta(
        path, dedupe=dedupe, drop_self_loops=drop_self_loops, relabel=relabel
    )
    return graph


def edge_list_sha256(graph: nx.Graph) -> str:
    """SHA-256 over the sorted canonical edge list of ``graph``."""
    canonical = sorted(
        (u, v) if u <= v else (v, u) for u, v in graph.edges()
    )
    payload = f"n={graph.number_of_nodes()}\n" + "\n".join(f"{u}-{v}" for u, v in canonical)
    return sha256_text(payload)


def graph_fingerprint(graph: nx.Graph, *, path: str | Path | None = None,
                      duplicates_removed: int | None = None,
                      raw_edge_records: int | None = None,
                      self_loops_removed: int | None = None) -> dict[str, Any]:
    """Build the fingerprint record of ``graph``."""
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
    """Load a graph from a ``configs/graphs.json``-style spec and verify it."""
    if "path" not in spec:
        raise GraphLoadError(f"graph spec without 'path': {spec!r}")
    graph = load_graph(spec["path"])
    fingerprint = graph_fingerprint(graph, path=spec["path"])
    name = spec.get("dataset") or spec.get("name") or Path(str(spec["path"])).stem

    mismatches: list[str] = []
    for key in ("n", "m", "duplicates_removed", "raw_edge_records"):
        if key in spec and spec[key] is not None and int(spec[key]) != int(fingerprint[key]):
            mismatches.append(
                f"{key}: expected {spec[key]}, measured {fingerprint[key]} (diff {int(fingerprint[key]) - int(spec[key])})"
            )
    if "sha256" in spec and spec["sha256"] and str(spec["sha256"]) != fingerprint["sha256"]:
        mismatches.append(
            f"sha256: expected {spec['sha256']}, measured {fingerprint['sha256']}"
        )
    if mismatches:
        raise GraphSpecMismatch(
            f"{name} does not match the pinned fingerprint: " + "; ".join(mismatches)
        )
    return graph, fingerprint
