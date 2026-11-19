"""Substrate-construction benchmark on all thirty graphs, several seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bit_hyrl.analysis.methods import build_method_graph   # noqa: E402
from bit_hyrl.io.graph_io import load_graph_spec            # noqa: E402
from bit_hyrl.topology.interval import bimodal_interval as bimodal_c  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "family" / "shards"

#: row order of the manuscript's construction table
METHODS = ("Original", "Random", "BA", "WS",
           "BiT-R", "BiT-HyRL", "BiT-E",
           "UNITY", "FRED-ABL", "QDLM", "Eff-R-Net", "ONION", "ROMEN")

CONSTRUCT = {
    "Original": None,
    "Random": "random_sameNM",
    "BA": "ba_sameNM",
    "WS": "ws_sameNM",
    "BiT-R": "bit",
    "BiT-HyRL": "bit",
    "BiT-E": "bit",
    "UNITY": "UNITY",
    "FRED-ABL": "FRED-ABL",
    "QDLM": "QDLM",
    "Eff-R-Net": "Eff-R-Net",
    "ONION": "ONION",
    "ROMEN": "ROMEN",
}

POINTS = 41
FIELDS = ("dataset", "family", "method", "seed", "c", "n", "m", "kbar", "k_max", "k_min",
          "kappa", "E_glob", "L", "D", "E0",
          "R_gcc_tar", "R_gcc_ran", "R_e_tar", "R_e_ran", "A",
          "construct_s", "status")


def family(name: str) -> str:
    prefix = name.split("-")[0]
    return prefix if prefix in ("BA", "ER", "WS", "AS") else "ISP"


def _adjust_edges(graph: nx.Graph, target_m: int, seed: int) -> nx.Graph:
    """Add or drop edges until the graph has exactly ``target_m``, staying connected."""
    rng = np.random.default_rng(seed)
    graph = graph.copy()
    nodes = list(graph.nodes())
    budget = target_m - graph.number_of_edges()
    if budget > 0:
        missing = [(u, v) for index, u in enumerate(nodes) for v in nodes[index + 1:]
                   if not graph.has_edge(u, v)]
        for position in rng.permutation(len(missing))[:budget]:
            graph.add_edge(*missing[position])
    elif budget < 0:
        bridges = set(map(frozenset, nx.bridges(graph)))
        candidates = [edge for edge in graph.edges() if frozenset(edge) not in bridges]
        for position in rng.permutation(len(candidates))[: -budget]:
            graph.remove_edge(*candidates[position])
    return graph


def _ba_same_nm(n: int, m_edge: int, seed: int) -> nx.Graph:
    attach = max(1, int(round(m_edge / max(n - 1, 1))))
    return _adjust_edges(nx.barabasi_albert_graph(n, attach, seed=seed), m_edge, seed)


def _ws_same_nm(n: int, m_edge: int, seed: int) -> nx.Graph:
    degree = max(2, 2 * int(round(m_edge / max(n, 1) / 2)))
    for _ in range(20):
        try:
            graph = nx.connected_watts_strogatz_graph(n, min(degree, n - 1), 0.1, seed=seed)
            return _adjust_edges(graph, m_edge, seed)
        except Exception:                                   # noqa: BLE001
            degree = max(2, degree - 2)
    return _adjust_edges(nx.path_graph(n), m_edge, seed)


def _random_same_nm(n: int, m_edge: int, seed: int) -> nx.Graph:
    from generate_random_topology import random_connected_graph
    return random_connected_graph(n, m_edge, seed=seed)


def efficiency(graph: nx.Graph) -> float:
    """Global efficiency ``1/[N(N-1)] sum 1/d_ij`` (unreachable pairs contribute 0)."""
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    nodes = np.fromiter(graph.nodes(), dtype=np.int64, count=n)
    index = {int(node): position for position, node in enumerate(nodes)}
    edges = np.array([(index[int(u)], index[int(v)]) for u, v in graph.edges()],
                     dtype=np.int64).reshape(-1, 2)
    ones = np.ones(len(edges))
    adjacency = csr_matrix((ones, (edges[:, 0], edges[:, 1])), shape=(n, n))
    distance = shortest_path(adjacency, method="D", unweighted=True, directed=False)
    finite = np.isfinite(distance) & (distance > 0)
    return float(np.where(finite, 1.0 / np.where(finite, distance, 1.0), 0.0).sum()
                 / (n * (n - 1.0)))


def gcc(graph: nx.Graph, alive: list[int]) -> float:
    n = graph.number_of_nodes()
    if not alive:
        return 0.0
    sub = graph.subgraph(alive)
    return max((len(c) for c in nx.connected_components(sub)), default=0) / n


def curves(graph: nx.Graph, order: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """GCC/N and E_glob on the common 41-point fraction grid."""
    n = graph.number_of_nodes()
    grid = np.linspace(0.0, 1.0, POINTS)
    gcc_values = np.zeros(POINTS)
    eff_values = np.zeros(POINTS)
    for index, fraction in enumerate(grid):
        alive = list(order[int(round(fraction * n)):])
        if not alive:
            continue
        gcc_values[index] = gcc(graph, alive)
        eff_values[index] = efficiency(graph.subgraph(alive)) if len(alive) > 1 else 0.0
    return gcc_values, eff_values


def _auc(values: np.ndarray, grid: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values, grid))


def degree_order(graph: nx.Graph) -> list[int]:
    """Degree-descending order, ties broken by node id (deterministic)."""
    return [int(node) for node in sorted(graph.nodes(),
                                         key=lambda v: (-graph.degree(v), int(v)))]


def random_order(graph: nx.Graph, seed: int) -> list[int]:
    rng = np.random.default_rng(10_000 + seed)
    nodes = np.array(sorted(int(v) for v in graph.nodes()))
    return [int(v) for v in rng.permutation(nodes)]


def run_method(spec: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    dataset = spec["dataset"]
    original, _meta = load_graph_spec(spec)
    n, m = original.number_of_nodes(), original.number_of_edges()
    row: dict[str, Any] = {
        "dataset": dataset, "family": family(dataset), "method": method, "seed": seed,
        "c": "", "n": n, "m": m, "kbar": round(2.0 * m / n, 3),
        "status": "ok",
    }
    try:
        e0 = efficiency(original)
        started = time.perf_counter()
        if method == "Original":
            graph = original
        elif method == "Random":
            graph = _random_same_nm(n, m, seed)
        elif method == "BA":
            graph = _ba_same_nm(n, m, seed)
        elif method == "WS":
            graph = _ws_same_nm(n, m, seed)
        elif method.startswith("BiT-"):
            bounds = bimodal_c(n, m, e0)
            c = {"BiT-R": bounds["c_min"], "BiT-HyRL": bounds["c_mid"],
                 "BiT-E": bounds["c_up"]}[method]
            row["c"] = round(float(c), 3)
            graph = build_method_graph(original, "bit", seed=seed, c=row["c"], h=1)
        else:
            graph = build_method_graph(original, CONSTRUCT[method], seed=seed)
        row["construct_s"] = round(time.perf_counter() - started, 4)

        degrees = np.array([d for _, d in graph.degree()], dtype=float)
        row.update({
            "n": graph.number_of_nodes(), "m": graph.number_of_edges(),
            "kbar": round(2.0 * graph.number_of_edges() / graph.number_of_nodes(), 3),
            "k_max": int(degrees.max()), "k_min": int(degrees.min()),
            "kappa": round(float((degrees ** 2).mean() / degrees.mean()), 3),
            "E0": round(e0, 4),
        })

        grid = np.linspace(0.0, 1.0, POINTS)
        gcc_tar, eff_tar = curves(graph, degree_order(graph))
        gcc_ran, eff_ran = curves(graph, random_order(graph, seed))
        areas = {
            "R_gcc_tar": _auc(gcc_tar, grid), "R_gcc_ran": _auc(gcc_ran, grid),
            "R_e_tar": _auc(eff_tar, grid), "R_e_ran": _auc(eff_ran, grid),
        }
        row.update({key: round(value, 4) for key, value in areas.items()})
        row["A"] = round(sum(areas.values()) / 4.0, 4)
        row["E_glob"] = round(efficiency(graph), 4)
        try:
            row["L"] = round(nx.average_shortest_path_length(graph), 4)
            row["D"] = int(nx.diameter(graph))
        except Exception:                                   # noqa: BLE001
            row["L"], row["D"] = "", ""
    except Exception as error:                              # noqa: BLE001
        row["status"] = f"{type(error).__name__}: {error}"[:180]
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="family bench (seeded, thirty graphs)")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--label", default="", help="suffix tag: <dataset>__s<seed>__<label>.csv")
    parser.add_argument("--force", action="store_true", help="re-run shards that already exist")
    args = parser.parse_args(argv)

    specs = json.loads((REPO_ROOT / "configs" / "graphs.json")
                       .read_text(encoding="utf-8"))["graphs"]
    if args.datasets.strip().lower() != "all":
        wanted = {name.strip() for name in args.datasets.split(",") if name.strip()}
        specs = [spec for spec in specs if spec["dataset"] in wanted]
    methods = list(METHODS) if args.methods.strip() == "all" else [
        name.strip() for name in args.methods.split(",") if name.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        for seed in seeds:
            tag = f"__{args.label}" if args.label else ""
            path = out_dir / f"{spec['dataset']}__s{seed}{tag}.csv"
            done: set[str] = set()
            if path.exists() and not args.force:
                with path.open(encoding="utf-8") as handle:
                    done = {row["method"] for row in csv.DictReader(handle)}
            todo = [name for name in methods if name not in done]
            if not todo:
                print(f"[skip] {path.name}", flush=True)
                continue
            print(f"[run ] {path.name}  methods={len(todo)}", flush=True)
            rows: list[dict[str, Any]] = []
            if args.workers > 1 and len(todo) > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    rows = list(pool.map(run_method, [spec] * len(todo), todo, [seed] * len(todo)))
            else:
                for name in todo:
                    rows.append(run_method(spec, name, seed))
                    print(f"       {name:<10} A={rows[-1].get('A', '--')} "
                          f"t={rows[-1].get('construct_s', '--')}", flush=True)

            existing: list[dict[str, Any]] = []
            if path.exists() and not args.force:
                with path.open(encoding="utf-8") as handle:
                    existing = [row for row in csv.DictReader(handle)
                                if row["method"] not in {r["method"] for r in rows}]
            merged = existing + [{key: row.get(key, "") for key in FIELDS} for row in rows]
            merged.sort(key=lambda row: METHODS.index(row["method"])
                        if row["method"] in METHODS else 99)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
                writer.writeheader()
                writer.writerows(merged)
            failures = [row for row in rows if row["status"] != "ok"]
            print(f"[done] {path.name}  {len(rows) - len(failures)}/{len(rows)} ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
