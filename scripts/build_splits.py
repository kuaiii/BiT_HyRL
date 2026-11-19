"""Build the train/val/test splits from ``dataset/all`` minus the 30 eval graphs.

Output: ``configs/splits.json`` with mutually-exclusive ``test`` / ``val`` /
``train`` path lists (no path and no ``edge_list_sha256`` overlap across the three
sets) plus a ``meta`` block (per-set counts, size histogram, real/syn ratio,
``leakage_excluded``).

Rules
-----
* TEST  = the 30 ``path`` entries of ``configs/graphs.json`` (unchanged).
* POOL  = ``dataset/all/syn/*.gml`` + ``dataset/all/real/*.gml``.
* Leakage: any pool graph whose ``edge_list_sha256`` equals a test graph's hash (or
  whose path is a test path) is **excluded** and counted.
* Pool duplicates by sha are de-duplicated (first wins) so the three sets are
  sha-disjoint by construction.
* VAL   = stratified ``~10%`` per ``(source, size_bucket)`` stratum, total capped
  at ``--val-cap``.
* TRAIN = all remaining ``real`` graphs + stratified ``syn`` fill up to
  ``--train-cap``.

Run from ``codes/v3``::

    /d/ProgramData/Anaconda/python.exe scripts/build_splits.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bit_hyrl import rng as _rng  # noqa: E402
from bit_hyrl.io.graph_io import edge_list_sha256, load_graph  # noqa: E402

#: ``(lo, hi)`` size buckets on the actual node count ``N``.
SIZE_BUCKETS: list[tuple[int, int]] = [
    (20, 49), (50, 99), (100, 199), (200, 299),
    (300, 499), (500, 999), (1000, 10 ** 9),
]


def bucket_of(n: int) -> str:
    """Return the size-bucket label of a node count ``n``."""
    for lo, hi in SIZE_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
    return f"<{SIZE_BUCKETS[0][0]}"


def _load_candidate(item: tuple[str, str]) -> tuple[str, str, int | None, str | None]:
    """Worker: ``(rel, source)`` -> ``(rel, source, n|None, sha|None)``; None on error."""
    rel, source = item
    try:
        graph = load_graph(PROJECT_ROOT / rel)
    except Exception:
        return (rel, source, None, None)
    return (rel, source, graph.number_of_nodes(), edge_list_sha256(graph))


def _load_all(candidates: list[tuple[str, str]], workers: int) -> list:
    """Load+hash every candidate in parallel (process pool), serial fallback."""
    try:
        from concurrent.futures import ProcessPoolExecutor

        count = max(1, min(workers, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=count) as executor:
            return list(executor.map(_load_candidate, candidates, chunksize=64))
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        print(f"[warn] 并行加载不可用（{exc}），回退串行（较慢）", flush=True)
        return [_load_candidate(item) for item in candidates]


def _sample_stratified(strata: dict, cap: int, gen) -> list:
    """Proportional stratified sample down to ``cap`` (seeded by ``gen``)."""
    total = sum(len(v) for v in strata.values())
    if total <= cap:
        out: list = []
        for key in sorted(strata):
            out.extend(strata[key])
        return out
    out: list = []
    for key in sorted(strata):
        items = strata[key]
        share = max(1, round(cap * len(items) / total))
        perm = gen.permutation(len(items))
        out.extend(items[int(i)] for i in perm[: min(share, len(items))])
    if len(out) > cap:
        perm = gen.permutation(len(out))
        out = [out[int(i)] for i in perm[:cap]]
    return out


def _size_hist(nodes: list[int]) -> dict:
    """``{bucket: count}`` over a list of node counts."""
    hist: dict = defaultdict(int)
    for n in nodes:
        hist[bucket_of(n)] += 1
    return {k: hist[k] for k in sorted(hist, key=lambda b: (len(b), b))}


def build_splits(train_cap: int, val_cap: int, seed: int,
                 graphs_path: Path, out_path: Path) -> dict:
    """Compute the splits dict and write it to ``out_path``."""
    registry = json.loads(graphs_path.read_text(encoding="utf-8"))
    test_paths = [entry["path"] for entry in registry["graphs"]]

    # Test hashes + node counts.
    test_shas: set[str] = set()
    test_nodes: list[int] = []
    test_path_set = set(test_paths)
    for rel in test_paths:
        graph = load_graph(PROJECT_ROOT / rel)
        test_shas.add(edge_list_sha256(graph))
        test_nodes.append(graph.number_of_nodes())

    # Candidate pool.
    candidates: list[tuple[str, str]] = []
    for source, sub in (("syn", "dataset/all/syn"), ("real", "dataset/all/real")):
        for path in sorted((PROJECT_ROOT / sub).glob("*.gml")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            candidates.append((rel, source))

    loaded = _load_all(candidates, workers=8)
    leakage_excluded = 0
    unreadable = 0
    sha_duplicates = 0
    seen_shas: set[str] = set()
    rows: list[tuple[str, str, int, str]] = []  # (rel, source, n, bucket)
    for rel, source, node_count, sha in loaded:
        if rel in test_path_set:
            leakage_excluded += 1
            continue
        if node_count is None or sha is None:
            unreadable += 1
            continue
        if sha in test_shas:
            leakage_excluded += 1
            continue
        if sha in seen_shas:
            sha_duplicates += 1  # keep the pool sha-unique for cross-set disjointness
            continue
        seen_shas.add(sha)
        rows.append((rel, source, node_count, bucket_of(node_count)))

    # VAL: stratified ~10% per (source, bucket), capped.
    val_target = min(val_cap, round(0.1 * len(rows)))
    strata_all: dict = defaultdict(list)
    for row in rows:
        strata_all[(row[1], row[3])].append(row)
    gen_val = _rng.derive(seed, "splits", "val")
    val_rows = _sample_stratified(strata_all, val_target, gen_val)
    val_rel = {row[0] for row in val_rows}

    # TRAIN: all remaining real + stratified syn fill to train_cap.
    remaining = [row for row in rows if row[0] not in val_rel]
    real_rem = [row for row in remaining if row[1] == "real"]
    syn_rem = [row for row in remaining if row[1] == "syn"]
    syn_strata: dict = defaultdict(list)
    for row in syn_rem:
        syn_strata[row[3]].append(row)
    syn_budget = max(0, train_cap - len(real_rem))
    gen_train = _rng.derive(seed, "splits", "train")
    train_rows = list(real_rem) + _sample_stratified(syn_strata, syn_budget, gen_train)

    val_paths = sorted(row[0] for row in val_rows)
    train_paths = sorted(row[0] for row in train_rows)

    def _ratio(paths: list[str], rows_by_rel: dict) -> dict:
        real = sum(1 for p in paths if rows_by_rel.get(p) == "real")
        syn = len(paths) - real
        return {"real": real, "syn": syn}

    rows_by_rel = {row[0]: row[1] for row in rows}
    val_nodes = [row[2] for row in val_rows]
    train_nodes = [row[2] for row in train_rows]

    payload = {
        "test": test_paths,
        "val": val_paths,
        "train": train_paths,
        "meta": {
            "counts": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
            "size_hist": {
                "train": _size_hist(train_nodes),
                "val": _size_hist(val_nodes),
                "test": _size_hist(test_nodes),
            },
            "source_ratio": {
                "train": _ratio(train_paths, rows_by_rel),
                "val": _ratio(val_paths, rows_by_rel),
            },
            "leakage_excluded": leakage_excluded,
            "pool_sha_duplicates_removed": sha_duplicates,
            "pool_unreadable": unreadable,
            "params": {"train_cap": train_cap, "val_cap": val_cap, "seed": seed},
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test splits.")
    parser.add_argument("--train-cap", type=int, default=2000)
    parser.add_argument("--val-cap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graphs", default="configs/graphs.json")
    parser.add_argument("--out", default="configs/splits.json")
    args = parser.parse_args()

    payload = build_splits(
        args.train_cap, args.val_cap, args.seed,
        PROJECT_ROOT / args.graphs, PROJECT_ROOT / args.out,
    )
    meta = payload["meta"]
    print(f"counts: {meta['counts']}")
    print(f"leakage_excluded: {meta['leakage_excluded']}  "
          f"pool_sha_duplicates_removed: {meta['pool_sha_duplicates_removed']}  "
          f"pool_unreadable: {meta['pool_unreadable']}")
    for split in ("train", "val", "test"):
        print(f"{split:>5} size_hist: {meta['size_hist'][split]}")
    for split in ("train", "val"):
        print(f"{split:>5} source_ratio: {meta['source_ratio'][split]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
