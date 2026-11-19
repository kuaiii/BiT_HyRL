"""Fallback designs for cells whose feasible window is empty (Fig. 2).

Uses ``k_max = lo`` for those cells and measures ``E`` and ``f_c`` so the curves
can be extended beyond the non-empty window.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import bimodal_solve as bs
from input_sweep import measure


def fallback_rows(n: int, kbar: float, lo: int, hi: int, e0: float, f0: float,
                  seeds: int = 4, gseeds: int = 2) -> List[Dict]:
    design = bs.design(n, kbar, 1, lo)
    if not design["ok"]:
        return [dict(n=n, kbar=kbar, k=lo, lo=lo, hi=hi, e0=e0, f0=f0,
                     status="unbuildable", why=design.get("why", ""))]

    es, frs, fts = [], [], []
    for gs in range(gseeds):
        graph = bs.build_seq(n, design["degs"], seed=gs)
        if not graph["ok"]:
            continue
        got = measure(graph["adj"], seeds=seeds)
        es.append(got["e"])
        frs.append(got["fr"])
        fts.append(got["ft"])
    if not es:
        return [dict(n=n, kbar=kbar, k=lo, lo=lo, hi=hi, e0=e0, f0=f0,
                     status="unbuildable", why="build_seq failed for every candidate")]

    e = sum(es) / len(es)
    fr = sum(frs) / len(frs)
    ft = sum(fts) / len(fts)
    ee, ff = e / e0, (fr + ft) / f0
    lam = lo * (lo + 3) / (2.0 * (n - 1) * e0)
    return [dict(n=n, kbar=round(kbar, 4), k=lo, lo=lo, hi=hi, e0=round(e0, 6),
                 f0=round(f0, 6), e=round(e, 6), fr=round(fr, 6), ft=round(ft, 6),
                 ee=round(ee, 6), ff=round(ff, 6), phi=round(0.5 * ee + 0.5 * ff, 6),
                 lambda_star=round(lam, 6), status="ok")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", default="results/input_sweep.csv")
    parser.add_argument("--out", dest="dst", default="results/input_sweep_fallback.csv")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--gseeds", type=int, default=2)
    args = parser.parse_args()

    cells = [r for r in csv.DictReader(open(args.src, encoding="utf-8"))
             if r["status"] == "empty"]
    rows: List[Dict] = []
    for cell in cells:
        got = fallback_rows(int(float(cell["n"])), float(cell["kbar"]),
                            int(cell["lo"]), int(cell["hi"]),
                            float(cell["e0"]), float(cell["f0"]),
                            seeds=args.seeds, gseeds=args.gseeds)
        rows.extend(got)
        for row in got:
            if row["status"] == "ok":
                print(f"n={row['n']:>5} <k>={row['kbar']:>5} k={row['k']:>3} "
                      f"E/E0={row['ee']:.4f} f/f0={row['ff']:.4f} "
                      f"Phi={row['phi']:.4f} lambda*={row['lambda_star']:.3f}")
            else:
                print(f"n={row['n']:>5} <k>={row['kbar']:>5} k={row['k']:>3} FAILED: {row['why']}")

    fields = ["n", "kbar", "k", "lo", "hi", "e0", "f0", "e", "fr", "ft",
              "ee", "ff", "phi", "lambda_star", "status", "why"]
    with open(args.dst, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print(f"-> {args.dst}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
