"""Sweep over the input variables ``(n, <k>)`` behind Fig. 1.

For each cell it builds the original graph, derives the feasible interval of
``k_max``, builds the bimodal graph for every integer in it and records ``E``,
``f_c`` and ``Phi``.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from typing import Dict, List

import bimodal_solve as bs
from isp_construct import (bfs_efficiency, fc_random, fc_targeted,
                           random_conn)


def measure(adj, seeds: int = 4) -> Dict:
    e = bfs_efficiency(adj)
    fr = fc_random(adj, seeds=seeds)["fc"]
    ft = fc_targeted(adj)["fc"]
    return dict(e=e, fr=fr, ft=ft)


def cell(n: int, kbar: float, seeds: int = 4, gseeds: int = 2) -> List[Dict]:
    M = int(round(n * kbar / 2))
    if M < n - 1:
        return [dict(n=n, kbar=kbar, M=M, status="no-graph")]
    base = random_conn(n, M, seed=0)
    if not base.get("adj"):
        return [dict(n=n, kbar=kbar, M=M, status="no-graph")]
    kbar = 2.0 * M / n
    m0 = measure(base["adj"], seeds=seeds)
    e0, f0 = m0["e"], m0["fr"] + m0["ft"]
    lo = math.ceil(kbar + 1 - 1.0 / n)
    cap = 2.0 * (n - 1) * e0
    hi = int(math.floor((math.sqrt(9.0 + 4.0 * cap) - 3.0) / 2.0))
    hi = min(hi, n - 1)
    if hi < lo:
        return [dict(n=n, kbar=kbar, M=M, status="empty", lo=lo, hi=hi,
                     e0=e0, f0=f0)]

    rows: List[Dict] = []
    for k in range(lo, hi + 1):
        d = bs.design(n, kbar, 1, k)
        if not d["ok"]:
            continue
        es, frs, fts = [], [], []
        for gs in range(gseeds):
            g = bs.build_seq(n, d["degs"], seed=gs)
            if not g["ok"]:
                continue
            mm = measure(g["adj"], seeds=seeds)
            es.append(mm["e"])
            frs.append(mm["fr"])
            fts.append(mm["ft"])
        if not es:
            continue
        e = sum(es) / len(es)
        fr = sum(frs) / len(frs)
        ft = sum(fts) / len(fts)
        ok_phi = (e > 0 and e0 > 0 and fr + ft > 0 and f0 > 0)
        rows.append(dict(n=n, kbar=round(kbar, 4), M=M, status="ok" if ok_phi else "degen",
                         lo=lo, hi=hi, k=k, kmin=d["kmin"], k_eff=d["k_eff"],
                         e0=e0, f0=f0, e=e, fr=fr, ft=ft,
                         phi=(0.5 * (e / e0) + 0.5 * ((fr + ft) / f0))
                         if ok_phi else float("nan"),
                         phi_loggeo=(0.5 * math.log(e / e0) + 0.5 * math.log((fr + ft) / f0))
                         if ok_phi else float("nan")))
    if not rows:
        return [dict(n=n, kbar=kbar, M=M, status="nobuild", lo=lo, hi=hi, e0=e0, f0=f0)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="100,300,1000")
    ap.add_argument("--kbar", default="2,2.5,3,4,5,6,8,10,12,16")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--gseeds", type=int, default=2)
    ap.add_argument("--out", default="input_sweep.csv")
    a = ap.parse_args()

    t0 = time.time()
    rows: List[Dict] = []
    for ns in a.n.split(","):
        n = int(ns)
        for ks in a.kbar.split(","):
            kb = float(ks)
            rs = cell(n, kb, seeds=a.seeds, gseeds=a.gseeds)
            rows += rs
            st = rs[0]["status"]
            if st == "ok":
                ph = [r["phi"] for r in rs if r["status"] == "ok"]
                print(f"n={n:5d} <k>={kb:5.1f} domain=[{rs[0]['lo']},{rs[0]['hi']}] "
                      f"pts={len(ph)} e/e0 {rs[0]['e']/rs[0]['e0']:.3f}-{rs[-1]['e']/rs[-1]['e0']:.3f} "
                      f"phi {min(ph):+.4f}..{max(ph):+.4f}", flush=True)
            else:
                print(f"n={n:5d} <k>={kb:5.1f} {st} [{rs[0].get('lo')},{rs[0].get('hi')}]", flush=True)

    if not rows:
        print("empty result")
        return
    keys = ["n", "kbar", "M", "status", "lo", "hi", "k", "kmin", "k_eff",
            "e0", "f0", "e", "fr", "ft", "phi", "phi_loggeo"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
