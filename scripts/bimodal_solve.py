"""Parametrise, build and evaluate bimodal degree distributions.

Enumerates the feasible ``(h, k_max, k_min)`` combinations for a given ``(N, M)``,
builds each graph, measures ``E`` and ``R``, and reports the Pareto front.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from isp_construct import bfs_efficiency, fc_random, fc_targeted, measure


def graphical(N: int, h: int, kmax: int, kmin: int) -> Tuple[bool, str]:
    """Whether the two-point sequence (kmax x h, kmin x (N-h)) is realisable (Erdos-Gallai)."""
    if h < 1 or h >= N:
        return False, "h out of range"
    if kmin < 1:
        return False, "k_min < 1"
    if kmax > N - 1:
        return False, f"k_max={kmax} > N-1={N - 1}"
    if kmax < kmin:
        return False, "k_max < k_min"
    S = h * kmax + (N - h) * kmin
    if S % 2:
        return False, f"degree sum {S} is odd"
    for k in range(1, N + 1):
        left = k * kmax if k <= h else h * kmax + (k - h) * kmin
        na = max(0, h - k)
        nb = (N - k) - na
        right = k * (k - 1) + na * min(kmax, k) + nb * min(kmin, k)
        if left > right:
            return False, f"Erdos-Gallai fails at k={k} ({left} > {right})"
    return True, "ok"


def params(N: int, M: int, strict: bool = True) -> List[Dict]:
    """Enumerate the integer bimodal parameter sets (h, k_max, k_min)."""
    S = 2 * M
    kbar = S / N
    out: List[Dict] = []
    for kmin in range(1, int(math.floor(kbar)) + 1):
        for h in range(1, N):
            num = S - (N - h) * kmin
            if num <= 0:
                break
            if num % h:
                continue
            kmax = num // h
            if strict and kmax == kmin:
                continue
            ok, _ = graphical(N, h, kmax, kmin)
            if not ok:
                continue
            hub_stubs = h * kmax
            leaf_stubs = (N - h) * kmin
            max_hh = h * (h - 1)
            out.append(dict(h=h, kmax=kmax, kmin=kmin, rho=h / N,
                            rho_min=1 - h / N, edges=M, kbar=kbar,
                            c=kmax / kbar,
                            hub_stubs=hub_stubs, leaf_stubs=leaf_stubs,
                            hub_slack=max_hh + leaf_stubs - hub_stubs,
                            hub_hub_max=max_hh // 2))
    out.sort(key=lambda d: (d["kmin"], -d["kmax"]))
    return out


def min_h(N: int, M: int, kmin: int) -> Optional[int]:
    """Smallest hub count that keeps the plan realisable for a given k_min, or None."""
    S = 2 * M
    for h in range(1, N):
        num = S - (N - h) * kmin
        if num <= 0:
            break
        if num % h:
            continue
        if graphical(N, h, num // h, kmin)[0]:
            return h
    return None


def _havel_hakimi(N: int, deg: Sequence[int]) -> Optional[List[set]]:
    """Build one simple graph deterministically; succeeds whenever Erdos-Gallai holds."""
    d = list(deg)
    adj = [set() for _ in range(N)]
    alive = [v for v in range(N) if d[v] > 0]
    while alive:
        alive.sort(key=lambda v: -d[v])
        u = alive[0]
        k = d[u]
        if k == 0:
            break
        cand = [v for v in alive if v != u and d[v] > 0]
        if len(cand) < k:
            return None
        for v in cand[:k]:
            adj[u].add(v)
            adj[v].add(u)
            d[v] -= 1
        d[u] = 0
        alive = [v for v in alive if d[v] > 0]
    if any(d):
        return None
    return adj


def _randomize(adj: List[set], swaps: int, rng: random.Random) -> int:
    """Degree-preserving double-edge swaps: randomise the structure, keep every degree."""
    edges = [(u, v) for u in range(len(adj)) for v in adj[u] if u < v]
    E = len(edges)
    if E < 2:
        return 0
    done = 0
    for _ in range(swaps):
        i, j = rng.randrange(E), rng.randrange(E)
        if i == j:
            continue
        u, v = edges[i]
        x, y = edges[j]
        if rng.random() < 0.5:
            x, y = y, x
        if len({u, v, x, y}) < 4:
            continue
        if x in adj[u] or y in adj[v]:
            continue
        adj[u].discard(v)
        adj[v].discard(u)
        adj[x].discard(y)
        adj[y].discard(x)
        adj[u].add(x)
        adj[x].add(u)
        adj[v].add(y)
        adj[y].add(v)
        edges[i] = (u, x)
        edges[j] = (v, y)
        done += 1
    return done


def _connected(adj: Sequence[Sequence[int]]) -> bool:
    N = len(adj)
    seen = [False] * N
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                cnt += 1
                stack.append(v)
    return cnt == N


def _hh_edges(n: int, degs: Sequence[int]) -> Optional[List[Tuple[int, int]]]:
    """Edge-list Havel-Hakimi; succeeds whenever the sequence is graphical."""
    d = list(degs)
    edges: List[Tuple[int, int]] = []
    order = sorted(range(n), key=lambda v: -d[v])
    while True:
        order = [v for v in order if d[v] > 0]
        if not order:
            break
        order.sort(key=lambda v: -d[v])
        u = order[0]
        k = d[u]
        cand = [v for v in order[1:] if d[v] > 0]
        if len(cand) < k:
            return None
        for v in cand[:k]:
            edges.append((u, v))
            d[v] -= 1
        d[u] = 0
    return None if any(d) else edges


def build(N: int, h: int, kmax: int, kmin: int, seed: int = 0,
          swaps_factor: int = 8) -> Dict:
    """Realise the three edge classes directly from the bimodal degree sequence."""
    ok, why = graphical(N, h, kmax, kmin)
    if not ok:
        return dict(ok=False, why=why)
    L = N - h
    hub_stubs = h * kmax
    leaf_stubs = L * kmin

    lh = L if (hub_stubs - L) % 2 == 0 else L + 1
    if lh > leaf_stubs:
        lh = leaf_stubs if (hub_stubs - leaf_stubs) % 2 == 0 else leaf_stubs - 1
    hh = (hub_stubs - lh) // 2
    if hh > h * (h - 1) // 2:
        hh = h * (h - 1) // 2
        lh = hub_stubs - 2 * hh
    ll = (leaf_stubs - lh) // 2
    if min(hh, lh, ll) < 0 or hh + lh + ll != (hub_stubs + leaf_stubs) // 2:
        return dict(ok=False, why="the three edge counts are inconsistent")

    rng = random.Random(seed)
    adj = [set() for _ in range(N)]

    def add(u: int, v: int) -> bool:
        if u == v or v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    hd = [2 * hh // h] * h
    for i in range(2 * hh % h):
        hd[i] += 1
    got_hh = None
    for t in range(30):
        trial = hd[:]
        random.Random(seed * 977 + t).shuffle(trial)
        got_hh = _hh_edges(h, trial)
        if got_hh is not None:
            break
    if got_hh is None:
        return dict(ok=False, why="the hub subgraph degree sequence is not graphical")
    for u, v in got_hh:
        add(u, v)

    cap = [kmax - len(adj[i]) for i in range(h)]
    leaves = list(range(h, N))
    rng.shuffle(leaves)
    need = {v: 1 for v in leaves}
    done = 0

    def assign(v: int) -> bool:
        """Give leaf v a hub edge, choosing the hub with the most capacity left."""
        best = -1
        for i in range(h):
            if cap[i] > 0 and i not in adj[v] and (best < 0 or cap[i] > cap[best]):
                best = i
        if best < 0:
            return False
        add(best, v)
        cap[best] -= 1
        need[v] -= 1
        return True

    for v in leaves:
        if not assign(v):
            return dict(ok=False, why="hub capacity cannot cover every leaf")
    done = L
    base_cap, base_adj = cap[:], [set(x) for x in adj]
    for attempt in range(40):
        cap[:] = base_cap[:]
        adj[:] = [set(x) for x in base_adj]
        need = {v: kmin - 1 for v in leaves}
        order = leaves[:]
        random.Random(seed * 31 + attempt).shuffle(order)
        cnt = 0
        guard = 0
        while cnt < lh - L and guard < 4 * L:
            guard += 1
            moved = False
            for v in order:
                if need[v] <= 0 or cnt >= lh - L:
                    continue
                if assign(v):
                    cnt += 1
                    moved = True
            if not moved:
                break
        if cnt == lh - L:
            done = lh
            break
    if done != lh:
        return dict(ok=False, why=f"hub-leaf edges only reach {lh - (lh - done)}/{lh}")

    ldeg = [need[v] for v in leaves]
    if sum(ldeg):
        sub = _hh_edges(len(leaves), ldeg)
        if sub is None:
            return dict(ok=False, why="the leaf subgraph degree sequence is not graphical")
        for a, b in sub:
            if not add(leaves[a], leaves[b]):
                return dict(ok=False, why="a leaf-leaf edge is reused")
        _randomize(adj, swaps_factor * (sum(ldeg) // 2), random.Random(seed * 53 + 11))

    if not _connected(adj):
        return dict(ok=False, why="disconnected")

    degs = [len(a) for a in adj]
    if set(degs) != {kmax, kmin}:
        return dict(ok=False, why=f"degrees changed: {sorted(set(degs))}")
    keep = [set(x) for x in adj]
    _randomize(adj, swaps_factor * (sum(degs) // 2), random.Random(seed * 131 + 7))
    if not _connected(adj):
        adj = keep
    lst = [sorted(x) for x in adj]
    return dict(ok=True, adj=lst, N=N, h=h, kmax=kmax, kmin=kmin,
                rho=h / N, edges=sum(degs) // 2, connected=True, attempts=1,
                mix=dict(hh=hh, lh=lh, ll=ll))


def design(n: int, kbar: float, nmax: int, kmax: int) -> Dict:
    """From (n_max, k_max) to an integer degree sequence."""
    rho = nmax / n
    two_m = n * kbar
    if abs(two_m - round(two_m)) > 1e-6:
        two_m = 2 * round(two_m / 2)
        kbar = two_m / n
    two_m = int(round(two_m))
    if not (1 <= nmax <= n - 1):
        return dict(ok=False, why="n_max out of range")
    if not (kbar < kmax <= n - 1):
        return dict(ok=False, why=f"k_max={kmax} must satisfy <k>={kbar} < k_max <= n-1")
    kmin = (kbar - rho * kmax) / (1 - rho)
    if kmin < 1:
        return dict(ok=False, why=f"k_min={kmin:.3f} < 1: the hubs cannot carry that many stubs")
    if nmax * kmax > nmax * (nmax - 1) + (n - nmax) * min(kmin, nmax):
        return dict(ok=False, why="Erdos-Gallai fails")
    L = n - nmax
    lo = int(math.floor(kmin))
    rem = two_m - nmax * kmax - L * lo
    if not (0 <= rem <= L):
        return dict(ok=False, why=f"rounding residual {rem} outside [0,{L}]")
    degs = [kmax] * nmax + [lo + 1] * rem + [lo] * (L - rem)
    return dict(ok=True, kmin=kmin, rho=rho, nmax=nmax, kmax=kmax,
                degs=degs, n_hi=rem, k_lo=lo, two_m=two_m,
                k_eff=2 * kmin - kbar / (1 - rho))


def build_seq(n: int, degs: Sequence[int], seed: int = 0,
              swaps_factor: int = 6) -> Dict:
    """Build a connected simple graph from an arbitrary integer sequence."""
    adj = _havel_hakimi(n, list(degs))
    if adj is None:
        return dict(ok=False, why="the sequence is not graphical")
    rng = random.Random(seed)
    keep = [set(x) for x in adj]
    _randomize(adj, swaps_factor * sum(degs) // 2, rng)
    if not _connected(adj):
        adj = keep
    return dict(ok=True, adj=[sorted(x) for x in adj], n=n,
                edges=sum(degs) // 2, connected=True)


def edge_mix(N: int, h: int, kmax: int, kmin: int) -> Dict:
    """Edge counts of the three classes: hub-hub, hub-leaf and leaf-leaf."""
    hub_stubs = h * kmax
    leaf_stubs = (N - h) * kmin
    M = (hub_stubs + leaf_stubs) // 2
    hh = min(h * (h - 1) // 2, max(0, hub_stubs - (N - h)) // 2)
    lh = hub_stubs - 2 * hh
    if lh > leaf_stubs:
        hh = max(0, h * (h - 1) // 2)
        lh = hub_stubs - 2 * hh
    ll = (leaf_stubs - lh) // 2
    return dict(hh=hh, lh=lh, ll=ll, total=hh + lh + ll, M=M,
                ok=(hh + lh + ll == M and lh >= 0 and ll >= 0))


def _front(rows: Sequence[Dict]) -> List[Dict]:
    nd = [d for d in rows
          if not any(e["E"] >= d["E"] - 1e-12 and e["R"] >= d["R"] - 1e-12
                     and (e["E"] > d["E"] + 1e-12 or e["R"] > d["R"] + 1e-12)
                     for e in rows)]
    return sorted(nd, key=lambda d: d["E"])


def hull(rows: Sequence[Dict]) -> List[Dict]:
    """Upper convex hull in the log-log plane, i.e. the efficient frontier."""
    p = _front(rows)
    h: List[Dict] = []
    for d in p:
        if d["E"] <= 0 or d["R"] <= 0:
            continue
        while len(h) >= 2:
            m1 = math.log(d["R"] / h[-1]["R"]) / math.log(d["E"] / h[-1]["E"])
            m2 = math.log(h[-1]["R"] / h[-2]["R"]) / math.log(h[-1]["E"] / h[-2]["E"])
            if abs(m1) <= abs(m2) + 1e-15:
                h.pop()
            else:
                break
        h.append(d)
    return h


def slope_hull(rows: Sequence[Dict]) -> float:
    """S = the steepest |dln R / dln E| between adjacent hull points."""
    h = hull(rows)
    s = 0.0
    for a, b in zip(h, h[1:]):
        dx = math.log(b["E"]) - math.log(a["E"])
        dy = math.log(b["R"]) - math.log(a["R"])
        if abs(dx) > 1e-15:
            s = max(s, abs(dy / dx))
    return s if s > 0 else 1.0


def lam_breaks(rows: Sequence[Dict]) -> List[Dict]:
    """Lambda breakpoints: each segment slope divided by S, one interval per hull point."""
    h = hull(rows)
    if len(h) < 2:
        return [{"pt": h[0] if h else None, "lo": 0.0, "hi": math.inf, "s": 0.0}] if h else []
    segs = []
    for a, b in zip(h, h[1:]):
        segs.append(abs(math.log(b["R"] / a["R"]) / math.log(b["E"] / a["E"])))
    S = max(segs)
    out = []
    for i, d in enumerate(h):
        lo = segs[i - 1] / S if i > 0 else 0.0
        hi = segs[i] / S if i < len(segs) else math.inf
        out.append({"pt": d, "lo": lo, "hi": hi, "s": segs[i] if i < len(segs) else None})
    return out


def pick_lam(rows: Sequence[Dict], lam: float) -> Optional[Dict]:
    """Pick a hull point with w = lam * S / (1 + lam * S)."""
    h = hull(rows)
    if not h:
        return None
    S = slope_hull(rows)
    w = lam * S / (1.0 + lam * S)
    return max(h, key=lambda d: w * math.log(d["E"]) + (1 - w) * math.log(d["R"]))


def fmt_lam_scale(rows: Sequence[Dict], n: int, kbar: float) -> List[str]:
    """Lambda scale: S, the segment slopes, the breakpoints and the choice at typical lambdas."""
    h = hull(rows)
    if len(h) < 2:
        return [f"n={n}, <k>={kbar}: fewer than two hull points, lambda is undefined"]
    S = slope_hull(rows)
    L = [f"n={n}, <k>={kbar}  S={S:.4f}  hull {len(h)}/{len(rows)} points",
         f"{'k_max':>7}{'E':>9}{'R':>9}{'slope':>9}{'lam_lo':>9}{'lam_hi':>9}{'note':>10}"]
    for r in lam_breaks(rows):
        d = r["pt"]
        km = d.get("kmax") or d.get("K") or 0
        lo = "0" if r["lo"] == 0 else f"{r['lo']:.3f}"
        hi = "inf" if r["hi"] == math.inf else f"{r['hi']:.3f}"
        seg = "—" if r["s"] is None else f"{r['s']:.4f}"
        tag = "R end" if r["lo"] == 0 else ("E end" if r["hi"] == math.inf else "")
        L.append(f"{km:7}{d['E']:9.4f}{d['R']:9.4f}{seg:>9}{lo:>9}{hi:>9}{tag:>10}")
    L.append(f"{'lam':>7}{'w':>9}{'k_max':>9}")
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 10.0):
        d = pick_lam(rows, lam)
        w = lam * S / (1.0 + lam * S)
        km = d.get("kmax") or d.get("K") or 0
        L.append(f"{lam:7g}{w:9.3f}{km:9d}")
    return L


def _slope(front: Sequence[Dict]) -> float:
    """Legacy interface: the older definition of S (Pareto filter only). Use slope_hull."""
    s = 0.0
    for a, b in zip(front, front[1:]):
        dx = math.log(b["E"]) - math.log(a["E"])
        dy = math.log(b["R"]) - math.log(a["R"])
        if abs(dx) > 1e-15 and a["R"] > 0 and b["R"] > 0:
            s = max(s, abs(dy / dx))
    return s if s > 0 else 1.0


def default_grid(N: int, M: int, cap: int = 12) -> List[Tuple[int, int]]:
    """Sample an (h, k_min) grid that covers the shape space."""
    ps = params(N, M)
    if not ps:
        return []
    by_k: Dict[int, List[Dict]] = {}
    for d in ps:
        by_k.setdefault(d["kmin"], []).append(d)
    picked: List[Tuple[int, int]] = []
    for kmin, rows in sorted(by_k.items()):
        rows.sort(key=lambda d: d["h"])
        idx = sorted({0, len(rows) // 3, (2 * len(rows)) // 3, len(rows) - 1})
        for i in idx:
            picked.append((rows[i]["h"], kmin))
    if len(picked) > cap:
        step = len(picked) / cap
        picked = [picked[int(i * step)] for i in range(cap)]
    return picked


def evaluate(N: int, M: int, grid: Optional[Sequence[Tuple[int, int]]] = None,
             seeds: int = 8, gseed: int = 2) -> List[Dict]:
    """Build and measure on the (h, k_min) grid; the mean degree fixes k_max from h."""
    if grid is None:
        grid = default_grid(N, M)
    rows: List[Dict] = []
    for h, kmin in grid:
        S = 2 * M
        num = S - (N - h) * kmin
        if num <= 0 or num % h:
            continue
        kmax = num // h
        if not graphical(N, h, kmax, kmin)[0]:
            continue
        ms = []
        for s in range(gseed):
            g = build(N, h, kmax, kmin, seed=17 + s)
            if g.get("ok"):
                ms.append(measure(g, seeds=seeds))
        if not ms:
            continue
        avg = lambda f: sum(m[f] for m in ms) / len(ms)
        rows.append(dict(h=h, kmax=kmax, kmin=kmin, rho=h / N,
                         c=kmax / (S / N),
                         E=avg("E"), fc_rand=avg("fc_rand"),
                         fc_tar=avg("fc_tar"),
                         R=0.5 * (avg("fc_rand") + avg("fc_tar"))))
    return rows


def pick(rows: Sequence[Dict], lam: float) -> Dict:
    """Map lambda onto the slope scale: w = lam*S/(1+lam*S), objective w*lnE + (1-w)*lnR."""
    S = _slope(_front(rows))
    w = lam * S / (1 + lam * S)
    best = max(rows, key=lambda d: (w * math.log(max(d["E"], 1e-300))
                                    + (1 - w) * math.log(max(d["R"], 1e-300)),
                                    d["E"]))
    return dict(best, w=w, S=S, lam=lam)


def fmt_params(rows: Sequence[Dict], N: int) -> List[str]:
    L = [f"N={N}  integer bimodal parameter sets (two distinct degrees)"]
    L.append(f"{'h':>5}{'rho_h':>8}{'k_max':>7}{'k_min':>7}{'c':>7}"
             f"{'hub stubs':>10}{'leaf stubs':>11}{'hub-internal max':>17}{'slack':>7}")
    for d in rows:
        L.append(f"{d['h']:5d}{d['rho']:8.4f}{d['kmax']:7d}{d['kmin']:7d}"
                 f"{d['c']:7.1f}{d['hub_stubs']:8d}{d['leaf_stubs']:7d}"
                 f"{d['hub_hub_max']:13d}{d['hub_slack']:7d}")
    return L


def fmt_trace(N: int, M: int, h: int, kmin: int) -> List[str]:
    S = 2 * M
    num = S - (N - h) * kmin
    if num <= 0 or num % h:
        return [f"h={h}, k_min={kmin} has no integer solution for N={N}, M={M}"
                f" (2M-(N-h)k_min = {num} is not divisible by h={h})"]
    kmax = num // h
    ok, why = graphical(N, h, kmax, kmin)
    em = edge_mix(N, h, kmax, kmin)
    L = [f"N={N}, M={M} (<k>={S / N:.2f}), with h={h}, k_min={kmin}",
         "  1. mean-degree conservation (fixes k_max)",
         f"     h*k_max + (N-h)*k_min = 2M",
         f"     {h}*k_max + {N - h}*{kmin} = {S}  =>  k_max = {kmax}",
         f"     rho_h = {h}/{N} = {h / N:.4f}，rho_min = {1 - h / N:.4f}",
         "  2. realisability (Erdos-Gallai)",
         f"     k_max <= N-1 ?  {kmax} <= {N - 1}  ->  {kmax <= N - 1}",
         f"     degree sum even?  {S} % 2 = {S % 2}  ->  {S % 2 == 0}",
         f"     can the {h * kmax} hub stubs be absorbed? upper bound = 2*C(h,2)={h * (h - 1)}"
         f" + {N * 0 + (N - h) * kmin} leaf stubs = {h * (h - 1) + (N - h) * kmin}"
         f"  ->  {'yes' if h * (h - 1) + (N - h) * kmin >= h * kmax else 'no'}",
         f"     verdict: {why}",
         f"  3. degree sequence  [{kmax}]*{h} + [{kmin}]*{N - h}, {S} stubs over {M} edges"]
    if ok:
        L.append(f"  4. the three edge classes (fixed by the degree sequence)")
        L.append(f"     hub-hub {em['hh']} | hub-leaf {em['lh']} | "
                 f"leaf-leaf {em['ll']} | total {em['total']}")
        g = build(N, h, kmax, kmin, seed=17)
        if g.get("ok"):
            m = measure(g, seeds=8)
            degs = [len(a) for a in g["adj"]]
            L.append(f"  5. build  Havel-Hakimi -> {12 * M} degree-preserving swaps"
                     f" -> connected={g['connected']}"
                     f" (attempt {g['attempts']})")
            L.append(f"     check: degrees {sorted(set(degs))}, edges {g['edges']}")
            L.append(f"  6. measured  E={m['E']:.4f}  f_c^rand={m['fc_rand']:.4f}  "
                     f"f_c^tar={m['fc_tar']:.4f}  R={m['R']:.4f}")
    return L


def fmt_kmax_sweep(n: int, kbar: float, nmax: int, G: int = 3,
                   seeds: int = 8, points: int = 8) -> List[str]:
    """Fix n_max and sweep k_max over (<k>, n-1], measuring E, f_c and R."""
    has = _measure_ok()
    L = [f"n={n}, <k>={kbar}, n_max={nmax} (rho_max={nmax/n:.4f})  sweeping k_max",
         f"{'k_max':>6}{'k_min':>8}{'k_eff':>7}{'<k_hub>':>8}"
         f"{'E':>8}{'f_c^r':>8}{'f_c^t':>8}{'R':>8}"]
    rows: List[Dict] = []
    lo_k = int(math.ceil(kbar))
    ks = sorted({int(round(x)) for x in _logspace(max(lo_k + 1, 2), n - 1, points)} | {n - 1})
    for kmax in ks:
        d = design(n, kbar, nmax, kmax)
        if not d["ok"]:
            L.append(f"{kmax:6d}   infeasible: {d['why']}")
            continue
        khub = nmax * kmax / (n - nmax)
        if not has:
            L.append(f"{kmax:6d}{d['kmin']:8.3f}{d['k_eff']:7.3f}{khub:8.3f}"
                     f"{'  - (needs isp_construct)':>32}")
            continue
        acc = dict(E=0.0, fc_rand=0.0, fc_tar=0.0)
        got = 0
        bad = 0
        for g in range(G):
            gg = build_seq(n, d["degs"], seed=101 + 37 * g)
            if not gg["ok"]:
                bad += 1
                continue
            mt = measure(gg, seeds=seeds)
            if any(isinstance(mt[k], float) and math.isnan(mt[k])
                   for k in ("E", "fc_rand", "fc_tar")):
                bad += 1
                continue
            for k in acc:
                acc[k] += mt[k]
            got += 1
        if not got:
            L.append(f"{kmax:6d}{d['kmin']:8.3f}{d['k_eff']:7.3f}{khub:8.3f}   every realisation unusable")
            continue
        E = acc["E"] / got
        fr = acc["fc_rand"] / got
        ft = acc["fc_tar"] / got
        tag = f"  ({bad}/{G} realisations unusable)" if bad else ""
        L.append(f"{kmax:6d}{d['kmin']:8.3f}{d['k_eff']:7.3f}{khub:8.3f}"
                 f"{E:8.4f}{fr:8.4f}{ft:8.4f}{0.5 * (fr + ft):8.4f}{tag}")
        rows.append(dict(kmax=kmax, E=E, R=0.5 * (fr + ft),
                         fc_rand=fr, fc_tar=ft, kmin=d["kmin"], k_eff=d["k_eff"]))
    if rows:
        L.append("")
        L += fmt_lam_scale(rows, n, kbar)
    return L


def _logspace(a: float, b: float, k: int) -> List[float]:
    if k <= 1 or a >= b:
        return [b]
    r = (b / a) ** (1.0 / (k - 1))
    return [a * r ** i for i in range(k)]


def _measure_ok() -> bool:
    try:
        from isp_construct import measure  # noqa: F401
        return True
    except Exception:
        return False


def fmt_solve(rows: Sequence[Dict], lam: float) -> List[str]:
    fr = _front(rows)
    fset = {(d["h"], d["kmax"], d["kmin"]) for d in fr}
    L = [f"{'h':>5}{'k_max':>7}{'k_min':>7}{'c':>7}{'E':>8}{'fcr':>8}{'fct':>8}{'R':>8}  front"]
    for d in sorted(rows, key=lambda x: x["E"]):
        mark = "*" if (d["h"], d["kmax"], d["kmin"]) in fset else ""
        L.append(f"{d['h']:5d}{d['kmax']:7d}{d['kmin']:7d}{d['c']:7.1f}"
                 f"{d['E']:8.4f}{d['fc_rand']:8.4f}{d['fc_tar']:8.4f}"
                 f"{d['R']:8.4f}  {mark}")
    if rows:
        p = pick(rows, lam)
        L.append("")
        L.append(f"lambda={lam:.2f}  S_max={p['S']:.3f}  w={p['w']:.3f}  =>  "
                 f"h={p['h']}, k_max={p['kmax']}, k_min={p['kmin']}, c={p['c']:.1f}  "
                 f"E={p['E']:.4f}  R={p['R']:.4f}")
    return L


def _selftest() -> int:
    bad = 0

    def chk(name: str, cond: bool, extra: str = "") -> None:
        nonlocal bad
        if not cond:
            bad += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}"
              + (f"  ({extra})" if extra else ""))

    print("realisability")
    chk("k_max > N-1 rejected", not graphical(10, 1, 10, 1)[0])
    chk("odd degree sum rejected", not graphical(10, 1, 4, 1)[0], "4+9=13")
    chk("unabsorbable hub stubs rejected", not graphical(100, 1, 60, 1)[0], "60 > 0+99")
    chk("valid point accepted", graphical(100, 2, 53, 3)[0])

    print("single-hub (h=1) integer bimodal solutions")
    for N, kb in ((1000, 3.0), (1000, 5.0), (100, 4.0)):
        M = int(round(kb * N / 2))
        chk(f"N={N},<k>={kb} has no h=1 integer solution",
            not any(d["h"] == 1 for d in params(N, M)),
            f"k_max must exceed N-1 when kmin<=2")

    print("realisable parameter sets")
    for N, kb in ((1000, 3.0), (1000, 5.0)):
        M = int(round(kb * N / 2))
        ps = params(N, M)
        chk(f"N={N},<k>={kb} has solutions", len(ps) > 0, f"{len(ps)} sets")
        chk(f"N={N},<k>={kb} all conserve the mean degree",
            all(d["h"] * d["kmax"] + (N - d["h"]) * d["kmin"] == 2 * M for d in ps))
        chk(f"N={N},<k>={kb} all have k_min < k_max",
            all(d["kmin"] < d["kmax"] for d in ps))
        chk(f"N={N},<k>={kb} all pass Erdos-Gallai",
            all(graphical(N, d["h"], d["kmax"], d["kmin"])[0] for d in ps))

    print("construction invariants")
    cases = [(100, 200, 2, 3), (100, 200, 5, 2), (100, 200, 20, 1), (200, 400, 8, 2)]
    for (N, M, h, kmin) in cases:
        S = 2 * M
        num = S - (N - h) * kmin
        if num <= 0 or num % h:
            continue
        kmax = num // h
        g = build(N, h, kmax, kmin, seed=17)
        if not g.get("ok"):
            chk(f"N={N} h={h} kmin={kmin} builds a connected graph", False, g.get("why", ""))
            continue
        degs = [len(a) for a in g["adj"]]
        chk(f"N={N} h={h} kmin={kmin} degrees strictly bimodal, edge count conserved",
            set(degs) == {kmax, kmin} and sum(degs) == S and g["edges"] == M,
            f"degs={sorted(set(degs))} edges={g['edges']}")

    print("edge-class counts")
    for (N, M, h, kmin) in ((1000, 1500, 2, 2), (1000, 1500, 40, 1),
                            (1000, 2500, 2, 4), (200, 400, 8, 2)):
        num = 2 * M - (N - h) * kmin
        if num <= 0 or num % h:
            continue
        kmax = num // h
        em = edge_mix(N, h, kmax, kmin)
        chk(f"N={N} h={h} kmin={kmin} the three classes add up to M and are non-negative",
            em["ok"] and em["total"] == M,
            f"hh={em['hh']} lh={em['lh']} ll={em['ll']}")
        chk(f"N={N} h={h} kmin={kmin} every leaf has at least one hub edge",
            em["lh"] >= N - h, f"lh={em['lh']} vs leaves {N - h}")

    print("measurement: E is not E_dir")
    g = build(1000, 2, 502, 2, seed=17)
    if g.get("ok"):
        m = measure(g, seeds=8)
        e_dir = 502 * 505 / (2 * 1000 * 999)
        chk("measured E well above the single-hub bound", m["E"] > 3 * e_dir,
            f"E={m['E']:.4f} vs E_dir={e_dir:.4f}")

    print("solver")
    rows = evaluate(100, 200, grid=[(2, 3), (4, 3), (5, 2), (10, 2), (20, 1)],
                    seeds=6, gseed=1)
    chk("evaluate returns points", len(rows) >= 3, f"{len(rows)} points")
    if len(rows) >= 3:
        fr = _front(rows)
        chk("E increases along the front", all(b["E"] >= a["E"] for a, b in zip(fr, fr[1:])))
        p0, p1 = pick(rows, 0.0), pick(rows, 1.0)
        chk("lambda=0 favours R", p0["R"] >= p1["R"] - 1e-9,
            f"R {p0['R']:.3f} vs {p1['R']:.3f}")
        chk("lambda=1 favours E", p1["E"] >= p0["E"] - 1e-9,
            f"E {p1['E']:.3f} vs {p0['E']:.3f}")

    print(f"\n{'all checks passed' if bad == 0 else f'{bad} checks failed'}")
    return 0 if bad == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--kbar", type=float, default=3.0)
    ap.add_argument("--params", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--kmin", type=int, default=None)
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--nmax", type=int, default=1)
    ap.add_argument("--kmax-sweep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest())

    N = a.N
    M = int(round(a.kbar * N / 2))
    if a.params:
        ps = params(N, M)
        print("\n".join(fmt_params(ps, N)))
        print(f"\n{len(ps)} sets; h=1 {'exists' if any(d['h'] == 1 for d in ps) else 'does not exist'}")
        for kmin in sorted({d['kmin'] for d in ps}):
            print(f"  k_min={kmin}: minimum h = {min_h(N, M, kmin)}")
    if a.trace:
        h = a.h or 2
        kmin = a.kmin if a.kmin is not None else 2
        print("\n".join(fmt_trace(N, M, h, kmin)))
    if a.kmax_sweep:
        print("\n".join(fmt_kmax_sweep(N, a.kbar, a.nmax)))
    if a.solve:
        print("\n".join(fmt_solve(evaluate(N, M, seeds=a.seeds), a.lam)))


if __name__ == "__main__":
    main()
