#!/usr/bin/env python3
"""Construct the ISP graphs at their real ``(N, M)`` and measure percolation.

Compares the measured ``E`` and ``f_c`` against the closed-form estimates.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from fc_robust import fc_random_moments, fc_targeted_moments
from param_window import efficiency_ceiling

#: ``design_map`` and ``domain_map`` are internal exploration helpers that are not
#: part of the released measurement path. They are imported inside the two
#: functions that use them so that ``measure``/``fc_targeted`` stay importable
#: without them.

EPS = 1e-12

ISP: List[Tuple[str, int, int, str]] = [
    ("COGENTCO", 197, 243, "pop"),
    ("COLT", 153, 177, "pop"),
    ("ION", 125, 146, "pop"),
    ("TATANLD", 145, 186, "pop"),
    ("DELTACOM", 113, 161, "pop"),
    ("NTT", 32, 65, "pop"),
    ("UUNET", 49, 84, "pop"),
    ("TW", 71, 115, "pop"),
    ("DFN", 50, 78, "pop"),
    ("CHINANET", 42, 66, "pop"),
    ("UNINETT", 71, 97, "pop"),
    ("SURFNET", 50, 68, "pop"),
    ("PALMETTO", 45, 64, "pop"),
    ("BELLSOUTH", 51, 66, "pop"),
    ("REDBESTEL", 84, 93, "pop"),
    ("VTLWAVENET2008", 88, 92, "pop"),
    ("TELCOVE", 71, 70, "pop"),
    ("INTEROUTE-PHYS", 110, 158, "opt"),
    ("DELTACOM-PHYS", 113, 183, "opt"),
    ("KENTUCKY-DATALINK", 754, 899, "opt"),
    ("BESTEL-PHYS", 84, 101, "opt"),
    ("OTEGLOBE-PHYS", 93, 106, "opt"),
    ("US-SIGNAL-PHYS", 63, 79, "opt"),
    ("VIATEL-PHYS", 88, 92, "opt"),
]

SRC_LABEL = {"pop": "PoP-level ISP", "opt": "optical transport layer"}


def _spread(n: int, r: int) -> List[int]:
    """Draw r distinct indices from [0, n)."""
    if r <= 0:
        return []
    if r >= n:
        return list(range(n))
    return [int(round(j * n / r)) % n for j in range(r)]


def _empty(n: int) -> List[List[int]]:
    return [[] for _ in range(n)]


def _add(adj: List[List[int]], seen: set, u: int, v: int) -> bool:
    if u == v:
        return False
    if u > v:
        u, v = v, u
    if (u, v) in seen:
        return False
    adj[u].append(v)
    adj[v].append(u)
    seen.add((u, v))
    return True


def extra_edges(n: int, count: int, base: int,
                adj: List[List[int]], seen: set) -> int:
    """Add count equidistant chords inside [base, base+n); returns the number added."""
    cnt = 0
    d = 2
    while cnt < count and d <= (n - 1) // 2:
        cap = n if 2 * d < n else n // 2
        need = min(cap, count - cnt)
        for i in _spread(cap, need):
            if cnt < count and _add(adj, seen, base + i, base + (i + d) % n):
                cnt += 1
        d += 1
    return cnt


def ring_graph(n: int, e: int, base: int = 0,
               adj: Optional[List[List[int]]] = None,
               seen: Optional[set] = None) -> Tuple[List[List[int]], int]:
    """Connected near-regular graph on n nodes with e edges: a ring plus chords."""
    if e < n - 1:
        raise ValueError(f"ring_graph: {n} nodes need at least {n - 1} edges, got {e}")
    if adj is None:
        adj = _empty(base + n)
    if seen is None:
        seen = set()
    cnt = 0
    for i in range(n - 1):
        if _add(adj, seen, base + i, base + i + 1):
            cnt += 1
    if cnt < e and n > 2:
        if _add(adj, seen, base + n - 1, base):
            cnt += 1
    cnt += extra_edges(n, e - cnt, base, adj, seen)
    if cnt < e:
        for u in range(base, base + n):
            if cnt >= e:
                break
            for v in range(u + 1, base + n):
                if cnt >= e:
                    break
                if _add(adj, seen, u, v):
                    cnt += 1
    return adj, cnt


def hub1(N: int, M: int, K: int, seed: int = 7) -> Dict:
    """Single-hub family: one node of degree K, the other N-1 form a connected residual."""
    res_n, res_e = N - 1, M - K
    if K < 1 or K > N - 1 or res_e < res_n - 1:
        return dict(ok=False, why=f"K={K} infeasible (res_e={res_e} < {res_n - 1})")
    sub = random_conn(res_n, int(res_e), seed=seed)
    adj = _empty(N)
    seen: set = set()
    for u in range(res_n):
        for v in sub["adj"][u]:
            if u < v:
                _add(adj, seen, u + 1, v + 1)
    for j in _spread(res_n, K):
        _add(adj, seen, 0, 1 + j)
    degs = [len(a) for a in adj]
    return dict(ok=True, adj=adj, edges=(sum(degs) // 2), N=N, M=M, K=K,
                hub=0, hub_deg=degs[0], k_min=min(degs[1:]) if N > 1 else 0,
                k_max=max(degs), connected=_connected(adj))


def flower(N: int, M: int, h: int, seed: int = 7) -> Dict:
    """Flower family: h hubs form a near-regular core, every leaf hangs off one hub."""
    L = N - h
    e_core = M - L
    if L <= 0 or e_core < h - 1:
        return dict(ok=False, why=f"h={h}: core budget too small (e_core={e_core})")
    cap = h * (h - 1) // 2
    extra_leaf = max(0, e_core - cap)
    e_core = min(e_core, cap)
    sub = random_conn(h, int(e_core), seed=seed + 1)
    adj = _empty(N)
    seen: set = set()
    for u in range(h):
        for v in sub["adj"][u]:
            if u < v:
                _add(adj, seen, u, v)
    for j in range(L):
        _add(adj, seen, h + j, j % h)
    if extra_leaf:
        extra_edges(L, extra_leaf, h, adj, seen)
    degs = [len(a) for a in adj]
    return dict(ok=True, adj=adj, edges=sum(degs) // 2, N=N, M=M, h=h,
                hub_deg=sum(degs[:h]) / h, k_min=min(degs[h:]), k_max=max(degs),
                connected=_connected(adj))


def random_conn(N: int, M: int, seed: int = 0) -> Dict:
    """Random connected graph: random recursive tree plus extra edges, exactly M edges."""
    rng = random.Random(seed)
    adj = _empty(N)
    seen: set = set()
    for i in range(1, N):
        _add(adj, seen, i, rng.randrange(i))
    guard = 0
    while sum(len(a) for a in adj) // 2 < M and guard < 200 * M + 10000:
        guard += 1
        _add(adj, seen, rng.randrange(N), rng.randrange(N))
    degs = [len(a) for a in adj]
    return dict(ok=True, adj=adj, edges=sum(degs) // 2, N=N, M=M,
                hub_deg=max(degs), k_min=min(degs), k_max=max(degs),
                connected=_connected(adj))


def bimodal_cm(N: int, r: float, K: float, kmin: float, seed: int = 0) -> Optional[List[List[int]]]:
    """Standard bimodal configuration model: rN hubs of degree K, the rest of degree kmin."""
    rng = random.Random(seed)
    nh = max(1, int(round(r * N)))
    deg = [int(K)] * nh + [int(kmin)] * (N - nh)
    if sum(deg) % 2:
        deg[-1] += 1
    stubs = []
    for v, d in enumerate(deg):
        stubs += [v] * d
    for _ in range(30):
        rng.shuffle(stubs)
        adj = _empty(N)
        seen: set = set()
        i, bad, ok = 0, 0, True
        while i < len(stubs) - 1:
            u, v = stubs[i], stubs[i + 1]
            i += 2
            if u == v or (min(u, v), max(u, v)) in seen:
                bad += 1
                if bad > 4 * len(stubs):
                    ok = False
                    break
                j = rng.randrange(i, len(stubs))
                stubs[i - 1], stubs[j] = stubs[j], stubs[i - 1]
                i -= 2
                continue
            seen.add((min(u, v), max(u, v)))
            adj[u].append(v)
            adj[v].append(u)
        if ok and _connected(adj):
            return adj
    return None


def circ(N: int, M: int) -> Dict:
    """Hub-free limit: a near-regular ring plus chords."""
    adj, cnt = ring_graph(N, M)
    degs = [len(a) for a in adj]
    return dict(ok=True, adj=adj, edges=cnt, N=N, M=M,
                hub_deg=max(degs), k_min=min(degs), k_max=max(degs),
                connected=_connected(adj))


def star(N: int) -> Dict:
    """Pure star (empty residual): K = N-1, feasible only when M >= 2N-3."""
    adj = _empty(N)
    for v in range(1, N):
        adj[0].append(v)
        adj[v].append(0)
    degs = [len(a) for a in adj]
    return dict(ok=True, adj=adj, edges=N - 1, N=N, M=N - 1, K=N - 1,
                hub=0, hub_deg=N - 1, k_min=1, k_max=N - 1, connected=True)


def _connected(adj: List[List[int]]) -> bool:
    n = len(adj)
    if n == 0:
        return True
    seen = [False] * n
    seen[0] = True
    st = [0]
    while st:
        u = st.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                st.append(v)
    return all(seen)


def bfs_efficiency(adj: List[List[int]]) -> float:
    """Global efficiency E = 1/(N(N-1)) * sum 1/d(s,t); nan when disconnected."""
    N = len(adj)
    tot = 0.0
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            du = dist[u]
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = du + 1
                    q.append(v)
        if head < N:
            return float("nan")
        tot += sum(1.0 / d for d in dist if d > 0)
    return tot / (N * (N - 1))


class DSU:
    __slots__ = ("p", "sz")

    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> Optional[Tuple[int, int]]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        sa, sb = self.sz[ra], self.sz[rb]
        self.p[rb] = ra
        self.sz[ra] = sa + sb
        return sa, sb


def giant_curve(adj: List[List[int]], order: Sequence[int]) -> Tuple[List[float], List[float]]:
    """Remove nodes in the given order and replay it backwards with union-find.

    Returns (S, chi) indexed by the number of surviving nodes.
    """
    N = len(adj)
    dsu = DSU(N)
    alive = [False] * N
    S = [0.0] * (N + 1)
    chi = [0.0] * (N + 1)
    sumsq = 0.0
    smax = 0
    n = 0
    for v in reversed(order):
        alive[v] = True
        n += 1
        sumsq += 1.0
        if smax < 1:
            smax = 1
        for u in adj[v]:
            if alive[u]:
                r = dsu.union(v, u)
                if r is not None:
                    a, b = r
                    sumsq += (a + b) ** 2 - a * a - b * b
                    if a + b > smax:
                        smax = a + b
        S[n] = smax / N
        chi[n] = (sumsq - smax * smax) / (n - smax) if n > smax else 0.0
    return S, chi


def _fc_from_S(Sbar: Sequence[float], N: int, floor: float) -> float:
    """Where the mean giant-component curve first drops below floor (linear interpolation)."""
    if Sbar[N] < floor:
        return 0.0
    for n in range(N - 1, 1, -1):
        if Sbar[n] < floor:
            s_hi, s_lo = Sbar[n + 1], Sbar[n]
            if s_hi - s_lo < 1e-15:
                return 1.0 - n / N
            t = (s_hi - floor) / (s_hi - s_lo)
            return 1.0 - (n + 1) / N + t / N
    return 1.0 - 1.0 / N


def fc_random(adj: List[List[int]], seeds: int = 24, floor: Optional[float] = None) -> Dict:
    """Random-failure threshold: average the giant-component curve over random permutations."""
    N = len(adj)
    if floor is None:
        floor = 1.0 / math.sqrt(N)
    acc = [0.0] * (N + 1)
    rng = random.Random(12345)
    for _ in range(seeds):
        order = list(range(N))
        rng.shuffle(order)
        S, _ = giant_curve(adj, order)
        for i in range(N + 1):
            acc[i] += S[i]
    Sbar = [a / seeds for a in acc]
    return dict(fc=_fc_from_S(Sbar, N, floor), S=Sbar)


def fc_targeted(adj: List[List[int]], floor: Optional[float] = None,
                seed: int = 7) -> Dict:
    """Targeted threshold: remove in descending initial degree.

    Equal-degree nodes follow a fixed seeded permutation rather than the node
    identifier, so the removal order is reproducible bit for bit (seed 7). The
    evaluation curves of the main experiments use the identifier order instead;
    this sweep only needs a per-graph scalar, not an adversarial permutation.
    """
    N = len(adj)
    if floor is None:
        floor = 1.0 / math.sqrt(N)
    rng = random.Random(seed)
    order = list(range(N))
    rng.shuffle(order)
    order.sort(key=lambda v: -len(adj[v]))
    S, chi = giant_curve(adj, order)
    return dict(fc=_fc_from_S(S, N, floor), S=S, chi=chi, order=order)


def measure(g: Dict, seeds: int = 24) -> Dict:
    """Measure E, f_c^rand, f_c^tar and R on one graph, next to the closed forms."""
    adj = g["adj"]
    N = len(adj)
    E = bfs_efficiency(adj)
    fr = fc_random(adj, seeds)
    ft = fc_targeted(adj)
    degs = [len(a) for a in adj]
    m1 = sum(degs) / N
    m2 = sum(d * d for d in degs) / N
    kappa = m2 / m1 if m1 > 0 else float("nan")
    fc_mf = 1.0 - 1.0 / (kappa - 1.0) if kappa > 1.0 else 0.0
    auc_r = sum(fr["S"][1:]) / N
    auc_t = sum(ft["S"][1:]) / N
    return dict(E=E, fc_rand=fr["fc"], fc_tar=ft["fc"],
                R=0.5 * (fr["fc"] + ft["fc"]), fc_mf=fc_mf,
                kappa=kappa, m1=m1, m2=m2,
                auc_rand=auc_r, auc_tar=auc_t,
                S_rand=fr["S"], S_tar=ft["S"], order=ft["order"])


def predict_from_moments(m1: float, m2: float, r: float, K: float) -> Dict:
    """Feed the first two degree moments into (7.64)-(7.66)."""
    fr = fc_random_moments(r, K, m1, m2)
    ft, br = fc_targeted_moments(r, K, m1, m2)
    if not (ft == ft):
        ft, br = r, -1
    return dict(fc_rand=fr, fc_tar=ft, branch=br, R=0.5 * (fr + ft))


def predict(adj: List[List[int]], r: float, K: float) -> Dict:
    """Graph -> degree moments -> closed forms."""
    N = len(adj)
    degs = [len(a) for a in adj]
    return predict_from_moments(sum(degs) / N,
                                sum(d * d for d in degs) / N, r, K)


HS = (2, 3, 4, 6, 8, 12, 16, 24, 32)


def designs_for(N: int, M: int, gseeds: int = 3) -> List[Tuple[str, Dict]]:
    """List the candidate designs for one ISP at its exact (N, M)."""
    import design_map as dz

    kbar = 2.0 * M / N
    out: List[Tuple[str, Dict]] = []
    Kcap = int(min(N - 1, M - (N - 2))) if M >= N - 1 else 0
    if Kcap < 1:
        return out

    def push(tag, g):
        if g.get("ok"):
            g["c"] = (g.get("K") or 0) / kbar
            out.append((tag, g))

    for s in range(gseeds):
        push(f"hub1(K={Kcap},c={Kcap / kbar:.1f})", hub1(N, M, Kcap, seed=7 + s))
    rec = dz.recommend(N, kbar, 0.5)
    if rec.get("status") in ("ok", "degenerate"):
        K5 = int(rec["K"])
        if K5 != Kcap:
            for s in range(gseeds):
                push(f"hub1(K={K5},c={K5 / kbar:.1f})[λ=.5]",
                     hub1(N, M, K5, seed=7 + s))
    for h in HS:
        if h < N - 2:
            for s in range(gseeds):
                push(f"flower(h={h})", flower(N, M, h, seed=7 + s))
    for s in range(gseeds):
        push("rand", random_conn(N, M, seed=7 + s))
    return out


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def family_of(tag: str) -> str:
    if tag.startswith("hub1"):
        return "hub1"
    if tag.startswith("flower"):
        return "flower"
    return "rand"


def evaluate(name: str, N: int, M: int, seeds: int = 16,
             gseeds: int = 3) -> List[Dict]:
    """Average several realisations of the same design; one sample jitters by about 0.03."""
    bucket: Dict[str, List[Dict]] = {}
    for tag, g in designs_for(N, M, gseeds):
        m = measure(g, seeds)
        m.update(design=tag, c=g.get("c"), K=g.get("K"), h=g.get("h"),
                 k_max=g["k_max"], k_min=g["k_min"], hub_deg=g["hub_deg"],
                 edges=g["edges"])
        bucket.setdefault(tag, []).append(m)

    rows = []
    for tag, ms in bucket.items():
        g0 = ms[0]
        if tag.startswith("hub1"):
            pr = predict_from_moments(_mean([m["m1"] for m in ms]),
                                      _mean([m["m2"] for m in ms]),
                                      1.0 / N, g0["hub_deg"])
        elif tag.startswith("flower"):
            pr = predict_from_moments(_mean([m["m1"] for m in ms]),
                                      _mean([m["m2"] for m in ms]),
                                      g0["h"] / N, g0["hub_deg"])
        else:
            pr = dict(fc_rand=float("nan"), fc_tar=float("nan"),
                      R=float("nan"), branch=0)
        rows.append(dict(isp=name, design=tag, N=N, M=M, kbar=2.0 * M / N,
                         c=g0["c"], K=g0["K"], h=g0["h"],
                         k_max=g0["k_max"], k_min=g0["k_min"],
                         hub_deg=g0["hub_deg"], edges=g0["edges"],
                         E=_mean([m["E"] for m in ms]),
                         fc_rand=_mean([m["fc_rand"] for m in ms]),
                         fc_tar=_mean([m["fc_tar"] for m in ms]),
                         auc_rand=_mean([m["auc_rand"] for m in ms]),
                         auc_tar=_mean([m["auc_tar"] for m in ms]),
                         fc_mf=_mean([m["fc_mf"] for m in ms]),
                         kappa=_mean([m["kappa"] for m in ms]),
                         pr_rand=pr["fc_rand"], pr_tar=pr["fc_tar"],
                         pr_R=pr["R"]))
        rows[-1]["R"] = 0.5 * (rows[-1]["fc_rand"] + rows[-1]["fc_tar"])
    return rows


def profile(N: int, M: int) -> Dict:
    import domain_map as dm

    kbar = 2.0 * M / N
    Kcap = int(min(N - 1, M - (N - 2)))
    st = dm.stage_of(N, kbar)
    w = dm.window(N, kbar)
    return dict(N=N, M=M, kbar=kbar, Kcap=Kcap, c_hi=Kcap / kbar,
                stage=st, E_dir=dm.E_dir(N, Kcap), E_cap=efficiency_ceiling(N),
                R_hi=w.get("R_hi"), R_lo=w.get("R_lo"), dR=w.get("dR"))


def table(rows: Sequence[Dict]) -> List[str]:
    L = [f"{'ISP':<18}{'<k>':>5}{'design':<22}{'E':>7}{'fcr':>7}{'fct':>7}"
         f"{'R':>7}{'auc_r':>7}{'auc_t':>7}"]
    cur = None
    for r in rows:
        if r["isp"] != cur:
            cur = r["isp"]
            L.append("-" * 88)
        L.append(f"{cur:<18}{r['kbar']:5.2f}{r['design']:<22}{r['E']:7.3f}"
                 f"{r['fc_rand']:7.3f}{r['fc_tar']:7.3f}{r['R']:7.3f}"
                 f"{r['auc_rand']:7.3f}{r['auc_tar']:7.3f}")
    return L


def summary(rows: Sequence[Dict]) -> List[str]:
    """Per ISP: single hub (budget-capped), best flower, random graph."""
    L = [f"{'ISP':<18}{'<k>':>5} | {'single hub':^22} | {'best flower':^26} | "
         f"{'random':^16} | gain"]
    L.append(f"{'':<18}{'':>5} | {'E':>7}{'R':>7}{'fct':>7} | "
             f"{'h*':>3}{'E':>7}{'R':>7}{'fct':>7} | {'E':>7}{'R':>7} | "
             f"{'ΔE':>6}{'ΔR':>7}")
    L.append("-" * 108)
    for name in dict.fromkeys(r["isp"] for r in rows):
        rs = [r for r in rows if r["isp"] == name]
        hub = next((r for r in rs if r["design"].startswith("hub1(K=")
                    and "λ" not in r["design"]), None)
        fl = [r for r in rs if r["design"].startswith("flower")]
        rd = next((r for r in rs if r["design"] == "rand"), None)
        if not hub or not fl:
            continue
        bf = max(fl, key=lambda r: (r["E"], r["R"]))
        dE = (bf["E"] - hub["E"]) / hub["E"] if hub["E"] else float("nan")
        L.append(f"{name:<18}{hub['kbar']:5.2f} | {hub['E']:7.3f}{hub['R']:7.3f}"
                 f"{hub['fc_tar']:7.3f} | {bf['h']:3d}{bf['E']:7.3f}"
                 f"{bf['R']:7.3f}{bf['fc_tar']:7.3f} | "
                 f"{rd['E']:7.3f}{rd['R']:7.3f} | {100 * dE:+5.0f}%"
                 f"{bf['R'] - hub['R']:+7.3f}")
    return L


def h_star(N: int, M: int, seeds: int = 12) -> List[Dict]:
    """Sweep h over the flower family on a single realisation."""
    out = []
    Kcap = int(min(N - 1, M - (N - 2)))
    for h in (1,) + HS:
        g = hub1(N, M, Kcap) if h == 1 else flower(N, M, h)
        if not g.get("ok"):
            continue
        m = measure(g, seeds)
        out.append(dict(h=h, E=m["E"], fc_rand=m["fc_rand"], fc_tar=m["fc_tar"],
                        R=m["R"], auc=0.5 * (m["auc_rand"] + m["auc_tar"])))
    return out


def err_stats(rows: Sequence[Dict]) -> List[str]:
    """Deviation of the closed forms (7.64)-(7.66) from the measurements."""
    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")

    L = ["closed forms (7.64)-(7.66) vs measurement on the same graph:"]
    for fam, label in (("hub1", "single hub"), ("flower", "flower")):
        rs = [r for r in rows if family_of(r["design"]) == fam
              and r["pr_R"] == r["pr_R"] and r["pr_R"] <= 1.0]
        if not rs:
            continue
        L.append(f"  {label:<5} n={len(rs):3d}  "
                 f"median d_fcr {med([r['pr_rand'] - r['fc_rand'] for r in rs]):+.3f}  "
                 f"median d_fct {med([r['pr_tar'] - r['fc_tar'] for r in rs]):+.3f}  "
                 f"median dR {med([r['pr_R'] - r['R'] for r in rs]):+.3f}")
    rs = [r for r in rows if r["design"] == "rand"]
    if rs:
        L.append(f"  random   n={len(rs):3d}  "
                 f"kappa criterion 1-1/(kappa-1) vs measured: "
                 f"median {med([r['fc_mf'] - r['fc_rand'] for r in rs]):+.3f}")
    return L


def aggregate(rows: Sequence[Dict]) -> List[str]:
    """Compare the families by density, i.e. how much room the c knob still has."""
    L = []
    lo = [r for r in rows if r["kbar"] < 3.0]
    hi = [r for r in rows if r["kbar"] >= 3.0]
    L.append(f"density split: {len(set(r['isp'] for r in lo))} ISPs with <k> < 3, "
             f"{len(set(r['isp'] for r in hi))} with <k> >= 3")
    for grp, label in ((lo, "<k><3"), (hi, "<k>>=3")):
        if not grp:
            continue
        dE, dR, e5, r5 = [], [], [], []
        for name in dict.fromkeys(r["isp"] for r in grp):
            rs = [r for r in grp if r["isp"] == name]
            hub = next((r for r in rs if r["design"].startswith("hub1(K=")
                        and "λ" not in r["design"]), None)
            fl = [r for r in rs if r["design"].startswith("flower")]
            rd = next((r for r in rs if r["design"] == "rand"), None)
            if not (hub and fl):
                continue
            bf = max(fl, key=lambda r: (r["E"], r["R"]))
            dE.append(bf["E"] / hub["E"] - 1 if hub["E"] else float("nan"))
            dR.append(bf["R"] - hub["R"])
            e5.append(rd["E"] / hub["E"] - 1 if rd and hub["E"] else float("nan"))
            r5.append(rd["R"] - hub["R"] if rd else float("nan"))
        L.append(f"  {label:<10} flower h=2 vs single hub: dE {_mean(dE) * 100:+5.0f}%   "
                 f"dR {_mean(dR):+.3f}   |  random vs single hub: "
                 f"ΔE {_mean(e5) * 100:+5.0f}%   ΔR {_mean(r5):+.3f}")
    return L


def validate() -> List[str]:
    """Check the closed forms against the standard bimodal configuration model."""
    L = ["closed-form check on the bimodal configuration model (one Monte Carlo each):"]
    L.append(f"  {'setting':<28}{'fcr num':>8}{'fcr cf':>8}{'kappa':>8}"
             f"{'fct num':>8}{'fct cf':>8}")
    for N, r, K, kmin in ((2000, 0.10, 20, 2), (2000, 0.50, 10, 2),
                          (1000, 0.02, 40, 2)):
        adj = bimodal_cm(N, r, K, kmin, seed=1)
        if adj is None:
            continue
        m = measure(dict(adj=adj), seeds=16)
        pr = predict(adj, r, K)
        L.append(f"  N={N} r={r:.2f} K={K} k_min={kmin}"
                 f"{'':<6}{m['fc_rand']:8.3f}{pr['fc_rand']:8.3f}"
                 f"{m['fc_mf']:8.3f}{m['fc_tar']:8.3f}{pr['fc_tar']:8.3f}")
    st = star(200)
    m = measure(st, seeds=40)
    L.append(f"  {'star N=200':<28}{m['fc_rand']:8.3f}{'1.000':>8}"
             f"{m['fc_mf']:8.3f}{m['fc_tar']:8.3f}{'0.005':>8}")
    L.append("  -> the closed forms overestimate; the kappa criterion is better when r is not small")
    return L


def selftest() -> int:
    fails = 0

    def chk(msg, cond, extra=""):
        nonlocal fails
        if not cond:
            fails += 1
            print(f"  FAIL {msg}  {extra}")

    for n, e in ((10, 9), (10, 12), (50, 60), (196, 195), (196, 300)):
        adj, c = ring_graph(n, e)
        chk(f"ring({n},{e}) edge count", c == e, str(c))
        chk(f"ring({n},{e}) connected", _connected(adj))
    for N, M, K in ((197, 243, 48), (153, 177, 26), (100, 200, 46), (100, 200, 99)):
        g = hub1(N, M, K)
        chk(f"hub1({N},{M},{K}) edge count", g["ok"] and g["edges"] == M, str(g.get("edges")))
        chk(f"hub1({N},{M},{K}) hub degree", g["hub_deg"] == K, str(g.get("hub_deg")))
        chk(f"hub1({N},{M},{K}) connected", g["connected"])
    for N, M, h in ((197, 243, 8), (153, 177, 8), (754, 899, 16)):
        g = flower(N, M, h)
        if not g["ok"]:
            chk(f"flower({N},{M},{h})", False, g["why"])
            continue
        chk(f"flower({N},{M},{h}) edge count", g["edges"] == M, str(g["edges"]))
        chk(f"flower({N},{M},{h}) connected", g["connected"])

    N = 100
    st = star(N)
    chk("star: E = (N+2)/2N", abs(bfs_efficiency(st["adj"]) - (N + 2) / (2 * N)) < 1e-12,
        str(bfs_efficiency(st["adj"])))

    st200 = star(200)
    ms = measure(st200, seeds=40)
    chk("star: high f_c^rand", ms["fc_rand"] > 0.65,
        f"{ms['fc_rand']:.3f}")
    chk("star: f_c^tar around 1/N", ms["fc_tar"] < 0.03, f"{ms['fc_tar']:.3f}")
    chk("star: R in [0.35, 0.55]", 0.35 < ms["R"] < 0.55, f"{ms['R']:.3f}")

    er = random_conn(200, 400, seed=3)
    me = measure(er, seeds=40)
    chk("random graph, <k>=4: f_c^rand around 0.75",
        0.62 < me["fc_rand"] < 0.84, f"{me['fc_rand']:.3f}")
    chk("targeted attack much stronger than random", me["fc_tar"] < me["fc_rand"] - 0.15,
        f"{me['fc_tar']:.3f} vs {me['fc_rand']:.3f}")

    pr = predict(st200["adj"], 1.0 / 200, 199)
    chk("star: closed-form R around 0.5", abs(pr["R"] - 0.5) < 0.02, f"{pr['R']:.3f}")

    for tag, g in designs_for(197, 243, gseeds=1):
        E = bfs_efficiency(g["adj"])
        chk(f"197/243 {tag} E∈(0,1]", 0 < E <= 1.0, str(E))

    print(f"selftest: {'OK' if fails == 0 else f'{fails} FAILED'}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description="build and compare designs at the real ISP size and budget")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", action="store_true", help="per-design detail")
    ap.add_argument("--summary", action="store_true", help="single hub vs best flower vs random")
    ap.add_argument("--profile", action="store_true", help="geometric profile")
    ap.add_argument("--hsweep", action="store_true", help="sweep h")
    ap.add_argument("--validate", action="store_true", help="check the closed forms")
    ap.add_argument("--isp", default=None)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--gseeds", type=int, default=3)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(1 if selftest() else 0)

    if a.validate:
        print("\n".join(validate()))
        return

    if a.hsweep:
        for name, N, M, kind in [x for x in ISP
                                 if a.isp is None or x[0].upper() == a.isp.upper()]:
            print(f"=== {name} N={N} M={M} <k>={2 * M / N:.2f}")
            for d in h_star(N, M, a.seeds):
                print(f"  h={d['h']:2d} E={d['E']:.3f} fcr={d['fc_rand']:.3f} "
                      f"fct={d['fc_tar']:.3f} R={d['R']:.3f} auc={d['auc']:.3f}")
        return

    sel = [x for x in ISP if a.isp is None or x[0].upper() == a.isp.upper()]
    rows: List[Dict] = []
    for name, N, M, kind in sel:
        rows.extend(evaluate(name, N, M, a.seeds, a.gseeds))

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"written {a.csv} ({len(rows)} rows)")

    if a.profile:
        print("\n".join(profile_lines(rows)))
        print()
    if a.table:
        print("\n".join(table(rows)))
        print()
    if a.summary or not (a.profile or a.table):
        print("\n".join(summary(rows)))
        print()
        print("\n".join(aggregate(rows)))
        print()
        print("\n".join(err_stats(rows)))


def profile_lines(rows: Sequence[Dict]) -> List[str]:
    L = [f"{'ISP':<18}{'N':>5}{'M':>5}{'<k>':>6}{'Kcap':>5}{'c_hi':>6}"
         f"{'stage':>6}{'E_dir':>7}{'R_lo':>7}{'R_hi':>7}{'dR':>7}"]
    for name in dict.fromkeys(r["isp"] for r in rows):
        r = next(x for x in rows if x["isp"] == name)
        p = profile(r["N"], r["M"])
        L.append(f"{name:<18}{p['N']:5d}{p['M']:5d}{p['kbar']:6.2f}"
                 f"{p['Kcap']:5d}{p['c_hi']:6.1f}{str(p['stage']):>6}"
                 f"{p['E_dir']:7.3f}{p['R_lo']:7.3f}{p['R_hi']:7.3f}{p['dR']:7.3f}")
    return L


if __name__ == "__main__":
    main()
