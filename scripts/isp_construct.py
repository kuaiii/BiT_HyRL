#!/usr/bin/env python3
"""在真实 ISP 的 (N, M) 上"真正把图造出来"，实测 E 与渗流 f_c，和闭式比对。

为什么要造图：手里只有每张 ISP 拓扑的 (N, M)（Zoo 的 PoP 级与光网络物理层），
没有边表，所以无法直接测原图。但设计问题问的正是"在这个规模、这个预算下，
应该造哪一种结构"，因此可以把同 (N, M) 的四种族都造出来，用同一个度量比较：

    hub1(K)   单枢纽 + 近正则残余（论文族，c = K/<k>）
    flower(h)  h 个枢纽 + 叶子各挂一个（低密度族的替代）
    circ       近正则环 + 弦（"无枢纽"极限）
    rand       随机连通图（同一预算下的最大熵零模型）

度量：
    E      = 1/(N(N-1)) Σ 1/d(s,t)          BFS 精确算
    f_c    渗流阈值                         反向并查集 + 平均巨分量曲线
    R      = ½(f_c^rand + f_c^tar)          与论文 (7.64)-(7.66) 同一口径

用法：
    python isp_construct.py --selftest
    python isp_construct.py --table --csv isp_design.csv
    python isp_construct.py --isp COGENTCO --detail
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from fc_robust import fc_random_moments, fc_targeted_moments
from param_window import efficiency_ceiling
import design_map as dz
import domain_map as dm

EPS = 1e-12

# ------------------------------------------------------------------ ISP 表
# (name, N, M, kind)   kind: pop = PoP 级 ISP, opt = 光网络物理层
ISP: List[Tuple[str, int, int, str]] = [
    # ---- Topology Zoo, PoP 级 ISP ----
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
    # ---- 光网络物理拓扑（含业务多纤）----
    ("INTEROUTE-PHYS", 110, 158, "opt"),
    ("DELTACOM-PHYS", 113, 183, "opt"),
    ("KENTUCKY-DATALINK", 754, 899, "opt"),
    ("BESTEL-PHYS", 84, 101, "opt"),
    ("OTEGLOBE-PHYS", 93, 106, "opt"),
    ("US-SIGNAL-PHYS", 63, 79, "opt"),
    ("VIATEL-PHYS", 88, 92, "opt"),
]

SRC_ZH = {"pop": "PoP 级 ISP", "opt": "光网络物理层"}


# ------------------------------------------------------------------ 造图
def _spread(n: int, r: int) -> List[int]:
    """在 [0, n) 上均匀取 r 个互不相同的下标。"""
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
    """在 [base, base+n) 内加 count 条"等距弦"，不要求连通。返回实际条数。"""
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
    """n 个点、e 条边的连通近正则图：环 + 等距弦。返回 (adj, 实际边数)。

    度数差 ≤ 1，且弦沿环均匀铺开，不会挤在一段局部。
    """
    if e < n - 1:
        raise ValueError(f"ring_graph: {n} 点需要至少 {n - 1} 条边，给的是 {e}")
    if adj is None:
        adj = _empty(base + n)
    if seen is None:
        seen = set()
    cnt = 0
    for i in range(n - 1):                       # 路径
        if _add(adj, seen, base + i, base + i + 1):
            cnt += 1
    if cnt < e and n > 2:                        # 补一条成环
        if _add(adj, seen, base + n - 1, base):
            cnt += 1
    cnt += extra_edges(n, e - cnt, base, adj, seen)
    if cnt < e:                                  # 兜底：任意未用点对
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
    """单枢纽族：hub 度 = K，其余 N-1 个点构成随机连通残余。

    残余用 G(n,m) 的随机实现而不是环格点：环格点的直径是 O(n)，会把 E 压低一个
    量级，那是"残余选得差"而不是"族不好"。
    """
    res_n, res_e = N - 1, M - K
    if K < 1 or K > N - 1 or res_e < res_n - 1:
        return dict(ok=False, why=f"K={K} 不可行（res_e={res_e} < {res_n - 1}）")
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
    """花形族：h 个枢纽组成核心（近正则），其余叶子各挂一个枢纽。"""
    L = N - h
    e_core = M - L
    if L <= 0 or e_core < h - 1:
        return dict(ok=False, why=f"h={h}: 核心预算不足 (e_core={e_core})")
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
    if extra_leaf:                                # 预算有剩，只能在叶间补弦
        extra_edges(L, extra_leaf, h, adj, seen)
    degs = [len(a) for a in adj]
    return dict(ok=True, adj=adj, edges=sum(degs) // 2, N=N, M=M, h=h,
                hub_deg=sum(degs[:h]) / h, k_min=min(degs[h:]), k_max=max(degs),
                connected=_connected(adj))


def random_conn(N: int, M: int, seed: int = 0) -> Dict:
    """随机连通图：随机递归树 + 随机补边，边数精确等于 M。"""
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
    """标准双峰配置模型：rN 个度为 K 的枢纽 + 其余度为 kmin。用于校验闭式本身。"""
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
    """无枢纽极限：近正则环 + 弦。"""
    adj, cnt = ring_graph(N, M)
    degs = [len(a) for a in adj]
    return dict(ok=True, adj=adj, edges=cnt, N=N, M=M,
                hub_deg=max(degs), k_min=min(degs), k_max=max(degs),
                connected=_connected(adj))


def star(N: int) -> Dict:
    """纯星形（残余无边）：K = N-1。只有 M ≥ 2N-3 时才落在可行域内。"""
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


# ------------------------------------------------------------------ 度量
def bfs_efficiency(adj: List[List[int]]) -> float:
    """E = 1/(N(N-1)) Σ_{s≠t} 1/d(s,t)；不连通返回 nan。"""
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
    """按 order 依次移除，反向并查集回放。返回按"存活节点数 n"索引的 (S, chi)。

    S(n) = s_max/N（相对**原始**规模，即标准 P∞ 口径），
    chi(n) = (Σs² − s_max²)/(n − s_max)。
    注意：S 若按 s_max/n 定义，小 n 段会有 1/n 的地板，f_c 永远测不出来。
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
    """巨分量均值曲线首次跌破 floor 的位置（线性内插）。

    n 从 N 递减（=f 从 0 递增），S 单调降，找第一个 S<floor 的 n。
    """
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
    """随机失效阈值：随机置换取平均巨分量曲线，再取跌破 1/√N 的位置。"""
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
    """定向失效阈值：按初始度降序移除，同度随机打破（否则近正则图会退化成按编号割弧）。"""
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
    """对一张造出来的图测 E、f_c^rand、f_c^tar、R，并给出两种闭式预测。"""
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
    """把度的一二阶矩喂进 (7.64)-(7.66)。"""
    fr = fc_random_moments(r, K, m1, m2)
    ft, br = fc_targeted_moments(r, K, m1, m2)
    if not (ft == ft):                     # k_min=1 时 (7.65)/(7.66) 奇点 → 结构极限 ρ_max
        ft, br = r, -1
    return dict(fc_rand=fr, fc_tar=ft, branch=br, R=0.5 * (fr + ft))


def predict(adj: List[List[int]], r: float, K: float) -> Dict:
    """图 → 度矩 → 闭式。"""
    N = len(adj)
    degs = [len(a) for a in adj]
    return predict_from_moments(sum(degs) / N,
                                sum(d * d for d in degs) / N, r, K)


# ------------------------------------------------------------------ 设计
HS = (2, 3, 4, 6, 8, 12, 16, 24, 32)


def designs_for(N: int, M: int, gseeds: int = 3) -> List[Tuple[str, Dict]]:
    """给一张 ISP 的 (N, M)，列出要比较的候选设计（严格同预算）。"""
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
    """同一设计的多张实现取平均（残余与补边是随机的，单张会有 ±0.03 抖动）。"""
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
    kbar = 2.0 * M / N
    Kcap = int(min(N - 1, M - (N - 2)))
    st = dm.stage_of(N, kbar)
    w = dm.window(N, kbar)
    return dict(N=N, M=M, kbar=kbar, Kcap=Kcap, c_hi=Kcap / kbar,
                stage=st, E_dir=dm.E_dir(N, Kcap), E_cap=efficiency_ceiling(N),
                R_hi=w.get("R_hi"), R_lo=w.get("R_lo"), dR=w.get("dR"))


# ------------------------------------------------------------------ 打印
def table(rows: Sequence[Dict]) -> List[str]:
    L = [f"{'ISP':<18}{'<k>':>5}{'设计':<22}{'E':>7}{'fcr':>7}{'fct':>7}"
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
    """每个 ISP：单枢纽(cap) vs 最优花形 vs 随机图。"""
    L = [f"{'ISP':<18}{'<k>':>5} | {'单枢纽':^22} | {'最优花形':^26} | "
         f"{'随机图':^16} | 增益"]
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
    """花形族 h 扫描（单张实现，用于看趋势）。"""
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
    """闭式 (7.64)-(7.66) 对实测的偏差。"""
    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")

    L = ["闭式 (7.64)-(7.66) vs 实测（同一张图，把实测度矩喂进公式）："]
    for fam, label in (("hub1", "单枢纽"), ("flower", "花形")):
        rs = [r for r in rows if family_of(r["design"]) == fam
              and r["pr_R"] == r["pr_R"] and r["pr_R"] <= 1.0]
        if not rs:
            continue
        L.append(f"  {label:<5} n={len(rs):3d}  "
                 f"Δfcr 中位 {med([r['pr_rand'] - r['fc_rand'] for r in rs]):+.3f}  "
                 f"Δfct 中位 {med([r['pr_tar'] - r['fc_tar'] for r in rs]):+.3f}  "
                 f"ΔR 中位 {med([r['pr_R'] - r['R'] for r in rs]):+.3f}")
    rs = [r for r in rows if r["design"] == "rand"]
    if rs:
        L.append(f"  随机图   n={len(rs):3d}  "
                 f"κ判据 1−1/(κ−1) vs 实测："
                 f"中位 {med([r['fc_mf'] - r['fc_rand'] for r in rs]):+.3f}")
    return L


def aggregate(rows: Sequence[Dict]) -> List[str]:
    """按密度分组统计族间差异，回答"c 拨盘在 ISP 上到底还剩多少空间"。"""
    L = []
    lo = [r for r in rows if r["kbar"] < 3.0]
    hi = [r for r in rows if r["kbar"] >= 3.0]
    L.append(f"密度分组：<k><3 的 ISP 有 {len(set(r['isp'] for r in lo))} 张，"
             f">=3 的有 {len(set(r['isp'] for r in hi))} 张")
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
        L.append(f"  {label:<7} 花形 h=2 vs 单枢纽： ΔE {_mean(dE) * 100:+5.0f}%   "
                 f"ΔR {_mean(dR):+.3f}   |  随机图 vs 单枢纽： "
                 f"ΔE {_mean(e5) * 100:+5.0f}%   ΔR {_mean(r5):+.3f}")
    return L


def validate() -> List[str]:
    """闭式本身准不准：在标准双峰配置模型上对答案。"""
    L = ["闭式校验（标准双峰配置模型，随机/定向各一次蒙特卡洛）："]
    L.append(f"  {'配置':<28}{'fcr测':>8}{'fcr式':>8}{'κ判据':>8}"
             f"{'fct测':>8}{'fct式':>8}")
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
    L.append(f"  {'星形 N=200':<28}{m['fc_rand']:8.3f}{'1.000':>8}"
             f"{m['fc_mf']:8.3f}{m['fc_tar']:8.3f}{'0.005':>8}")
    L.append("  → 闭式在两处都偏高：r 很小时偏高最多；κ 判据在 r 不小的时候更准")
    return L


# ------------------------------------------------------------------ 自检
def selftest() -> int:
    fails = 0

    def chk(msg, cond, extra=""):
        nonlocal fails
        if not cond:
            fails += 1
            print(f"  FAIL {msg}  {extra}")

    # 1) 造图：边数精确、连通
    for n, e in ((10, 9), (10, 12), (50, 60), (196, 195), (196, 300)):
        adj, c = ring_graph(n, e)
        chk(f"ring({n},{e}) 边数", c == e, str(c))
        chk(f"ring({n},{e}) 连通", _connected(adj))
    for N, M, K in ((197, 243, 48), (153, 177, 26), (100, 200, 46), (100, 200, 99)):
        g = hub1(N, M, K)
        chk(f"hub1({N},{M},{K}) 边数", g["ok"] and g["edges"] == M, str(g.get("edges")))
        chk(f"hub1({N},{M},{K}) hub 度", g["hub_deg"] == K, str(g.get("hub_deg")))
        chk(f"hub1({N},{M},{K}) 连通", g["connected"])
    for N, M, h in ((197, 243, 8), (153, 177, 8), (754, 899, 16)):
        g = flower(N, M, h)
        if not g["ok"]:
            chk(f"flower({N},{M},{h})", False, g["why"])
            continue
        chk(f"flower({N},{M},{h}) 边数", g["edges"] == M, str(g["edges"]))
        chk(f"flower({N},{M},{h}) 连通", g["connected"])

    # 2) E 的解析锚点：星形 = (N+2)/2N
    N = 100
    st = star(N)
    chk("星形 E = (N+2)/2N", abs(bfs_efficiency(st["adj"]) - (N + 2) / (2 * N)) < 1e-12,
        str(bfs_efficiency(st["adj"])))

    # 3) 渗流测量器在已知答案上校准
    st200 = star(200)
    ms = measure(st200, seeds=40)
    chk("星形 f_c^rand 高（随机攻击几乎打不烂）", ms["fc_rand"] > 0.65,
        f"{ms['fc_rand']:.3f}")
    chk("星形 f_c^tar ≈ 1/N（一枪毙命）", ms["fc_tar"] < 0.03, f"{ms['fc_tar']:.3f}")
    chk("星形 R 落在 [0.35,0.55]", 0.35 < ms["R"] < 0.55, f"{ms['R']:.3f}")

    er = random_conn(200, 400, seed=3)
    me = measure(er, seeds=40)
    chk("随机图 <k>=4 的 f_c^rand ≈ 1−1/<k>=0.75",
        0.62 < me["fc_rand"] < 0.84, f"{me['fc_rand']:.3f}")
    chk("随机图上定向比随机有效得多", me["fc_tar"] < me["fc_rand"] - 0.15,
        f"{me['fc_tar']:.3f} vs {me['fc_rand']:.3f}")

    # 4) 星形闭式：k_min=1 是奇点，结构极限 R_t→ρ_max，故 R→0.5
    pr = predict(st200["adj"], 1.0 / 200, 199)
    chk("星形闭式 R ≈ 0.5（奇点取结构极限）", abs(pr["R"] - 0.5) < 0.02, f"{pr['R']:.3f}")

    # 5) 同一 (N,M) 下各族都能造出来，且 E 落在 (0,1]
    for tag, g in designs_for(197, 243, gseeds=1):
        E = bfs_efficiency(g["adj"])
        chk(f"197/243 {tag} E∈(0,1]", 0 < E <= 1.0, str(E))

    print(f"selftest: {'OK' if fails == 0 else f'{fails} FAILED'}")
    return fails


# ------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="在 ISP 规模/预算上造图并对比")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", action="store_true", help="逐设计明细")
    ap.add_argument("--summary", action="store_true", help="单枢纽 vs 最优花形 vs 随机")
    ap.add_argument("--profile", action="store_true", help="几何画像")
    ap.add_argument("--hsweep", action="store_true", help="h 扫描")
    ap.add_argument("--validate", action="store_true", help="闭式自身校验")
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
         f"{'阶段':>6}{'E_dir':>7}{'R_lo':>7}{'R_hi':>7}{'ΔR':>7}"]
    for name in dict.fromkeys(r["isp"] for r in rows):
        r = next(x for x in rows if x["isp"] == name)
        p = profile(r["N"], r["M"])
        L.append(f"{name:<18}{p['N']:5d}{p['M']:5d}{p['kbar']:6.2f}"
                 f"{p['Kcap']:5d}{p['c_hi']:6.1f}{str(p['stage']):>6}"
                 f"{p['E_dir']:7.3f}{p['R_lo']:7.3f}{p['R_hi']:7.3f}{p['dR']:7.3f}")
    return L


if __name__ == "__main__":
    main()
