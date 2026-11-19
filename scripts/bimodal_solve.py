"""bimodal_solve.py —— 只用双峰度分布建模与求解。

    p_k = (1 - rho_h) * delta(k - k_min) + rho_h * delta(k - k_max)

不引入任何其它族。所谓"花形"只是这张表里 rho_h = h/N、k_min = 1 的那几行，
只要度数只取两个值就写在同一个参数表里。

三件事：
  1) params(N, M)   枚举所有"整数度、恰好可实现"的 (h, k_max, k_min)
  2) build(...)     按 Havel-Hakimi 造简单图，再做保度随机交换（度数严格不变）
  3) solve(...)     逐参数实测 E 与 R，取 Pareto 前沿，再按 lambda 选点

命令行：
    python bimodal_solve.py --N 1000 --kbar 3 --params
    python bimodal_solve.py --N 1000 --kbar 3 --trace --h 2 --kmin 2
    python bimodal_solve.py --N 1000 --kbar 3 --solve --lam 0.5
    python bimodal_solve.py --selftest
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from isp_construct import bfs_efficiency, fc_random, fc_targeted, measure

# ---------------------------------------------------------------- 参数枚举


def graphical(N: int, h: int, kmax: int, kmin: int) -> Tuple[bool, str]:
    """两点度序列 (kmax x h, kmin x (N-h)) 能否由简单图实现（Erdos-Gallai 充要）。

    对两点序列，真正会失效的只有 k = h 那一行（枢纽的桩数超过"叶桩 + 枢纽内部边"），
    但这里仍走完整 EG，代价 O(N)，换来的是不会漏判。
    """
    if h < 1 or h >= N:
        return False, "h 越界"
    if kmin < 1:
        return False, "k_min < 1"
    if kmax > N - 1:
        return False, f"k_max={kmax} > N-1={N - 1}"
    if kmax < kmin:
        return False, "k_max < k_min"
    S = h * kmax + (N - h) * kmin
    if S % 2:
        return False, f"度数和 {S} 为奇数"
    for k in range(1, N + 1):
        left = k * kmax if k <= h else h * kmax + (k - h) * kmin
        na = max(0, h - k)                     # 前 k 个之后的 kmax 节点数
        nb = (N - k) - na                      # 前 k 个之后的 kmin 节点数
        right = k * (k - 1) + na * min(kmax, k) + nb * min(kmin, k)
        if left > right:
            return False, f"EG 在 k={k} 失效（{left} > {right}）"
    return True, "ok"


def params(N: int, M: int, strict: bool = True) -> List[Dict]:
    """枚举所有整数可实现的双峰参数 (h, k_max, k_min)。

    strict=True 时丢掉 k_max == k_min 的退化点（那是单峰正则图，c=1）。
    """
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
            max_hh = h * (h - 1)                     # 枢纽内部最多能吃掉这么多桩
            out.append(dict(h=h, kmax=kmax, kmin=kmin, rho=h / N,
                            rho_min=1 - h / N, edges=M, kbar=kbar,
                            c=kmax / kbar,
                            hub_stubs=hub_stubs, leaf_stubs=leaf_stubs,
                            hub_slack=max_hh + leaf_stubs - hub_stubs,
                            hub_hub_max=max_hh // 2))
    out.sort(key=lambda d: (d["kmin"], -d["kmax"]))
    return out


def min_h(N: int, M: int, kmin: int) -> Optional[int]:
    """给定 k_min，能满足可实现性的最小枢纽数（没有则 None）。"""
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


# ---------------------------------------------------------------- 构造


def _havel_hakimi(N: int, deg: Sequence[int]) -> Optional[List[set]]:
    """确定性造一张简单图（EG 成立则必成功）。"""
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
    """保度双边交换：随机化结构，每个节点的度一点不变。"""
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
    """Havel-Hakimi 的边表版本（序列可图则必成功）。"""
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
    """按双峰度序列直接实现三类边（枢纽-枢纽 / 枢纽-叶 / 叶-叶）。

    先定 lh（每个叶子至少一条边连到枢纽，否则整张图必然不连通），
    剩下的枢纽桩在枢纽内部配对，剩下的叶桩在叶之间配对。
    """
    ok, why = graphical(N, h, kmax, kmin)
    if not ok:
        return dict(ok=False, why=why)
    L = N - h
    hub_stubs = h * kmax
    leaf_stubs = L * kmin

    lh = L if (hub_stubs - L) % 2 == 0 else L + 1      # 与枢纽桩同奇偶
    if lh > leaf_stubs:
        lh = leaf_stubs if (hub_stubs - leaf_stubs) % 2 == 0 else leaf_stubs - 1
    hh = (hub_stubs - lh) // 2
    if hh > h * (h - 1) // 2:                          # 枢纽内部装不下
        hh = h * (h - 1) // 2
        lh = hub_stubs - 2 * hh
    ll = (leaf_stubs - lh) // 2
    if min(hh, lh, ll) < 0 or hh + lh + ll != (hub_stubs + leaf_stubs) // 2:
        return dict(ok=False, why="三类边计数不自洽")

    rng = random.Random(seed)
    adj = [set() for _ in range(N)]

    def add(u: int, v: int) -> bool:
        if u == v or v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    # 1) 枢纽内部：先定每枢纽在枢纽子图里的度（尽量均匀），再用 HH 连
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
        return dict(ok=False, why="枢纽子图度序列不可图")
    for u, v in got_hh:
        add(u, v)

    # 2) 枢纽-叶：二部图 Havel-Hakimi（枢纽按剩余度、叶子按剩余需求配对）
    cap = [kmax - len(adj[i]) for i in range(h)]
    leaves = list(range(h, N))
    rng.shuffle(leaves)
    need = {v: 1 for v in leaves}                  # 先保证每叶 1 条
    done = 0

    def assign(v: int) -> bool:
        """给叶子 v 找一条枢纽边：在还有容量的枢纽里挑剩余容量最大的。"""
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

    for v in leaves:                               # 第一阶段：每叶 1 条
        if not assign(v):
            return dict(ok=False, why="枢纽容量不足以覆盖所有叶子")
    done = L
    base_cap, base_adj = cap[:], [set(x) for x in adj]
    for attempt in range(40):                      # 第二阶段：补到 k_min
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
        return dict(ok=False, why=f"枢纽-叶只连到 {lh - (lh - done)}/{lh}")

    # 3) 叶-叶：对叶子子图单独跑一次 HH（度数 = k_min 减去已连的枢纽边）
    ldeg = [need[v] for v in leaves]
    if sum(ldeg):
        sub = _hh_edges(len(leaves), ldeg)
        if sub is None:
            return dict(ok=False, why="叶子子图度序列不可图")
        for a, b in sub:
            if not add(leaves[a], leaves[b]):
                return dict(ok=False, why="叶-叶复用同一条边")
        # 随机化叶子子图
        _randomize(adj, swaps_factor * (sum(ldeg) // 2), random.Random(seed * 53 + 11))

    # 4) 连通性
    if not _connected(adj):
        return dict(ok=False, why="不连通")

    degs = [len(a) for a in adj]
    if set(degs) != {kmax, kmin}:
        return dict(ok=False, why=f"度数被破坏：{sorted(set(degs))}")
    keep = [set(x) for x in adj]
    _randomize(adj, swaps_factor * (sum(degs) // 2), random.Random(seed * 131 + 7))
    if not _connected(adj):                        # 双边交换可能拆断桥
        adj = keep
    lst = [sorted(x) for x in adj]
    return dict(ok=True, adj=lst, N=N, h=h, kmax=kmax, kmin=kmin,
                rho=h / N, edges=sum(degs) // 2, connected=True, attempts=1,
                mix=dict(hh=hh, lh=lh, ll=ll))


# ------------------------------------------------- (rho_max, k_max) 参数化


def design(n: int, kbar: float, nmax: int, kmax: int) -> Dict:
    """参数化 (n_max, k_max) → 整数度序列。对应 优化目标函数.md 的 (F1)-(F4)。

    与 params() 的区别：这里允许 k_min 是**实数**（由均值守恒解出），
    再把它落成 floor/ceil 两个整数档。这样 n_max=1 才可用——
    params() 强制 k_min 为整数，会把单枢纽整片砍掉（那不是图不存在，是度分布被卡死）。

    k_eff = k_min − rho*k_max/(1-rho) = 2*k_min − kbar/(1-rho) 是对向攻击的结构判据：
    撤掉全部枢纽后叶层剩 k_eff 度，k_eff < 2 时叶层不成巨分量，f_c^tar ≈ 0。
    """
    rho = nmax / n
    # ⟨k⟩ 通常由实测的 2m/n 四舍五入而来，所以 2m=n⟨k⟩ 一般不是整数。按最近的
    # 整边数吸附（2m 必为偶数），再把 ⟨k⟩ 回代成吸附后的值——否则 n=149、
    # ⟨k⟩=2.201342 这种真实输入会被整盘拒绝。
    two_m = n * kbar
    if abs(two_m - round(two_m)) > 1e-6:
        two_m = 2 * round(two_m / 2)
        kbar = two_m / n
    two_m = int(round(two_m))
    if not (1 <= nmax <= n - 1):
        return dict(ok=False, why="n_max 越界")
    if not (kbar < kmax <= n - 1):
        return dict(ok=False, why=f"k_max={kmax} 须满足 <k>={kbar}<k_max<=n-1")
    kmin = (kbar - rho * kmax) / (1 - rho)
    if kmin < 1:
        return dict(ok=False, why=f"k_min={kmin:.3f}<1（枢纽吃不下这么多桩）")
    if nmax * kmax > nmax * (nmax - 1) + (n - nmax) * min(kmin, nmax):
        return dict(ok=False, why="Erdos-Gallai 失效")
    L = n - nmax
    lo = int(math.floor(kmin))
    rem = two_m - nmax * kmax - L * lo
    if not (0 <= rem <= L):
        return dict(ok=False, why=f"取整残差 {rem} 超出 [0,{L}]")
    degs = [kmax] * nmax + [lo + 1] * rem + [lo] * (L - rem)
    return dict(ok=True, kmin=kmin, rho=rho, nmax=nmax, kmax=kmax,
                degs=degs, n_hi=rem, k_lo=lo, two_m=two_m,
                k_eff=2 * kmin - kbar / (1 - rho))


def build_seq(n: int, degs: Sequence[int], seed: int = 0,
              swaps_factor: int = 6) -> Dict:
    """按任意整数度序列造连通简单图（Havel-Hakimi + 保度交换 + 连通性回滚）。"""
    adj = _havel_hakimi(n, list(degs))
    if adj is None:
        return dict(ok=False, why="序列不可图")
    rng = random.Random(seed)
    keep = [set(x) for x in adj]
    _randomize(adj, swaps_factor * sum(degs) // 2, rng)
    if not _connected(adj):
        adj = keep
    return dict(ok=True, adj=[sorted(x) for x in adj], n=n,
                edges=sum(degs) // 2, connected=True)


def edge_mix(N: int, h: int, kmax: int, kmin: int) -> Dict:
    """三型边：枢纽-枢纽 hh / 枢纽-叶 lh / 叶-叶 ll。

    规范构造：先保证每个叶子至少一条边连到枢纽（否则整张图不连通），
    剩下的枢纽桩在枢纽之间两两配对，剩下的叶桩在叶之间配对。
    """
    hub_stubs = h * kmax
    leaf_stubs = (N - h) * kmin
    M = (hub_stubs + leaf_stubs) // 2
    hh = min(h * (h - 1) // 2, max(0, hub_stubs - (N - h)) // 2)
    lh = hub_stubs - 2 * hh
    if lh > leaf_stubs:                        # 枢纽内部边不够吃，只能多连叶
        hh = max(0, h * (h - 1) // 2)
        lh = hub_stubs - 2 * hh
    ll = (leaf_stubs - lh) // 2
    return dict(hh=hh, lh=lh, ll=ll, total=hh + lh + ll, M=M,
                ok=(hh + lh + ll == M and lh >= 0 and ll >= 0))


# ---------------------------------------------------------------- 求解


def _front(rows: Sequence[Dict]) -> List[Dict]:
    nd = [d for d in rows
          if not any(e["E"] >= d["E"] - 1e-12 and e["R"] >= d["R"] - 1e-12
                     and (e["E"] > d["E"] + 1e-12 or e["R"] > d["R"] + 1e-12)
                     for e in rows)]
    return sorted(nd, key=lambda d: d["E"])


def hull(rows: Sequence[Dict]) -> List[Dict]:
    """log-log 平面上的上凸包（"有效前沿"）。

    Pareto 前沿上的点不一定在凸包上：凸包外的点会被某条支撑线同时越过的两点支配，
    任何 w 都选不到它。S（最陡斜率）必须只在凸包上算，否则会被这类点抬高。
    """
    p = _front(rows)
    h: List[Dict] = []
    for d in p:
        if d["E"] <= 0 or d["R"] <= 0:
            continue
        while len(h) >= 2:
            m1 = math.log(d["R"] / h[-1]["R"]) / math.log(d["E"] / h[-1]["E"])
            m2 = math.log(h[-1]["R"] / h[-2]["R"]) / math.log(h[-1]["E"] / h[-2]["E"])
            if abs(m1) <= abs(m2) + 1e-15:      # 斜率没变陡 → 中间点冗余
                h.pop()
            else:
                break
        h.append(d)
    return h


def slope_hull(rows: Sequence[Dict]) -> float:
    """S = 凸包上相邻点的最大 |Δln R̄ / Δln Ē|（§4.2）。"""
    h = hull(rows)
    s = 0.0
    for a, b in zip(h, h[1:]):
        dx = math.log(b["E"]) - math.log(a["E"])
        dy = math.log(b["R"]) - math.log(a["R"])
        if abs(dx) > 1e-15:
            s = max(s, abs(dy / dx))
    return s if s > 0 else 1.0


def lam_breaks(rows: Sequence[Dict]) -> List[Dict]:
    """λ 的分界点 = 各段斜率 / S（§4.3）。返回凸包每点的 λ 区间。"""
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
    """按 (K1) w=λS/(1+λS) 在凸包上取点。"""
    h = hull(rows)
    if not h:
        return None
    S = slope_hull(rows)
    w = lam * S / (1.0 + lam * S)
    return max(h, key=lambda d: w * math.log(d["E"]) + (1 - w) * math.log(d["R"]))


def fmt_lam_scale(rows: Sequence[Dict], n: int, kbar: float) -> List[str]:
    """λ 刻度表：S、各段斜率、λ 分界、典型 λ 的选点。"""
    h = hull(rows)
    if len(h) < 2:
        return [f"n={n}, <k>={kbar}: 凸包不足 2 点，λ 无意义"]
    S = slope_hull(rows)
    L = [f"n={n}, <k>={kbar}  S={S:.4f}  凸包 {len(h)}/{len(rows)} 点",
         f"{'k_max':>7}{'E':>9}{'R':>9}{'段斜率':>9}{'lam_lo':>9}{'lam_hi':>9}{'注释':>10}"]
    for r in lam_breaks(rows):
        d = r["pt"]
        km = d.get("kmax") or d.get("K") or 0
        lo = "0" if r["lo"] == 0 else f"{r['lo']:.3f}"
        hi = "inf" if r["hi"] == math.inf else f"{r['hi']:.3f}"
        seg = "—" if r["s"] is None else f"{r['s']:.4f}"
        tag = "R 端" if r["lo"] == 0 else ("E 端" if r["hi"] == math.inf else "")
        L.append(f"{km:7}{d['E']:9.4f}{d['R']:9.4f}{seg:>9}{lo:>9}{hi:>9}{tag:>10}")
    L.append(f"{'lam':>7}{'w':>9}{'选 k_max':>9}")
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 10.0):
        d = pick_lam(rows, lam)
        w = lam * S / (1.0 + lam * S)
        km = d.get("kmax") or d.get("K") or 0
        L.append(f"{lam:7g}{w:9.3f}{km:9d}")
    return L


def _slope(front: Sequence[Dict]) -> float:
    """兼容旧接口：S 的旧口径（只做 Pareto 过滤）。新代码请用 slope_hull。"""
    s = 0.0
    for a, b in zip(front, front[1:]):
        dx = math.log(b["E"]) - math.log(a["E"])
        dy = math.log(b["R"]) - math.log(a["R"])
        if abs(dx) > 1e-15 and a["R"] > 0 and b["R"] > 0:
            s = max(s, abs(dy / dx))
    return s if s > 0 else 1.0


def default_grid(N: int, M: int, cap: int = 12) -> List[Tuple[int, int]]:
    """从可实现参数里抽一个能覆盖形状的 (h, k_min) 网格。"""
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
    """在给定 (h, k_min) 网格上造图实测；h 决定 k_max（均值守恒唯一确定）。"""
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
    """lambda 落在斜率刻度上：w = lam*S/(1+lam*S)，目标 w*lnE + (1-w)*lnR。"""
    S = _slope(_front(rows))
    w = lam * S / (1 + lam * S)
    best = max(rows, key=lambda d: (w * math.log(max(d["E"], 1e-300))
                                    + (1 - w) * math.log(max(d["R"], 1e-300)),
                                    d["E"]))
    return dict(best, w=w, S=S, lam=lam)


# ---------------------------------------------------------------- 输出


def fmt_params(rows: Sequence[Dict], N: int) -> List[str]:
    L = [f"N={N}  双峰可实现参数（整数度、只取两个值）"]
    L.append(f"{'h':>5}{'rho_h':>8}{'k_max':>7}{'k_min':>7}{'c':>7}"
             f"{'枢纽桩':>8}{'叶桩':>7}{'枢纽内边上限':>13}{'余量':>7}")
    for d in rows:
        L.append(f"{d['h']:5d}{d['rho']:8.4f}{d['kmax']:7d}{d['kmin']:7d}"
                 f"{d['c']:7.1f}{d['hub_stubs']:8d}{d['leaf_stubs']:7d}"
                 f"{d['hub_hub_max']:13d}{d['hub_slack']:7d}")
    return L


def fmt_trace(N: int, M: int, h: int, kmin: int) -> List[str]:
    S = 2 * M
    num = S - (N - h) * kmin
    if num <= 0 or num % h:
        return [f"h={h}, k_min={kmin} 在 N={N}, M={M} 下无整数解"
                f"（2M-(N-h)k_min = {num} 不能被 h={h} 整除）"]
    kmax = num // h
    ok, why = graphical(N, h, kmax, kmin)
    em = edge_mix(N, h, kmax, kmin)
    L = [f"N={N}, M={M}（<k>={S / N:.2f}），取 h={h}, k_min={kmin}",
         "  1. 均值守恒（唯一确定 k_max）",
         f"     h*k_max + (N-h)*k_min = 2M",
         f"     {h}*k_max + {N - h}*{kmin} = {S}  =>  k_max = {kmax}",
         f"     rho_h = {h}/{N} = {h / N:.4f}，rho_min = {1 - h / N:.4f}",
         "  2. 可实现性（Erdos-Gallai）",
         f"     k_max <= N-1 ?  {kmax} <= {N - 1}  ->  {kmax <= N - 1}",
         f"     度数和为偶 ?  {S} % 2 = {S % 2}  ->  {S % 2 == 0}",
         f"     枢纽桩 {h * kmax} 能否被吃掉？上限 = 枢纽内部 2*C(h,2)={h * (h - 1)}"
         f" + 叶桩 {N * 0 + (N - h) * kmin} = {h * (h - 1) + (N - h) * kmin}"
         f"  ->  {'够' if h * (h - 1) + (N - h) * kmin >= h * kmax else '不够'}",
         f"     判定：{why}",
         f"  3. 度序列  [{kmax}]*{h} + [{kmin}]*{N - h}，共 {S} 个桩 / {M} 条边"]
    if ok:
        L.append(f"  4. 边的三类（由度序列唯一确定）")
        L.append(f"     枢纽-枢纽 {em['hh']} 条 | 枢纽-叶 {em['lh']} 条 | "
                 f"叶-叶 {em['ll']} 条 | 合计 {em['total']}")
        g = build(N, h, kmax, kmin, seed=17)
        if g.get("ok"):
            m = measure(g, seeds=8)
            degs = [len(a) for a in g["adj"]]
            L.append(f"  5. 造图  Havel-Hakimi 连边 -> 保度双边交换 {12 * M} 次"
                     f"（度数一点不变）-> 连通={g['connected']}"
                     f"（第 {g['attempts']} 轮）")
            L.append(f"     校验：度数取值为 {sorted(set(degs))}，边数 {g['edges']}")
            L.append(f"  6. 实测  E={m['E']:.4f}  f_c^rand={m['fc_rand']:.4f}  "
                     f"f_c^tar={m['fc_tar']:.4f}  R={m['R']:.4f}")
    return L


def fmt_kmax_sweep(n: int, kbar: float, nmax: int, G: int = 3,
                   seeds: int = 8, points: int = 8) -> List[str]:
    """固定 n_max，扫 k_max ∈ (<k>, n-1]，实测 E / f_c / R。"""
    has = _measure_ok()
    L = [f"n={n}, <k>={kbar}, n_max={nmax}（rho_max={nmax/n:.4f}）  扫 k_max",
         f"{'k_max':>6}{'k_min':>8}{'k_eff':>7}{'<k_hub>':>8}"
         f"{'E':>8}{'f_c^r':>8}{'f_c^t':>8}{'R':>8}"]
    rows: List[Dict] = []
    lo_k = int(math.ceil(kbar))
    ks = sorted({int(round(x)) for x in _logspace(max(lo_k + 1, 2), n - 1, points)} | {n - 1})
    for kmax in ks:
        d = design(n, kbar, nmax, kmax)
        if not d["ok"]:
            L.append(f"{kmax:6d}   不可行：{d['why']}")
            continue
        khub = nmax * kmax / (n - nmax)          # 每个叶平均挂几个枢纽桩
        if not has:
            L.append(f"{kmax:6d}{d['kmin']:8.3f}{d['k_eff']:7.3f}{khub:8.3f}"
                     f"{'  —（需 isp_construct 才能实测）':>32}")
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
            # 稀疏图上个别实现会返回 nan（连通但测量退化）；把 nan 平均进去会
            # 让整行进凸包，最终 S=1.0。这里丢弃该实现并如实报出弃用数。
            if any(isinstance(mt[k], float) and math.isnan(mt[k])
                   for k in ("E", "fc_rand", "fc_tar")):
                bad += 1
                continue
            for k in acc:
                acc[k] += mt[k]
            got += 1
        if not got:
            L.append(f"{kmax:6d}{d['kmin']:8.3f}{d['k_eff']:7.3f}{khub:8.3f}   全部实现不可用")
            continue
        E = acc["E"] / got
        fr = acc["fc_rand"] / got
        ft = acc["fc_tar"] / got
        tag = f"  （{bad}/{G} 实现不可用）" if bad else ""
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
    L = [f"{'h':>5}{'k_max':>7}{'k_min':>7}{'c':>7}{'E':>8}{'fcr':>8}{'fct':>8}{'R':>8}  前沿"]
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


# ---------------------------------------------------------------- 自检


def _selftest() -> int:
    bad = 0

    def chk(name: str, cond: bool, extra: str = "") -> None:
        nonlocal bad
        if not cond:
            bad += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}"
              + (f"  ({extra})" if extra else ""))

    print("可实现性判定")
    chk("k_max > N-1 被拒", not graphical(10, 1, 10, 1)[0])
    chk("度数和为奇被拒", not graphical(10, 1, 4, 1)[0], "4+9=13")
    chk("枢纽桩吃不完被拒", not graphical(100, 1, 60, 1)[0], "60 > 0+99")
    chk("合法点通过", graphical(100, 2, 53, 3)[0])

    print("单枢纽（h=1）在整数双峰下是否存在")
    for N, kb in ((1000, 3.0), (1000, 5.0), (100, 4.0)):
        M = int(round(kb * N / 2))
        chk(f"N={N},<k>={kb} 无 h=1 的整数解",
            not any(d["h"] == 1 for d in params(N, M)),
            f"kmin<=2 时 k_max 必超 N-1")

    print("可实现参数集")
    for N, kb in ((1000, 3.0), (1000, 5.0)):
        M = int(round(kb * N / 2))
        ps = params(N, M)
        chk(f"N={N},<k>={kb} 有解", len(ps) > 0, f"{len(ps)} 组")
        chk(f"N={N},<k>={kb} 全部均值守恒",
            all(d["h"] * d["kmax"] + (N - d["h"]) * d["kmin"] == 2 * M for d in ps))
        chk(f"N={N},<k>={kb} 全部 k_min<k_max",
            all(d["kmin"] < d["kmax"] for d in ps))
        chk(f"N={N},<k>={kb} 全部 EG 通过",
            all(graphical(N, d["h"], d["kmax"], d["kmin"])[0] for d in ps))

    print("构造守恒")
    cases = [(100, 200, 2, 3), (100, 200, 5, 2), (100, 200, 20, 1), (200, 400, 8, 2)]
    for (N, M, h, kmin) in cases:
        S = 2 * M
        num = S - (N - h) * kmin
        if num <= 0 or num % h:
            continue
        kmax = num // h
        g = build(N, h, kmax, kmin, seed=17)
        if not g.get("ok"):
            chk(f"N={N} h={h} kmin={kmin} 造出连通图", False, g.get("why", ""))
            continue
        degs = [len(a) for a in g["adj"]]
        chk(f"N={N} h={h} kmin={kmin} 度数严格双峰、边数守恒",
            set(degs) == {kmax, kmin} and sum(degs) == S and g["edges"] == M,
            f"degs={sorted(set(degs))} edges={g['edges']}")

    print("三类边计数")
    for (N, M, h, kmin) in ((1000, 1500, 2, 2), (1000, 1500, 40, 1),
                            (1000, 2500, 2, 4), (200, 400, 8, 2)):
        num = 2 * M - (N - h) * kmin
        if num <= 0 or num % h:
            continue
        kmax = num // h
        em = edge_mix(N, h, kmax, kmin)
        chk(f"N={N} h={h} kmin={kmin} 三类边相加 = M 且非负",
            em["ok"] and em["total"] == M,
            f"hh={em['hh']} lh={em['lh']} ll={em['ll']}")
        chk(f"N={N} h={h} kmin={kmin} 每个叶子至少一条枢纽边",
            em["lh"] >= N - h, f"lh={em['lh']} vs 叶子数 {N - h}")

    print("度量：E 不能被 E_dir 代替")
    g = build(1000, 2, 502, 2, seed=17)
    if g.get("ok"):
        m = measure(g, seeds=8)
        e_dir = 502 * 505 / (2 * 1000 * 999)
        chk("实测 E 远高于单枢纽下界 E_dir", m["E"] > 3 * e_dir,
            f"E={m['E']:.4f} vs E_dir={e_dir:.4f}")

    print("求解")
    rows = evaluate(100, 200, grid=[(2, 3), (4, 3), (5, 2), (10, 2), (20, 1)],
                    seeds=6, gseed=1)
    chk("evaluate 出点", len(rows) >= 3, f"{len(rows)} 点")
    if len(rows) >= 3:
        fr = _front(rows)
        chk("前沿 E 单调", all(b["E"] >= a["E"] for a, b in zip(fr, fr[1:])))
        p0, p1 = pick(rows, 0.0), pick(rows, 1.0)
        chk("lambda=0 偏 R", p0["R"] >= p1["R"] - 1e-9,
            f"R {p0['R']:.3f} vs {p1['R']:.3f}")
        chk("lambda=1 偏 E", p1["E"] >= p0["E"] - 1e-9,
            f"E {p1['E']:.3f} vs {p0['E']:.3f}")

    print(f"\n{'全部通过' if bad == 0 else f'{bad} 条失败'}")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------- CLI


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
        print(f"\n共 {len(ps)} 组；h=1 {'存在' if any(d['h'] == 1 for d in ps) else '不存在'}")
        for kmin in sorted({d['kmin'] for d in ps}):
            print(f"  k_min={kmin}: 最小 h = {min_h(N, M, kmin)}")
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
