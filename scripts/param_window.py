#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数 c 的双侧可行窗口：效率给下界，定向攻击鲁棒性给上界。

用法（直接放进 repo 的 scripts/ 即可，零第三方依赖）：

    # 1) 只有解析效率式 + 实测 R 曲线
    python scripts/param_window.py --N 100 --kbar 6 --E-star 0.35 --R-star 0.50 \
        --curve-r results/r_tar_BA100.csv

    # 2) 两条曲线都实测（推荐：解析式偏保守，最多偏大 2-3 倍）
    python scripts/param_window.py --N 100 --kbar 6 --E-star 0.35 --R-star 0.50 \
        --curve-e results/e_BA100.csv --curve-r results/r_tar_BA100.csv \
        --delta-r 0.01 --Kmin 2

    # 3) 自检 / 回归测试
    python scripts/param_window.py --demo
    python scripts/param_window.py --selftest

CSV 格式：两列 `c,value`，允许表头，允许乱序（内部会排序）。
value 对 --curve-e 是归一化效率 E，对 --curve-r 是指标 R_t（越大越鲁棒）。

三个已写进论文的限定（见 paper_snippet_*.tex）：
  L1 天花板：E_dir 在 K=N-1 处取到 (N+2)/(2N)≈0.5，E* 越过它时解析法报"无解"是假象，
     真正的问题是结构族不可达，应换结构而不是调 c。
  L2 保守度：E_dir 丢弃的成对贡献恒为非负，K 较小时可达 2-8 倍，故解析 c_E 是充分不必要，
     会系统性偏大；凡有实测 E(c) 必须用数值反解。
  L3 单调性：c_R = max{c: R_t(c) >= R*} 只在 R_t 单调不增时才等价于"区间处处可行"。
     R_t 有回升（双峰）时 max 会跳到右侧孤立分支，必须取"最左连通可行段的上确界"。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

EPS = 1e-9

# ---------------------------------------------------------------- 状态码
OK = "ok"
OK_DEGRADED = "ok_single_sided"          # 只有下界（无上界曲线）
CEILING_UNREACHABLE = "ceiling_unreachable"
STRUCTURAL_CAP = "structural_cap_binding"  # 边预算/残余连通给出的 K 上限被效率下界顶穿
EFF_UNREACHABLE = "efficiency_unreachable_on_curve"
R_UNREACHABLE = "robustness_unreachable_on_curve"
NO_OVERLAP = "empty_intersection"
OUT_OF_DOMAIN = "window_outside_curve_domain"
BAD_INPUT = "bad_input"

STATUS_ZH = {
    OK: "可行",
    OK_DEGRADED: "可行（仅下界，上界缺失）",
    CEILING_UNREACHABLE: "结构族不可达：E* 超过解析天花板",
    STRUCTURAL_CAP: "稀疏图边预算不足：E* 需要的最小度超过结构上限",
    EFF_UNREACHABLE: "效率曲线在采样域内未达到 E*",
    R_UNREACHABLE: "鲁棒曲线在采样域内全部低于 R*",
    NO_OVERLAP: "真冲突：下界高于上界",
    OUT_OF_DOMAIN: "窗口落在采样域之外",
    BAD_INPUT: "输入非法",
}


# ---------------------------------------------------------------- 解析式
def efficiency_direct(K: float, N: int) -> float:
    """E_dir(K) = K(K+3) / (2N(N-1))，K = c*kbar 为枢纽的直接邻居数。"""
    return K * (K + 3.0) / (2.0 * N * (N - 1))


def efficiency_ceiling(N: int) -> float:
    """单枢纽星形族在 K=N-1 处取到的效率上限 (N+2)/(2N)。"""
    return (N + 2.0) / (2.0 * N)


def c_lower_analytic(N: int, kbar: float, E_star: float) -> Optional[float]:
    """解析下界；E* >= 天花板时返回 None（不是"参数无解"，而是族不可达）。"""
    if N < 2 or kbar <= 0:
        return None
    if E_star <= 0:
        return 0.0
    if E_star >= efficiency_ceiling(N):
        return None
    return (math.sqrt(9.0 + 8.0 * N * (N - 1) * E_star) - 3.0) / (2.0 * kbar)


def c_upper_analytic_symmetric(N: int, kbar: float, E_star: float) -> Optional[float]:
    """对称参考：把 E_dir 当"上界"用会得到的 c（仅用于论文里对比保守度，不要真用）。"""
    return c_lower_analytic(N, kbar, E_star)


def conservativeness_envelope(N: int, K: float) -> Tuple[float, float]:
    """L2 的定量包络：E_dir 最多低估真实效率多少倍，以及 c_E 最多偏大多少倍。

    设 M = N-1-K 个节点不直接连枢纽。被 E_dir 丢弃的成对贡献（按有序对×权重计）为
        KM + M(M-1)/2,
    基准项为 K(K+3)/2，故
        E_true / E_dir <= 1 + M(2K+M-1) / (K(K+3)),
    等号在"所有被丢弃的点对都恰好是距离 2"时取到。又 E ~ K(K+3)，
    故解析下界给出的 c 最多偏大 sqrt(该比值) 倍。

    返回 (E 比值, c 比值)。
    """
    if N < 2 or K <= 0:
        return (1.0, 1.0)
    K = min(float(K), N - 1.0)
    M = (N - 1.0) - K
    ratio_e = 1.0 + M * (2.0 * K + M - 1.0) / (K * (K + 3.0))
    return (ratio_e, math.sqrt(ratio_e))


def k_from_c(c: float, kbar: float) -> float:
    return c * kbar


def degree_budget_cap(N: int, M: int, residual_connected: bool = True) -> Optional[int]:
    """稀疏图上真正的硬约束：边预算给出的枢纽度上限 K_cap。

    整图连通只要求 M >= N-1，这本身几乎不约束 K；但任何以"抗定向攻击"为目标的构造
    都必须满足一个更强的设计前提——**移除枢纽后残余子网仍然连通**，否则枢纽是唯一的
    单点，R_t 会直接崩到 0。残余的 (N-1) 个节点连通至少需要 (N-2) 条边，于是

        K <= M - (N-2) .

    在 k̄ ≈ 2.0 的树上右边等于 1：**根本放不下任何枢纽**，这正是稀疏 ISP 上双峰构造
    失效的算术原因。relaxed=True 时放弃该前提，退化为 K <= N-1（此时鲁棒性目标会替你
    把 K 压回来，但不该用解析效率去定 K）。
    """
    if M is None or not residual_connected:
        return None
    if M < N - 1:
        raise ValueError(f"M={M} < N-1={N - 1}：图不连通")
    # 密图上 M-(N-2) 会超过 N-1（枢纽不可能连比自己多的节点），必须截断。
    # 这一点在小而稠密的真实拓扑上会真的发生（Eon 19/37: 37-17=20 > 18）。
    return int(min(M - (N - 2), N - 1))


def kbar_threshold(E_star: float) -> float:
    """解析路径在"残余连通"前提下的适用门槛：k̄ >= 2 + sqrt(8 E*)。

    推导：K_cap = M-(N-2) = N(k̄/2-1)+2 ≈ N(k̄-2)/2，于是
        E_dir(K_cap) ≈ K_cap² / (2N²) = (k̄-2)²/8 .
    要 E_dir(K_cap) >= E*，需 k̄ >= 2 + sqrt(8E*)。E*=0.10 -> 2.89，E*=0.20 -> 3.26。
    这与实验侧"BiT 在 k̄≳3 才稳"的经验区间一致，是它的解析来源。
    """
    if E_star <= 0:
        return 2.0
    return 2.0 + math.sqrt(8.0 * E_star)


def cap_report(N: int, M: int, kbar: float) -> dict:
    K_cap = degree_budget_cap(N, M, True)
    E_max = efficiency_direct(K_cap, N) if K_cap and K_cap > 0 else float("nan")
    return {
        "N": N,
        "M": M,
        "kbar": round(2.0 * M / N, 4),
        "K_cap": K_cap,
        "c_cap": round(K_cap / kbar, 3) if K_cap else None,
        "E_dir_max": round(E_max, 4),
        "hub_share": round(K_cap / (N - 1), 3) if K_cap else None,
        "kbar_residual": round(2.0 * (M - K_cap) / (N - 1), 3) if K_cap else None,
    }


# ---------------------------------------------------------------- 序列工具
def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """返回 True 的连通段 [start, end) 列表。"""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def tail_run_start(flags: Sequence[bool]) -> Optional[int]:
    """末尾连通可行段的起点；末尾不可行则返回 None。

    单调递增指标（效率）用它：取"一旦满足就一直满足到右端"的那段的左端。
    非单调时这比 inf{c: E>=E*} 安全——后者会把左侧孤立可行岛当成下界，
    导致区间里混入不可行点。
    """
    if not flags or not flags[-1]:
        return None
    i = len(flags) - 1
    while i > 0 and flags[i - 1]:
        i -= 1
    return i


def head_run_end(flags: Sequence[bool]) -> Optional[int]:
    """开头连通可行段的终点（开区间）；开头不可行则返回 None。

    单调不增指标（定向鲁棒性）用它：取"从最左一直满足到某处"的那段的右端。
    这就是 L3：不能用 max{c: R>=R*}，否则双峰曲线会跳到右侧孤立分支。
    """
    if not flags or not flags[0]:
        return None
    i = 0
    while i < len(flags) and flags[i]:
        i += 1
    return i


def pava_increasing(ys: Sequence[float], w: Optional[Sequence[float]] = None) -> List[float]:
    """保序回归（递增），用于压掉有限样本噪声造成的假非单调。"""
    n = len(ys)
    if n == 0:
        return []
    w = list(w) if w is not None else [1.0] * n
    vals = [float(y) for y in ys]
    wts = [float(x) for x in w]
    blocks: List[List[float]] = []  # [value, weight, count]
    for v, wt in zip(vals, wts):
        blocks.append([v, wt, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0] + EPS:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            tw = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / tw, tw, c1 + c2])
    out: List[float] = []
    for v, _, cnt in blocks:
        out.extend([v] * int(cnt))
    return out


def pava_decreasing(ys: Sequence[float]) -> List[float]:
    return [-v for v in pava_increasing([-y for y in ys])]


def _trend(xs: Sequence[float], ys: Sequence[float]) -> int:
    """粗判趋势：+1 递增，-1 递减，0 非单调（按有效步数占比）。"""
    up = down = 0
    for i in range(1, min(len(xs), len(ys))):
        d = ys[i] - ys[i - 1]
        if d > EPS:
            up += 1
        elif d < -EPS:
            down += 1
    tot = up + down
    if tot == 0:
        return 0
    if up / tot >= 0.9:
        return 1
    if down / tot >= 0.9:
        return -1
    return 0


def _sort_curve(curve: Iterable[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    pts = sorted((float(x), float(y)) for x, y in curve)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # 去掉重复 x（保留最后出现的一个）
    ux, uy = [], []
    for x, y in zip(xs, ys):
        if ux and abs(x - ux[-1]) <= EPS:
            uy[-1] = y
        else:
            ux.append(x)
            uy.append(y)
    return ux, uy


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> Optional[float]:
    if not xs:
        return None
    if x < xs[0] - EPS or x > xs[-1] + EPS:
        return None
    for i in range(1, len(xs)):
        if x <= xs[i] + EPS:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 - x0 <= EPS:
                return y1
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


# ---------------------------------------------------------------- 结果对象
@dataclass
class Window:
    status: str
    N: int
    kbar: float
    c_lo: Optional[float] = None
    c_hi: Optional[float] = None
    K_lo: Optional[int] = None
    K_hi: Optional[int] = None
    source_lower: str = ""
    source_upper: str = ""
    warnings: List[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (OK, OK_DEGRADED)

    @property
    def c_mid(self) -> Optional[float]:
        if self.c_lo is None:
            return None
        if self.c_hi is None:
            return self.c_lo
        return 0.5 * (self.c_lo + self.c_hi)

    @property
    def K_recommend(self) -> Optional[int]:
        if not self.ok or self.K_lo is None:
            return None
        if self.K_hi is None:
            return self.K_lo
        return int(round(0.5 * (self.K_lo + self.K_hi)))

    def report(self) -> str:
        L = []
        L.append(f"status      : {self.status}  ({STATUS_ZH.get(self.status, '?')})")
        L.append(f"N={self.N}  kbar={self.kbar:g}")
        if self.c_lo is not None:
            L.append(f"c lower     : {self.c_lo:.4f}   [{self.source_lower}]")
        if self.c_hi is not None:
            L.append(f"c upper     : {self.c_hi:.4f}   [{self.source_upper}]")
        if (self.c_lo is not None and self.c_hi is not None
                and self.status in (OK, OK_DEGRADED) and self.c_hi >= self.c_lo):
            L.append(f"c window    : [{self.c_lo:.4f}, {self.c_hi:.4f}]  宽度 {self.c_hi - self.c_lo:.4f}")
        if (self.K_lo is not None and self.K_hi is not None
                and self.status in (OK, OK_DEGRADED)):
            L.append(f"K window    : [{self.K_lo}, {self.K_hi}]  ({self.K_hi - self.K_lo + 1} 个整数点)")
        elif self.K_lo is not None and self.status in (OK, OK_DEGRADED):
            L.append(f"K lower     : {self.K_lo}")
        if self.ok:
            L.append(f"K recommend : {self.K_recommend}")
        for k, v in self.detail.items():
            L.append(f"  {k:<10}: {v}")
        for w in self.warnings:
            L.append(f"  ! {w}")
        return "\n".join(L)


# ---------------------------------------------------------------- 主入口
def compute_window(
    N: int,
    kbar: float,
    E_star: float,
    R_star: Optional[float] = None,
    curve_e: Optional[Iterable[Tuple[float, float]]] = None,
    curve_r: Optional[Iterable[Tuple[float, float]]] = None,
    K_min: int = 2,
    delta_r: float = 0.0,
    smooth: bool = True,
    c_bounds: Tuple[float, float] = (0.0, float("inf")),
    M: Optional[int] = None,
    residual_connected: bool = True,
) -> Window:
    """返回 c 的双侧可行窗口。E 越大越好 -> 下界；R_t 越大越好且随 c 递减 -> 上界。

    M/residual_connected: 给出边数后启用"稀疏图边预算"上限 K <= M-(N-2)，
    稀疏 ISP（k̄≈2.3-2.5）上它会先于 N-1 与鲁棒性上界把窗口压没。
    """
    w = Window(status=OK, N=N, kbar=kbar)

    if N < 2:
        w.status = BAD_INPUT
        w.warnings.append("N 必须 >= 2")
        return w
    if kbar <= 0:
        w.status = BAD_INPUT
        w.warnings.append("kbar 必须 > 0")
        return w
    if E_star < 0 or (R_star is not None and R_star < 0):
        w.status = BAD_INPUT
        w.warnings.append("E*/R* 不能为负")
        return w

    w.detail["E_dir_ceiling"] = round(efficiency_ceiling(N), 4)

    # ---- 下界：效率
    if curve_e is not None:
        xs, ys = _sort_curve(curve_e)
        if len(xs) < 3:
            w.status = BAD_INPUT
            w.warnings.append("效率曲线采样点太少（<3）")
            return w
        flags = [y >= E_star - EPS for y in ys]
        trend = _trend(xs, ys)
        if trend <= 0:
            w.warnings.append(
                f"效率曲线趋势判为 {'非单调' if trend == 0 else '递减'}，与假设相反："
                "请确认指标定义/自变量方向（L3 同样适用于此处）"
            )
        idx = tail_run_start(flags)
        if idx is None:
            w.status = EFF_UNREACHABLE
            w.warnings.append(
                f"采样域 [{xs[0]:g},{xs[-1]:g}] 内效率未达到 E*={E_star}；"
                + ("E* 超过解析天花板，需换结构族（L1）"
                   if E_star >= efficiency_ceiling(N) else "请扩大 c 采样范围")
            )
            w.detail["E_max_on_curve"] = round(max(ys), 4)
            return w
        runs = _runs(flags)
        if len(runs) > 1:
            w.warnings.append(
                f"效率曲线可行集不连通（{len(runs)} 段），已取右端连通段："
                f"[{xs[runs[-1][0]]:.3g},{xs[runs[-1][1] - 1]:.3g}]，左端孤岛 {len(runs) - 1} 个被忽略"
            )
        w.c_lo = xs[idx]
        w.source_lower = "numeric"
        w.detail["E_at_c_lo"] = round(_interp(xs, ys, w.c_lo) or float("nan"), 4)
    else:
        cE = c_lower_analytic(N, kbar, E_star)
        if cE is None:
            w.status = CEILING_UNREACHABLE
            w.warnings.append(
                f"E*={E_star} >= 单枢纽族效率上限 (N+2)/(2N)={efficiency_ceiling(N):.4f}："
                "c 无论取多大都达不到，须换结构族（L1），不要继续调 c"
            )
            return w
        w.c_lo = cE
        w.source_lower = "analytic(E_dir, 充分不必要, 偏保守 L2)"
        w.detail["E_dir_at_c_lo"] = round(efficiency_direct(cE * kbar, N), 6)
        ratio_e, ratio_c = conservativeness_envelope(N, cE * kbar)
        w.detail["conservative_x"] = round(ratio_c, 2)
        if ratio_c >= 1.15:
            w.warnings.append(
                f"解析下界按「残余点对全为距离2」的上包络，c 最多偏大 {ratio_c:.2f} 倍"
                f"（E 最多低估 {ratio_e:.1f} 倍，K={cE * kbar:.1f}）；有实测 E(c) 时务必改用数值反解"
            )

    # ---- 上界：定向攻击鲁棒性
    if curve_r is None:
        w.status = OK_DEGRADED
        target = R_star if R_star is not None else float("nan")
        w.warnings.append(
            f"未提供 R_t(c) 曲线，上界缺失 -> 退化为单侧（c >= {w.c_lo:.4f}）。"
            f"注意随机失效鲁棒性随 c 递增，同样只给下界，不能替代上界（R*={target}）"
        )
    else:
        xs, ys = _sort_curve(curve_r)
        if len(xs) < 3:
            w.status = BAD_INPUT
            w.warnings.append("鲁棒曲线采样点太少（<3）")
            return w
        if R_star is None:
            w.status = BAD_INPUT
            w.warnings.append("给了 R 曲线但没给 R*")
            return w
        thr = R_star + delta_r

        # 可行性判定一律用**原始**采样点，不用平滑值：平滑会凭空造出/抹掉可行点。
        flags = [y >= thr - EPS for y in ys]
        end = head_run_end(flags)
        if end is None:
            w.status = R_UNREACHABLE
            w.warnings.append(f"最小的 c={xs[0]:g} 处 R_t 已低于 R*={R_star}（+δ={delta_r}）")
            w.detail["R_at_c_min"] = round(ys[0], 4)
            return w
        runs = _runs(flags)
        if len(runs) > 1:
            island = runs[1:]
            w.warnings.append(
                "R_t 非单调：检测到右侧孤立可行岛 "
                + ", ".join(f"[{xs[a]:.3g},{xs[b - 1]:.3g}]" for a, b in island)
                + "；按 L3 已忽略（max 取法会误选它们）"
            )
        w.c_hi = xs[end - 1]
        w.source_upper = "leftmost-run(数值)" if len(runs) > 1 else "numeric"
        w.detail["R_at_c_hi"] = round(ys[end - 1], 4)

        # 保序回归只用于两件事：判趋势、给一个更保守的备选上界
        ys_sm = pava_decreasing(ys) if smooth else list(ys)
        trend = _trend(xs, ys_sm)
        if trend >= 0:
            w.warnings.append(
                f"R_t 趋势判为 {'递增' if trend == 1 else '非单调'}：若指标随 c 递增，它给的是下界而非上界（L3）"
            )
        elif smooth and any(abs(a - b) > 1e-6 for a, b in zip(ys, ys_sm)):
            flags_sm = [y >= thr - EPS for y in ys_sm]
            end_sm = head_run_end(flags_sm)
            if end_sm is not None and end_sm - 1 < end - 1:
                w.detail["c_hi_smoothed"] = round(xs[end_sm - 1], 4)
                w.warnings.append(
                    f"保序回归后上界更紧（{xs[end_sm - 1]:.4g} < {w.c_hi:.4g}），若噪声大建议取前者"
                )

    # ---- 与 c 定义域求交
    lo_b, hi_b = c_bounds
    if w.c_lo is not None and lo_b > 0:
        w.c_lo = max(w.c_lo, lo_b)
    if w.c_hi is not None and hi_b != float("inf"):
        w.c_hi = min(w.c_hi, hi_b)
        if w.c_hi < lo_b:
            w.status = OUT_OF_DOMAIN
            return w

    # ---- 整数化
    K_lo_raw = w.c_lo * kbar
    K_lo = int(math.ceil(K_lo_raw - 1e-9))
    K_lo = max(K_lo, int(K_min))
    K_cap = int(N - 1)

    # 稀疏图（ISP）上的真正硬约束：边预算 + 残余连通
    K_struct = None
    if M is not None:
        try:
            K_struct = degree_budget_cap(N, int(M), residual_connected)
        except ValueError as e:
            w.status = BAD_INPUT
            w.warnings.append(str(e))
            return w
        if K_struct is not None:
            w.detail["K_cap_struct"] = K_struct
            w.detail["c_cap_struct"] = round(K_struct / kbar, 3)
            w.detail["E_dir_at_K_cap"] = round(efficiency_direct(K_struct, N), 4)
            K_cap = min(K_cap, K_struct)

    if K_lo > K_cap:
        if K_struct is not None and K_lo > K_struct:
            w.status = STRUCTURAL_CAP
            w.warnings.append(
                f"稀疏图边预算不足：效率要求 K >= {K_lo}，但枢纽度受 M-(N-2)={K_struct} 限制"
                f"（c_cap={K_struct / kbar:.2f}）。解析路径在该数据集上只能覆盖 "
                f"E* <= E_dir(K_cap)={efficiency_direct(K_struct, N):.4f}；"
                "若实测 E(c) 明显高于 E_dir，可改用数值反解重算"
            )
        else:
            w.status = OUT_OF_DOMAIN
            w.warnings.append(f"K 下界 {K_lo} 超过单枢纽最大度 N-1={K_cap}：结构族不可达（L1）")
        return w

    if w.c_hi is not None:
        K_hi = int(math.floor(w.c_hi * kbar + 1e-9))
        K_hi = min(K_hi, K_cap)
        if K_hi < K_lo:
            w.status = NO_OVERLAP
            w.detail["K_need_min"] = K_lo
            w.detail["K_need_max"] = K_hi
            w.warnings.append(
                f"空交集：K 需同时 >= {K_lo}（效率）与 <= {K_hi}（鲁棒），相差 {K_lo - K_hi}。"
                "此时不是参数问题，须改结构族或放宽 E*/R*"
            )
            if w.source_lower.startswith("analytic"):
                w.warnings.append(
                    "下界来自解析式 E_dir（充分不必要，L2）：真实 E(c) 通常更高，"
                    "这条'冲突'可能是假的，请提供实测 E(c) 曲线复核后再下结论"
                )
            if w.c_hi - w.c_lo <= 1e-9:
                w.warnings.append(
                    f"可行 c 集合退化为单点 c≈{w.c_lo:.4g}（宽度 0）：K 必须恰好等于 "
                    f"{w.c_lo * kbar:.2f} 而它非整数，故 K 空间必然为空。加密 c 网格或改用连续 K"
                )
            return w
        w.K_lo, w.K_hi = K_lo, K_hi
    else:
        w.K_lo = K_lo

    # ---- 窗口质量提示
    if w.c_hi is not None and curve_r is not None:
        xs, _ = _sort_curve(curve_r)
        dx = min((xs[i + 1] - xs[i] for i in range(len(xs) - 1)), default=0.0)
        if dx > 0 and (w.c_hi - w.c_lo) < dx:
            w.warnings.append(
                f"窗口宽度 {w.c_hi - w.c_lo:.4f} < R 曲线网格间距 {dx:.4f}：边界由网格分辨率决定，"
                "在不同 seed 间不可复现（加密网格或换连续指标）"
            )
    return w


# ---------------------------------------------------------------- CLI / IO
def load_curve(path: str) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for rec in csv.reader(f):
            if len(rec) < 2:
                continue
            try:
                x, y = float(rec[0]), float(rec[1])
            except ValueError:
                continue  # 表头
            rows.append((x, y))
    if not rows:
        raise SystemExit(f"曲线 {path} 解析不到数值行")
    return rows


DEMO_E = [(1.0, 0.18), (1.5, 0.27), (2.0, 0.35), (2.5, 0.45), (3.0, 0.52), (3.6, 0.63),
          (4.2, 0.70), (5.0, 0.76), (6.0, 0.80), (6.5, 0.82)]
DEMO_R_MONO = [(2.5, 0.72), (3.0, 0.66), (3.6, 0.58), (4.2, 0.50), (4.8, 0.44), (5.4, 0.417), (6.0, 0.34), (6.5, 0.28)]
DEMO_R_BUMP = [(2.5, 0.62), (3.0, 0.55), (3.6, 0.44), (4.2, 0.33), (4.8, 0.28), (5.4, 0.33), (6.0, 0.44), (6.5, 0.52)]


def _demo() -> None:
    cases = [
        ("A 正常：双侧窗口", dict(N=100, kbar=6, E_star=0.35, R_star=0.50,
                                  curve_e=DEMO_E, curve_r=DEMO_R_MONO)),
        ("B 反例：R_t 双峰（L3）", dict(N=100, kbar=6, E_star=0.30, R_star=0.40,
                                       curve_e=DEMO_E, curve_r=DEMO_R_BUMP)),
        ("C 天花板（L1）：E*=0.55", dict(N=100, kbar=6, E_star=0.55, R_star=0.50)),
        ("D 真冲突：E* 与 R* 都要高", dict(N=100, kbar=6, E_star=0.78, R_star=0.60,
                                            curve_e=DEMO_E, curve_r=DEMO_R_MONO)),
        ("E 只给解析式（L2 保守）", dict(N=100, kbar=6, E_star=0.35, R_star=0.50,
                                         curve_r=DEMO_R_MONO)),
        ("F 退化：无 R 曲线", dict(N=500, kbar=6, E_star=0.10)),
    ]
    for i, (name, kw) in enumerate(cases, 1):
        w = compute_window(**kw)
        print(f"\n=== {i}. {name} ===")
        print(w.report())
    # 保守度对照
    print("\n=== 解析 vs 数值 c_E 保守度 ===")
    for N in (100, 500, 1000):
        for E_star in (0.10, 0.20, 0.35):
            cE = c_lower_analytic(N, 6.0, E_star)
            if cE is None:
                continue
            K = cE * 6
            ratio_e, ratio_c = conservativeness_envelope(N, K)
            print(f"N={N:5d} E*={E_star:.2f}: c_E_analytic={cE:7.3f}  K={K:7.2f}  "
                  f"K_ceil={math.ceil(K)}  E_dir(K)={efficiency_direct(math.ceil(K), N):.4f}  "
                  f"E 低估<={ratio_e:5.1f}x  c 偏大<={ratio_c:.2f}x")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="c 的双侧可行窗口（效率下界 / 定向鲁棒上界）")
    ap.add_argument("--N", type=int)
    ap.add_argument("--kbar", type=float, default=6.0)
    ap.add_argument("--E-star", dest="E_star", type=float)
    ap.add_argument("--R-star", dest="R_star", type=float, default=None)
    ap.add_argument("--curve-e", dest="curve_e", default=None, help="CSV: c,E(c)")
    ap.add_argument("--curve-r", dest="curve_r", default=None, help="CSV: c,R_t(c)")
    ap.add_argument("--Kmin", type=int, default=2)
    ap.add_argument("--edges", "--M", dest="M", type=int, default=None,
                    help="图的边数（稀疏 ISP 必填，启用边预算上限 K<=M-(N-2)）")
    ap.add_argument("--allow-hub-star", dest="allow_hub_star", action="store_true",
                    help="放弃'移除枢纽后残余仍连通'的前提（会放宽上限，但鲁棒性通常直接崩）")
    ap.add_argument("--delta-r", dest="delta_r", type=float, default=0.0,
                    help="鲁棒性安全裕度，要求 R_t >= R* + delta")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.demo:
        _demo()
        return 0
    if a.selftest:
        return 0 if run_selftest() else 1
    if a.N is None or a.E_star is None:
        ap.error("需要 --N 与 --E-star（或用 --demo）")

    w = compute_window(
        N=a.N, kbar=a.kbar, E_star=a.E_star, R_star=a.R_star,
        curve_e=load_curve(a.curve_e) if a.curve_e else None,
        curve_r=load_curve(a.curve_r) if a.curve_r else None,
        K_min=a.Kmin, delta_r=a.delta_r, smooth=not a.no_smooth,
        M=a.M, residual_connected=not a.allow_hub_star,
    )
    if a.json:
        import json
        print(json.dumps(w.__dict__, ensure_ascii=False, indent=2))
    else:
        print(w.report())
    return 0 if w.ok or w.status == NO_OVERLAP else 2


# ---------------------------------------------------------------- 自检
def run_selftest() -> bool:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and cond

    # 1) 反解精确
    for N, kbar, Es in [(100, 6, 0.20), (500, 6, 0.10), (50, 4, 0.45)]:
        cE = c_lower_analytic(N, kbar, Es)
        chk(cE is not None and abs(efficiency_direct(cE * kbar, N) - Es) < 1e-12,
            f"c_E 反解精确 N={N} E*={Es}")

    # 2) ceil 恰好够、减一不够
    N, kbar, Es = 200, 6, 0.12
    K = math.ceil(c_lower_analytic(N, kbar, Es) * kbar)
    chk(efficiency_direct(K, N) >= Es, f"K={K} 满足 E*")
    chk(efficiency_direct(K - 1, N) < Es, f"K={K - 1} 不满足 E*（ceil 无多余）")

    # 3) 天花板
    chk(c_lower_analytic(100, 6, 0.55) is None, "E*=0.55 在 N=100 判定为族不可达")
    chk(abs(efficiency_direct(99, 100) - efficiency_ceiling(100)) < 1e-12, "K=N-1 处取到天花板")

    # 4) L3 反例：双峰 R_t，max 取法会跳到右侧岛
    w = compute_window(N=100, kbar=6, E_star=0.30, R_star=0.40, curve_e=DEMO_E, curve_r=DEMO_R_BUMP)
    chk(w.c_hi is not None and w.c_hi <= 3.6 + 1e-9,
        f"双峰 R_t 下 c_hi={w.c_hi} 取最左段而非右侧岛（max 会给出 6.5）")
    chk(any("孤立可行岛" in x for x in w.warnings), "已报告右侧孤立可行岛")

    # 5) PAVA
    chk(pava_decreasing([0.5, 0.4, 0.45, 0.3]) == [0.45, 0.45, 0.45, 0.3]
        or all(a >= b - 1e-9 for a, b in zip(pava_decreasing([0.5, 0.4, 0.45, 0.3]),
                                             pava_decreasing([0.5, 0.4, 0.45, 0.3])[1:])),
        "PAVA 输出单调不增")

    # 6) 空交集
    w = compute_window(N=100, kbar=6, E_star=0.78, R_star=0.60, curve_e=DEMO_E, curve_r=DEMO_R_MONO)
    chk(w.status == NO_OVERLAP, f"高 E*/R* 报空交集（得到 {w.status}）")

    # 7) 退化单侧
    w = compute_window(N=500, kbar=6, E_star=0.10)
    chk(w.status == OK_DEGRADED and w.c_hi is None, "无 R 曲线时退化为单侧并告警")
    return ok


if __name__ == "__main__":
    sys.exit(_main())
