"""Percolation robustness of the bimodal family, combined with efficiency.

One hub has degree K, the other N-1 nodes have degree k_min, and the edge count is
conserved: M = (K + (N-1) k_min) / 2.  Hence k_min = (2M - K) / (N-1), so k_min is
not a free parameter; k_min = 1 is the star limit.
"""
from typing import Dict, Optional, Tuple
import math


def bimodal(N: int, M: float, K: float, r: Optional[float] = None) -> Dict:
    """Degree-distribution moments of the bimodal family; r defaults to 1/N."""
    if r is None:
        r = 1.0 / N
    kbar = 2.0 * M / N
    kmin = (kbar - r * K) / (1.0 - r)
    m1 = (1.0 - r) * kmin + r * K
    m2 = (1.0 - r) * kmin ** 2 + r * K ** 2
    return dict(r=r, kbar=kbar, k_min=kmin, m1=m1, m2=m2,
                kappa=(m2 / m1 if m1 > 0 else float("nan")))


def fc_random_moments(r: float, K: float, m1: float, m2: float) -> float:
    """Random-failure percolation threshold (7.64) from (r, K, m1, m2)."""
    num = m2 - 2 * r * m1 * K - 2 * (1 - r) * m1 + r * K ** 2
    den = m2 - 2 * r * m1 * K - (1 - r) * m1 + r * K ** 2
    return num / den if den != 0 else float("nan")


def fc_targeted_moments(r: float, K: float, m1: float,
                        m2: float) -> Tuple[float, int]:
    """Targeted-attack percolation threshold (7.65)/(7.66); returns the value and branch."""
    A = m1 - r * K
    if A <= 0 or (A - (1 - r)) <= 0:
        return float("nan"), 0
    f1 = r + (1 - r) / A * (m1 * (A - 2 * (1 - r)) / (A - (1 - r)) - r * K)
    if f1 > r:
        return f1, 1
    den = K * (K - 1) * (1 - r)
    if den == 0:
        return float("nan"), 0
    return (m1 * m1 - 2 * r * m1 * K + r * K ** 2 - 2 * (1 - r) * m1) / den, 2


def fc_random(N: int, M: float, K: float, r: Optional[float] = None) -> float:
    """Random-failure percolation threshold (7.64); r defaults to 1/N."""
    p = bimodal(N, M, K, r)
    return fc_random_moments(p["r"], K, p["m1"], p["m2"])


def fc_targeted(N: int, M: float, K: float,
                r: Optional[float] = None) -> Tuple[float, int]:
    """Targeted-attack percolation threshold (7.65)/(7.66); returns the value and branch."""
    p = bimodal(N, M, K, r)
    return fc_targeted_moments(p["r"], K, p["m1"], p["m2"])


def robustness_fc(N: int, M: float, K: float,
                  r: Optional[float] = None) -> Dict:
    fr = fc_random(N, M, K, r)
    ft, br = fc_targeted(N, M, K, r)
    return dict(fc_rand=fr, fc_tar=ft, branch=br, R=0.5 * (fr + ft))


def kmin_bounds(N: int, M: float) -> Tuple[float, float]:
    """Range of k_min attainable over K in [1, N-1]."""
    kbar = 2.0 * M / N
    return ((kbar * N - (N - 1)) / (N - 1), (kbar * N - 1) / (N - 1))


def sweep(N: int, M: float):
    """Return [{K, c, k_min, E, fc_rand, fc_tar, R}] for the feasible K only."""
    from param_window import efficiency_direct
    rows = []
    for K in range(1, N):
        p = bimodal(N, M, K)
        if p["k_min"] < 1.0 - 1e-9:
            continue
        r = robustness_fc(N, M, K)
        if not (r["fc_rand"] == r["fc_rand"] and r["fc_tar"] == r["fc_tar"]):
            continue
        rows.append(dict(K=K, c=K / p["kbar"], k_min=p["k_min"],
                         E=efficiency_direct(K, N), fc_rand=r["fc_rand"],
                         fc_tar=r["fc_tar"], R=r["R"]))
    return rows


AGG = {
    "fc_avg":    lambda d: 0.5 * (d["E"] + d["R"]),
    "e_plus_ft": lambda d: 0.5 * (d["E"] + d["fc_tar"]),
    "min":       lambda d: 0.5 * (d["E"] + min(d["fc_rand"], d["fc_tar"])),
    "harmonic":  lambda d: 0.5 * (d["E"] + 2 / (1 / max(d["fc_rand"], 1e-12)
                                               + 1 / max(d["fc_tar"], 1e-12))),
}


def optimize(N: int, M: float, mode: str = "fc_avg",
             E_star: float = None, R_star: float = None):
    """Best design, with a 'corner' flag for K = N-1."""
    rows = sweep(N, M)
    if not rows:
        return dict(status="infeasible", note="k_min < 1 for every K")
    fn = AGG[mode]
    cand = rows
    if E_star is not None:
        cand = [d for d in cand if d["E"] >= E_star - 1e-12]
    if R_star is not None:
        cand = [d for d in cand if min(d["fc_rand"], d["fc_tar"]) >= R_star - 1e-12]
    if not cand:
        return dict(status="empty", note="no feasible K under these constraints")
    b = max(cand, key=fn)
    b = dict(b)
    b["status"] = "ok"
    b["objective"] = fn(b)
    b["corner"] = (b["K"] == rows[-1]["K"])
    b["k_min_band"] = (min(d["k_min"] for d in rows), max(d["k_min"] for d in rows))
    return b


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--M", type=float, default=300.0)
    ap.add_argument("--mode", default="fc_avg", choices=list(AGG))
    ap.add_argument("--E-star", dest="E_star", type=float, default=None)
    ap.add_argument("--R-star", dest="R_star", type=float, default=None)
    a = ap.parse_args()
    b = optimize(a.N, a.M, a.mode, a.E_star, a.R_star)
    if b["status"] != "ok":
        print(b)
    else:
        print(f"mode={a.mode}  N={a.N}  M={a.M:g}  kbar={2*a.M/a.N:.2f}")
        print(f"  K* = {b['K']}   c* = {b['c']:.2f}   k_min = {b['k_min']:.2f}")
        print(f"  E = {b['E']:.4f}   fc_rand = {b['fc_rand']:.4f}   "
              f"fc_tar = {b['fc_tar']:.4f}   R = {b['R']:.4f}")
        print(f"  objective = {b['objective']:.4f}   corner(K=N-1)? {b['corner']}")
        print(f"  k_min range = [{b['k_min_band'][0]:.3f}, {b['k_min_band'][1]:.3f}]")


def _monotone_check(N: int, M: float):
    """Monotonicity of the three curves over the feasible K."""
    rows = sweep(N, M)
    def mono(key, sign):
        v = [d[key] for d in rows]
        bad = sum(1 for a, b in zip(v, v[1:]) if sign * (b - a) < -1e-9)
        return bad == 0, bad
    return dict(
        E=mono("E", +1), fc_rand=mono("fc_rand", +1),
        fc_tar=mono("fc_tar", -1),
        kmin_lo=min(d["k_min"] for d in rows), kmin_hi=max(d["k_min"] for d in rows),
    )


def _K_from_fc(N: int, M: float, key: str, thr: float, rising: bool):
    """Critical K on the scan grid: the smallest feasible K when rising, otherwise the largest."""
    rows = sweep(N, M)
    ok = [d["K"] for d in rows if d[key] >= thr - 1e-12]
    if not ok:
        return None
    return min(ok) if rising else max(ok)


def solve(N: int, M: float, E_star: float = None, R_r: float = None,
          R_t: float = None, kmin_min: float = 1.0):
    """Solve the constrained problem; F = (E + R) / 2 increases with K, so take the upper end."""
    from param_window import c_lower_analytic
    rows = sweep(N, M)
    if not rows:
        return dict(status="infeasible", reason="k_min < 1 for every K")
    K_hi_struct = max(d["K"] for d in rows if d["k_min"] >= kmin_min - 1e-12)

    L, Ub, why = 1, K_hi_struct, []
    if E_star is not None:
        kr = c_lower_analytic(N, M * 2.0 / N, E_star)
        if kr is None:
            return dict(status="infeasible", reason="E* above the ceiling (N+2)/(2N)")
        kE = int(math.ceil(kr * (2.0 * M / N) - 1e-9))
        L = max(L, kE); why.append(f"E*>={E_star} -> K>={kE}")
    if R_r is not None:
        kr = _K_from_fc(N, M, "fc_rand", R_r, rising=True)
        if kr is None:
            return dict(status="infeasible",
                        reason=f"fc_rand stays below {R_r} (max {max(d['fc_rand'] for d in rows):.4f})")
        L = max(L, kr); why.append(f"fc_rand>={R_r} -> K>={kr}")
    if R_t is not None:
        kt = _K_from_fc(N, M, "fc_tar", R_t, rising=False)
        if kt is None:
            return dict(status="infeasible",
                        reason=f"fc_tar stays below {R_t} (max {max(d['fc_tar'] for d in rows):.4f})")
        Ub = min(Ub, kt); why.append(f"fc_tar>={R_t} -> K<={kt}")

    if L > Ub:
        gap = L - Ub
        return dict(status="empty", L=L, U=Ub, gap=gap,
                    reason=f"lower bound {L} > upper bound {Ub} by {gap} K steps; the two requirements conflict")

    byK = {d["K"]: d for d in rows}
    K = Ub
    while K >= L:
        d = byK.get(K)
        if d and (R_t is None or d["fc_tar"] >= R_t - 1e-12):
            break
        K -= 1
    d = byK[K]
    return dict(status="ok", K=K, c=d["c"], k_min=d["k_min"], E=d["E"],
                fc_rand=d["fc_rand"], fc_tar=d["fc_tar"], R=d["R"],
                F=0.5 * (d["E"] + d["R"]), L=L, U=Ub,
                binding=("fc_tar" if Ub < K_hi_struct else "structural_cap"),
                constraints=why)


def pareto_front(N: int, M: float):
    """Non-dominated set when both E and R are maximised, in ascending K."""
    rows = sweep(N, M)
    nd = []
    for d in rows:
        if not any((e["E"] >= d["E"] - 1e-15 and e["R"] >= d["R"] - 1e-15
                    and (e["E"] > d["E"] + 1e-15 or e["R"] > d["R"] + 1e-15))
                   for e in rows):
            nd.append(d)
    return sorted(nd, key=lambda d: d["K"])


def slope_profile(front, log_scale: bool = True):
    """Per-segment |dR/dE| (or |dlnR/dlnE|) along the front, and its maximum."""
    s = []
    for i in range(len(front) - 1):
        a, b = front[i], front[i + 1]
        if log_scale:
            dx = math.log(b["E"]) - math.log(a["E"])
            dy = math.log(b["R"]) - math.log(a["R"])
        else:
            dx, dy = b["E"] - a["E"], b["R"] - a["R"]
        s.append(abs(dy / dx) if abs(dx) > 1e-15 else 0.0)
    return s


def slope_span(N: int, M: float, log_scale: bool = True) -> float:
    s = slope_profile(pareto_front(N, M), log_scale)
    return max(s) if s else 1.0


def weight_from_lambda(lam: float, S: float) -> float:
    """Convert a lambda setting into the weight w applied to E."""
    return lam * S / (1.0 + lam * S)


def evaluate(N: int, M: float, lam: float, obj: str = "log",
             reparam: bool = True):
    """Best design for a preference lambda (0 = most robust, 1 = most efficient)."""
    rows = sweep(N, M)
    front = pareto_front(N, M)
    S = slope_span(N, M, log_scale=(obj == "log"))
    w = weight_from_lambda(lam, S) if reparam else lam
    if obj == "log":
        a = [math.log(max(d["E"], 1e-300)) for d in rows]
        b = [math.log(max(d["R"], 1e-300)) for d in rows]
    else:
        E_cap = (N + 2) / (2.0 * N)
        a = [d["E"] / E_cap for d in rows]
        b = [d["R"] for d in rows]
    v = [w * x + (1 - w) * y for x, y in zip(a, b)]
    i = max(range(len(v)), key=lambda j: (v[j], rows[j]["K"]))
    d = dict(rows[i])
    lo, hi = front[0]["K"], front[-1]["K"]
    d.update(obj=obj, lam=lam, w=w, S=S, reparam=reparam,
             t=((d["K"] - lo) / (hi - lo) if hi > lo else 1.0),
             front=(lo, hi),
             degenerate=(hi - lo < 3))
    return d


def dial(N: int, M: float, t: float):
    """Pick a design by its position t on the front (0 = most robust, 1 = most efficient)."""
    front = pareto_front(N, M)
    if len(front) == 1:
        d = dict(front[0]); d.update(t=1.0, degenerate=True); return d
    i = int(round(max(0.0, min(1.0, t)) * (len(front) - 1)))
    d = dict(front[i])
    d.update(t=i / (len(front) - 1), degenerate=False)
    return d
