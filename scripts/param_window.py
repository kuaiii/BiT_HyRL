#!/usr/bin/env python3
"""Two-sided feasible window for ``c``.

Efficiency gives the lower bound and targeted-attack robustness the upper bound;
the script reports both plus the envelope between them.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

EPS = 1e-9

OK = "ok"
OK_DEGRADED = "ok_single_sided"
CEILING_UNREACHABLE = "ceiling_unreachable"
STRUCTURAL_CAP = "structural_cap_binding"
EFF_UNREACHABLE = "efficiency_unreachable_on_curve"
R_UNREACHABLE = "robustness_unreachable_on_curve"
NO_OVERLAP = "empty_intersection"
OUT_OF_DOMAIN = "window_outside_curve_domain"
BAD_INPUT = "bad_input"

STATUS_ZH = {
    OK: "feasible",
    OK_DEGRADED: "feasible, lower bound only",
    CEILING_UNREACHABLE: "family unreachable: E* above the analytic ceiling",
    STRUCTURAL_CAP: "edge budget too small: the degree E* needs exceeds the structural cap",
    EFF_UNREACHABLE: "the efficiency curve never reaches E* on the sampled range",
    R_UNREACHABLE: "the robustness curve stays below R* on the sampled range",
    NO_OVERLAP: "conflict: the lower bound exceeds the upper bound",
    OUT_OF_DOMAIN: "the window lies outside the sampled range",
    BAD_INPUT: "invalid input",
}


def efficiency_direct(K: float, N: int) -> float:
    """E_dir(K) = K(K+3) / (2N(N-1)), where K = c*kbar is the hub degree."""
    return K * (K + 3.0) / (2.0 * N * (N - 1))


def efficiency_ceiling(N: int) -> float:
    """Efficiency ceiling (N+2)/(2N), reached by the single-hub family at K = N-1."""
    return (N + 2.0) / (2.0 * N)


def c_lower_analytic(N: int, kbar: float, E_star: float) -> Optional[float]:
    """Analytic lower bound; None when E* is at or above the ceiling, i.e. the family is unreachable."""
    if N < 2 or kbar <= 0:
        return None
    if E_star <= 0:
        return 0.0
    if E_star >= efficiency_ceiling(N):
        return None
    return (math.sqrt(9.0 + 8.0 * N * (N - 1) * E_star) - 3.0) / (2.0 * kbar)


def c_upper_analytic_symmetric(N: int, kbar: float, E_star: float) -> Optional[float]:
    """Symmetric reference: the c obtained by treating E_dir as an upper bound."""
    return c_lower_analytic(N, kbar, E_star)


def conservativeness_envelope(N: int, K: float) -> Tuple[float, float]:
    """How far E_dir underestimates efficiency, and how far c_E is off, in the worst case."""
    if N < 2 or K <= 0:
        return (1.0, 1.0)
    K = min(float(K), N - 1.0)
    M = (N - 1.0) - K
    ratio_e = 1.0 + M * (2.0 * K + M - 1.0) / (K * (K + 3.0))
    return (ratio_e, math.sqrt(ratio_e))


def k_from_c(c: float, kbar: float) -> float:
    return c * kbar


def degree_budget_cap(N: int, M: int, residual_connected: bool = True) -> Optional[int]:
    """Hard constraint on sparse graphs: the hub degree cap K_cap implied by the edge budget."""
    if M is None or not residual_connected:
        return None
    if M < N - 1:
        raise ValueError(f"M={M} < N-1={N - 1}: the graph cannot be connected")
    return int(min(M - (N - 2), N - 1))


def kbar_threshold(E_star: float) -> float:
    """Applicability threshold of the analytic path with a connected residual: kbar >= 2 + sqrt(8 E*)."""
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


def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """Contiguous True runs, as [start, end) pairs."""
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
    """Start of the trailing feasible run, or None when the tail is infeasible."""
    if not flags or not flags[-1]:
        return None
    i = len(flags) - 1
    while i > 0 and flags[i - 1]:
        i -= 1
    return i


def head_run_end(flags: Sequence[bool]) -> Optional[int]:
    """End (exclusive) of the leading feasible run, or None when the head is infeasible."""
    if not flags or not flags[0]:
        return None
    i = 0
    while i < len(flags) and flags[i]:
        i += 1
    return i


def pava_increasing(ys: Sequence[float], w: Optional[Sequence[float]] = None) -> List[float]:
    """Isotonic regression, used to suppress spurious non-monotonicity from finite samples."""
    n = len(ys)
    if n == 0:
        return []
    w = list(w) if w is not None else [1.0] * n
    vals = [float(y) for y in ys]
    wts = [float(x) for x in w]
    blocks: List[List[float]] = []
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
    """Coarse trend: +1 increasing, -1 decreasing, 0 non-monotone."""
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
            L.append(f"c window    : [{self.c_lo:.4f}, {self.c_hi:.4f}]  width {self.c_hi - self.c_lo:.4f}")
        if (self.K_lo is not None and self.K_hi is not None
                and self.status in (OK, OK_DEGRADED)):
            L.append(f"K window    : [{self.K_lo}, {self.K_hi}]  ({self.K_hi - self.K_lo + 1} integer points)")
        elif self.K_lo is not None and self.status in (OK, OK_DEGRADED):
            L.append(f"K lower     : {self.K_lo}")
        if self.ok:
            L.append(f"K recommend : {self.K_recommend}")
        for k, v in self.detail.items():
            L.append(f"  {k:<10}: {v}")
        for w in self.warnings:
            L.append(f"  ! {w}")
        return "\n".join(L)


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
    """Two-sided feasible window for c: efficiency gives the lower bound, robustness the upper."""
    w = Window(status=OK, N=N, kbar=kbar)

    if N < 2:
        w.status = BAD_INPUT
        w.warnings.append("N must be >= 2")
        return w
    if kbar <= 0:
        w.status = BAD_INPUT
        w.warnings.append("kbar must be > 0")
        return w
    if E_star < 0 or (R_star is not None and R_star < 0):
        w.status = BAD_INPUT
        w.warnings.append("E* and R* cannot be negative")
        return w

    w.detail["E_dir_ceiling"] = round(efficiency_ceiling(N), 4)

    if curve_e is not None:
        xs, ys = _sort_curve(curve_e)
        if len(xs) < 3:
            w.status = BAD_INPUT
            w.warnings.append("too few samples on the efficiency curve (<3)")
            return w
        flags = [y >= E_star - EPS for y in ys]
        trend = _trend(xs, ys)
        if trend <= 0:
            w.warnings.append(
                f"the efficiency curve looks {'non-monotone' if trend == 0 else 'decreasing'}, contrary to the assumption: "
                "check the metric definition and the direction of the independent variable"
            )
        idx = tail_run_start(flags)
        if idx is None:
            w.status = EFF_UNREACHABLE
            w.warnings.append(
                f"efficiency does not reach E*={E_star} on [{xs[0]:g}, {xs[-1]:g}]; "
                + ("E* is at or above the analytic ceiling, so the family must change"
                   if E_star >= efficiency_ceiling(N) else "widen the c sampling range")
            )
            w.detail["E_max_on_curve"] = round(max(ys), 4)
            return w
        runs = _runs(flags)
        if len(runs) > 1:
            w.warnings.append(
                f"the feasible set of the efficiency curve is disconnected ({len(runs)} runs); using the rightmost: "
                f"[{xs[runs[-1][0]]:.3g}, {xs[runs[-1][1] - 1]:.3g}]; {len(runs) - 1} isolated runs on the left ignored"
            )
        w.c_lo = xs[idx]
        w.source_lower = "numeric"
        w.detail["E_at_c_lo"] = round(_interp(xs, ys, w.c_lo) or float("nan"), 4)
    else:
        cE = c_lower_analytic(N, kbar, E_star)
        if cE is None:
            w.status = CEILING_UNREACHABLE
            w.warnings.append(
                f"E*={E_star} >= the single-hub ceiling (N+2)/(2N)={efficiency_ceiling(N):.4f}: "
                "no value of c reaches it, so the family must change rather than c"
            )
            return w
        w.c_lo = cE
        w.source_lower = "analytic(E_dir, sufficient but not necessary)"
        w.detail["E_dir_at_c_lo"] = round(efficiency_direct(cE * kbar, N), 6)
        ratio_e, ratio_c = conservativeness_envelope(N, cE * kbar)
        w.detail["conservative_x"] = round(ratio_c, 2)
        if ratio_c >= 1.15:
            w.warnings.append(
                f"the analytic bound assumes distance 2 for every residual pair, so c can be up to {ratio_c:.2f}x too large"
                f" (E underestimated by up to {ratio_e:.1f}x, K={cE * kbar:.1f}); use the numeric inverse when E(c) is measured"
            )

    if curve_r is None:
        w.status = OK_DEGRADED
        target = R_star if R_star is not None else float("nan")
        w.warnings.append(
            f"no R_t(c) curve, so there is no upper bound; the window is one-sided (c >= {w.c_lo:.4f})."
            f"random-failure robustness increases with c, so it gives another lower bound (R*={target})"
        )
    else:
        xs, ys = _sort_curve(curve_r)
        if len(xs) < 3:
            w.status = BAD_INPUT
            w.warnings.append("too few samples on the robustness curve (<3)")
            return w
        if R_star is None:
            w.status = BAD_INPUT
            w.warnings.append("an R curve was given without R*")
            return w
        thr = R_star + delta_r

        flags = [y >= thr - EPS for y in ys]
        end = head_run_end(flags)
        if end is None:
            w.status = R_UNREACHABLE
            w.warnings.append(f"R_t is already below R*={R_star} at the smallest c={xs[0]:g} (delta={delta_r})")
            w.detail["R_at_c_min"] = round(ys[0], 4)
            return w
        runs = _runs(flags)
        if len(runs) > 1:
            island = runs[1:]
            w.warnings.append(
                "R_t is non-monotone: an isolated feasible run on the right "
                + ", ".join(f"[{xs[a]:.3g},{xs[b - 1]:.3g}]" for a, b in island)
                + "; ignored, since taking the maximum would select it"
            )
        w.c_hi = xs[end - 1]
        w.source_upper = "leftmost run (numeric)" if len(runs) > 1 else "numeric"
        w.detail["R_at_c_hi"] = round(ys[end - 1], 4)

        ys_sm = pava_decreasing(ys) if smooth else list(ys)
        trend = _trend(xs, ys_sm)
        if trend >= 0:
            w.warnings.append(
                f"R_t looks {'increasing' if trend == 1 else 'non-monotone'}: an increasing metric gives a lower bound, not an upper one"
            )
        elif smooth and any(abs(a - b) > 1e-6 for a, b in zip(ys, ys_sm)):
            flags_sm = [y >= thr - EPS for y in ys_sm]
            end_sm = head_run_end(flags_sm)
            if end_sm is not None and end_sm - 1 < end - 1:
                w.detail["c_hi_smoothed"] = round(xs[end_sm - 1], 4)
                w.warnings.append(
                    f"the isotonic fit gives a tighter upper bound ({xs[end_sm - 1]:.4g} < {w.c_hi:.4g})"
                )

    lo_b, hi_b = c_bounds
    if w.c_lo is not None and lo_b > 0:
        w.c_lo = max(w.c_lo, lo_b)
    if w.c_hi is not None and hi_b != float("inf"):
        w.c_hi = min(w.c_hi, hi_b)
        if w.c_hi < lo_b:
            w.status = OUT_OF_DOMAIN
            return w

    K_lo_raw = w.c_lo * kbar
    K_lo = int(math.ceil(K_lo_raw - 1e-9))
    K_lo = max(K_lo, int(K_min))
    K_cap = int(N - 1)

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
                f"edge budget too small: efficiency needs K >= {K_lo}, but the hub degree is capped at M-(N-2)={K_struct}"
                f" (c_cap={K_struct / kbar:.2f}); the analytic path covers "
                f"E* <= E_dir(K_cap)={efficiency_direct(K_struct, N):.4f}；"
                "if the measured E(c) is well above E_dir, use the numeric inverse"
            )
        else:
            w.status = OUT_OF_DOMAIN
            w.warnings.append(f"lower bound K={K_lo} exceeds the hub degree cap {K_cap}: the family is unreachable")
        return w

    if w.c_hi is not None:
        K_hi = int(math.floor(w.c_hi * kbar + 1e-9))
        K_hi = min(K_hi, K_cap)
        if K_hi < K_lo:
            w.status = NO_OVERLAP
            w.detail["K_need_min"] = K_lo
            w.detail["K_need_max"] = K_hi
            w.warnings.append(
                f"empty intersection: K must be >= {K_lo} (efficiency) and <= {K_hi} (robustness), a gap of {K_lo - K_hi}."
                "this is not a matter of tuning: change the family or relax E*/R*"
            )
            if w.source_lower.startswith("analytic"):
                w.warnings.append(
                    "the lower bound comes from E_dir (sufficient but not necessary); the real E(c) is usually higher, "
                    "so this conflict may be an artefact; check it against a measured E(c) curve"
                )
            if w.c_hi - w.c_lo <= 1e-9:
                w.warnings.append(
                    f"the feasible set collapses to the single point c={w.c_lo:.4g}: K would have to be exactly "
                    f"{w.c_lo * kbar:.2f}, which is not an integer, so the K set is empty by construction"
                )
            return w
        w.K_lo, w.K_hi = K_lo, K_hi
    else:
        w.K_lo = K_lo

    if w.c_hi is not None and curve_r is not None:
        xs, _ = _sort_curve(curve_r)
        dx = min((xs[i + 1] - xs[i] for i in range(len(xs) - 1)), default=0.0)
        if dx > 0 and (w.c_hi - w.c_lo) < dx:
            w.warnings.append(
                f"window width {w.c_hi - w.c_lo:.4f} < grid spacing {dx:.4f}: the bounds are set by the grid, "
                "so they will not reproduce across seeds"
            )
    return w


def load_curve(path: str) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for rec in csv.reader(f):
            if len(rec) < 2:
                continue
            try:
                x, y = float(rec[0]), float(rec[1])
            except ValueError:
                continue
            rows.append((x, y))
    if not rows:
        raise SystemExit(f"no numeric rows in {path}")
    return rows


DEMO_E = [(1.0, 0.18), (1.5, 0.27), (2.0, 0.35), (2.5, 0.45), (3.0, 0.52), (3.6, 0.63),
          (4.2, 0.70), (5.0, 0.76), (6.0, 0.80), (6.5, 0.82)]
DEMO_R_MONO = [(2.5, 0.72), (3.0, 0.66), (3.6, 0.58), (4.2, 0.50), (4.8, 0.44), (5.4, 0.417), (6.0, 0.34), (6.5, 0.28)]
DEMO_R_BUMP = [(2.5, 0.62), (3.0, 0.55), (3.6, 0.44), (4.2, 0.33), (4.8, 0.28), (5.4, 0.33), (6.0, 0.44), (6.5, 0.52)]


def _demo() -> None:
    cases = [
        ("A normal: two-sided window", dict(N=100, kbar=6, E_star=0.35, R_star=0.50,
                                  curve_e=DEMO_E, curve_r=DEMO_R_MONO)),
        ("B counter-example: bimodal R_t", dict(N=100, kbar=6, E_star=0.30, R_star=0.40,
                                       curve_e=DEMO_E, curve_r=DEMO_R_BUMP)),
        ("C ceiling: E*=0.55", dict(N=100, kbar=6, E_star=0.55, R_star=0.50)),
        ("D genuine conflict: high E* and high R*", dict(N=100, kbar=6, E_star=0.78, R_star=0.60,
                                            curve_e=DEMO_E, curve_r=DEMO_R_MONO)),
        ("E analytic only (conservative)", dict(N=100, kbar=6, E_star=0.35, R_star=0.50,
                                         curve_r=DEMO_R_MONO)),
        ("F degenerate: no R curve", dict(N=500, kbar=6, E_star=0.10)),
    ]
    for i, (name, kw) in enumerate(cases, 1):
        w = compute_window(**kw)
        print(f"\n=== {i}. {name} ===")
        print(w.report())
    print("\n=== analytic vs numeric c_E ===")
    for N in (100, 500, 1000):
        for E_star in (0.10, 0.20, 0.35):
            cE = c_lower_analytic(N, 6.0, E_star)
            if cE is None:
                continue
            K = cE * 6
            ratio_e, ratio_c = conservativeness_envelope(N, K)
            print(f"N={N:5d} E*={E_star:.2f}: c_E_analytic={cE:7.3f}  K={K:7.2f}  "
                  f"K_ceil={math.ceil(K)}  E_dir(K)={efficiency_direct(math.ceil(K), N):.4f}  "
                  f"E underestimated by <= {ratio_e:5.1f}x, c too large by <= {ratio_c:.2f}x")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="two-sided feasible window for c (efficiency lower bound, robustness upper bound)")
    ap.add_argument("--N", type=int)
    ap.add_argument("--kbar", type=float, default=6.0)
    ap.add_argument("--E-star", dest="E_star", type=float)
    ap.add_argument("--R-star", dest="R_star", type=float, default=None)
    ap.add_argument("--curve-e", dest="curve_e", default=None, help="CSV: c,E(c)")
    ap.add_argument("--curve-r", dest="curve_r", default=None, help="CSV: c,R_t(c)")
    ap.add_argument("--Kmin", type=int, default=2)
    ap.add_argument("--edges", "--M", dest="M", type=int, default=None,
                    help="edge count (required for sparse ISPs; caps K at M-(N-2))")
    ap.add_argument("--allow-hub-star", dest="allow_hub_star", action="store_true",
                    help="drop the connected-residual assumption (loosens the bound, but robustness usually collapses)")
    ap.add_argument("--delta-r", dest="delta_r", type=float, default=0.0,
                    help="robustness margin: require R_t >= R* + delta")
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
        ap.error("--N and --E-star are required (or use --demo)")

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


def run_selftest() -> bool:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and cond

    for N, kbar, Es in [(100, 6, 0.20), (500, 6, 0.10), (50, 4, 0.45)]:
        cE = c_lower_analytic(N, kbar, Es)
        chk(cE is not None and abs(efficiency_direct(cE * kbar, N) - Es) < 1e-12,
            f"exact inverse for N={N}, E*={Es}")

    N, kbar, Es = 200, 6, 0.12
    K = math.ceil(c_lower_analytic(N, kbar, Es) * kbar)
    chk(efficiency_direct(K, N) >= Es, f"K={K} meets E*")
    chk(efficiency_direct(K - 1, N) < Es, f"K={K - 1} misses E*, so the ceiling is tight")

    chk(c_lower_analytic(100, 6, 0.55) is None, "E*=0.55 is unreachable at N=100")
    chk(abs(efficiency_direct(99, 100) - efficiency_ceiling(100)) < 1e-12, "the ceiling is reached at K=N-1")

    w = compute_window(N=100, kbar=6, E_star=0.30, R_star=0.40, curve_e=DEMO_E, curve_r=DEMO_R_BUMP)
    chk(w.c_hi is not None and w.c_hi <= 3.6 + 1e-9,
        f"with a bimodal R_t, c_hi={w.c_hi} comes from the leftmost run, not the right-hand island")
    chk(any("isolated feasible run" in x for x in w.warnings), "the isolated run is reported")

    chk(pava_decreasing([0.5, 0.4, 0.45, 0.3]) == [0.45, 0.45, 0.45, 0.3]
        or all(a >= b - 1e-9 for a, b in zip(pava_decreasing([0.5, 0.4, 0.45, 0.3]),
                                             pava_decreasing([0.5, 0.4, 0.45, 0.3])[1:])),
        "the isotonic fit is non-increasing")

    w = compute_window(N=100, kbar=6, E_star=0.78, R_star=0.60, curve_e=DEMO_E, curve_r=DEMO_R_MONO)
    chk(w.status == NO_OVERLAP, f"high E*/R* reports an empty intersection (got {w.status})")

    w = compute_window(N=500, kbar=6, E_star=0.10)
    chk(w.status == OK_DEGRADED and w.c_hi is None, "without an R curve the window is one-sided")
    return ok


if __name__ == "__main__":
    sys.exit(_main())
