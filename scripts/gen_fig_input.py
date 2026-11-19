"""gen_fig_input.py —— 由 input_sweep.csv 生成论文图 2（设计点随输入的变化）。

三幅子图，横轴都是输入 <k>（对数刻度），三色对应 n=100/300/1000：
  (a) 可行域：下端 k-（灰点线）与上端 k+（彩色实线/虚线），两线相交即关窗
  (b) 设计点处的 E/E0
  (c) 设计点处的 f/f0

绘图约定（2026-11-18 改版，替换原手写 SVG + Edge 打印路线）：
  1. 用 matplotlib 直出 PDF，rcParams 与 paper_figures.py 完全一致
     （Times New Roman 8 pt、轴线 0.7 pt、去掉上/右边框、无网格），
     与正文其余 matplotlib 插图风格统一；
  2. 画布宽度 = \\textwidth = 494.5 pt，避免 LaTeX 缩放导致字号偏小；
  3. 三色：n=100 (255,230,148)、n=300 (239,147,158)、n=1000 (168,233,227)；
  4. 线型/记号编码的是"设计规则"，不是 n：
       实线 + 实心点 = 非空窗口，几何中点设计；
       虚线 + 空心点 = 空窗口，下界回退设计（k_max = k-，见 input_sweep_fallback.py）；
     曲线因此从非空窗口连续延伸到空窗口，不会在关窗处无解释地断掉；
  5. <k>=2 的退化窗（窗口非空但构造违反连通性，正文明确排除）既无设计点也无
     性能点：曲线上有位置，三个面板都不落点。

用法：
    python gen_fig_input.py --csv ../results/input_sweep.csv --out fig.pdf [--png p.png]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "mathtext.fontset": "stix",         # 数学字形用 STIX（Times 风格），避免 DejaVu 无衬线混入
    "mathtext.default": "bf",           # 数学标签同样加粗
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.95,
    "xtick.major.width": 0.95,
    "ytick.major.width": 0.95,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
})

# n -> 颜色（用户指定三色）
COLOURS = {100: "#FFE694", 300: "#EF939E", 1000: "#A8E9E3"}
KBAR_INK = "#5A5A5A"
REF_INK = "#BEBEBE"


def darken(colour: str, factor: float = 0.80) -> str:
    """同色压暗，用作实心标记的描边（深灰描边会让浅色填充看起来像空心圆环）。"""
    value = colour.lstrip("#")
    return "#" + "".join(
        f"{round(int(value[i:i + 2], 16) * factor):02X}" for i in (0, 2, 4))

KBARS = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
XR = (1.92, 17.2)
WIDTH_PT = 494.50888          # \textwidth, 见 main.log
FIGSIZE = (WIDTH_PT / 72.0, 1.98)

#: 空窗段（回退设计）的虚线样式，与 k- 的点线 (0, (1, 1.6)) 拉开距离。
EMPTY_DASH = (0, (4.0, 1.6))


def load(path: Path, fallback: Path | None = None) -> dict:
    cells = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cells[(int(float(row["n"])), round(float(row["kbar"]), 3))].append(row)

    out = {}
    for key, rows in cells.items():
        first = rows[0]
        lo, hi = int(first["lo"]), int(first["hi"])
        e0, f0 = float(first["e0"]), float(first["f0"])
        state = first["status"]                     # ok / degen / empty
        # 几何中点设计点：k = round(sqrt(lo*hi))，仅在窗口非空时有定义
        kmid = round(math.sqrt(lo * hi)) if hi >= lo else None
        at_mid = [r for r in rows
                  if kmid is not None and r["k"] and int(r["k"]) == kmid
                  and r["phi"] not in ("nan", "")]
        cell = {"lo": lo, "hi": hi, "state": state, "ok": False}
        if at_mid:
            row = at_mid[0]
            cell.update(ok=True,
                        ee=float(row["e"]) / e0,
                        ff=(float(row["fr"]) + float(row["ft"])) / f0)
        out[key] = cell

    # 空窗口格补上"下界回退设计"的实测值（k_max = k-，input_sweep_fallback.py）
    if fallback is not None and Path(fallback).exists():
        with open(fallback, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                key = (int(float(row["n"])), round(float(row["kbar"]), 3))
                if key in out:
                    out[key].update(fb=True, ee_fb=float(row["ee"]),
                                    ff_fb=float(row["ff"]),
                                    phi_fb=float(row["phi"]))
    return out


def draw(data: dict, out_pdf: Path, out_png: Path | None) -> None:
    ns = [100, 300, 1000]
    xs = sorted({key[1] for key in data})

    figure, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    # top 留一条给整幅图共用的单行图例（6 条：三种 n + k- + 两种设计规则）
    figure.subplots_adjust(left=0.056, right=0.998, bottom=0.176, top=0.845, wspace=0.26)

    def style_logx(axis) -> None:
        axis.set_xscale("log")
        axis.set_xlim(*XR)
        axis.set_xticks(KBARS)
        axis.set_xticklabels([f"{v:g}" for v in KBARS])
        axis.xaxis.set_minor_locator(NullLocator())

    # ---------------- (a) 可行域 ----------------
    ax = axes[0]
    style_logx(ax)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel(r"$k_{\max}$")
    ax.set_title("(a)", loc="left", pad=2)

    # 下端 k-：三种 n 的取值完全相同（只依赖 <k>），画一条灰点线
    lower = [(v, data[(1000, v)]["lo"]) for v in xs if (1000, v) in data]
    ax.plot([p[0] for p in lower], [p[1] for p in lower],
            color=KBAR_INK, lw=1.1, ls=(0, (1, 1.6)), zorder=2)

    legend_handles = []
    for n in ns:
        cells = [(v, data[(n, v)]) for v in xs if (n, v) in data]
        colour = COLOURS[n]
        # 曲线分两段，编码的是设计规则而不是 n：
        #   实线 + 实心点 = 非空窗口，几何中点设计；
        #   虚线 + 空心点 = 空窗口，下界回退设计（k_max = k-）。
        # 两段在最后一个非空格处共点，曲线因此连续，不会在关窗处无解释地断掉。
        keep = next((i for i, (_, c) in enumerate(cells) if c["state"] == "empty"),
                    len(cells))
        live, dead = cells[:keep], cells[keep:]
        if len(live) > 1:
            ax.plot([v for v, _ in live], [c["hi"] for _, c in live],
                    color=colour, lw=1.5, ls="-", solid_capstyle="round", zorder=3)
        if len(dead) >= 1:
            seg = (live[-1:] if live else []) + dead
            if len(seg) > 1:
                ax.plot([v for v, _ in seg], [c["hi"] for _, c in seg],
                        color=colour, lw=1.5, ls=EMPTY_DASH, dash_capstyle="round",
                        zorder=3)
        # 实心点：窗口非空、且实测到几何中点指标；degen（<k>=2，正文已排除）不画
        for v, cell in live:
            if cell["ok"]:
                ax.plot([v], [cell["hi"]], marker="o", ms=4.0, mfc=colour,
                        mec=colour, mew=0.8, ls="none", zorder=4)
        # 空心点：空窗口（k+ < k-）。描边用同色压暗版，浅黄在白底上才看得见
        for v, cell in dead:
            ax.plot([v], [cell["hi"]], marker="o", ms=4.2, mfc="white",
                    mec=darken(colour, 0.86), mew=1.2, ls="none", zorder=4)
        legend_handles.append(Line2D([], [], color=colour, lw=1.5, marker="o", ms=4.0,
                                     mfc=colour, mec=colour, mew=0.8,
                                     label=f"$n={n}$"))

    # 图例统一放到整幅图顶部、排成一行（用户要求）：6 条——三种 n、k- 点线、
    # 以及两种设计规则（实线+实心=几何中点，虚线+空心=下界回退）。
    # 面板内不再放任何图例，三个面板的绘图区因此完全干净。
    legend_handles.append(Line2D([], [], color=KBAR_INK, lw=1.1, ls=(0, (1, 1.6)),
                                 label=r"$k^{-}$"))
    legend_handles.append(Line2D([], [], color="#4D4D4D", lw=1.3, marker="o", ms=4.0,
                                 mfc="#4D4D4D", label="midpoint"))
    legend_handles.append(Line2D([], [], color="#4D4D4D", lw=1.3, ls=EMPTY_DASH,
                                 marker="o", ms=4.2, mfc="white", mew=1.1,
                                 label="fallback"))

    # ---------------- (b)(c) 设计点处的指标 ----------------
    for index, key, ylim, ticks, label, tag in (
            (1, "ee", (0.65, 1.02), [0.7, 0.8, 0.9, 1.0],
             r"$\mathcal{E}/\mathcal{E}_0$", "(b)"),
            (2, "ff", (0.90, 1.30), [0.9, 1.0, 1.1, 1.2, 1.3],
             r"$f/f_0$", "(c)")):
        axis = axes[index]
        style_logx(axis)
        axis.set_ylim(*ylim)
        axis.set_yticks(ticks)
        axis.set_ylabel(label)
        axis.set_title(tag, loc="left", pad=2)
        # 参考线用细点线：回退段是长虚线，两者若都用短划，(b) 里回退点又恰好
        # 落在 y=1 上（E/E0≈1.00），会糊成一条线。
        axis.axhline(1.0, color=REF_INK, lw=0.9, ls=(0, (1, 2.2)), zorder=1)
        for n in ns:
            colour = COLOURS[n]
            # 非空窗口：实线 + 实心点（几何中点设计）
            pts = [(v, data[(n, v)][key]) for v in xs
                   if (n, v) in data and data[(n, v)]["ok"]
                   and data[(n, v)]["state"] == "ok"]
            # 空窗口：虚线 + 空心点（下界回退设计，k_max = k-）
            fb = [(v, data[(n, v)][key + "_fb"]) for v in xs
                  if (n, v) in data and data[(n, v)].get("fb")
                  and data[(n, v)]["state"] == "empty"]
            if not pts and not fb:
                continue
            if len(pts) > 1:
                axis.plot([v for v, _ in pts], [p for _, p in pts], color=colour,
                          lw=1.5, ls="-", marker="o", ms=4.0, mfc=colour,
                          mec=colour, mew=0.8, zorder=3)
            elif len(pts) == 1:
                axis.plot([pts[0][0]], [pts[0][1]], color=colour, ls="none",
                          marker="o", ms=4.0, mfc=colour, mec=colour, mew=0.8,
                          zorder=3)
            if fb:
                seg = (pts[-1:] if pts else []) + fb
                if len(seg) > 1:
                    axis.plot([v for v, _ in seg], [p for _, p in seg], color=colour,
                              lw=1.5, ls=EMPTY_DASH, dash_capstyle="round",
                              zorder=3)
                for v, value in fb:
                    axis.plot([v], [value], marker="o", ms=4.2, mfc="white",
                              mec=darken(colour, 0.86), mew=1.2, ls="none", zorder=4)

    # 关键数值标注：只标两端——(b) 最稀输入处的效率损失、(c) 的最大增益，
    # 中间点数值由曲线形状已能读出，全部标出会在 <k>=4~5 处相互压字。
    def note(axis, x, y, text, offset, align="left"):
        axis.annotate(text, (x, y), textcoords="offset points", xytext=offset,
                      ha=align, va="center", fontsize=7, color="#333333", zorder=6,
                      arrowprops=dict(arrowstyle="-", lw=0.6, color="#8A8A8A",
                                      shrinkA=1.0, shrinkB=1.6))

    for n, offset, align in ((100, (0, 13), "center"), (300, (13, 0), "left"),
                             (1000, (9, -7), "left")):
        cell = data[(n, 2.5)]
        note(axes[1], 2.5, cell["ee"], f"{cell['ee']:.2f}", offset, align)

    peak = max(((n, v, data[(n, v)]) for n in ns for v in xs if (n, v) in data
                and data[(n, v)]["ok"]), key=lambda item: item[2]["ff"])
    note(axes[2], peak[1], peak[2]["ff"], f"{peak[2]['ff']:.2f}", (-4, 9), align="right")

    for axis in axes:
        axis.set_xlabel(r"$\langle k\rangle$ (log scale)", labelpad=2)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontweight("bold")

    # 单行图例，横跨整幅图（6 条），不占任何面板的绘图区
    figure.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles),
                  frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.7,
                  handletextpad=0.5, columnspacing=1.7, borderaxespad=0.2,
                  fontsize=8)

    figure.savefig(out_pdf, format="pdf")
    if out_png:
        figure.savefig(out_png, format="png", dpi=300)
    plt.close(figure)
    print("->", out_pdf)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(root / "results" / "input_sweep.csv"))
    parser.add_argument("--fallback",
                        default=str(root / "results" / "input_sweep_fallback.csv"),
                        help="空窗口格的下界回退实测（缺省则空窗段不画）")
    parser.add_argument("--out", default="fig_input_sensitivity.pdf")
    parser.add_argument("--png", default=None, help="可选：同时输出 PNG 预览")
    args = parser.parse_args()

    data = load(Path(args.csv), Path(args.fallback) if args.fallback else None)
    draw(data, Path(args.out), Path(args.png) if args.png else None)


if __name__ == "__main__":
    main()
