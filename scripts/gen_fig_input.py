"""Draw the design-point figure from ``input_sweep.csv``."""

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
    "mathtext.fontset": "stix",         
    "mathtext.default": "bf",           
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

COLOURS = {100: "#FFE694", 300: "#EF939E", 1000: "#A8E9E3"}
KBAR_INK = "#5A5A5A"
REF_INK = "#BEBEBE"


def darken(colour: str, factor: float = 0.80) -> str:
    """Darken a colour; used as the rim of filled markers."""
    value = colour.lstrip("#")
    return "#" + "".join(
        f"{round(int(value[i:i + 2], 16) * factor):02X}" for i in (0, 2, 4))

KBARS = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
XR = (1.92, 17.2)
WIDTH_PT = 494.50888
FIGSIZE = (WIDTH_PT / 72.0, 1.98)

#: Dash pattern for the fallback (empty-window) segment.
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
        state = first["status"]
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
    figure.subplots_adjust(left=0.056, right=0.998, bottom=0.176, top=0.845, wspace=0.26)

    def style_logx(axis) -> None:
        axis.set_xscale("log")
        axis.set_xlim(*XR)
        axis.set_xticks(KBARS)
        axis.set_xticklabels([f"{v:g}" for v in KBARS])
        axis.xaxis.set_minor_locator(NullLocator())

    ax = axes[0]
    style_logx(ax)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel(r"$k_{\max}$")
    ax.set_title("(a)", loc="left", pad=2)

    lower = [(v, data[(1000, v)]["lo"]) for v in xs if (1000, v) in data]
    ax.plot([p[0] for p in lower], [p[1] for p in lower],
            color=KBAR_INK, lw=1.1, ls=(0, (1, 1.6)), zorder=2)

    legend_handles = []
    for n in ns:
        cells = [(v, data[(n, v)]) for v in xs if (n, v) in data]
        colour = COLOURS[n]
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
        for v, cell in live:
            if cell["ok"]:
                ax.plot([v], [cell["hi"]], marker="o", ms=4.0, mfc=colour,
                        mec=colour, mew=0.8, ls="none", zorder=4)
        for v, cell in dead:
            ax.plot([v], [cell["hi"]], marker="o", ms=4.2, mfc="white",
                    mec=darken(colour, 0.86), mew=1.2, ls="none", zorder=4)
        legend_handles.append(Line2D([], [], color=colour, lw=1.5, marker="o", ms=4.0,
                                     mfc=colour, mec=colour, mew=0.8,
                                     label=f"$n={n}$"))

    legend_handles.append(Line2D([], [], color=KBAR_INK, lw=1.1, ls=(0, (1, 1.6)),
                                 label=r"$k^{-}$"))
    legend_handles.append(Line2D([], [], color="#4D4D4D", lw=1.3, marker="o", ms=4.0,
                                 mfc="#4D4D4D", label="midpoint"))
    legend_handles.append(Line2D([], [], color="#4D4D4D", lw=1.3, ls=EMPTY_DASH,
                                 marker="o", ms=4.2, mfc="white", mew=1.1,
                                 label="fallback"))

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
        axis.axhline(1.0, color=REF_INK, lw=0.9, ls=(0, (1, 2.2)), zorder=1)
        for n in ns:
            colour = COLOURS[n]
            pts = [(v, data[(n, v)][key]) for v in xs
                   if (n, v) in data and data[(n, v)]["ok"]
                   and data[(n, v)]["state"] == "ok"]
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
                        help="measured fallback values for empty windows (the segment is omitted without them)")
    parser.add_argument("--out", default="fig_input_sensitivity.pdf")
    parser.add_argument("--png", default=None, help="also write a PNG preview")
    args = parser.parse_args()

    data = load(Path(args.csv), Path(args.fallback) if args.fallback else None)
    draw(data, Path(args.out), Path(args.png) if args.png else None)


if __name__ == "__main__":
    main()
