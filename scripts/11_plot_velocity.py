#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""速度 V 与各测量量的关系（θr、ρ）。

    python scripts/11_plot_velocity.py "D:\\data\\20261112_8group"

速度来源：优先 <组>/baseline.json 里的 velocity_mps，其次组名（能转成数字时）。
输出：fig_velocity_panels.png（双联）、fig_velocity_theta_r.png、fig_velocity_rho.png
"""
import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.ticker import MultipleLocator                      # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SUMMARY = "jet_angle_summary.csv"
RED = "#d62728"
GREEN = "#2ca02c"
GREY = "#8a939c"


def _f(row, key, default=float("nan")):
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def velocity_of(root, row):
    g = row["group"]
    p = os.path.join(root, g, "baseline.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                v = json.load(f).get("velocity_mps")
            if v is not None:
                return float(v), "baseline.json"
        except (OSError, ValueError, TypeError):
            pass
    try:
        return float(g.replace(",", ".")), "folder"
    except ValueError:
        return None, None


def load(root):
    path = os.path.join(root, SUMMARY)
    if not os.path.exists(path):
        raise SystemExit("找不到 %s —— 先跑 04_report.py" % path)
    rows, skipped = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v, src = velocity_of(root, r)
            if v is None:
                continue
            n_valid = _f(r, "n_valid", 0)
            ratio = _f(r, "valid_ratio", 1.0)
            if not np.isfinite(_f(r, "theta_r_mean")):
                skipped.append((r["group"], v))
                continue
            rows.append(dict(
                group=r["group"], v=v, v_src=src,
                ti=_f(r, "theta_i_mean"), tr=_f(r, "theta_r_mean"),
                tr_sd=_f(r, "theta_r_std"), rho=_f(r, "rho_mean"),
                rho_sd=_f(r, "rho_std"), n_valid=n_valid,
                weak=(ratio < 0.2 or n_valid < 10)))
    rows.sort(key=lambda d: d["v"])
    for g, v in skipped:
        print("  [skip] %-10s V=%.2f  没有有效帧（全部 none/invalid）" % (g, v))
    return rows


def style(ax, xlabel="射流速度 V (m/s)"):
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10)


def draw_tr(ax, rows, fs=9):
    strong = [r for r in rows if not r["weak"]]
    weak = [r for r in rows if r["weak"]]
    ax.errorbar([r["v"] for r in strong], [r["tr"] for r in strong],
                yerr=[r["tr_sd"] for r in strong], fmt="o", ms=6, capsize=3.5,
                lw=1.2, color=RED, label="θr（有效组）")
    if weak:
        ax.errorbar([r["v"] for r in weak], [r["tr"] for r in weak],
                    yerr=[r["tr_sd"] for r in weak], fmt="o", ms=6,
                    capsize=3.5, mfc="white", mec=GREY, ecolor=GREY,
                    label="铺展/低有效率组")
    ti = np.nanmean([r["ti"] for r in rows])
    ax.axhline(ti, ls=":", lw=1.1, color=GREY)
    ax.text(rows[0]["v"], ti + 0.5, "θi ≈ %.1f°（板角）" % ti,
            fontsize=fs - 0.5, color=GREY, va="bottom")
    ax.set_ylabel("反射角 θr (deg)", fontsize=11)
    ax.legend(fontsize=fs, frameon=False, loc="lower right")
    style(ax)
    ax.set_xlim(rows[0]["v"] - 0.22, rows[-1]["v"] + 0.22)
    ax.set_ylim(20.0, 80.0)                    # 角度轴统一 20–80
    ax.yaxis.set_major_locator(MultipleLocator(10))
    for r in rows:                             # 超出上限的点画在顶边
        if r["tr"] > 79.4:
            ax.plot([r["v"]], [79.0], marker="^", ms=7, mfc="white",
                    mec=GREY, mew=1.2, ls="none", zorder=5)
            ax.annotate("%.1f" % r["tr"], xy=(r["v"], 79.0), xytext=(-9, 0),
                        textcoords="offset points", fontsize=8, color=GREY,
                        va="center", ha="right")


def draw_rho(ax, rows, fs=9):
    strong = [r for r in rows if not r["weak"]]
    weak = [r for r in rows if r["weak"]]
    ax.errorbar([r["v"] for r in strong], [r["rho"] for r in strong],
                yerr=[r["rho_sd"] for r in strong], fmt="s", ms=6, capsize=3.5,
                lw=1.2, color=GREEN, label=r"$\rho$（有效组）")
    if weak:
        ax.errorbar([r["v"] for r in weak], [r["rho"] for r in weak],
                    yerr=[r["rho_sd"] for r in weak], fmt="s", ms=6,
                    capsize=3.5, mfc="white", mec=GREY, ecolor=GREY,
                    label="铺展/低有效率组")
    ax.axhline(1.0, ls=":", lw=1.1, color=GREY)
    ax.text(rows[0]["v"], 1.04, "ρ = 1（镜面反射）", fontsize=fs - 0.5,
            color=GREY, va="bottom")
    ax.set_ylabel(r"$\rho$", fontsize=12)
    ax.legend(fontsize=fs, frameon=False, loc="upper right")
    style(ax)
    ax.set_xlim(rows[0]["v"] - 0.22, rows[-1]["v"] + 0.22)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="11_plot_velocity",
                                 description="θr / ρ 随射流速度的变化")
    ap.add_argument("root", help="数据集根目录（含 jet_angle_summary.csv）")
    ap.add_argument("--out", default=None, help="输出目录（默认同 root）")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    rows = load(root)
    if len(rows) < 2:
        raise SystemExit("至少需要两个速度组")
    outdir = os.path.abspath(args.out) if args.out else root
    os.makedirs(outdir, exist_ok=True)

    # ---- 双联 ----
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=200)
    draw_tr(ax[0], rows)
    draw_rho(ax[1], rows)
    fig.tight_layout()
    p0 = os.path.join(outdir, "fig_velocity_panels.png")
    fig.savefig(p0, facecolor="white")
    plt.close(fig)

    # ---- 单张 ----
    for key, fn, draw, size in (("theta_r", "fig_velocity_theta_r.png",
                                 draw_tr, (7.4, 4.6)),
                                ("rho", "fig_velocity_rho.png",
                                 draw_rho, (7.4, 4.6))):
        fig, ax = plt.subplots(figsize=size, dpi=220)
        draw(ax, rows, fs=9.5)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fn), facecolor="white")
        plt.close(fig)
        print("figure ->", os.path.join(outdir, fn))
    print("figure ->", p0)

    print("\n%-8s %8s %8s %8s %8s  %s" % ("V(m/s)", "θi", "θr", "ρ", "n_ok",
                                          "速度来源"))
    for r in rows:
        print("%-8.2f %8.2f %8.2f %8.2f %8d  %s%s"
              % (r["v"], r["ti"], r["tr"], r["rho"], r["n_valid"], r["v_src"],
                 "  (铺展/低有效率)" if r["weak"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
