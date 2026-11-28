# -*- coding: utf-8 -*-
"""单张图：θr–θi 与 ρ–θi（只画测得的组，默认按入射角升序）。

    python 10_plot_single.py <数据集根目录> [--out 输出目录] [--fit none|quadratic]

输入：<根目录>/jet_angle_summary.csv（04_report.py 的产物）
输出：fig_single_theta_r.png、fig_single_rho.png
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.ticker import MultipleLocator                       # noqa: E402
import numpy as np                                                 # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SUMMARY = "jet_angle_summary.csv"
FIG_TR = "fig_single_theta_r.png"
FIG_RHO = "fig_single_rho.png"

RED = "#d62728"
RED_D = "#7f0000"
GREEN = "#2ca02c"
GREY = "#8a939c"


def _f(row, key, default=0.0):
    v = row.get(key)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("theta_i_mean") or not r.get("theta_r_mean"):
                continue
            if not np.isfinite(_f(r, "theta_r_mean")):
                print("  [skip] %-24s 没有有效帧" % r["group"])
                continue
            rows.append(dict(group=r["group"],
                             ti=_f(r, "theta_i_mean"),
                             tr=_f(r, "theta_r_mean"),
                             tr_sd=_f(r, "theta_r_std"),
                             rho=_f(r, "rho_mean"),
                             rho_sd=_f(r, "rho_std"),
                             n=int(_f(r, "n_valid"))))
    rows.sort(key=lambda d: d["ti"])
    return rows


def fit_curve(rows, order):
    x = np.array([r["ti"] for r in rows], float)
    y = np.array([r["tr"] for r in rows], float)
    sd = np.array([r["tr_sd"] for r in rows], float)
    sd[~np.isfinite(sd) | (sd <= 0)] = np.nan
    w = (1.0 / np.maximum(sd, 1e-3)) if np.isfinite(sd).sum() >= order + 2 \
        else None
    coeff = np.polyfit(x, y, order, w=w)
    yhat = np.polyval(coeff, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coeff, r2


def style(ax):
    ax.grid(alpha=0.3)
    ax.set_xlabel("入射角 θi (deg)", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=10)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="09_plot_single",
                                 description="θr–θi / ρ–θi 单张图")
    ap.add_argument("root", help="数据集根目录（含 jet_angle_summary.csv）")
    ap.add_argument("--out", default=None, help="输出目录（默认同 root）")
    ap.add_argument("--fit", default="none",
                    choices=["none", "linear", "quadratic", "cubic"])
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    path = os.path.join(root, SUMMARY)
    if not os.path.exists(path):
        raise SystemExit("找不到 %s —— 先跑 04_report.py" % path)
    rows = load(path)
    if len(rows) < 2:
        raise SystemExit("至少需要两组数据")

    outdir = os.path.abspath(args.out) if args.out else root
    os.makedirs(outdir, exist_ok=True)

    x = np.array([r["ti"] for r in rows], float)
    tr = np.array([r["tr"] for r in rows], float)
    e_tr = np.array([r["tr_sd"] for r in rows], float)
    rho = np.array([r["rho"] for r in rows], float)
    e_rho = np.array([r["rho_sd"] for r in rows], float)

    order = {"none": None, "linear": 1, "quadratic": 2, "cubic": 3}[args.fit]
    coeff, r2 = (None, float("nan"))
    if order is not None and len(rows) > order + 1:
        coeff, r2 = fit_curve(rows, order)
    fit_name = {"linear": "一次", "quadratic": "二次", "cubic": "三次"}

    # 两张图共用：角度两轴统一取 20–80°，画布保持原来的横版比例
    xlo, xhi = 20.0, 80.0
    FIG_SIZE = (7.4, 4.6)

    # ---------------- 单张 1：θr–θi（两轴都取同一角度范围，不做等比例） ----------------
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=220)
    ax.errorbar(x, tr, yerr=e_tr, fmt="o", ms=6, capsize=3.5, lw=1.2,
                color=RED, label="反射角 θr（%d 组）" % len(rows))
    if coeff is not None:
        xs = np.linspace(x.min() - 0.6, x.max() + 0.6, 240)
        ax.plot(xs, np.polyval(coeff, xs), "-", lw=1.6, color=RED_D,
                alpha=0.8,
                label="拟合（%s）  R² = %.2f" % (fit_name[args.fit], r2))
    ax.set_ylabel("反射角 θr (deg)", fontsize=11)
    style(ax)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(xlo, xhi)                  # 角度两轴都取同一范围（20–80）
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.plot([xlo, xhi], [xlo, xhi], ls=":", lw=1.1, color=GREY, zorder=0)
    ax.text(xlo + 0.40 * (xhi - xlo), xlo + 0.67 * (xhi - xlo),
            "θr = θi（镜面反射）", fontsize=8.5, color=GREY,
            ha="left", va="bottom")
    # 超出上限的点：在顶边画三角标记并标数值（不剪掉信息）
    for xi, yi in zip(x, tr):
        if yi > xhi:
            ax.plot([xi], [xhi - 0.9], marker="^", ms=7, mfc="white",
                    mec=GREY, mew=1.2, ls="none", zorder=5)
            ax.annotate("%.1f" % yi, xy=(xi, xhi - 0.9), xytext=(9, 0),
                        textcoords="offset points", fontsize=8, color=GREY,
                        va="center", ha="left")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    p1 = os.path.join(outdir, FIG_TR)
    fig.savefig(p1, facecolor="white")
    plt.close(fig)

    # ---------------- 单张 2：ρ–θi（画布与横轴和上面完全一致） ----------------
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=220)
    ax.errorbar(x, rho, yerr=e_rho, fmt="s", ms=6, capsize=3.5, lw=1.2,
                color=GREEN,
                label=r"$\rho=\tan\theta_r/\tan\theta_i$（%d 组）" % len(rows))
    ax.axhline(1.0, ls=":", lw=1.1, color="gray")
    ax.text(xlo + 0.04 * (xhi - xlo), 1.06, "ρ = 1（镜面反射）", fontsize=8.5,
            color="gray", va="bottom")
    ax.set_ylabel(r"$\rho$", fontsize=12)
    style(ax)
    ax.set_xlim(xlo, xhi)                      # 与 θr 图同一横轴范围 20–80
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylim(0.6, max(rho + 1.2))
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    p2 = os.path.join(outdir, FIG_RHO)
    fig.savefig(p2, facecolor="white")
    plt.close(fig)

    print("figure -> %s" % p1)
    print("figure -> %s" % p2)
    if coeff is not None:
        print("fit: R2 = %.3f" % r2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
