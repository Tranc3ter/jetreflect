#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 6 - rho and d(theta_r)/d(theta_i) against the incident angle.

    python scripts/06_plot_rho.py "D:\\data\\20261119_11group"
    python scripts/06_plot_rho.py "D:\\data\\20261119_11group" --min-dx 1.5
    python scripts/06_plot_rho.py "D:\\data\\20261119_11group" --fit linear

Reads ``<dataset>/jet_angle_summary.csv`` (written by ``04_report.py``), sorts
the groups by the mean incident angle and writes

    <dataset>/fig_rho_vs_theta_i.png    theta_r / rho / d(theta_r)/d(theta_i)
    <dataset>/rho_vs_theta_i.csv        the sorted table + the slopes

The rate of change is reported twice:

* discrete: ``(theta_r[i+1] - theta_r[i]) / (theta_i[i+1] - theta_i[i])``,
  drawn as stems at the interval midpoint.  Intervals narrower than
  ``--min-dx`` (default 1 deg) are skipped -- two groups with the same plate
  angle would otherwise produce a meaningless spike.
* smooth: the analytic derivative of a weighted polynomial fit of
  ``theta_r(theta_i)`` (quadratic by default), drawn as a dashed curve.
  Its R^2 is printed and put on the figure.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import matplotlib                                                   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from jetreflect import pipeline                                     # noqa: E402

SUMMARY = "jet_angle_summary.csv"
FIG = "fig_rho_vs_theta_i.png"
TABLE = "rho_vs_theta_i.csv"


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
            rows.append(dict(group=r["group"],
                             ti=_f(r, "theta_i_mean"),
                             ti_sd=_f(r, "theta_i_std"),
                             tr=_f(r, "theta_r_mean"),
                             tr_sd=_f(r, "theta_r_std"),
                             rho=_f(r, "rho_mean"),
                             rho_sd=_f(r, "rho_std"),
                             n=int(_f(r, "n_valid"))))
    rows.sort(key=lambda d: d["ti"])
    return rows


def intervals(rows, min_dx):
    out = []
    for a, b in zip(rows, rows[1:]):
        dx = b["ti"] - a["ti"]
        dy = b["tr"] - a["tr"]
        sk = np.hypot(a["tr_sd"], b["tr_sd"]) / abs(dx) if dx else float("nan")
        out.append(dict(x=0.5 * (a["ti"] + b["ti"]), dx=dx, dy=dy,
                        k=(dy / dx) if abs(dx) >= min_dx else float("nan"),
                        sk=sk, a=a["group"], b=b["group"]))
    return out


def fit_curve(rows, order):
    x = np.array([r["ti"] for r in rows], float)
    y = np.array([r["tr"] for r in rows], float)
    sd = np.array([r["tr_sd"] for r in rows], float)
    sd[~np.isfinite(sd) | (sd <= 0)] = np.nan
    good = np.isfinite(sd)
    if good.sum() < order + 2:                     # not enough spread -> equal weights
        w = None
    else:
        w = 1.0 / np.maximum(sd, 1e-3)
    coeff = np.polyfit(x, y, order, w=w)
    yhat = np.polyval(coeff, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coeff, r2


def main(argv=None):
    ap = argparse.ArgumentParser(prog="06_plot_rho",
                                 description="rho and d(theta_r)/d(theta_i) "
                                             "versus the incident angle")
    ap.add_argument("root", help="dataset folder (contains the summary csv)")
    ap.add_argument("--out", default=None, help="output folder (default: root)")
    ap.add_argument("--min-dx", type=float, default=1.0,
                    help="skip discrete slopes over a narrower theta_i gap "
                         "(deg, default 1.0)")
    ap.add_argument("--fit", default="quadratic",
                    choices=["none", "linear", "quadratic", "cubic"],
                    help="smooth trend used for the derivative (default quadratic)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    path = os.path.join(root, SUMMARY)
    if not os.path.exists(path):
        raise SystemExit("missing %s -- run 04_report.py first" % path)
    rows = load(path)
    if len(rows) < 2:
        raise SystemExit("need at least two measured groups")

    order = {"none": None, "linear": 1, "quadratic": 2, "cubic": 3}[args.fit]
    coeff = None
    r2 = float("nan")
    if order is not None and len(rows) > order + 1:
        coeff, r2 = fit_curve(rows, order)
    iv = intervals(rows, args.min_dx)

    # ---- table ---------------------------------------------------------
    outdir = os.path.abspath(args.out) if args.out else root
    os.makedirs(outdir, exist_ok=True)
    tbl = os.path.join(outdir, TABLE)
    with open(tbl, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_valid", "theta_i_deg", "theta_i_std",
                    "theta_r_deg", "theta_r_std", "rho", "rho_std",
                    "gap_to_next_deg", "dtheta_r_dtheta_i", "note"])
        for i, r in enumerate(rows):
            gap = iv[i]["dx"] if i < len(iv) else ""
            k = iv[i]["k"] if i < len(iv) else ""
            note = ""
            if i < len(iv) and not np.isfinite(iv[i]["k"]):
                note = "gap < %.1f deg, slope skipped" % args.min_dx
                k = ""
            w.writerow([r["group"], r["n"], round(r["ti"], 3),
                        round(r["ti_sd"], 3), round(r["tr"], 3),
                        round(r["tr_sd"], 3), round(r["rho"], 3),
                        round(r["rho_sd"], 3),
                        "" if gap == "" else round(gap, 3),
                        "" if k == "" else round(k, 3), note])
    print("table -> %s" % tbl)

    # ---- figure --------------------------------------------------------
    cjk = pipeline.cjk()
    T = pipeline.T
    x = np.array([r["ti"] for r in rows], float)
    tr = np.array([r["tr"] for r in rows], float)
    e_tr = np.array([r["tr_sd"] for r in rows], float)
    rho = np.array([r["rho"] for r in rows], float)
    e_rho = np.array([r["rho_sd"] for r in rows], float)

    fig, ax = plt.subplots(3, 1, figsize=(9.2, 11.0), dpi=170, sharex=True)
    fig.suptitle(T("结果随入射角的变化（按入射角升序）",
                   "results versus the incident angle (sorted)"), fontsize=12)

    # (1) theta_r
    ax[0].errorbar(x, tr, yerr=e_tr, fmt="o", ms=5, capsize=3,
                   color="#d62728", label=T("反射角", "reflected"))
    if coeff is not None:
        xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
        ax[0].plot(xs, np.polyval(coeff, xs), "-", lw=1.4, color="#7f0000",
                   alpha=0.75,
                   label=T("拟合（%s）R²=%.2f"
                          % ({"linear": "一次", "quadratic": "二次",
                              "cubic": "三次"}[args.fit], r2),
                          "%s fit, R2=%.2f"
                          % (args.fit, r2)))
    ax[0].set_ylabel(T("反射角 θr (deg)", "theta_r (deg)"))
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="best")

    # (2) rho
    ax[1].errorbar(x, rho, yerr=e_rho, fmt="s", ms=5, capsize=3,
                   color="#2ca02c", label=r"$\rho=\tan\theta_r/\tan\theta_i$")
    ax[1].axhline(1.0, ls=":", lw=1, color="gray")
    ax[1].set_ylabel(r"$\rho$")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8, loc="best")

    # (3) d theta_r / d theta_i
    skipped = False
    for seg in iv:
        if not np.isfinite(seg["k"]):
            ax[2].axvspan(seg["x"] - 0.2, seg["x"] + 0.2, color="#ffdddd",
                          alpha=0.7, lw=0)
            skipped = True
            continue
        col = "#1f77b4" if seg["k"] >= 0 else "#d62728"
        ax[2].plot([seg["x"] - 0.5 * seg["dx"], seg["x"] + 0.5 * seg["dx"]],
                   [seg["k"], seg["k"]], "-", lw=2.0, color=col, alpha=0.85)
        ax[2].errorbar([seg["x"]], [seg["k"]], yerr=[seg["sk"]], fmt="o",
                       ms=4, capsize=2, color=col)
    if coeff is not None:
        xs = np.linspace(x.min(), x.max(), 200)
        ax[2].plot(xs, np.polyval(np.polyder(coeff), xs), "--", lw=1.6,
                   color="#333333",
                   label=T("拟合曲线的导数", "derivative of the fit"))
        ax[2].legend(fontsize=8, loc="best")
    ax[2].axhline(0.0, ls="-", lw=0.8, color="gray")
    ax[2].axhline(1.0, ls=":", lw=1, color="gray",
                  label=T("1:1（θr 与 θi 等量变化）", "1:1"))
    handles, labels = ax[2].get_legend_handles_labels()
    if skipped:
        from matplotlib.patches import Patch
        handles.append(Patch(color="#ffdddd", alpha=0.7))
        labels.append(T("两组入射角相同，无法求斜率", "same angle, skipped"))
    ax[2].legend(handles, labels, fontsize=8, loc="lower right")
    ax[2].set_ylabel(T("dθr / dθi", "d(theta_r)/d(theta_i)"))
    ax[2].set_xlabel(T("入射角 θi (deg)", "theta_i (deg)"))
    ax[2].set_xlim(x.min() - 1.5, x.max() + 1.5)
    ax[2].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(outdir, FIG)
    fig.savefig(png)
    plt.close(fig)
    print("figure -> %s" % png)

    # ---- console -------------------------------------------------------
    print("\n%-24s %6s %7s %7s %6s" % ("group", "th_i", "th_r", "rho", "k"))
    for i, r in enumerate(rows):
        k = iv[i]["k"] if i < len(iv) else float("nan")
        print("%-24s %6.2f %7.2f %7.3f %6s"
              % (r["group"], r["ti"], r["tr"], r["rho"],
                 "  -  " if not np.isfinite(k) else "%5.2f" % k))
    if coeff is not None:
        print("\nfit: theta_r = " + " + ".join(
            "%.4g*x^%d" % (c, len(coeff) - 1 - i) for i, c in enumerate(coeff))
            + "   R2 = %.3f" % r2)
        print("derivative at theta_i = %.1f deg: %.3f"
              % (x.mean(), np.polyval(np.polyder(coeff), x.mean())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
