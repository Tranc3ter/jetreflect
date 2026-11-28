#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 8 - Weber / Reynolds numbers of one group.

    python scripts/08_jet_physics.py "D:\\data\\20261119_11group"
    python scripts/08_jet_physics.py "D:\\data\\20261119_11group" --velocity 2.0
    python scripts/08_jet_physics.py "D:\\data\\20261119_11group" --group M005 ^
        --diameter-px 19.4          (testing without the manual marking)

Inputs
------
* ``<group>/baseline.json``            manual initial diameter (07_mark_diameter)
* ``<group>/jet_angle/result.json``    theta_i, theta_r, r and the frame counts
* ``jet_angle_config.json``            px_per_mm and the ``physics`` block

Quantities (water properties from the config, 20 degC by default):

    We      = rho V^2 D / gamma
    We_perp = rho R (V cos theta_i)^2 / gamma      (R = D/2, docs/02 convention)
    Re      = rho V D / mu
    r       = tan(theta_r) / tan(theta_i)          (from the angles only)

Writes ``<group>/jet_physics.json``; the old json keeps the base measurements.
"""
import argparse
import csv
import datetime
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from jetreflect import pipeline                                    # noqa: E402
from jetreflect.io_utils import read_json, write_json              # noqa: E402

SCHEMA = "jetreflect/physics-2"
OUT = "jet_physics.json"
PHYS_DEFAULTS = {
    "velocity_ms": 2.0,
    "rho": 998.0,          # kg/m^3   (water, 20 degC)
    "gamma": 0.072,        # N/m      (water/air, 20 degC)
    "mu": 0.001002,        # Pa s     (water, 20 degC)
    "temperature_C": 20.0,
}
FORMULAS = {
    "We": "rho * V^2 * D / gamma",
    "We_perp": "rho * R * (V cos(theta_i))^2 / gamma,  R = D/2",
    "Re": "rho * V * D / mu",
    "r": "tan(theta_r) / tan(theta_i)",
}


def _iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def compute(base, res, cfg, phys, diameter_px=None):
    d_px = float(diameter_px or base.get("jet_diameter_px") or 0)
    if d_px <= 0:
        return None, ("no jet diameter in baseline.json -- run "
                      "06_mark_diameter.py first")
    scale = float(base.get("diameter_px_per_mm") or cfg.get("px_per_mm") or 0)
    if scale <= 0:
        return None, "no px_per_mm (run 05_calibrate.py)"
    d_mm = d_px / scale
    r_mm = d_mm / 2.0
    d_m = d_mm / 1000.0
    r_m = r_mm / 1000.0

    st = (res or {}).get("stats") or {}
    try:
        theta_i = float(st["theta_i"]["mean"])
        theta_r = float(st["theta_r"]["mean"])
        r_ang = float(st["rho"]["mean"])
    except (KeyError, TypeError):
        return None, "no theta_i / theta_r in result.json"

    v = float(phys["velocity_ms"])
    rho_l = float(phys["rho"])
    gam = float(phys["gamma"])
    mu = float(phys["mu"])
    ti = math.radians(theta_i)

    we = rho_l * v ** 2 * d_m / gam
    we_perp = rho_l * r_m * (v * math.cos(ti)) ** 2 / gam
    re = rho_l * v * d_m / mu

    out = {
        "schema": SCHEMA,
        "dataset": cfg.get("dataset"),
        "group": base.get("group"),
        "created": _iso(),
        "algorithm": "jetreflect " + pipeline._version(),
        "config": {
            "velocity_ms": v,
            "velocity_source": phys.get("velocity_source", "config"),
            "rho_kg_m3": rho_l, "gamma_N_m": gam, "mu_Pa_s": mu,
            "temperature_C": phys.get("temperature_C"),
            "px_per_mm": scale,
            "diameter_px": round(d_px, 3),
            "diameter_mm": round(d_mm, 5),
            "diameter_source": base.get("diameter_source"),
            "diameter_frame": base.get("diameter_frame"),
        },
        "angles": {
            "theta_i": round(theta_i, 4),
            "theta_r": round(theta_r, 4),
            "r": round(r_ang, 4),
            "n_valid": (res or {}).get("n_valid"),
        },
        "results": {
            "We": round(we, 3),
            "We_perp": round(we_perp, 3),
            "Re": round(re, 1),
        },
        "formulas": FORMULAS,
        "sources": {
            "baseline": os.path.join(base.get("_dir", ""), "baseline.json"),
            "result": os.path.join(base.get("_dir", ""), "jet_angle",
                                   "result.json"),
        },
    }
    return out, ""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="08_jet_physics",
                                 description="Weber / Reynolds numbers per group")
    ap.add_argument("root")
    ap.add_argument("--group", default="")
    ap.add_argument("--velocity", type=float, default=None,
                    help="jet velocity (m/s); default from the config")
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--diameter-px", type=float, default=None,
                    help="override the manual diameter (testing)")
    ap.add_argument("--csv", action="store_true",
                    help="also write jet_physics_summary.csv in the root")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit("not a folder: %s" % root)
    cfg = pipeline.load_config(root)
    phys = dict(PHYS_DEFAULTS)
    phys.update(cfg.get("physics") or {})
    for k, v in (("velocity_ms", args.velocity), ("rho", args.rho),
                 ("gamma", args.gamma), ("mu", args.mu)):
        if v is not None:
            phys[k] = v
    groups = pipeline.groups_of(root, cfg)
    if args.group:
        keys = [s.strip() for s in args.group.split(",") if s.strip()]
        groups = [g for g in groups if any(k in g["name"] for k in keys)]
    if not groups:
        raise SystemExit("no groups selected")

    rows, failed = [], []
    for g in groups:
        base = read_json(os.path.join(g["dir"], "baseline.json")) or {}
        base["_dir"] = g["dir"]
        base.setdefault("group", g["name"])
        res = read_json(os.path.join(g["dir"], "jet_angle", "result.json"))
        if res is None:
            failed.append((g["label"], "no result.json (run 02_measure.py)"))
            print("  [fail] %-22s no result.json" % g["label"])
            continue
        phys_g = dict(phys)
        bv = base.get("velocity_mps")
        if args.velocity is None and bv not in (None, ""):
            try:
                phys_g["velocity_ms"] = float(bv)
                phys_g["velocity_source"] = "baseline.json"
            except (TypeError, ValueError):
                pass
        out, why = compute(base, res, cfg, phys_g, args.diameter_px)
        if out is None:
            failed.append((g["label"], why))
            print("  [fail] %-22s %s" % (g["label"], why))
            continue
        out["dataset"] = cfg.get("dataset")
        out["group"] = g["name"]
        out["sources"] = {
            "baseline": os.path.join(g["dir"], "baseline.json"),
            "result": os.path.join(g["dir"], "jet_angle", "result.json"),
        }
        write_json(os.path.join(g["dir"], OUT), out)
        rows.append(out)
        print("  [ ok ] %-22s D=%.2f px = %.4f mm   We=%6.1f  We_perp=%5.2f  "
              "Re=%5.0f" % (g["label"], out["config"]["diameter_px"],
                            out["config"]["diameter_mm"],
                            out["results"]["We"], out["results"]["We_perp"],
                            out["results"]["Re"]))

    print("\n%d/%d groups done" % (len(rows), len(groups)))
    for lbl, why in failed:
        print("  %-24s %s" % (lbl, why))

    if args.csv and rows:
        path = os.path.join(root, "jet_physics_summary.csv")
        cols = ["group", "theta_i", "theta_r", "r", "diameter_px",
                "diameter_mm", "velocity_ms", "We", "We_perp", "Re"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for o in rows:
                w.writerow([o["group"], o["angles"]["theta_i"],
                            o["angles"]["theta_r"], o["angles"]["r"],
                            o["config"]["diameter_px"],
                            o["config"]["diameter_mm"],
                            o["config"]["velocity_ms"],
                            o["results"]["We"], o["results"]["We_perp"],
                            o["results"]["Re"]])
        print("table -> %s" % path)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
