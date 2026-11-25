#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 7 - mark the initial jet diameter by hand (one value per group).

    python scripts/07_mark_diameter.py "D:\\data\\20261119_11group"
    python scripts/07_mark_diameter.py "D:\\data\\20261119_11group" --group M005
    python scripts/07_mark_diameter.py "D:\\data\\20261119_11group" ^
        --diameter-px 19.4 --group "MX-5 2058-1-M005-N000"    (no window)

Mark the **contracted liquid interface at the nozzle exit**: two clicks on the
two interfaces, the distance is the jet diameter.  The value is stored in the
group's ``baseline.json`` (the old json, together with the surface line):

    "jet_diameter_px": 19.40,
    "jet_diameter_mm": 0.4725,
    "diameter_p1": [x1, y1], "diameter_p2": [x2, y2],
    "diameter_frame": "..._00001000.tif",
    "diameter_px_per_mm": 41.063,
    "diameter_source": "manual",
    "diameter_created": "...", "diameter_operator": "..."

Keys: click twice = set / redo, U = undo, N/P = prev/next frame,
      S or Enter = save & next group, Q or Esc = quit.
"""
import argparse
import datetime
import getpass
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from jetreflect import geom, pipeline                                  # noqa: E402
from jetreflect.gui import _View                                        # noqa: E402
from jetreflect.io_utils import imread, read_json, write_json           # noqa: E402

COL_PT = (0, 220, 255)


def _iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _operator():
    try:
        return getpass.getuser()
    except Exception:                                        # noqa: BLE001
        return ""


def _px_per_mm(root, cfg, override=None):
    if override:
        return float(override), "command line"
    cal = cfg.get("calibration") or {}
    v = cfg.get("px_per_mm")
    if v:
        src = os.path.basename(cal.get("image") or "") or "config"
        return float(v), src
    raise SystemExit("no px_per_mm in jet_angle_config.json -- run "
                     "05_calibrate.py or pass --px-per-mm")


def _save(group, frame, p1, p2, scale, src):
    path = os.path.join(group["dir"], "baseline.json")
    rec = read_json(path) or {}
    d_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    rec.update({
        "jet_diameter_px": round(d_px, 2),
        "jet_diameter_mm": round(d_px / scale, 4),
        "diameter_p1": [round(float(p1[0]), 2), round(float(p1[1]), 2)],
        "diameter_p2": [round(float(p2[0]), 2), round(float(p2[1]), 2)],
        "diameter_frame": os.path.basename(frame),
        "diameter_px_per_mm": round(float(scale), 4),
        "diameter_source": src,
        "diameter_created": _iso(),
        "diameter_operator": _operator(),
    })
    write_json(path, rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="07_mark_diameter",
                                 description="mark the initial jet diameter")
    ap.add_argument("root")
    ap.add_argument("--group", default="", help="only these groups (substring)")
    ap.add_argument("--frame-ratio", type=float, default=0.5,
                    help="which frame to open (0-1, default 0.5 = middle)")
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="override the calibration in the config")
    ap.add_argument("--diameter-px", type=float, default=None,
                    help="write this value without opening a window")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit("not a folder: %s" % root)
    cfg = pipeline.load_config(root)
    groups = pipeline.groups_of(root, cfg)
    if args.group:
        keys = [s.strip() for s in args.group.split(",") if s.strip()]
        groups = [g for g in groups
                  if any(k in g["name"] for k in keys)]
    if not groups:
        raise SystemExit("no groups selected")
    scale, src = _px_per_mm(root, cfg, args.px_per_mm)

    if args.diameter_px:                       # batch / non-interactive
        px = float(args.diameter_px)
        for g in groups:
            rec = read_json(os.path.join(g["dir"], "baseline.json")) or {}
            rec.update({
                "jet_diameter_px": round(px, 2),
                "jet_diameter_mm": round(px / scale, 4),
                "diameter_p1": rec.get("diameter_p1"),
                "diameter_p2": rec.get("diameter_p2"),
                "diameter_px_per_mm": round(scale, 4),
                "diameter_source": "command line",
                "diameter_created": _iso(),
                "diameter_operator": _operator(),
            })
            write_json(os.path.join(g["dir"], "baseline.json"), rec)
            print("  [saved] %-24s D = %.2f px = %.4f mm"
                  % (g["label"], px, px / scale))
        return 0

    title = "jetreflect diameter"
    canvas = (1024, 768)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, canvas[0], canvas[1])
    state = {"i": 0, "quit": False}
    pts = {"cur": []}

    def load(i):
        g = groups[i]
        frames = g["images"]
        k = int(round((len(frames) - 1) * args.frame_ratio))
        img = imread(frames[k])
        rec = read_json(os.path.join(g["dir"], "baseline.json")) or {}
        old = None
        if rec.get("diameter_p1") and rec.get("diameter_p2"):
            old = (rec["diameter_p1"], rec["diameter_p2"])
        return g, frames, k, img, _View(img, canvas), old

    g, frames, k, img, view, old = load(state["i"])
    pts["cur"] = list(old) if old else []

    def hud():
        t = ["[%d/%d] %s  frame %d/%d  zoom %.2fx  %dx%d  px/mm %.3f (%s)"
             % (state["i"] + 1, len(groups), g["label"], k + 1, len(frames),
                view.scale, img.shape[1], img.shape[0], scale, src)]
        if len(pts["cur"]) == 2:
            a, b = pts["cur"]
            d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            t.append("D = %.2f px = %.4f mm" % (d, d / scale))
        else:
            t.append("click the two interfaces at the nozzle exit (2 points)")
        t.append("click/2 pts  U=undo  N/P=frame  S=save&next  Q=quit")
        return t

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            view.zoom_at(mx, my, 1.15 if flags > 0 else 1 / 1.15)
        elif event == cv2.EVENT_LBUTTONDOWN:
            p = view.to_img(mx, my)
            if len(pts["cur"]) >= 2:
                pts["cur"] = [p]
            else:
                pts["cur"].append(p)

    cv2.setMouseCallback(title, on_mouse)

    while not state["quit"]:
        draft = None
        if len(pts["cur"]) == 2:
            draft = (pts["cur"][0], pts["cur"][1])
        vis = view.render(lines=[(draft, COL_PT, 1)] if draft else [],
                          draft=None, hud=hud())
        for p in pts["cur"]:
            cv2.circle(vis, view.to_view(*p), 4, COL_PT, -1, cv2.LINE_AA)
        cv2.imshow(title, vis)
        key = cv2.waitKeyEx(20)
        kk = key & 0xFF
        LEFT, UP, RIGHT, DOWN = 2424832, 2490368, 2555904, 2621440
        if key in (RIGHT,) or kk in (ord("n"), ord("N")):
            k = min(k + 1, len(frames) - 1)
            img = imread(frames[k])
            view = _View(img, canvas)
        elif key in (LEFT,) or kk in (ord("p"), ord("P")):
            k = max(k - 1, 0)
            img = imread(frames[k])
            view = _View(img, canvas)
        elif key in (DOWN,) or kk == ord("j"):
            state["i"] = (state["i"] + 1) % len(groups)
            g, frames, k, img, view, old = load(state["i"])
            pts["cur"] = list(old) if old else []
        elif key in (UP,) or kk == ord("k"):
            state["i"] = (state["i"] - 1) % len(groups)
            g, frames, k, img, view, old = load(state["i"])
            pts["cur"] = list(old) if old else []
        elif kk == ord("u"):
            pts["cur"] = pts["cur"][:-1]
        elif kk in (ord("s"), 13, 10):
            if len(pts["cur"]) != 2:
                continue
            rec = _save(g, frames[k], pts["cur"][0], pts["cur"][1],
                        scale, "manual")
            print("  [saved] %-24s D = %.2f px = %.4f mm"
                  % (g["label"], rec["jet_diameter_px"],
                     rec["jet_diameter_mm"]))
            state["i"] = (state["i"] + 1) % len(groups)
            g, frames, k, img, view, old = load(state["i"])
            pts["cur"] = list(old) if old else []
        elif kk in (ord("q"), 27):
            state["quit"] = True
    cv2.destroyWindow(title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
