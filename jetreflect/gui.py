# -*- coding: utf-8 -*-
"""OpenCV interactive windows: total-manual surface marking and px/mm.

The reflection surface is stored as an **undirected infinite line**
``a*x + b*y + c = 0``: the two clicked points only define where it passes, not
a direction, and it is drawn across the whole image.

Controls (baseline window)
--------------------------
    left drag / left click twice   draw a straight line on the reflection surface
    U or Ctrl+Z                    undo the line
    R                              clear it
    N / P  or  -> / <-             next / previous frame
    J / K  or  Down / Up           next / previous group
    S or Enter                     save baseline.json and go to next group
    Z / X  or mouse wheel          zoom in / out
    F                              fit the whole image
    right/middle drag              pan
    Q or Esc                       quit (already saved groups are kept)
"""
from __future__ import annotations

import datetime
import getpass
import math
import os

import cv2
import numpy as np

from . import geom
from .io_utils import imread, write_json, read_json

COL_LINE = (0, 255, 255)      # BGR yellow
COL_SAVED = (0, 200, 0)       # green
COL_NORMAL = (255, 0, 255)    # magenta
COL_DRAFT = (0, 128, 255)     # orange


class _View(object):
    """Zoomable, pannable viewport drawn into a fixed-size canvas."""

    def __init__(self, gray, canvas=(1024, 768)):
        self.gray = gray
        self.bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.h, self.w = gray.shape[:2]
        self.cw, self.ch = int(canvas[0]), int(canvas[1])
        self.smin = max(self.cw / float(self.w), self.ch / float(self.h))
        self.scale = self.smin
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.ix0 = self.iy0 = 0
        self.iw, self.ih = self.w, self.h

    # ---- geometry --------------------------------------------------------
    def rect(self):
        vw = min(self.cw / self.scale, float(self.w))
        vh = min(self.ch / self.scale, float(self.h))
        x0 = min(max(self.cx - vw / 2.0, 0.0), self.w - vw)
        y0 = min(max(self.cy - vh / 2.0, 0.0), self.h - vh)
        return x0, y0, vw, vh

    def to_img(self, mx, my):
        return (self.ix0 + mx * self.iw / float(self.cw),
                self.iy0 + my * self.ih / float(self.ch))

    def to_view(self, x, y):
        return (int(round((x - self.ix0) * self.cw / float(self.iw))),
                int(round((y - self.iy0) * self.ch / float(self.ih))))

    def zoom_at(self, mx, my, factor):
        px, py = self.to_img(mx, my)
        self.scale = float(np.clip(self.scale * factor, self.smin, 8.0))
        vw, vh = self.cw / self.scale, self.ch / self.scale
        self.cx = px - mx / self.scale + vw / 2.0
        self.cy = py - my / self.scale + vh / 2.0

    def pan(self, dmx, dmy):
        self.cx -= dmx / self.scale
        self.cy -= dmy / self.scale

    def fit(self):
        self.scale = self.smin
        self.cx, self.cy = self.w / 2.0, self.h / 2.0

    # ---- drawing ---------------------------------------------------------
    def render(self, lines=(), draft=None, normals=(), hud=None):
        x0, y0, vw, vh = self.rect()
        self.ix0, self.iy0 = int(round(x0)), int(round(y0))
        self.iw = max(1, int(round(vw)))
        self.ih = max(1, int(round(vh)))
        crop = self.bgr[self.iy0:self.iy0 + self.ih,
                        self.ix0:self.ix0 + self.iw]
        crop = crop[:self.ih, :self.iw]
        disp = cv2.resize(crop, (self.cw, self.ch),
                          interpolation=cv2.INTER_LINEAR)
        n_h = 0
        if hud:
            n_h = 22 * len(hud) + 10
            bar = np.full_like(disp, 25)
            disp = cv2.addWeighted(bar, 0.55, disp, 0.45, 0)
        for (p1, p2), col, thick in lines:
            a = self.to_view(p1[0], p1[1])
            b = self.to_view(p2[0], p2[1])
            cv2.line(disp, a, b, col, thick, cv2.LINE_AA)
            for p in (a, b):
                if 0 <= p[0] < self.cw and 0 <= p[1] < self.ch:
                    cv2.circle(disp, p, 3, col, -1, cv2.LINE_AA)
        if draft is not None:
            a = self.to_view(draft[0][0], draft[0][1])
            b = self.to_view(draft[1][0], draft[1][1])
            cv2.line(disp, a, b, COL_DRAFT, 1, cv2.LINE_AA)
        for p, d in normals:
            a = self.to_view(p[0], p[1])
            b = self.to_view(p[0] + d[0] * 40, p[1] + d[1] * 40)
            cv2.arrowedLine(disp, a, b, COL_NORMAL, 1, cv2.LINE_AA,
                            tipLength=0.25)
        if hud:
            for i, t in enumerate(hud):
                cv2.putText(disp, t, (8, self.ch - n_h + 18 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (210, 230, 210) if i == 0 else (170, 175, 170),
                            1, cv2.LINE_AA)
        return disp


def _iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _operator():
    try:
        return getpass.getuser()
    except Exception:                                     # noqa: BLE001
        return ""


def _warn(title, text):
    bar = np.zeros((70, 900, 3), np.uint8)
    cv2.putText(bar, text, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(title, bar)
    cv2.waitKey(900)


def mark_baselines(root, groups=None, frame_ratio=0.5, canvas=(1024, 768),
                   offset_default=0.0, start_group=0, dataset=None):
    """Total-manual marking.  One undirected line per group -> baseline.json."""
    from .io_utils import discover_groups, list_images

    groups = groups if groups is not None else discover_groups(root)
    if not groups:
        raise RuntimeError("no image groups found under %s" % root)
    title = "jetreflect surface"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, canvas[0], canvas[1])

    state = {"i": 0, "lines": [], "saved": [], "quit": False}
    pts = {"start": None, "cur": None, "pan": None}

    def load_group(i):
        g = groups[i]
        frames = g["images"] or list_images(g["dir"])
        k = int(round((len(frames) - 1) * frame_ratio))
        img = imread(frames[k])
        view = _View(img, canvas)
        rec = read_json(os.path.join(g["dir"], "baseline.json"))
        lines = []
        if rec and "p1" in rec and "p2" in rec:
            lines = [((rec["p1"][0], rec["p1"][1]),
                      (rec["p2"][0], rec["p2"][1]))]
        return g, frames, k, img, view, lines, rec

    g, frames, k, img, view, lines, rec = load_group(state["i"])
    state["lines"] = lines

    def current_line():
        if not state["lines"]:
            return None
        p1, p2 = state["lines"][-1]
        try:
            return geom.Line.from_points(p1, p2)
        except ValueError:
            return None

    def hud_text():
        t = ["[%d/%d] %s  frame %d/%d  zoom %.2fx  %dx%d" % (
            state["i"] + 1, len(groups), g["label"], k + 1, len(frames),
            view.scale, img.shape[1], img.shape[0])]
        line = current_line()
        if line is None:
            t.append("surface: NOT SET - drag a straight line on it")
        else:
            t.append("surface: tilt %+.2f deg from vertical   "
                     "(undirected, infinite)" % line.tilt_deg())
        t.append("drag=line  U=undo  R=clear  N/P=frame  J/K=group  "
                 "S=save&next  Z/X=zoom  right=drag  Q=quit")
        return t

    def commit(p0, p1):
        state["lines"] = [(p0, p1)]

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            view.zoom_at(mx, my, 1.15 if flags > 0 else 1 / 1.15)
            return
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            pts["pan"] = (mx, my)
            return
        if event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            pts["pan"] = None
            return
        if event == cv2.EVENT_MOUSEMOVE:
            if pts["pan"] is not None:
                view.pan(mx - pts["pan"][0], my - pts["pan"][1])
                pts["pan"] = (mx, my)
            if pts["start"] is not None:
                pts["cur"] = view.to_img(mx, my)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            p = view.to_img(mx, my)
            if pts["start"] is not None:
                d = float(np.hypot(p[0] - pts["start"][0],
                                   p[1] - pts["start"][1]))
                if d > 6:
                    commit(pts["start"], p)
                    pts["start"] = pts["cur"] = None
                    return
            pts["start"] = p
            pts["cur"] = p
            return
        if event == cv2.EVENT_LBUTTONUP:
            if pts["start"] is None:
                return
            p = view.to_img(mx, my)
            d = float(np.hypot(p[0] - pts["start"][0],
                               p[1] - pts["start"][1]))
            if d > 6:
                commit(pts["start"], p)
                pts["start"] = pts["cur"] = None
            return

    cv2.setMouseCallback(title, on_mouse)

    while not state["quit"]:
        g = groups[state["i"]]
        draft = None
        if pts["start"] is not None and pts["cur"] is not None:
            draft = (pts["start"], pts["cur"])
        line = current_line()
        drawn = []
        if line is not None:
            seg = line.clip(img.shape[1], img.shape[0])
            if seg is not None:
                drawn = [((seg[0][0], seg[0][1]), (seg[1][0], seg[1][1]))]
        vis = view.render(lines=[(l, COL_SAVED, 2) for l in drawn],
                          draft=draft, hud=hud_text())
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
            g, frames, k, img, view, lines, rec = load_group(state["i"])
            state["lines"] = lines
        elif key in (UP,) or kk == ord("k"):
            state["i"] = (state["i"] - 1) % len(groups)
            g, frames, k, img, view, lines, rec = load_group(state["i"])
            state["lines"] = lines
        elif kk == ord("u") or key == 26:
            state["lines"] = state["lines"][:-1]
        elif kk == ord("r"):
            state["lines"] = []
        elif kk in (ord("z"),):
            view.zoom_at(canvas[0] // 2, canvas[1] // 2, 1.25)
        elif kk in (ord("x"),):
            view.zoom_at(canvas[0] // 2, canvas[1] // 2, 1 / 1.25)
        elif kk == ord("f"):
            view.fit()
        elif kk in (ord("s"), 13, 10):
            line = current_line()
            if line is None:
                _warn(title, "no line: drag on the reflection surface first")
                continue
            if not state["lines"]:
                continue
            raw1, raw2 = state["lines"][-1]
            rec_out = {
                "schema": "jetreflect/baseline-2",
                "dataset": dataset or os.path.basename(os.path.abspath(root)),
                "group": g["name"],
                "image_size": [int(img.shape[1]), int(img.shape[0])],
                "p1": [round(float(raw1[0]), 3), round(float(raw1[1]), 3)],
                "p2": [round(float(raw2[0]), 3), round(float(raw2[1]), 3)],
                "line": line.to_dict(),
                "tilt_deg": round(line.tilt_deg(), 3),
                "frame": os.path.basename(frames[k]),
                "source": "manual",
                "created": _iso(),
                "operator": _operator(),
            }
            if rec and rec.get("qc"):
                rec_out["qc"] = rec["qc"]
            write_json(os.path.join(g["dir"], "baseline.json"), rec_out)
            print("  [saved] %-24s tilt %+7.2f deg" % (g["label"],
                                                        rec_out["tilt_deg"]))
            state["i"] = (state["i"] + 1) % len(groups)
            g, frames, k, img, view, lines, rec = load_group(state["i"])
            state["lines"] = lines
        elif kk in (ord("q"), 27):
            state["quit"] = True
    cv2.destroyWindow(title)
    return groups


def pick_points(img, title, n=2, canvas=(1024, 768), hint=""):
    """Let the user click *n* points; returns a list of (x, y) in image coords."""
    view = _View(img, canvas)
    pts = []
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, canvas[0], canvas[1])

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append(view.to_img(mx, my))
        elif event == cv2.EVENT_MOUSEWHEEL:
            view.zoom_at(mx, my, 1.15 if flags > 0 else 1 / 1.15)

    cv2.setMouseCallback(title, on_mouse)
    while True:
        lines = []
        if len(pts) == 2:
            lines = [((pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]))]
        vis = view.render(lines=[(l, COL_LINE, 1) for l in lines],
                          hud=["click %d/%d points   %s" % (len(pts), n, hint),
                               "wheel=zoom  Q/Esc=cancel"])
        for p in pts:
            cv2.circle(vis, view.to_view(*p), 4, COL_LINE, -1)
        cv2.imshow(title, vis)
        key = cv2.waitKeyEx(20) & 0xFF
        if key in (ord("q"), 27):
            pts = []
            break
        if len(pts) >= n:
            break
    cv2.destroyWindow(title)
    return pts
