# -*- coding: utf-8 -*-
"""Engine v2: axis tracking without any assumption on the jet direction.

Why a second engine
-------------------
The legacy engine (``detect.py``) worked out the working frame from a global
Hough transform and then tracked **rows**.  On the 20261119 data that fails for
three independent reasons:

1. the longest straight dark edge is the *reflection surface*, not the jet, so
   the Hough tilt (-20 ... -56 deg) rotates the image by the wrong amount;
2. the outgoing stream leaves the impact point to the *down-left*, while the
   seed search only looked below-right of the impact;
3. a stream running 15-25 deg off the surface is nearly horizontal, so
   row-wise tracking is geometrically invalid.

This engine therefore

* derives the frame orientation from the tracked jet itself (two passes) and
  only falls back to the Hough estimate if the jet cannot be seeded at all;
* tracks both axes **along their own direction** (the image is rotated so the
  axis of interest is vertical), so near-horizontal streams are fine;
* searches the outgoing stream on **both sides along the surface**, on the
  fluid side only, starting next to the impact point, and requires the found
  segment to be *contiguous* with the impact (so a stream that breaks into
  droplets right away is rejected, while a short intact jet that breaks later
  is used up to the break);
* treats the surface as an **undirected infinite line** ``a*x + b*y + c = 0``;
* reports a frame without an outgoing stream as ``status = "none"`` instead of
  as a failure, so it is counted separately and kept out of the statistics.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from . import detect, geom
from .io_utils import imread

SCHEMA = "jetreflect/result-2"

#: tunables; every key can be overridden in ``jet_angle_config.json``
DEFAULT_PARAMS = {
    "cmin": 30.0,            # dark/bright contrast to accept an edge pair
    "dmin": 6.0,             # pair width range (px)
    "dmax": 45.0,
    "target_width": 15.0,
    "halfwin": 22.0,         # cross-axis search half window (px)
    "step": 2,               # step along the axis (px)
    "scan_step": 4,          # row step while seeding
    "max_rows": 520,         # max steps per direction
    "max_miss": 8,           # missed rows tolerated while tracking the jet
    "wall_margin": 10.0,     # keep this far from the surface (px)
    "seed_min_v": 18.0,      # incident seed must be this far from the surface
    "min_inc_points": 25,
    "rms_i_max": 2.0,
    "rms_r_max": 12.0,       # loose: the outgoing *boundary* undulates (see below)
    "se_max": 0.02,          # ... so gate on the standard error of its slope
    "theta_i_range": [5.0, 89.9],
    "theta_r_max": 89.5,
    "rho_max": 10.0,         # rho above this is reported as none
    "ref_v": [3.0, 140.0],   # band (distance from surface) searched for it
    "ref_u_max": 340.0,      # how far along the surface to follow it
    "ref_min_len": 60.0,     # contiguous length required (px)
    "bright_min_width": 2.0,   # bright core width range (px)
    "bright_max_width": 26.0,
    "bright_contrast": 12.0,   # core peak must be this close to bg (gray levels)
    "min_bright_points": 15,   # columns needed to fit the bright axis
    "min_axis_angle": 8.0,   # the two axes must not be collinear
    "min_frames": 20,
}


# --------------------------------------------------------------- helpers
def _params(params):
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    return p


def _pair(p, branch, width=None):
    """Pair-detection settings for one branch."""
    if branch == "inc":
        return {"cmin": p["cmin"], "dmin": p["dmin"], "dmax": p["dmax"]}
    w = float(width or p["target_width"])
    return {"cmin": p["cmin"] * p["ref_cmin_scale"],
            "dmin": max(4.0, 0.55 * w),
            "dmax": min(70.0, 2.4 * w)}


def _back(view, k, c):
    """``x = k*y + c`` in the rotated view -> undirected line in image coords."""
    h = view.size[1]
    q = geom.apply_matrix(view.minv, np.array([[c, 0.0],
                                               [k * (h - 1) + c, h - 1.0]]))
    return geom.Line.from_points(q[0], q[1])


class View(object):
    """One rotated, surface-aware copy of a frame."""

    def __init__(self, img, tau, base, sign):
        self.tau = float(tau)
        self.m, self.size = geom.fit_rot_canvas(img.shape[:2], -self.tau)
        self.minv = cv2.invertAffineTransform(self.m)
        self.img = cv2.warpAffine(img, self.m, self.size,
                                  flags=cv2.INTER_LINEAR,
                                  borderValue=int(np.median(img)))
        self.p0 = np.asarray(geom.apply_matrix(self.m, base.point_at(0.0)),
                             float)
        self.n = geom.unit(self.m[:, :2].dot(base.n))
        self.sign = float(sign)

    def dist(self, p):
        """Signed distance to the surface; ``> 0`` on the fluid side."""
        p = np.asarray(p, float)
        return self.sign * ((p[..., 0] - self.p0[0]) * self.n[0] +
                            (p[..., 1] - self.p0[1]) * self.n[1])

    def span(self, y, margin):
        """Column range on the fluid side of the surface at row ``y``."""
        w = self.size[0]
        ax = self.sign * self.n[0]
        ay = self.sign * self.n[1]
        if abs(ax) < 1e-9:
            if ay * (y - self.p0[1]) > margin:
                return 0, w - 1
            return 0, -1
        x = (margin + ax * self.p0[0] - ay * (y - self.p0[1])) / ax
        if ax > 0:
            return max(0, int(math.ceil(x))), w - 1
        return 0, min(w - 1, int(math.floor(x)))


def track(view, seed, prm, p, dirs=(1, -1), max_miss=None, avoid=None):
    """Follow a near-vertical axis in ``view``; returns ``(N, 3)`` x, y, width."""
    img = view.img
    h, w = img.shape[:2]
    halfwin = float(p["halfwin"])
    step = max(1, int(p["step"]))
    max_rows = int(p["max_rows"])
    if max_miss is None:
        max_miss = int(p["max_miss"])
    xs, ys, ws = float(seed[0]), float(seed[1]), float(seed[2])
    pts = [(xs, ys, ws)]
    for sgn in dirs:
        pred = xs
        hist = []
        widths = [ws]
        miss = 0
        for k in range(1, max_rows):
            y = int(round(ys + sgn * step * k))
            if y < 2 or y >= h - 2:
                break
            if miss > max_miss:
                break
            if len(hist) >= 3:
                yy = np.array([q[0] for q in hist[-6:]], float)
                xx = np.array([q[1] for q in hist[-6:]], float)
                slope = float(np.clip(np.polyfit(yy, xx, 1)[0], -3.0, 3.0))
                pred += sgn * step * slope
            lo, hi = view.span(y, float(p["wall_margin"]))
            lo = int(max(lo, pred - halfwin))
            hi = int(min(hi, pred + halfwin))
            if hi - lo < int(ws) + 8:
                break
            r = detect.find_pair(img[y, lo:hi + 1], lo, None,
                                 cmin=prm["cmin"], dmin=prm["dmin"],
                                 dmax=prm["dmax"], target=ws)
            if r is None:
                miss += 1
                continue
            med = float(np.median(widths[-10:]))
            if abs(r[2] - med) > max(6.0, 0.8 * med):
                miss += 1
                continue
            c = 0.5 * (r[0] + r[1])
            if abs(c - pred) > halfwin:
                miss += 1
                continue
            if avoid is not None and avoid(c, y):
                miss += 1
                continue
            miss = 0
            pts.append((c, float(y), r[2]))
            widths.append(r[2])
            hist.append((float(y), c))
            pred = c
    return np.asarray(pts, float)


# --------------------------------------------------------------- scene
class Scene(object):
    """Per-group measurement context: undirected surface + parameters."""

    def __init__(self, base, sign, params=None, image_size=None,
                 group="", dataset=""):
        self.base = base
        self.sign = float(sign)
        self.params = _params(params)
        self.image_size = list(image_size) if image_size else None
        self.group = group
        self.dataset = dataset

    def to_dict(self):
        return {"base": self.base.to_dict(), "sign": self.sign,
                "params": self.params, "image_size": self.image_size}


def build_scene(base_rec, img, params=None, group="", dataset=""):
    """Scene from a ``baseline.json`` record and one representative frame."""
    line = geom.Line.from_dict(base_rec)
    if line is None:
        return None, "baseline.json has no usable line (p1/p2 or a/b/c)"
    p0 = line.point_at(0.0)
    up = detect._side_brightness(img, p0, line.n, +1)
    dn = detect._side_brightness(img, p0, line.n, -1)
    sign = 1.0 if up >= dn else -1.0
    return Scene(line, sign, params, [img.shape[1], img.shape[0]],
                 group=group, dataset=dataset), ""


# --------------------------------------------------------------- branches
def _incident_seeds(view, p, limit=10):
    """Candidate seeds for the incoming jet, farthest from the surface first.

    A single "farthest pair" is not enough: a droplet flying away from the
    plate is farther than the jet, so several candidates are tracked until one
    behaves like a jet.
    """
    h = view.img.shape[0]
    cands = []
    for y in range(2, h - 2, int(p["scan_step"])):
        lo, hi = view.span(y, float(p["wall_margin"]) + float(p["seed_min_v"]))
        if hi - lo < 40:
            continue
        r = detect.find_pair(view.img[y, lo:hi + 1], lo, None,
                             cmin=p["cmin"], dmin=p["dmin"],
                             dmax=p["dmax"], target=p["target_width"])
        if r is None:
            continue
        cx = 0.5 * (r[0] + r[1])
        v = float(view.dist((cx, y)))
        if v < float(p["seed_min_v"]):
            continue
        cands.append((v, cx, float(y), r[2]))
    cands.sort(key=lambda c: -c[0])
    out, taken = [], []
    for v, cx, y, w in cands:
        if any(abs(y - ty) < 40 for ty in taken):
            continue
        taken.append(y)
        out.append((cx, y, w))
        if len(out) >= limit:
            break
    return out


def _incident(scene, img):
    """Track the incoming jet. Returns ``(dict, why)``."""
    p = scene.params
    tau = 0.0
    res = None
    why = "incident seed not found"
    tried_hough = False
    for _ in range(3):
        view = View(img, tau, scene.base, scene.sign)
        seeds = _incident_seeds(view, p)
        if not seeds:
            if not tried_hough:
                tried_hough = True
                t = detect.jet_tilt(img, scene.base.point_at(0.0),
                                    scene.base.n)
                if t is not None and abs(t) > 2.0:
                    tau = float(t)
                    continue
            return None, "incident seed not found"
        res = None
        for seed in seeds:
            pts = track(view, seed, _pair(p, "inc"), p)
            keep = pts[view.dist(pts) > float(p["wall_margin"]) + 8.0]
            if len(keep) < int(p["min_inc_points"]):
                why = "incident too short (%d points)" % len(keep)
                continue
            k, c, rms, n, inl = detect.fit_line(keep)
            if rms > float(p["rms_i_max"]):
                why = "incident fit rms %.2f px" % rms
                continue
            res = {"line": _back(view, k, c), "rms": float(rms), "n": int(n),
                   "width": float(np.median(inl[:, 2])), "tau": float(tau)}
            tilt = math.degrees(math.atan(k))
            break
        if res is None:
            return None, why
        if abs(tilt) < 1.0:
            break
        tau += tilt
    return res, ""


def _bright_axis(view, p, imp, inc_r, side, s, bg, guard=16.0):
    """Bright core of the outgoing column, sandwiched by two dark edges.

    The stream is a liquid cylinder lit from behind: two dark silhouettes and a
    bright core in between.  For every ``u`` we look at the profile across the
    stream, take the *last* two dark runs (the inner one next to the plate, the
    outer one on the fluid side) and, only if the gap between them is bright,
    locate the peak of that gap.  Columns without such a sandwiched core are
    skipped (glued to the plate, droplets, broken stream).

    Returns ``(m, b, n, rms, umin, umax, se)`` or ``None``.
    """
    h, w = view.img.shape[:2]
    v0, v1 = float(p["ref_v"][0]), float(p["ref_v"][1])
    xp, yp = float(imp[0]), float(imp[1])
    thr = bg - float(p["cmin"])
    step = max(1, int(p["step"]))
    wmin = float(p.get("bright_min_width", 2.0))
    wmax = float(p.get("bright_max_width", 26.0))
    cmin_core = float(p.get("bright_contrast", 12.0))
    need = int(p["min_bright_points"])
    us, cs = [], []
    for k in range(1, int(p["ref_u_max"] / step)):
        u = float(k * step)
        y = int(round(yp + side * u))
        if y < 2 or y >= h - 2:
            break
        xa = int(round(xp + s * v0))
        xb = int(round(xp + s * v1))
        xa, xb = min(xa, xb), max(xa, xb)
        xa, xb = max(1, xa), min(w - 2, xb)
        if xb - xa < 20:
            break
        prof = view.img[y, xa:xb + 1].astype(np.float32)
        if s < 0:
            prof = prof[::-1]                      # index 0 = next to the plate
        idx = np.nonzero(prof < thr)[0]
        if idx.size == 0:
            continue
        # split the dark samples into contiguous runs
        cut = np.nonzero(np.diff(idx) > 1)[0]
        runs = []
        st = 0
        for c in cut:
            runs.append((int(idx[st]), int(idx[c])))
            st = c + 1
        runs.append((int(idx[st]), int(idx[-1])))
        if len(runs) < 2:
            continue
        in0, in1 = runs[-2]                        # inner edge (plate side)
        out0, out1 = runs[-1]                      # outer edge (fluid side)
        if in0 < 3 or out1 > prof.size - 4:
            continue                               # not separated / cut by window
        c0, c1 = in1 + 1, out0 - 1                 # bright core between the edges
        if c1 - c0 < wmin or c1 - c0 > wmax:
            continue
        core = prof[c0:c1 + 1]
        peak = float(core.max())
        if peak < bg - cmin_core:                  # no bright core -> not a stream
            continue
        plate = np.nonzero(core >= peak - 3.0)[0]
        i_sub = float(plate.mean())                # plateau centre = axis
        scale = (v1 - v0) / float(max(1, prof.size - 1))
        v_core = v0 + scale * (c0 + i_sub)
        x = xp + s * v_core
        if inc_r.distance((x, y)) < guard:         # the incoming jet itself
            continue
        us.append(u)
        cs.append(v_core)
    if len(us) < need:
        return None
    u = np.asarray(us, float)
    c = np.asarray(cs, float)
    m, b = np.polyfit(u, c, 1)
    r = c - (m * u + b)
    mad = float(np.median(np.abs(r - np.median(r)))) + 1e-6
    keep = np.abs(r - np.median(r)) < max(1.0, 4.0 * 1.4826 * mad)
    if keep.sum() >= need:
        m, b = np.polyfit(u[keep], c[keep], 1)
        r = c[keep] - (m * u[keep] + b)
        u, c = u[keep], c[keep]
    rms = float(np.std(r))
    se = rms / max(1.0, math.sqrt(float(u.size))) / max(1.0, float(np.std(u)))
    return float(m), float(b), int(u.size), rms, float(u.min()), float(u.max()), \
        float(se)


def _reflected(scene, img, inc, impact):
    """Direction of the outgoing stream, from the bright core of the column."""
    p = scene.params
    base = scene.base
    view = View(img, base.tilt_deg(), base, scene.sign)
    imp = np.asarray(geom.apply_matrix(view.m, np.asarray(impact, float)),
                     float)
    h, w = view.img.shape[:2]
    if not (2 <= imp[0] <= w - 3 and 2 <= imp[1] <= h - 3):
        return None, "invalid", "impact outside the rotated frame"
    inc_r = geom.Line.from_points(
        geom.apply_matrix(view.m, inc["line"].point_at(0.0)),
        geom.apply_matrix(view.m, inc["line"].point_at(120.0)))

    xp, yp = float(imp[0]), float(imp[1])
    s = 1.0 if float(view.dist((xp + 2.0, yp))) >= \
        float(view.dist((xp - 2.0, yp))) else -1.0
    bg = float(np.percentile(view.img, 80))
    if bg < 90:                                    # underexposed frame
        return None, "invalid", "frame too dark (bg %.0f)" % bg

    tan_min = math.tan(math.radians(float(p["min_axis_angle"])))
    guard = max(12.0, 0.5 * float(inc.get("width") or 16.0) + 10.0)

    best = None
    for side in (1, -1):
        fit = _bright_axis(view, p, imp, inc_r, side, s, bg, guard=guard)
        if fit is None:
            continue
        m, b, n, rms, umin, umax, se = fit
        if umax - umin < float(p["ref_min_len"]):
            continue
        if umin > 140.0:                           # starts too far from impact
            continue
        if abs(m) < tan_min:                       # parallel: just the plate
            continue
        if abs(b) / math.hypot(1.0, m) > 30.0:     # not attached to the impact
            continue
        if se > float(p["se_max"]):                # direction not determined
            continue
        if best is None or n > best[2]:
            best = (side, m, b, n, rms, umin, umax, se)

    if best is None:
        return None, "none", "no bright core between two dark edges"

    side, m, b, n, rms, umin, umax, se = best
    if rms > float(p["rms_r_max"]):
        return None, "invalid", ("bright core is not straight (rms %.1f px)"
                                 % rms)

    n_ref = geom.unit(view.m[:, :2].dot(base.n))
    d = geom.unit(np.array([s * m, side], float))
    theta_r = math.degrees(math.acos(min(1.0, abs(float(np.dot(d, n_ref))))))

    q = np.array([[xp + s * (m * k + b), yp + side * k]
                  for k in (umin, umax)], float)
    q_img = geom.apply_matrix(view.minv, q)
    line = geom.Line.from_points(q_img[0], q_img[1])

    return {"line": line, "theta_r": float(theta_r), "rms": float(rms),
            "n": int(n), "span": float(umax - umin), "offset": float(b),
            "side": int(side), "source": "bright",
            "seg": [q_img[0].tolist(), q_img[1].tolist()]}, "ok", ""


# --------------------------------------------------------------- gates
def sanity(scene, row):
    """Physical plausibility; returns ``(ok, reason)``."""
    p = scene.params
    if row.get("theta_i") is None:
        return False, "no incident axis"
    lo, hi = p["theta_i_range"]
    if not (lo <= row["theta_i"] <= hi):
        return False, "theta_i %.1f outside [%.0f, %.0f]" % (
            row["theta_i"], lo, hi)
    if row["theta_r"] > float(p["theta_r_max"]):
        return False, "theta_r %.1f too close to the surface (film?)" % (
            row["theta_r"])
    if row["angle_between"] < float(p["min_axis_angle"]):
        return False, ("axes only %.1f deg apart (same jet?)"
                       % row["angle_between"])
    if row["rms_i"] > float(p["rms_i_max"]):
        return False, "incident fit rms %.2f px" % row["rms_i"]
    if row["rms_r"] > float(p["rms_r_max"]):
        return False, "outgoing fit rms %.2f px" % row["rms_r"]
    return True, ""


# --------------------------------------------------------------- measure
def new_row(path, frame=None):
    return {"file": os.path.basename(path), "frame": frame,
            "status": "invalid", "why": "", "note": "",
            "theta_i": None, "theta_r": None, "rho": None,
            "angle_between": None, "x_imp": None, "y_imp": None,
            "rms_i": None, "rms_r": None, "n_i": 0, "n_r": 0,
            "inc_line": None, "ref_line": None, "valid": False,
            "ref_seg": None, "ref_source": None}


def measure(scene, path, frame=None):
    """Measure one frame.  ``theta_r = None`` means *no outgoing stream*."""
    out = new_row(path, frame)
    try:
        img = imread(path)
    except Exception as e:                                  # noqa: BLE001
        out["why"] = "read: %s" % e
        return out
    if scene.image_size and list(scene.image_size) != [img.shape[1],
                                                       img.shape[0]]:
        out["why"] = ("image size changed: %s vs %dx%d"
                      % (scene.image_size, img.shape[1], img.shape[0]))
        return out

    inc, why = _incident(scene, img)
    if inc is None:
        out["why"] = why
        return out
    out["inc_line"] = inc["line"].to_dict()
    out["rms_i"] = inc["rms"]
    out["n_i"] = inc["n"]
    out["theta_i"] = float(geom.angle_to_normal(inc["line"].d, scene.base.n))

    imp = inc["line"].intersect(scene.base)
    h, w = img.shape[:2]
    if imp is None or not (0 <= imp[0] <= w - 1 and 0 <= imp[1] <= h - 1):
        out["why"] = "impact outside frame"
        return out
    out["x_imp"], out["y_imp"] = float(imp[0]), float(imp[1])

    ref, status, why = _reflected(scene, img, inc, imp)
    if ref is None:
        out["status"] = status
        out["note"] = why
        return out

    out["ref_line"] = ref["line"].to_dict()
    out["ref_seg"] = ref.get("seg")
    out["rms_r"] = ref["rms"]
    out["n_r"] = ref["n"]
    out["theta_r"] = float(ref["theta_r"])
    out["ref_source"] = ref.get("source")
    out["angle_between"] = float(geom.angle_line_to_line(inc["line"],
                                                         ref["line"]))

    ok, why = sanity(scene, out)
    if not ok:
        out["why"] = why
        return out

    tan_i = math.tan(math.radians(out["theta_i"]))
    tan_r = math.tan(math.radians(out["theta_r"]))
    rho = (tan_r / tan_i) if abs(tan_i) > 1e-9 else None
    if rho is not None and (rho > float(scene.params["rho_max"]) or rho <= 0):
        out["note"] = "rho %.1f > %.0f -> none" % (rho,
                                                   scene.params["rho_max"])
        rho = None
    out["rho"] = None if rho is None else float(rho)
    out["status"] = "ok"
    out["valid"] = True
    return out


# --------------------------------------------------------------- drawing
def draw(img, scene, row, group="", scale=1.0):
    """Annotated BGR overlay: surface, both axes, impact point, numbers."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    seg = scene.base.clip(w, h)
    if seg is not None:
        cv2.line(vis, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])),
                 (255, 255, 0), 1, cv2.LINE_AA)
    for key, col in (("inc_line", (0, 0, 255)),):
        d = row.get(key)
        if not d:
            continue
        ln = geom.Line.from_dict(d)
        seg = ln.clip(w, h) if ln is not None else None
        if seg is not None:
            cv2.line(vis, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])),
                     col, 1, cv2.LINE_AA)
    seg = row.get("ref_seg")
    if seg:
        a = tuple(np.int32(seg[0]))
        b = tuple(np.int32(seg[1]))
        cv2.line(vis, a, b, (0, 220, 0), 1, cv2.LINE_AA)
        for q in (a, b):
            cv2.circle(vis, q, 2, (0, 220, 0), -1, cv2.LINE_AA)
    elif row.get("ref_line"):
        ln = geom.Line.from_dict(row["ref_line"])
        seg = ln.clip(w, h) if ln is not None else None
        if seg is not None:
            cv2.line(vis, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])),
                     (0, 220, 0), 1, cv2.LINE_AA)
    if row.get("x_imp") is not None:
        cv2.circle(vis, (int(round(row["x_imp"])), int(round(row["y_imp"]))),
                   4, (0, 200, 255), 1, cv2.LINE_AA)
    if scale != 1.0:
        vis = cv2.resize(vis, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_LINEAR)
    lines = label_lines(row, group)
    hh, ww = vis.shape[:2]
    bar = vis.copy()
    cv2.rectangle(bar, (0, 0), (ww, 20 + 20 * len(lines)), (0, 0, 0), -1)
    vis = cv2.addWeighted(bar, 0.55, vis, 0.45, 0)
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (8, 24 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (90, 255, 90) if i == 0 else (220, 235, 220), 1,
                    cv2.LINE_AA)
    return vis


def label_lines(row, group=""):
    """Three short text lines for QC images."""
    st = row.get("status") or "invalid"
    if row.get("theta_r") is None:
        head = "th_i %6.2f   th_r  none (%s)" % (row.get("theta_i") or 0.0, st)
    else:
        head = "th_i %6.2f   th_r %6.2f   rho %5.2f" % (
            row["theta_i"], row["theta_r"],
            row["rho"] if row["rho"] is not None else float("nan"))
    tail = "rms %.2f/%.2f  n %d/%d  %s" % (
        row.get("rms_i") or 0.0, row.get("rms_r") or 0.0,
        row.get("n_i") or 0, row.get("n_r") or 0,
        (row.get("why") or row.get("note") or st))
    return ["%s   f%s" % (group, row.get("frame")), head, tail]
