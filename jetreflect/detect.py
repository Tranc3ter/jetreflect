# -*- coding: utf-8 -*-
"""Core detector: dark-edge pair -> jet axis -> angles to the surface normal.

The algorithm is orientation aware:
  1. the user gives the reflection surface (a straight line);
  2. the fluid side is found from the brightness of the two half planes;
  3. the image is rotated so that the two jets are close to vertical;
  4. the incident jet is seeded from the dark pair furthest from the surface,
     the reflected jet from a pair downstream of the impact point;
  5. both axes are tracked row by row (sub-pixel edge midpoints) and fitted by
     a robust straight line.

All thresholds live in ``DEFAULT_PARAMS`` and can be overridden per dataset in
``jet_angle_config.json``.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import geom
from .io_utils import imread

DEFAULT_PARAMS = {
    # dark-edge pair detection
    "cmin": 30.0,          # minimum dark/background contrast (8-bit levels)
    "dmin": 6.0,           # minimum pair width (px)
    "dmax": 45.0,          # maximum pair width (px)
    "target_width": 15.0,  # preferred pair width (px)
    # tracking
    "margin": 22.0,        # keep this far from the surface (px)
    "halfwin": 22.0,       # half search window across the jet (px)
    "step": 2,             # row step while tracking
    "max_rows": 460,
    # seeding
    "scan_step": 5,
    "min_inc_points": 25,
    "min_ref_points": 15,
    # quality gates
    "rms_i_max": 1.5,
    "rms_r_max": 2.5,
    "theta_r_max": 85.0,
    "theta_i_range": [5.0, 89.9],
    # the reflected axis must separate from the incident one: at 100 px from
    # the impact the perpendicular gap has to exceed this value (px).  This
    # rejects "land" frames where the spreading film is tracked as if it were
    # a reflected jet.
    "min_separation_px": 18.0,
    # dataset
    "min_frames": 20,
}


# ---------------------------------------------------------------- primitives
def _subpix(prof, i):
    if i <= 0 or i >= len(prof) - 1:
        return float(i)
    a, b, c = prof[i - 1], prof[i], prof[i + 1]
    den = a - 2.0 * b + c
    return float(i) + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)


def find_pair(seg, x_off=0.0, bg=None, cmin=30.0, dmin=6.0, dmax=45.0,
              target=15.0):
    """Find the best "dark edge - bright core - dark edge" pair in a profile.

    Returns ``(x_left, x_right, width, dark_left, dark_right)`` or ``None``.
    """
    seg = np.asarray(seg, np.float32)
    n = seg.size
    if n < 9:
        return None
    if bg is None:
        k = max(1, min(n - 1, int(round(0.10 * n))))
        bg = float(np.partition(seg, -k)[-k])
    if bg < 90:
        return None
    g = ndi.gaussian_filter1d(seg, 0.8)
    # local minima (cheaper than scipy.signal.find_peaks and good enough
    # after the 0.8 px smoothing)
    mins = np.nonzero((g[1:-1] <= g[:-2]) & (g[1:-1] < g[2:]))[0] + 1
    if mins.size < 2:
        return None
    mins = mins[g[mins] < bg - cmin]
    if mins.size < 2:
        return None
    vals = g.tolist()
    idl = mins.tolist()
    best = None
    for ia in range(len(idl)):
        i = idl[ia]
        for ib in range(ia + 1, len(idl)):
            j = idl[ib]
            width = j - i
            if width < dmin:
                continue
            if width > dmax:
                break
            core = max(vals[i:j + 1])
            contrast = core - (vals[i] if vals[i] > vals[j] else vals[j])
            if contrast < cmin:
                continue
            score = contrast - 2.0 * abs(width - target)
            if best is None or score > best[0]:
                best = (score, i, j)
    if best is None:
        return None
    _, i, j = best
    return (x_off + _subpix(g, i), x_off + _subpix(g, j), float(j - i),
            float(g[i]), float(g[j]))


def fit_line(pts, k_mad=4.0):
    """Robust ``x = k*y + c`` fit; returns (k, c, rms, n_inlier, inliers)."""
    pts = np.asarray(pts, float)
    k, c = np.polyfit(pts[:, 1], pts[:, 0], 1)
    r = pts[:, 0] - (k * pts[:, 1] + c)
    mad = float(np.median(np.abs(r - np.median(r)))) + 1e-6
    keep = np.abs(r - np.median(r)) < max(1.2, k_mad * 1.4826 * mad)
    if keep.sum() < 15:
        keep = np.ones(len(pts), bool)
    k, c = np.polyfit(pts[keep, 1], pts[keep, 0], 1)
    r = pts[keep, 0] - (k * pts[keep, 1] + c)
    return float(k), float(c), float(np.std(r)), int(keep.sum()), pts[keep]


def _dark_mask(img, k=25):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ker)
    return np.clip(bg.astype(np.int16) - img.astype(np.int16), 0, 255).astype(np.uint8)


def hough_segments(mask, threshold=25, min_len=45, max_gap=6):
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=threshold,
                           minLineLength=min_len, maxLineGap=max_gap)
    if segs is None:
        return np.zeros((0, 4), float)
    return np.asarray(segs, float).reshape(-1, 4)


def jet_tilt(img, p0, n, vmin=25.0):
    """Dominant jet tilt (deg from vertical, y down) used to build the frame."""
    d = _dark_mask(img)
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    v = (xs - p0[0]) * n[0] + (ys - p0[1]) * n[1]
    mask = (((d > 35) & (v > vmin)).astype(np.uint8)) * 255
    segs = hough_segments(mask)
    if len(segs) == 0:
        return None
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    L = np.hypot(dx, dy)
    T = np.degrees(np.arctan2(dx, dy))
    T = (T + 90.0) % 180.0 - 90.0
    if L.sum() <= 0:
        return None
    return float(np.sum(T * L) / np.sum(L))


# ---------------------------------------------------------------- context
class Ctx(object):
    """Everything needed to measure one frame of one group."""

    def __init__(self, p1, p2, params=None, n=None):
        self.p1 = np.asarray(p1, float)
        self.p2 = np.asarray(p2, float)
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.p0, self.d = geom.two_points_to_pd(self.p1, self.p2)
        self.n = geom.perp(self.d) if n is None else np.asarray(n, float)
        self.kref, self.cref = geom.kc_from_points(self.p1, self.p2)
        self.tau = 0.0
        self.m = None
        self.minv = None
        self.size = None
        self.q0 = None
        self.nq = None
        self.ms = None
        self.cs = None

    # ---- setup -----------------------------------------------------------
    def setup_fluid(self, img):
        """Pick the normal direction that points into the fluid (brighter)."""
        if _side_brightness(img, self.p0, self.n, +1) < \
                _side_brightness(img, self.p0, self.n, -1):
            self.n = -self.n
        return self.n

    def prepare_for(self, shape, tau):
        self.tau = float(tau)
        self.m, self.size = geom.fit_rot_canvas(shape, -self.tau)
        self.minv = cv2.invertAffineTransform(self.m)
        q1 = geom.apply_matrix(self.m, self.p1)
        q2 = geom.apply_matrix(self.m, self.p2)
        self.q0 = 0.5 * (q1 + q2)
        dq = geom.unit(q2 - q1)
        self.nq = self.m[:, :2] @ self.n          # rotation keeps orientation
        self.q0 = np.asarray(self.q0, float)
        self.ms = dq[0] / dq[1]
        self.cs = q2[0] - self.ms * q2[1]

    # ---- serialisation ---------------------------------------------------
    def to_dict(self):
        return {
            "p1": self.p1.tolist(), "p2": self.p2.tolist(),
            "n": self.n.tolist(), "params": self.params, "tau": self.tau,
            "m": None if self.m is None else self.m.tolist(),
            "size": self.size,
            "q0": None if self.q0 is None else self.q0.tolist(),
            "nq": None if self.nq is None else self.nq.tolist(),
            "ms": self.ms, "cs": self.cs,
        }

    @classmethod
    def from_dict(cls, d):
        c = cls(d["p1"], d["p2"], d.get("params"), d.get("n"))
        c.tau = float(d.get("tau", 0.0))
        c.m = None if d.get("m") is None else np.asarray(d["m"], float)
        c.size = None if d.get("size") is None else tuple(d["size"])
        c.q0 = None if d.get("q0") is None else np.asarray(d["q0"], float)
        c.nq = None if d.get("nq") is None else np.asarray(d["nq"], float)
        c.ms = d.get("ms")
        c.cs = d.get("cs")
        if c.m is not None:
            c.minv = cv2.invertAffineTransform(c.m)
        return c

    # ---- measurement -----------------------------------------------------
    def measure(self, path):
        return measure_frame(self, path)


def _side_brightness(img, p0, n, sign):
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    v = (xs - p0[0]) * n[0] + (ys - p0[1]) * n[1]
    vals = []
    for t in (45.0, 70.0, 100.0, 140.0):
        m = np.abs(v - sign * t) < 3.0
        if m.sum() > 80:
            vals.append(float(np.median(img[m])))
    return float(np.mean(vals)) if vals else 0.0


def _track(img, ctx, x_seed, y_seed, w_seed):
    """Follow one jet along y in the rotated frame."""
    h, w = img.shape[:2]
    p = ctx.params
    halfwin = float(p["halfwin"])
    margin = float(p["margin"])
    step = max(1, int(p["step"]))
    max_rows = int(p["max_rows"])

    pts = [(x_seed, float(y_seed), w_seed)]
    widths = [w_seed]
    for sgn in (+1, -1):
        pred = x_seed
        hist = []
        miss = 0
        for k in range(1, max_rows):
            y = int(round(y_seed + sgn * step * k))
            if y < 2 or y >= h - 2:
                break
            if miss > 14:
                break
            if len(hist) >= 3:
                yy = np.array([q[0] for q in hist[-6:]], float)
                xx = np.array([q[1] for q in hist[-6:]], float)
                slope = float(np.clip(np.polyfit(yy, xx, 1)[0], -3.0, 3.0))
                pred += sgn * step * slope
            lo = int(max(2, pred - halfwin))
            hi = int(min(w - 2, pred + halfwin))
            if hi - lo < int(w_seed) + 8:
                break
            r = find_pair(img[y, lo:hi], lo, None,
                          dmin=max(4.0, 0.4 * w_seed),
                          dmax=min(80.0, 3.0 * w_seed), target=w_seed)
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
            if geom.signed_distance((c, y), ctx.q0, ctx.nq) < margin:
                break
            miss = 0
            pts.append((c, float(y), r[2]))
            widths.append(r[2])
            hist.append((float(y), c))
            pred = c
    return np.asarray(pts, float)


def _fluid_span(ctx, y, vmin=10.0):
    nx = ctx.nq[0]
    w = ctx.size[0]
    if abs(nx) < 1e-6:
        return 0, w - 1
    x = ctx.q0[0] + (vmin - (y - ctx.q0[1]) * ctx.nq[1]) / nx
    if nx > 0:
        return int(max(0, x)), w - 1
    return 0, int(min(w - 1, x))


def measure_frame(ctx, path):
    """Measure one frame.  Returns a plain dict (JSON serialisable)."""
    from os.path import basename

    out = {"file": basename(path), "ok": False, "why": "", "note": ""}
    try:
        img = imread(path)
    except Exception as e:                                  # noqa: BLE001
        out["why"] = "read: %s" % e
        return out

    warped = cv2.warpAffine(img, ctx.m, ctx.size, flags=cv2.INTER_LINEAR,
                            borderValue=int(np.median(img)))
    h, w = warped.shape[:2]
    p = ctx.params

    # ---- 1. incident seed: the pair furthest from the surface -----------
    seed = None
    for y in range(2, h - 2, int(p["scan_step"])):
        lo, hi = _fluid_span(ctx, y)
        if hi - lo < 40:
            continue
        r = find_pair(warped[y, lo:hi + 1], lo, None, cmin=p["cmin"],
                      dmin=p["dmin"], dmax=p["dmax"],
                      target=p["target_width"])
        if r is None:
            continue
        cx = 0.5 * (r[0] + r[1])
        v = float(geom.signed_distance((cx, y), ctx.q0, ctx.nq))
        if v < 25.0:
            continue
        if seed is None or v > seed[0]:
            seed = (v, cx, float(y), r[2])
    if seed is None:
        out["why"] = "incident seed not found"
        return out
    _, sx, sy, sw = seed

    inc = _track(warped, ctx, sx, sy, sw)
    inc_f = inc[geom.signed_distance(inc, ctx.q0, ctx.nq) > 30.0]
    if len(inc_f) < int(p["min_inc_points"]):
        out["why"] = "incident too short (%d)" % len(inc_f)
        return out
    ki, ci, rms_i, n_i, inc_in = fit_line(inc_f)
    if rms_i > float(p["rms_i_max"]):
        out["why"] = "incident fit rms %.2f px" % rms_i
        return out

    # ---- 2. impact point = incident axis x surface ----------------------
    denom = ctx.ms - ki
    if abs(denom) < 1e-6:
        out["why"] = "incident axis parallel to surface"
        return out
    y_imp = (ci - ctx.cs) / denom
    x_imp = ki * y_imp + ci
    if not (0 <= y_imp < h and 0 <= x_imp < w):
        out["why"] = "impact outside frame"
        return out

    # ---- 3. reflected seed downstream of the impact ---------------------
    ref_seed = None
    y_lo = int(y_imp) + 30
    y_hi = min(h - 3, int(y_imp) + 250)
    for y in range(y_lo, y_hi, 6):
        lo, hi = _fluid_span(ctx, y)
        lo = max(lo, int(x_imp + 18))
        hi = min(hi, int(x_imp + 380))
        if hi - lo < 25:
            continue
        r = find_pair(warped[y, lo:hi], lo, None, cmin=max(20.0, 0.8 * p["cmin"]),
                      dmin=max(4.0, 0.55 * sw),
                      dmax=min(60.0, 2.0 * sw), target=sw)
        if r is None:
            continue
        cx = 0.5 * (r[0] + r[1])
        if geom.signed_distance((cx, y), ctx.q0, ctx.nq) < 8:
            continue
        dist = math.hypot(cx - x_imp, y - y_imp)
        if dist < 30 or dist > 300:
            continue
        score = r[3] + r[4]
        if ref_seed is None or score < ref_seed[0]:
            ref_seed = (score, cx, float(y), r[2])
    if ref_seed is None:
        out["why"] = "reflected seed not found"
        return out
    _, rx, ry, rw = ref_seed

    ref = _track(warped, ctx, rx, ry, rw)
    dist = np.hypot(ref[:, 0] - x_imp, ref[:, 1] - y_imp)
    ref_f = ref[(dist > 25) &
                (geom.signed_distance(ref, ctx.q0, ctx.nq) > 8)]
    if len(ref_f) < int(p["min_ref_points"]):
        out["why"] = "reflected too short (%d)" % len(ref_f)
        return out
    kr, cr, rms_r, n_r, ref_in = fit_line(ref_f)
    if rms_r > float(p["rms_r_max"]):
        out["why"] = "reflected fit rms %.2f px" % rms_r
        return out

    # ---- 4. back to image coordinates and angles -----------------------
    def back(k, c):
        q = geom.apply_matrix(ctx.minv, np.array([[c, 0.0],
                                                  [k * (h - 1) + c, h - 1.0]]))
        return geom.kc_from_points(q[0], q[1])

    ko, co = back(ki, ci)
    kro, cro = back(kr, cr)
    nref = geom.perp(geom.unit([ctx.kref, 1.0]))
    th_i = geom.angle_to_normal([ko, 1.0], nref)
    th_r = geom.angle_to_normal([kro, 1.0], nref)
    angle_between = geom.angle_between_lines([ko, 1.0], [kro, 1.0])
    rho = math.tan(math.radians(th_r)) / max(1e-6, math.tan(math.radians(th_i)))

    wi = float(np.median(inc_in[:, 2]) * math.cos(abs(math.atan(ki))))
    wr = float(np.median(ref_in[:, 2]) * math.cos(abs(math.atan(kr))))
    imp = geom.apply_matrix(ctx.minv, (x_imp, y_imp))

    out.update({
        "ok": True,
        "theta_i": float(th_i), "theta_r": float(th_r),
        "angle_between": float(angle_between), "rho": float(rho),
        "x_imp": float(imp[0]), "y_imp": float(imp[1]),
        "R_inc": wi / 2.0, "R_ref": wr / 2.0,
        "rms_i": float(rms_i), "rms_r": float(rms_r),
        "n_i": int(n_i), "n_r": int(n_r),
        "inc_k": float(ko), "inc_c": float(co),
        "ref_k": float(kro), "ref_c": float(cro),
        "inc_w": wi, "ref_w": wr,
    })
    return out


def sanity(ctx, row):
    """Physical plausibility gate; returns (ok, reason)."""
    p = ctx.params
    if not row.get("ok"):
        return False, row.get("why", "not measured")
    lo, hi = p["theta_i_range"]
    if not (lo <= row["theta_i"] <= hi):
        return False, "theta_i %.1f outside [%.0f, %.0f]" % (
            row["theta_i"], lo, hi)
    if row["theta_r"] > float(p["theta_r_max"]):
        return False, "theta_r %.1f too close to surface (land/splash)" % (
            row["theta_r"])
    sep = 100.0 * math.sin(math.radians(row["angle_between"]))
    if sep < float(p["min_separation_px"]):
        return False, "reflected axis only %.0f px from incident at 100 px " \
                      "(film/land, no jet)" % sep
    if row.get("rms_i", 0) > float(p["rms_i_max"]):
        return False, "incident fit rms %.2f px" % row["rms_i"]
    if row.get("rms_r", 0) > float(p["rms_r_max"]):
        return False, "reflected fit rms %.2f px" % row["rms_r"]
    return True, ""
