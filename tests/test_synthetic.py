# -*- coding: utf-8 -*-
"""Synthetic-image tests: the measured angles must match the drawing."""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jetreflect import detect, geom  # noqa: E402
from jetreflect.io_utils import imwrite  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _art():
    d = os.path.join(HERE, "_art")
    os.makedirs(d, exist_ok=True)
    return d


def synth(size=(768, 576), plate_deg=21.8, inc_deg=0.46, ref_deg=35.6,
          impact=(370.0, 290.0), d=15.0, bg=230, dark=25):
    """Draw a plate, an incident jet and a reflected jet with known geometry."""
    w, h = size
    img = np.full((h, w), bg, np.uint8)
    p0 = np.array(impact, float)
    dsurf = np.array([math.sin(math.radians(plate_deg)),
                      math.cos(math.radians(plate_deg))])
    nsurf = np.array([dsurf[1], -dsurf[0]])       # points to the fluid
    ys, xs = np.mgrid[0:h, 0:w]
    v = (xs - p0[0]) * nsurf[0] + (ys - p0[1]) * nsurf[1]
    img[v < 0] = dark
    img[(v < 0) & (v > -18)] = 150                # bright specular band
    cv2.line(img, (int(p0[0] - 600 * dsurf[0]), int(p0[1] - 600 * dsurf[1])),
             (int(p0[0] + 600 * dsurf[0]), int(p0[1] + 600 * dsurf[1])),
             200, 4)
    for k, y_from, y_to in (
            (math.tan(math.radians(inc_deg)), -10, impact[1]),
            (math.tan(math.radians(ref_deg)), impact[1], h + 10)):
        dd = np.array([k, 1.0]) / math.hypot(k, 1.0)
        nn = np.array([-dd[1], dd[0]])
        for s in (-1, 1):
            a = p0 + nn * s * d / 2.0 + dd * (y_from - impact[1])
            b = p0 + nn * s * d / 2.0 + dd * (y_to - impact[1])
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     0, 2, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    p1 = p0 - dsurf * 300
    p2 = p0 + dsurf * 300
    return img, p1, p2, dsurf, nsurf


def _measure(img, p1, p2, tau=17.0, name="a.png"):
    path = os.path.join(_art(), name)
    imwrite(path, img)
    ctx = detect.Ctx(p1, p2, dict(detect.DEFAULT_PARAMS))
    ctx.setup_fluid(img)
    ctx.prepare_for(img.shape[:2], tau)
    return ctx, ctx.measure(path)


def test_synthetic_angles():
    img, p1, p2, dsurf, nsurf = synth()
    ctx, row = _measure(img, p1, p2, name="synthetic.png")
    assert row["ok"], row.get("why")
    nref = geom.perp(geom.unit([math.tan(math.radians(21.8)), 1.0]))
    di = geom.unit([math.tan(math.radians(0.46)), 1.0])
    dr = geom.unit([math.tan(math.radians(35.6)), 1.0])
    exp_i = geom.angle_to_normal(di, nref)
    exp_r = geom.angle_to_normal(dr, nref)
    err_i = abs(row["theta_i"] - exp_i)
    err_r = abs(row["theta_r"] - exp_r)
    print("theta_i %.3f (exp %.3f)  theta_r %.3f (exp %.3f)  d=%.3f/%.3f"
          % (row["theta_i"], exp_i, row["theta_r"], exp_r, err_i, err_r))
    assert err_i < 0.2, err_i
    assert err_r < 0.2, err_r
    assert abs(row["x_imp"] - 370.0) < 3.0
    assert abs(row["y_imp"] - 290.0) < 3.0


def test_separation_rejects_land():
    """A film tracked almost along the incident jet must fail the QC gate."""
    img, p1, p2, dsurf, nsurf = synth(ref_deg=1.5)
    ctx, row = _measure(img, p1, p2, name="land.png")
    if not row.get("ok"):
        return
    ok, why = detect.sanity(ctx, row)
    assert not ok, "land case should be rejected, got %.2f deg" % row["theta_r"]


def test_find_pair():
    seg = np.full(120, 230.0)
    seg[40:43] = 30
    seg[55:58] = 30
    r = detect.find_pair(seg, 0.0, None, 30, 6, 45, 15)
    assert r is not None
    assert abs(r[0] - 41.0) < 1.0
    assert abs(r[1] - 56.0) < 1.0


if __name__ == "__main__":
    test_find_pair()
    test_synthetic_angles()
    test_separation_rejects_land()
    print("all tests passed")
