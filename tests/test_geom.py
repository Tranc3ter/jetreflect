# -*- coding: utf-8 -*-
"""Geometry and GUI-viewport tests (no window needed)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jetreflect import geom  # noqa: E402


def test_kc_roundtrip():
    k, c = geom.kc_from_points((100.0, 0.0), (200.0, 300.0))
    assert abs(k - 1.0 / 3.0) < 1e-9
    p0, d = geom.kc_to_pd(k, c, 150.0)
    assert abs(p0[0] - (k * 150 + c)) < 1e-9


def test_angle_to_normal():
    # horizontal surface, normal = +y
    n = np.array([0.0, -1.0])
    assert abs(geom.angle_to_normal([0.0, 1.0], n) - 0.0) < 1e-6
    assert abs(geom.angle_to_normal([1.0, 0.0], n) - 90.0) < 1e-6
    assert abs(geom.angle_to_normal([1.0, 1.0], n) - 45.0) < 1e-6


def test_tilt():
    assert abs(geom.tilt_from_vertical((0, 0), (0, 10))) < 1e-9
    assert abs(geom.tilt_from_vertical((0, 0), (10, 10)) - 45.0) < 1e-9


def test_viewport_roundtrip():
    from jetreflect.gui import _View
    img = np.zeros((576, 768), np.uint8)
    v = _View(img, (1024, 768))
    v.scale = 2.0
    v.cx, v.cy = 400.0, 300.0
    v.render()
    for p in ((10.0, 10.0), (400.0, 300.0), (760.0, 570.0)):
        q = v.to_img(*v.to_view(*p))
        assert abs(q[0] - p[0]) < 1.0 and abs(q[1] - p[1]) < 1.0, (p, q)


if __name__ == "__main__":
    test_kc_roundtrip()
    test_angle_to_normal()
    test_tilt()
    test_viewport_roundtrip()
    print("geometry tests passed")
