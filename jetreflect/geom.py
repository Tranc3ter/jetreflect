# -*- coding: utf-8 -*-
"""Plane geometry used by the jet-reflection measurement.

Conventions
-----------
* image coordinates: x to the right, y down.
* a straight line is stored either as two points or as ``x = k*y + c``
  (valid for any line that is not horizontal; checked by ``kc_ok``).
* the surface normal ``n`` always points into the fluid.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def perp(d):
    """Rotate a direction by +90 deg (in the x/y component convention)."""
    return np.array([-d[1], d[0]], float)


def two_points_to_pd(p1, p2):
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    return 0.5 * (p1 + p2), unit(p2 - p1)


def pd_to_kc(p0, d):
    """``x = k*y + c`` for a non-horizontal line through p0 with direction d."""
    k = d[0] / d[1]
    return float(k), float(p0[0] - k * p0[1])


def kc_to_pd(k, c, y=0.0):
    p0 = np.array([k * y + c, y], float)
    return p0, unit([k, 1.0])


def kc_from_points(p1, p2):
    p0, d = two_points_to_pd(p1, p2)
    return pd_to_kc(p0, d)


def kc_ok(p1, p2, min_dy=1.0):
    return abs(float(p2[1]) - float(p1[1])) >= min_dy


def offset_line_kc(k, c, t):
    """Shift ``x = k*y + c`` along its normal by ``t`` pixels (x>0 side)."""
    n = math.hypot(1.0, k)
    return c + t * n


def tilt_from_vertical(p1, p2):
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return math.degrees(math.atan2(dx, dy))


def angle_between_lines(d1, d2):
    """Acute angle in degrees between two undirected lines."""
    c = abs(float(np.dot(unit(d1), unit(d2))))
    return math.degrees(math.acos(min(1.0, c)))


def angle_to_normal(d, nref):
    """Acute angle (deg) between a jet direction and the surface normal."""
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(unit(d), unit(nref)))))))


def rot_matrix(angle_deg, center):
    return cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                   float(angle_deg), 1.0)


class Line(object):
    """Undirected infinite line ``a*x + b*y + c = 0`` with unit ``(a, b)``.

    The representation is canonical -- ``(a, b)`` has unit length and the sign
    is fixed (``a > 0``, or ``a == 0 and b > 0``) -- so ``p1 -> p2`` and
    ``p2 -> p1`` describe the *same* line.  A line therefore has no direction:
    ``n = (a, b)`` is one of the two normals and ``d = (-b, a)`` one of the two
    directions; all angles are computed with absolute values.
    """

    __slots__ = ("a", "b", "c")

    def __init__(self, a, b, c):
        norm = math.hypot(a, b)
        if norm < 1e-12:
            raise ValueError("degenerate line (a = b = 0)")
        a, b, c = a / norm, b / norm, c / norm
        if a < -1e-12 or (abs(a) <= 1e-12 and b < 0.0):
            a, b, c = -a, -b, -c
        self.a, self.b, self.c = float(a), float(b), float(c)

    # ---- constructors ----------------------------------------------------
    @classmethod
    def from_points(cls, p1, p2):
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        if math.hypot(x2 - x1, y2 - y1) < 1e-9:
            raise ValueError("two identical points do not define a line")
        return cls(y1 - y2, x2 - x1, x1 * y2 - x2 * y1)

    @classmethod
    def from_dict(cls, d):
        if not d:
            return None
        if "a" in d and "b" in d and "c" in d:
            return cls(d["a"], d["b"], d["c"])
        if "p1" in d and "p2" in d:
            return cls.from_points(d["p1"], d["p2"])
        return None

    @classmethod
    def from_kc(cls, k, c):
        """``x = k*y + c`` -> normalized line (handles huge / infinite k)."""
        return cls(1.0, -k, -c)

    # ---- properties ------------------------------------------------------
    @property
    def n(self):
        return np.array([self.a, self.b], float)

    @property
    def d(self):
        return np.array([-self.b, self.a], float)

    def to_dict(self):
        return {"a": self.a, "b": self.b, "c": self.c}

    def kc(self):
        """``(k, c)`` with ``x = k*y + c``; ``None`` when the line is horizontal."""
        if abs(self.b) < 1e-9:
            return None
        return -self.a / self.b, -self.c / self.b

    def is_horizontal(self, tol=1e-6):
        return abs(self.b) < tol

    # ---- geometry --------------------------------------------------------
    def signed(self, p):
        p = np.asarray(p, float)
        return p[..., 0] * self.a + p[..., 1] * self.b + self.c

    def distance(self, p):
        return np.abs(self.signed(p))

    def project(self, p):
        p = np.asarray(p, float)
        return p - self.signed(p)[..., None] * self.n

    def point_at(self, s=0.0):
        """Point on the line at arclength ``s`` from the foot of the origin."""
        return -self.c * self.n + float(s) * self.d

    def offset(self, t):
        """Parallel line shifted by ``t`` pixels along ``n``."""
        return Line(self.a, self.b, self.c - float(t))

    def tilt_deg(self):
        """Direction folded into ``(-90, 90]`` -- undirected, so no 137 deg."""
        t = math.degrees(math.atan2(self.d[0], self.d[1]))
        if t > 90.0:
            t -= 180.0
        elif t <= -90.0:
            t += 180.0
        return t

    def angle_deg(self):
        """Acute angle of the line against the x axis, in ``[0, 90]``."""
        return 90.0 - abs(self.tilt_deg())

    def intersect(self, other):
        det = self.a * other.b - other.a * self.b
        if abs(det) < 1e-12:
            return None
        x = (self.b * other.c - other.b * self.c) / det
        y = (self.c * other.a - other.c * self.a) / det
        return np.array([x, y], float)

    def clip(self, w, h, margin=0.0):
        """Segment of the line clipped to the image rectangle."""
        pts = []
        for x in (margin, w - 1 - margin):
            if abs(self.b) > 1e-9:
                y = -(self.a * x + self.c) / self.b
                if margin <= y <= h - 1 - margin:
                    pts.append((x, y))
        for y in (margin, h - 1 - margin):
            if abs(self.a) > 1e-9:
                x = -(self.b * y + self.c) / self.a
                if margin <= x <= w - 1 - margin:
                    pts.append((x, y))
        if len(pts) < 2:
            return None
        pts = sorted(set((round(px, 3), round(py, 3)) for px, py in pts))
        if len(pts) < 2:
            return None
        p, q = pts[0], pts[-1]
        return (np.array(p, float), np.array(q, float))


def angle_line_to_line(l1, l2):
    """Acute angle (deg) between two undirected lines."""
    return angle_between_lines(l1.d, l2.d)


def fit_rot_canvas(shape, angle_deg):
    """Build a rotation matrix that keeps the whole image inside the canvas."""
    h, w = shape[:2]
    m = rot_matrix(angle_deg, (w / 2.0 - 0.5, h / 2.0 - 0.5))
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    pts = (m[:, :2] @ corners.T).T + m[:, 2]
    lo, hi = pts.min(0), pts.max(0)
    m[0, 2] -= lo[0]
    m[1, 2] -= lo[1]
    size = (int(round(hi[0] - lo[0])) + 1, int(round(hi[1] - lo[1])) + 1)
    return m, size


def apply_matrix(m, p):
    """Apply a 2x3 affine matrix to a point or an (N,2) array."""
    arr = np.asarray(p, float)
    flat = arr.reshape(-1, 2)
    out = (m[:, :2] @ flat.T).T + m[:, 2]
    return out.reshape(arr.shape)


def signed_distance(p, p0, n):
    p = np.asarray(p, float)
    return (p[..., 0] - p0[0]) * n[0] + (p[..., 1] - p0[1]) * n[1]
