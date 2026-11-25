# -*- coding: utf-8 -*-
"""File helpers: Chinese/space-safe image IO, group discovery, JSON."""
from __future__ import annotations

import json
import os
import re
import tempfile

import cv2
import numpy as np

IMAGE_EXT = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg")

try:  # silence the harmless TIFF private-tag warnings (0x7900)
    try:
        import cv2.utils.logging as _cvl
        _cvl.setLogLevel(_cvl.LOG_LEVEL_SILENT)
    except Exception:                                   # noqa: BLE001
        cv2.setLogLevel(0)
except Exception:                                       # noqa: BLE001
    pass


def imread(path):
    """Read an image with a path containing any Unicode characters."""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError("cannot read image: %s" % path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[..., 0]
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("cannot encode image: %s" % path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    buf.tofile(path)


def list_images(folder):
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted([os.path.join(folder, n) for n in names
                   if n.lower().endswith(IMAGE_EXT)], key=natural_key)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(str(s)))]


def _is_group_dir(folder, min_frames):
    return len(list_images(folder)) >= min_frames


def discover_groups(root, min_frames=20, skip=None):
    """Immediate sub-folders of *root* that look like one measurement group."""
    skip = set(s.lower() for s in (skip or ()))
    default_skip = {"sample", "jet_angle", "results", "_summary", "doc", "docs",
                    "config", "tools", "deck", "paper", ".git", "__pycache__"}
    out = []
    try:
        names = os.listdir(root)
    except OSError as e:
        raise IOError("cannot list %s: %s" % (root, e))
    for name in sorted(names, key=natural_key):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if name.startswith(".") or name.lower() in default_skip \
                or name.lower() in skip:
            continue
        imgs = list_images(path)
        if len(imgs) >= min_frames:
            out.append({"name": name, "label": name.strip(),
                        "dir": path, "images": imgs})
    return out


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d or ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
