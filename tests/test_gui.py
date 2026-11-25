# -*- coding: utf-8 -*-
"""GUI smoke test: simulate a drag + save without opening a real window."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from jetreflect import gui  # noqa: E402
from jetreflect.io_utils import imwrite  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CANVAS = (512, 384)


def _dataset():
    root = os.path.join(HERE, "_art", "ds")
    gdir = os.path.join(root, "0.9")
    os.makedirs(gdir, exist_ok=True)
    img = np.full((576, 768), 210, np.uint8)
    img[:, :300] = 40
    images = []
    for i in range(3):
        p = os.path.join(gdir, "f%03d.png" % i)
        imwrite(p, img)
        images.append(p)
    return root, {"name": "0.9", "label": "0.9", "dir": gdir,
                  "images": images}


def test_mark_and_save():
    root, group = _dataset()
    base = os.path.join(group["dir"], "baseline.json")
    if os.path.exists(base):
        os.remove(base)

    state = {"shown": 0, "cb": None,
             "keys": [-1, -1, ord("s"), ord("q"), ord("q")]}

    def _set_cb(title, cb):
        state["cb"] = cb

    def _imshow(title, img):
        state["shown"] += 1
        if state["shown"] == 1:          # first paint -> replay a mouse drag
            cb = state["cb"]
            cb(cv2.EVENT_LBUTTONDOWN, 200, 100, 0, None)
            cb(cv2.EVENT_MOUSEMOVE, 300, 220, 1, None)
            cb(cv2.EVENT_LBUTTONUP, 400, 350, 0, None)

    def _wait(delay=0):
        return state["keys"].pop(0) if state["keys"] else ord("q")

    cv2.namedWindow = lambda *a, **k: None
    cv2.resizeWindow = lambda *a, **k: None
    cv2.destroyWindow = lambda *a, **k: None
    cv2.imshow = _imshow
    cv2.waitKeyEx = _wait
    cv2.setMouseCallback = _set_cb

    gui.mark_baselines(root, [group], canvas=CANVAS)
    assert os.path.exists(base), "baseline.json was not written"
    with open(base, encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["schema"] == "jetreflect/baseline-2"
    p1, p2 = rec["p1"], rec["p2"]
    # canvas 512x384 for a 768x576 image -> scale 2/3, mapping factor 1.5
    assert abs(p1[0] - 300.0) < 2 and abs(p1[1] - 150.0) < 2, p1
    assert abs(p2[0] - 600.0) < 2 and abs(p2[1] - 525.0) < 2, p2
    assert rec["image_size"] == [768, 576]
    assert abs(rec["tilt_deg"] - np.degrees(np.arctan2(300.0, 375.0))) < 0.1
    # the surface is an undirected line: a*x + b*y + c = 0, no offset anymore
    assert "offset_px" not in rec
    line = rec["line"]
    assert abs(np.hypot(line["a"], line["b"]) - 1.0) < 1e-6
    for q in (p1, p2):
        assert abs(line["a"] * q[0] + line["b"] * q[1] + line["c"]) < 1e-2


if __name__ == "__main__":
    test_mark_and_save()
    print("gui test passed")
