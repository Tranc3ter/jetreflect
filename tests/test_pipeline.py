# -*- coding: utf-8 -*-
"""End-to-end test on a small synthetic dataset (measure -> report)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jetreflect import pipeline  # noqa: E402
from jetreflect.io_utils import imwrite, write_json  # noqa: E402
from test_synthetic import _art, synth  # noqa: E402


def test_pipeline():
    root = os.path.join(_art(), "ds2")
    groups = []
    for name, ref in (("1", 35.6), ("2", 30.0)):
        gdir = os.path.join(root, name)
        os.makedirs(gdir, exist_ok=True)
        img, p1, p2, _, _ = synth(ref_deg=ref)
        images = []
        for i in range(3):
            p = os.path.join(gdir, "f%03d.png" % i)
            imwrite(p, img)
            images.append(p)
        groups.append({"name": name, "label": name, "dir": gdir,
                       "images": images})
        write_json(os.path.join(gdir, "baseline.json"), {
            "schema": "jetreflect/baseline-1", "dataset": "synthetic",
            "group": name, "image_size": [768, 576],
            "p1": [float(p1[0]), float(p1[1])],
            "p2": [float(p2[0]), float(p2[1])],
            "offset_px": 0.0, "source": "test",
        })

    cfg = pipeline.load_config(root)
    for g in groups:
        res = pipeline.measure_group(root, g, cfg, workers=1, verbose=False)
        assert res is not None
        assert res["n_valid"] == 3, res["n_valid"]
        exp_r = 90.0 - abs((35.6 if g["name"] == "1" else 30.0) - 21.8)
        got = res["stats"]["theta_r"]["mean"]
        assert abs(got - exp_r) < 0.5, (got, exp_r)

    pipeline.write_summary(root, groups, cfg)
    assert os.path.exists(os.path.join(root, "jet_angle_summary.csv"))
    assert os.path.exists(os.path.join(root, "fig_summary.png"))
    pipeline.make_samples(root, groups, cfg)
    assert os.path.exists(os.path.join(root, "sample", "1_overlay.png"))


if __name__ == "__main__":
    test_pipeline()
    print("pipeline test passed")
