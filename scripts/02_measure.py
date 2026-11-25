#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2/4 - measure every frame of every group.

    python scripts/02_measure.py "D:\\data\\20261112_8group"
    python scripts/02_measure.py "D:\\data\\20261112_8group" --group 2,3
    python scripts/02_measure.py "D:\\data\\20261112_8group" --stride 5 --force

First look at a few frames of every group before the full run:

    python scripts/02_measure.py "D:\\data\\20261119_11group" --sample 5 \
        --out jet_angle_sample --review

Writes, per group:  <group>/<out>/{result.json, per_frame.csv, angles.csv,
summary.csv, fig_timeseries.png, qc_montage.png}   (out defaults to jet_angle)
and, with --review,  <group>/<out>/review/f#####.png + check_sheet.png
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from jetreflect.cli import main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    sys.exit(main(["measure"] + sys.argv[1:]))
