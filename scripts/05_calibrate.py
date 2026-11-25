#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optional - px/mm calibration from one image and a known length.

    python scripts/05_calibrate.py "D:\\data\\20261112_8group" ruler.tif --mm 10

Click the two ends of a known length; the value is stored in
<dataset>/jet_angle_config.json as "px_per_mm".  Angles never need it; only
physical radii / velocities / We / Re do.
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
    sys.exit(main(["calibrate"] + sys.argv[1:]))
