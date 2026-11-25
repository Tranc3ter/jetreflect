#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 4/4 - summary tables, plots and one annotated sample per group.

    python scripts/04_report.py "D:\\data\\20261112_8group"

Writes in the dataset root:
    jet_angle_summary.csv / .json
    fig_summary.png
    sample/<group>_overlay.png
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
    sys.exit(main(["report"] + sys.argv[1:]))
