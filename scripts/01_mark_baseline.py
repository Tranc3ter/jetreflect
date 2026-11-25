#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 1/4 - mark one reflection-surface baseline per group (total manual).

Example
-------
    python scripts/01_mark_baseline.py "D:\\data\\20261112_8group"

Controls
--------
    left drag / two clicks   draw a straight line on the reflection surface
    U or Ctrl+Z              undo
    R                        clear
    N/P or <-/->             previous / next frame
    J/K or up/down           previous / next group
    S or Enter               save baseline.json and go to the next group
    [ / ]                    shift the line along its normal by +-0.5 px
    Z/X or mouse wheel       zoom
    right/middle drag        pan
    Q or Esc                 quit

Each saved line goes to  <group>/baseline.json  (same folder as the images).
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
    sys.exit(main(["mark"] + sys.argv[1:]))
