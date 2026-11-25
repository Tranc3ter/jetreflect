#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 3/4 - review the QC montage of every group and sign it off.

    python scripts/03_review.py "D:\\data\\20261112_8group"

Y = accept, N = reject, S/Esc = skip, Q = quit.
The decision is written to <group>/jet_angle/qc.json.
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
    sys.exit(main(["review"] + sys.argv[1:]))
