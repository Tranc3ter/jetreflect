#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-shot pipeline: measure -> report -> review (baselines must exist).

    python scripts/run_all.py "D:\\data\\20261112_8group"

Add --no-review to skip the interactive QC sign-off.
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
    sys.exit(main(["all"] + sys.argv[1:]))
