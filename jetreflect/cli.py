# -*- coding: utf-8 -*-
"""Command line interface."""
from __future__ import annotations

import argparse
import os
import sys

from . import pipeline


def _groups(root, cfg, spec=None):
    groups = pipeline.groups_of(root, cfg)
    if not spec:
        return groups
    names = [s.strip().strip("/") for s in str(spec).split(",") if s.strip()]
    exact = [g for g in groups
             if any(n in (g["name"], g["label"]) for n in names)]
    if exact:
        return exact
    return [g for g in groups
            if any(n in g["name"] for n in names)]


def _need_root(p):
    root = os.path.abspath(p)
    if not os.path.isdir(root):
        raise SystemExit("not a folder: %s" % root)
    return root


def _cfg(root, args, write=True):
    """Load the config, apply --engine and persist a migration once."""
    cfg = pipeline.load_config(root)
    if getattr(args, "engine", None):
        cfg = pipeline.with_engine(cfg, args.engine)
    if write:
        path = os.path.join(root, pipeline.CONFIG_NAME)
        disk = pipeline.read_json(path) or {}
        if (disk.get("engine") != cfg.get("engine")
                or disk.get("params_engine") != cfg.get("params_engine")):
            cfg["schema"] = "jetreflect/dataset-2"
            pipeline.save_config(root, cfg)
            print("config migrated -> %s  (engine=%s)" % (path, cfg["engine"]))
    return cfg


def cmd_info(args):
    root = _need_root(args.root)
    cfg = _cfg(root, args, write=False)
    groups = _groups(root, cfg, args.group)
    print("dataset : %s" % root)
    print("config  : %s" % os.path.join(root, pipeline.CONFIG_NAME))
    print("px/mm   : %s" % cfg.get("px_per_mm"))
    print("")
    print("%-24s %6s  %-26s %-16s %s" %
          ("group", "frames", "baseline", "result", "approved"))
    for g in groups:
        base = pipeline.read_json(os.path.join(g["dir"], "baseline.json"))
        res = pipeline.read_json(os.path.join(pipeline.out_dir(g),
                                              "result.json"))
        qc = pipeline.read_json(os.path.join(pipeline.out_dir(g), "qc.json"))
        b = ("yes  tilt %+6.2f" % base["tilt_deg"]) if base else "MISSING"
        if res:
            r = "%d/%d valid" % (res.get("n_valid", 0),
                                 res.get("frames_measured", 0))
        else:
            r = "-"
        print("%-24s %6d  %-26s %-16s %s" %
              (g["label"], len(g["images"]), b, r,
               ("" if qc is None else ("yes" if qc.get("approved")
                                       else "NO"))))
    return 0


def cmd_mark(args):
    from .gui import mark_baselines
    root = _need_root(args.root)
    cfg = _cfg(root, args)
    groups = _groups(root, cfg, args.group)
    if not groups:
        raise SystemExit("no groups found under %s" % root)
    print("marking %d group(s); one baseline per group" % len(groups))
    print("  left-drag = line   U = undo   N/P/<-/-> = frame   "
          "J/K/up/down = group")
    print("  S/Enter = save & next   [ ] = normal offset   Q = quit")
    mark_baselines(root, groups, frame_ratio=args.frame_ratio,
                   offset_default=args.offset, dataset=cfg.get("dataset"))
    return 0


def _frame_list(n_images, sample=None, frames=None):
    """Frame indices for a sample run (``--sample``) or an explicit list."""
    if frames:
        out = []
        for s in str(frames).replace(" ", "").split(","):
            if s == "":
                continue
            out.append(int(s))
        return sorted(set(i for i in out if 0 <= i < n_images))
    if sample:
        k = max(1, int(sample))
        if k == 1:
            return [int(round(0.5 * (n_images - 1)))]
        return sorted(set(int(round(i * (n_images - 1) / float(k - 1)))
                          for i in range(k)))
    return None


def cmd_measure(args):
    root = _need_root(args.root)
    cfg = _cfg(root, args)
    groups = _groups(root, cfg, args.group)
    missing = [g["label"] for g in groups
               if not pipeline.read_json(os.path.join(g["dir"],
                                                      "baseline.json"))]
    if missing:
        print("missing baseline.json for: %s" % ", ".join(missing))
        print("run:  python -m jetreflect mark \"%s\"" % root)
    todo = [g for g in groups if g["label"] not in missing]
    if not todo:
        return 1
    print("measuring %d group(s), engine=%s, stride=%d, workers=%s"
          % (len(todo), cfg.get("engine"), args.stride, args.workers))
    for g in todo:
        frames = _frame_list(len(g["images"]), args.sample, args.frames)
        if frames:
            print("  %-22s frames %s" % (g["label"], frames))
        pipeline.measure_group(root, g, cfg, stride=args.stride,
                               workers=args.workers, force=args.force,
                               out_name=args.out, frames=frames)
    if args.review:
        pipeline.make_review(root, todo, cfg, out_name=args.out)
    return 0


def cmd_review(args):
    root = _need_root(args.root)
    cfg = _cfg(root, args)
    groups = _groups(root, cfg, args.group)
    n = pipeline.review(root, groups, cfg)
    print("signed off %d group(s)" % n)
    return 0


def cmd_report(args):
    root = _need_root(args.root)
    cfg = _cfg(root, args)
    groups = _groups(root, cfg, args.group)
    # the dataset summary always covers every group that has a result, so a
    # filtered report never truncates the table
    pipeline.write_summary(root, None, cfg, out_name=args.out)
    pipeline.make_samples(root, groups, cfg, out_name=args.out)
    return 0


def cmd_all(args):
    root = _need_root(args.root)
    cfg = _cfg(root, args)
    groups = _groups(root, cfg, args.group)
    if not groups:
        raise SystemExit("no groups found under %s" % root)
    missing = [g["label"] for g in groups
               if not pipeline.read_json(os.path.join(g["dir"],
                                                      "baseline.json"))]
    if missing:
        print("missing baseline.json for: %s" % ", ".join(missing))
        print("run:  python -m jetreflect mark \"%s\"" % root)
        return 1
    for g in groups:
        pipeline.measure_group(root, g, cfg, stride=args.stride,
                               workers=args.workers, force=args.force)
    pipeline.write_summary(root, None, cfg)
    pipeline.make_samples(root, groups, cfg)
    if not args.no_review:
        pipeline.review(root, groups, cfg)
    return 0


def cmd_calibrate(args):
    from .gui import pick_points
    from .io_utils import imread, list_images
    from . import geom
    root = _need_root(args.root)
    cfg = _cfg(root, args)
    if args.px_per_mm is not None:
        cfg["px_per_mm"] = float(args.px_per_mm)
        pipeline.save_config(root, cfg)
        print("px_per_mm = %.4f" % cfg["px_per_mm"])
        return 0
    image = args.image
    if not image:
        groups = pipeline.groups_of(root, cfg)
        if not groups:
            raise SystemExit("no groups; give an image path explicitly")
        image = groups[0]["images"][0]
    image = os.path.abspath(image)
    img = imread(image)
    if args.p1 and args.p2:
        p1 = tuple(float(v) for v in args.p1.split(","))
        p2 = tuple(float(v) for v in args.p2.split(","))
    else:
        pts = pick_points(img, "jetreflect calibration", 2,
                          hint="click the two ends of a known length")
        if len(pts) < 2:
            print("cancelled")
            return 1
        p1, p2 = pts
    dist_px = geom.unit((p2[0] - p1[0], p2[1] - p1[1]))
    d = float(((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5)
    mm = args.mm
    if mm is None:
        try:
            mm = float(input("real distance between the two points [mm]: "))
        except Exception:                                  # noqa: BLE001
            print("no value given; cancelled")
            return 1
    if mm <= 0:
        print("distance must be positive")
        return 1
    cfg["px_per_mm"] = d / mm
    cfg["calibration"] = {"image": image, "p1": list(p1), "p2": list(p2),
                          "distance_px": d, "distance_mm": mm,
                          "created": pipeline.now_iso()}
    pipeline.save_config(root, cfg)
    print("%.3f px = %.3f mm  ->  %.4f px/mm" % (d, mm, d / mm))
    print("saved to %s" % os.path.join(root, pipeline.CONFIG_NAME))
    return 0


def cmd_import_legacy(args):
    root = _need_root(args.root)
    pipeline.import_legacy(root, os.path.abspath(args.legacy))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="jetreflect",
        description="liquid-jet reflection angle: manual baseline, per-frame "
                    "angles, QC, summary")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("info", help="show groups and their status")
    p.add_argument("root")
    p.add_argument("--group", default="")

    p = sub.add_parser("mark", help="interactively mark one baseline per group")
    p.add_argument("root")
    p.add_argument("--group", default="")
    p.add_argument("--frame-ratio", type=float, default=0.5,
                   help="which frame to open first (0-1, default 0.5)")
    p.add_argument("--offset", type=float, default=0.0,
                   help="normal offset of the reflection surface (px)")

    p = sub.add_parser("measure", help="measure frames of every group")
    p.add_argument("root")
    p.add_argument("--group", default="",
                   help="comma separated group names (default: all)")
    p.add_argument("--stride", type=int, default=1,
                   help="measure every N-th frame (default 1 = every frame)")
    p.add_argument("--sample", type=int, default=None,
                   help="measure N frames per group, evenly spread")
    p.add_argument("--frames", default=None,
                   help="explicit comma separated frame indices")
    p.add_argument("--out", default="jet_angle",
                   help="output folder inside each group (default jet_angle)")
    p.add_argument("--review", action="store_true",
                   help="also write per-frame overlays + a check sheet")
    p.add_argument("--engine", default=None, choices=["v2", "legacy"],
                   help="override the engine stored in the config")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel threads (default 1; 2-4 may help a little)")
    p.add_argument("--force", action="store_true",
                   help="re-measure even if result.json already exists")

    p = sub.add_parser("review", help="show QC montages and sign them off")
    p.add_argument("root")
    p.add_argument("--group", default="")

    p = sub.add_parser("report", help="summary tables, plots and samples")
    p.add_argument("root")
    p.add_argument("--group", default="",
                   help="limit the sample images (the summary always "
                        "covers every measured group)")
    p.add_argument("--out", default="jet_angle",
                   help="output folder inside each group (default jet_angle)")

    p = sub.add_parser("all", help="measure + report + review")
    p.add_argument("root")
    p.add_argument("--group", default="")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-review", action="store_true",
                   help="skip the interactive QC sign-off")

    p = sub.add_parser("calibrate", help="px/mm from one image and a known length")
    p.add_argument("root")
    p.add_argument("image", nargs="?",
                   help="image with the ruler (default: first frame)")
    p.add_argument("--mm", type=float, default=None,
                   help="real length between the two clicked points")
    p.add_argument("--p1", default=None, help="x,y of the first point")
    p.add_argument("--p2", default=None, help="x,y of the second point")
    p.add_argument("--px-per-mm", type=float, default=None,
                   help="set the value directly, no interaction")

    p = sub.add_parser("import-legacy",
                       help="convert old config/mirror_lines.json")
    p.add_argument("root")
    p.add_argument("legacy")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0
    fn = {
        "info": cmd_info, "mark": cmd_mark, "measure": cmd_measure,
        "review": cmd_review, "report": cmd_report, "all": cmd_all,
        "calibrate": cmd_calibrate, "import-legacy": cmd_import_legacy,
    }[args.cmd]
    try:
        return fn(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
