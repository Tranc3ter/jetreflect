# -*- coding: utf-8 -*-
"""Batch measurement, quality control, plots and summary tables.

Two engines are available and selected per dataset in ``jet_angle_config.json``:

``engine: "v2"`` (default)
    :mod:`jetreflect.v2` -- direction-free axis tracking, both-side outgoing
    search, ``theta_r = None`` for frames without an outgoing stream.
``engine: "legacy"``
    :mod:`jetreflect.detect` -- the first engine, kept for old datasets.

The reflection surface is an **undirected infinite line**; ``offset_px`` is
gone (it was a leftover of the old workflow).
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import detect, geom, v2  # noqa: E402
from .io_utils import (discover_groups, imread, imwrite, list_images,  # noqa: E402
                       read_json, write_json)

CONFIG_NAME = "jet_angle_config.json"
RESULT_SCHEMA = "jetreflect/result-2"
SUMMARY_SCHEMA = "jetreflect/summary-2"
OUT_NAME = "jet_angle"
REVIEW_NAME = "review"
ENGINES = ("v2", "legacy")


# ---------------------------------------------------------------- helpers
def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _operator():
    try:
        import getpass
        return getpass.getuser()
    except Exception:                                     # noqa: BLE001
        return ""


def _font_ok():
    from matplotlib import font_manager
    for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC"):
        try:
            if any(f.name == fam for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                return True
        except Exception:                                  # noqa: BLE001
            pass
    plt.rcParams["axes.unicode_minus"] = False
    return False


_FONT = None


def cjk():
    global _FONT
    if _FONT is None:
        _FONT = _font_ok()
    return _FONT


def T(zh, en):
    return zh if cjk() else en


def _version():
    from . import __version__
    return __version__


def default_config(root, engine="v2"):
    return {
        "schema": "jetreflect/dataset-2",
        "dataset": os.path.basename(os.path.abspath(root)),
        "engine": engine,
        "params_engine": engine,
        "image_formats": [".tif", ".tiff", ".bmp", ".png"],
        "min_frames": 20,
        "skip_dirs": ["sample", "jet_angle", "results", "_summary"],
        "px_per_mm": None,
        "params": dict(detect.DEFAULT_PARAMS if engine == "legacy"
                       else v2.DEFAULT_PARAMS),
        "created": now_iso(),
        "note": "surface = undirected line a*x+b*y+c=0; angles do not depend "
                "on px_per_mm.",
    }


def load_config(root, create=True):
    path = os.path.join(root, CONFIG_NAME)
    cfg = read_json(path)
    if cfg is None and create:
        cfg = default_config(root)
        write_json(path, cfg)
    cfg = cfg or default_config(root)
    engine = str(cfg.get("engine") or "v2")
    if engine not in ENGINES:
        engine = "v2"
    base = dict(detect.DEFAULT_PARAMS if engine == "legacy"
                else v2.DEFAULT_PARAMS)
    stored = dict(cfg.get("params") or {})
    if cfg.get("params_engine") != engine:
        stored = {}                       # the stored values belong to the other engine
    base.update({k: v for k, v in stored.items() if k in base})
    cfg["engine"] = engine
    cfg["params_engine"] = engine
    cfg["params"] = base
    return cfg


def save_config(root, cfg):
    write_json(os.path.join(root, CONFIG_NAME), cfg)


def with_engine(cfg, engine):
    """Return a copy of ``cfg`` bound to another engine (CLI override)."""
    if not engine or engine == cfg.get("engine"):
        return cfg
    if engine not in ENGINES:
        raise SystemExit("unknown engine %r (choose from %s)"
                         % (engine, ", ".join(ENGINES)))
    out = dict(cfg)
    base = dict(detect.DEFAULT_PARAMS if engine == "legacy"
                else v2.DEFAULT_PARAMS)
    if cfg.get("params_engine") == engine:
        base.update({k: v for k, v in (cfg.get("params") or {}).items()
                     if k in base})
    out["engine"] = engine
    out["params_engine"] = engine
    out["params"] = base
    return out


def group_name(group):
    return group["name"] if isinstance(group, dict) else group


def groups_of(root, cfg=None):
    cfg = cfg or load_config(root)
    skip = ["sample", "jet_angle", "results", "_summary"] + \
        list(cfg.get("skip_dirs") or [])
    groups = discover_groups(root, min_frames=int(cfg.get("min_frames", 20)),
                             skip=skip)
    if not groups and os.path.exists(os.path.join(root, "baseline.json")):
        # 传进来的是单个组目录：把它自己当成一组（方便单帧预览）
        imgs = list_images(root)
        if imgs:
            name = os.path.basename(os.path.abspath(root))
            groups = [{"name": name, "label": name.strip(), "dir": root,
                       "images": imgs}]
    return groups


def out_dir(group, name=OUT_NAME):
    return os.path.join(group["dir"], name or OUT_NAME)


# ---------------------------------------------------------------- context
def build_scene(group, cfg):
    """v2 scene: undirected surface + fluid side + parameters."""
    rec = read_json(os.path.join(group["dir"], "baseline.json"))
    if not rec:
        return None, "baseline.json missing"
    images = group["images"]
    if not images:
        return None, "no images"
    img = imread(images[len(images) // 2])
    h, w = img.shape[:2]
    if rec.get("image_size") and list(rec["image_size"]) != [w, h]:
        return None, ("image size changed: baseline %s vs frame %dx%d"
                      % (rec["image_size"], w, h))
    scene, why = v2.build_scene(rec, img, cfg["params"],
                                group=group["name"],
                                dataset=cfg.get("dataset"))
    return scene, why


def build_ctx(group, cfg, frames_for_tau=3):
    """Legacy context (kept for ``engine: legacy``)."""
    base = read_json(os.path.join(group["dir"], "baseline.json"))
    if not base or "p1" not in base or "p2" not in base:
        return None, "baseline.json missing"
    if not geom.kc_ok(base["p1"], base["p2"]):
        return None, "baseline is horizontal"
    images = group["images"]
    if not images:
        return None, "no images"
    img = imread(images[-1])
    h, w = img.shape[:2]
    if base.get("image_size") and list(base["image_size"]) != [w, h]:
        return None, ("image size changed: baseline %s vs frame %dx%d"
                      % (base["image_size"], w, h))
    ctx = detect.Ctx(base["p1"], base["p2"], cfg["params"],
                     n=base.get("normal"))
    ctx.offset = 0.0
    ctx.setup_fluid(img)
    n_tau = max(1, min(int(frames_for_tau), len(images)))
    taus = []
    for k in np.linspace(0, len(images) - 1, n_tau).astype(int):
        t = detect.jet_tilt(imread(images[k]), ctx.p0, ctx.n)
        if t is not None:
            taus.append(t)
    ctx.prepare_for((h, w), float(np.median(taus)) if taus else 0.0)
    ctx.image_size = [w, h]
    return ctx, ""


# ---------------------------------------------------------------- workers
def _run_one(ctx, path, idx):
    row = ctx.measure(path)
    row["frame"] = int(idx)
    ok, why = detect.sanity(ctx, row)
    row["valid"] = bool(ok)
    if row.get("ok") and not ok:
        row["note"] = why
    row["status"] = "ok" if row["valid"] else "invalid"
    return row


def _run_v2(scene, path, idx):
    """One frame; never raises, so a single bad frame cannot kill a 2000-frame
    group in the middle of an unattended run."""
    try:
        return v2.measure(scene, path, frame=int(idx))
    except Exception as e:                                # noqa: BLE001
        row = v2.new_row(path, int(idx))
        row["why"] = "exception: %s" % e
        return row


def measure_group(root, group, cfg, stride=1, workers=None, force=False,
                  verbose=True, out_name=OUT_NAME, frames=None):
    engine = cfg.get("engine", "v2")
    od = out_dir(group, out_name)
    res_path = os.path.join(od, "result.json")
    images = group["images"]
    idx = [int(i) for i in frames] if frames else \
        list(range(0, len(images), max(1, int(stride))))
    if os.path.exists(res_path) and not force and not frames:
        old = read_json(res_path) or {}
        if old.get("schema") == RESULT_SCHEMA and \
                int(old.get("frames_measured", -1)) == len(idx):
            if verbose:
                print("  [skip] %-22s already measured (%s)"
                      % (group["label"], os.path.basename(res_path)))
            return old

    if engine == "legacy":
        ctx, why = build_ctx(group, cfg)
    else:
        ctx, why = build_scene(group, cfg)
    if ctx is None:
        print("  [fail] %-22s %s" % (group["label"], why))
        return None

    jobs = [(images[i], i) for i in idx if 0 <= i < len(images)]
    rows = []
    if workers is None:
        workers = 1
    runner = _run_one if engine == "legacy" else _run_v2
    if workers <= 1 or len(jobs) <= 2:
        for n, (path, i) in enumerate(jobs):
            rows.append(runner(ctx, path, i))
            if verbose and (n + 1) % 40 == 0:
                print("    %s %d/%d" % (group["label"], n + 1, len(jobs)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(runner, ctx, path, i) for path, i in jobs]
            for n, f in enumerate(futs):
                rows.append(f.result())
                if verbose and (n + 1) % 40 == 0:
                    print("    %s %d/%d" % (group["label"], n + 1, len(jobs)))
    return _write_group_outputs(root, group, cfg, ctx, rows, od,
                                frames_expected=len(idx))


def _robust(v):
    v = np.asarray([x for x in v if x is not None], float)
    if v.size == 0:
        return dict(n=0)
    med = float(np.median(v))
    mad = float(1.4826 * np.median(np.abs(v - med)))
    return dict(n=int(v.size), mean=float(np.mean(v)),
                std=float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
                median=med, mad=mad,
                min=float(v.min()), max=float(v.max()))


def _write_group_outputs(root, group, cfg, ctx, rows, od,
                         frames_expected=None):
    os.makedirs(od, exist_ok=True)
    rows.sort(key=lambda r: r["frame"])
    good = [r for r in rows if r.get("valid")]
    none = [r for r in rows if r.get("status") == "none"]
    bad = [r for r in rows if r.get("status") not in ("ok", "none")]
    keys = ["theta_i", "theta_r", "angle_between", "rho"]
    stats = {k: _robust([r.get(k) for r in good]) for k in keys}
    n_total = len(group["images"])
    baseline = read_json(os.path.join(group["dir"], "baseline.json"))
    result = {
        "schema": RESULT_SCHEMA,
        "dataset": cfg.get("dataset") or os.path.basename(os.path.abspath(root)),
        "group": group["name"],
        "created": now_iso(),
        "algorithm": "jetreflect " + _version(),
        "engine": cfg.get("engine", "v2"),
        "baseline": baseline,
        "params": cfg["params"],
        "px_per_mm": cfg.get("px_per_mm"),
        "image_size": list(ctx.image_size) if getattr(ctx, "image_size", None)
        else None,
        "frames_expected": int(frames_expected or n_total),
        "frames_measured": len(rows),
        "n_valid": len(good),
        "n_none": len(none),
        "n_invalid": len(bad),
        "valid_ratio": (len(good) / float(len(rows))) if rows else 0.0,
        "none_ratio": (len(none) / float(len(rows))) if rows else 0.0,
        "stats": stats,
        "frames": rows,
    }
    write_json(os.path.join(od, "result.json"), result)

    # ---- per-frame CSV -------------------------------------------------
    cols = ["frame", "file", "status", "valid", "theta_i", "theta_r",
            "source", "angle_between", "rho",
            "x_imp", "y_imp", "rms_i", "rms_r", "n_i", "n_r", "why", "note"]
    with open(os.path.join(od, "per_frame.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get("ref_source") if c == "source"
                                else r.get(c)) for c in cols})

    # ---- human facing table -------------------------------------------
    with open(os.path.join(od, "angles.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["帧", "文件名", "入射角_deg", "反射角_deg",
                    "两轴夹角_deg", "rho", "来源", "状态", "备注"])
        for r in rows:
            w.writerow([r.get("frame"), r.get("file"),
                        _fmt(r.get("theta_i")), _fmt(r.get("theta_r")),
                        _fmt(r.get("angle_between")), _fmt(r.get("rho")),
                        r.get("ref_source") or "",
                        _status_text(r),
                        r.get("why") or r.get("note") or ""])

    # ---- one line summary ---------------------------------------------
    s = stats
    with open(os.path.join(od, "summary.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_frames", "n_measured", "n_valid", "n_none",
                    "n_invalid", "theta_i_mean", "theta_i_std",
                    "theta_i_median", "theta_r_mean", "theta_r_std",
                    "theta_r_median", "angle_between_mean",
                    "angle_between_std", "rho_mean", "rho_std",
                    "baseline_tilt_deg", "px_per_mm"])
        w.writerow([group["name"], n_total, len(rows), len(good), len(none),
                    len(bad),
                    _fmt(s["theta_i"].get("mean")), _fmt(s["theta_i"].get("std")),
                    _fmt(s["theta_i"].get("median")),
                    _fmt(s["theta_r"].get("mean")), _fmt(s["theta_r"].get("std")),
                    _fmt(s["theta_r"].get("median")),
                    _fmt(s["angle_between"].get("mean")),
                    _fmt(s["angle_between"].get("std")),
                    _fmt(s["rho"].get("mean")), _fmt(s["rho"].get("std")),
                    _fmt((baseline or {}).get("tilt_deg")),
                    cfg.get("px_per_mm")])

    _plot_group(od, group, rows, stats, ctx)
    _make_montage(od, group, rows, baseline)
    print("  [ ok ] %-22s valid %3d  none %3d  invalid %3d   "
          "theta_i=%.2f+-%.2f  theta_r=%.2f+-%.2f  rho=%.2f"
          % (group["label"], len(good), len(none), len(bad),
             s["theta_i"].get("mean", float("nan")),
             s["theta_i"].get("std", float("nan")),
             s["theta_r"].get("mean", float("nan")),
             s["theta_r"].get("std", float("nan")),
             s["rho"].get("mean", float("nan"))))
    return result


def _fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return round(v, 4)
    return v


def _status_text(r):
    st = r.get("status")
    if st == "ok":
        return "ok"
    if st == "none":
        return "none(无出流)"
    return "invalid"


def _imread_color(path):
    import cv2
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("cannot read image: %s" % path)
    return img


# ---------------------------------------------------------------- drawing
def draw_measurement(gray, baseline, row, baseline_color=(255, 255, 0)):
    """Surface, incident axis, outgoing axis, impact point, numbers."""
    import cv2
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape[:2]
    base = geom.Line.from_dict(baseline)
    if base is not None:
        seg = base.clip(w, h)
        if seg is not None:
            cv2.line(vis, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])),
                     baseline_color, 1, cv2.LINE_AA)
    had_line = False
    inc = (row or {}).get("inc_line")
    if inc:
        seg = geom.Line.from_dict(inc).clip(w, h)
        if seg is not None:
            cv2.line(vis, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])),
                     (0, 0, 255), 1, cv2.LINE_AA)
            had_line = True
    ref_seg = (row or {}).get("ref_seg")
    if ref_seg:
        # only the fitted span is drawn: extrapolating it back to the impact
        # would fake a three-line junction that does not exist in the data
        a = tuple(np.int32(ref_seg[0]))
        b = tuple(np.int32(ref_seg[1]))
        cv2.line(vis, a, b, (0, 220, 0), 1, cv2.LINE_AA)
        for q in (a, b):
            cv2.circle(vis, q, 2, (0, 220, 0), -1, cv2.LINE_AA)
        had_line = True
    elif (row or {}).get("ref_line"):
        seg = geom.Line.from_dict(row["ref_line"]).clip(w, h)
        if seg is not None:
            cv2.line(vis, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])),
                     (0, 220, 0), 1, cv2.LINE_AA)
            had_line = True
    if not had_line and (row or {}).get("ok"):        # legacy rows
        for key, col in (("inc_k", (0, 0, 255)), ("ref_k", (0, 220, 0))):
            if key not in row:
                continue
            k = row[key]
            c = row[key.replace("k", "c")]
            pts = [(int(round(k * y + c)), y) for y in (0, h - 1)
                   if 0 <= k * y + c <= w - 1]
            if len(pts) == 2:
                cv2.line(vis, pts[0], pts[1], col, 1, cv2.LINE_AA)
    if (row or {}).get("x_imp") is not None:
        cv2.circle(vis, (int(round(row["x_imp"])), int(round(row["y_imp"]))),
                   4, (0, 200, 255), 1, cv2.LINE_AA)
    return vis


def _row_caption(row):
    st = (row or {}).get("status") or ("ok" if (row or {}).get("valid")
                                       else "invalid")
    ti = (row or {}).get("theta_i")
    tr = (row or {}).get("theta_r")
    rho = (row or {}).get("rho")
    head = ("th_i %6.2f   th_r %6.2f   rho %5.2f"
            % (ti or 0.0, tr or 0.0, rho or 0.0)) if tr is not None else \
        ("th_i %6.2f   th_r  none   (%s)" % (ti or 0.0, st))
    return head


# ---------------------------------------------------------------- plots
def _plot_group(od, group, rows, stats, ctx):
    fi = [r["frame"] for r in rows if r.get("theta_i") is not None]
    ti = [r["theta_i"] for r in rows if r.get("theta_i") is not None]
    fr = [r["frame"] for r in rows if r.get("theta_r") is not None]
    tr = [r["theta_r"] for r in rows if r.get("theta_r") is not None]

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    ax[0].plot(fi, ti, ".", ms=3, color="#1f77b4",
               label=T("入射角", "incident"))
    ax[0].plot(fr, tr, ".", ms=3, color="#d62728",
               label=T("反射角", "reflected"))
    for r in rows:
        if r.get("status") == "none":
            ax[0].axvline(r["frame"], color="#cccccc", lw=0.4, alpha=0.6)
    ax[0].set_xlabel(T("帧号", "frame"))
    ax[0].set_ylabel(T("角度 (deg)", "angle (deg)"))
    ax[0].set_title(T("%s：逐帧角度", "%s: angle per frame") % group["label"])
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].hist(ti, bins=30, alpha=0.6,
               color="#1f77b4", label=T("入射角", "incident"))
    ax[1].hist(tr, bins=30, alpha=0.6,
               color="#d62728", label=T("反射角", "reflected"))
    ax[1].set_xlabel(T("角度 (deg)", "angle (deg)"))
    ax[1].set_ylabel(T("帧数", "count"))
    ax[1].set_title(T("角度分布", "angle distribution"))
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig_timeseries.png"))
    plt.close(fig)


def _tile(group, baseline, row, width=480, height=360, scale_text=0.5):
    import cv2
    img = imread(os.path.join(group["dir"], row["file"]))
    vis = draw_measurement(img, baseline, row)
    vis = cv2.resize(vis, (width, height))
    hh = 20
    bar = np.full((hh, width, 3), 25, np.uint8)
    cv2.putText(bar, "f%s  %s" % (row.get("frame"), _row_caption(row)),
                (6, 14), cv2.FONT_HERSHEY_SIMPLEX, scale_text,
                (200, 235, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def _make_montage(od, group, rows, baseline, name="qc_montage.png"):
    import cv2
    good = [r for r in rows if r.get("valid")]
    if not good:
        good = [r for r in rows if r.get("theta_i") is not None] or rows[:1]
    picks = []
    for q in (0.0, 0.33, 0.66, 1.0):
        picks.append(good[int(round(q * (len(good) - 1)))])
    seen, uniq = set(), []
    for r in picks:
        if r["frame"] not in seen:
            seen.add(r["frame"])
            uniq.append(r)
    picks = uniq[:4]
    tiles = [_tile(group, baseline, r) for r in picks]
    while len(tiles) < 4:
        tiles.append(np.full((380, 480, 3), 30, np.uint8))
    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4])
    grid = np.vstack([top, bot])
    out = np.vstack([_header(group, rows, grid.shape[1]), grid])
    imwrite(os.path.join(od, name), out)
    return out


def _header(group, rows, width):
    import cv2
    good = [r for r in rows if r.get("valid")]
    none = [r for r in rows if r.get("status") == "none"]
    bad = [r for r in rows if r.get("status") not in ("ok", "none")]
    st = _robust([r.get("theta_i") for r in good])
    sr = _robust([r.get("theta_r") for r in good])
    sr2 = _robust([r.get("rho") for r in good])
    lines = [
        "%s   measured %d   valid %d   none %d   invalid %d"
        % (group["label"], len(rows), len(good), len(none), len(bad)),
        "theta_i %.2f +- %.2f   theta_r %.2f +- %.2f   rho %.2f"
        % (st.get("mean", float("nan")), st.get("std", float("nan")),
           sr.get("mean", float("nan")), sr.get("std", float("nan")),
           sr2.get("mean", float("nan"))),
    ]
    hdr = np.full((24 + 22 * len(lines), width, 3), 25, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(hdr, t, (8, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (210, 240, 210), 1, cv2.LINE_AA)
    return hdr


# ---------------------------------------------------------------- summary
def _stat_row(root, group, cfg, out_name=OUT_NAME):
    od = out_dir(group, out_name)
    res = read_json(os.path.join(od, "result.json"))
    if not res:
        return None
    st = res.get("stats", {})
    qc = read_json(os.path.join(od, "qc.json")) or {}
    row = {
        "group": group["name"],
        "n_frames": res.get("frames_expected"),
        "n_measured": res.get("frames_measured"),
        "n_valid": res.get("n_valid"),
        "n_none": res.get("n_none", 0),
        "n_invalid": res.get("n_invalid", 0),
        "valid_ratio": round(res.get("valid_ratio", 0.0), 3),
    }
    for k in ("theta_i", "theta_r", "angle_between", "rho"):
        d = st.get(k, {})
        row[k + "_mean"] = _fmt(d.get("mean"))
        row[k + "_std"] = _fmt(d.get("std"))
        row[k + "_median"] = _fmt(d.get("median"))
    row["baseline_tilt_deg"] = _fmt((res.get("baseline") or {}).get("tilt_deg"))
    row["px_per_mm"] = cfg.get("px_per_mm")
    row["approved"] = qc.get("approved")
    row["operator"] = (qc.get("operator") or "")
    ratio = float(res.get("valid_ratio") or 0.0)
    row["status"] = ("ok" if ratio >= 0.5
                     else "low-validity(%d none)" % int(res.get("n_none") or 0))
    return row


def write_summary(root, groups=None, cfg=None, out_name=OUT_NAME):
    """Write the per-group data tables (csv + json).  No plotting."""
    cfg = cfg or load_config(root)
    groups = groups or groups_of(root, cfg)
    rows = []
    for g in groups:
        r = _stat_row(root, g, cfg, out_name)
        if r:
            rows.append(r)
    if not rows:
        print("no results yet - run measure first")
        return None
    cols = list(rows[0].keys())
    path = os.path.join(root, "jet_angle_summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    js = {"schema": SUMMARY_SCHEMA, "created": now_iso(),
          "dataset": cfg.get("dataset"), "groups": rows}
    write_json(os.path.join(root, "jet_angle_summary.json"), js)
    print("summary -> %s" % path)
    return rows


def read_summary_rows(root):
    """Rows written by :func:`write_summary` (from ``jet_angle_summary.json``)."""
    js = read_json(os.path.join(root, "jet_angle_summary.json"))
    if not js:
        return None
    return list(js.get("groups") or [])


def write_summary_figure(root, rows=None):
    """Draw ``fig_summary.png`` from the summary table (plotting only)."""
    rows = rows if rows is not None else read_summary_rows(root)
    if not rows:
        print("no summary yet - run 03_review.py first")
        return False
    _plot_summary(root, rows)
    print("figure  -> %s" % os.path.join(root, "fig_summary.png"))
    return True


def _plot_summary(root, rows):
    labels = [r["group"] for r in rows]
    x = np.arange(len(rows))
    ti = np.array([r["theta_i_mean"] or np.nan for r in rows], float)
    tr = np.array([r["theta_r_mean"] or np.nan for r in rows], float)
    ei = np.array([r["theta_i_std"] or 0.0 for r in rows], float)
    er = np.array([r["theta_r_std"] or 0.0 for r in rows], float)
    rho = np.array([r["rho_mean"] or np.nan for r in rows], float)
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=150)
    ax[0].errorbar(x - 0.08, ti, yerr=ei, fmt="o", ms=6, capsize=3,
                   color="#1f77b4", label=T("入射角", "incident"))
    ax[0].errorbar(x + 0.08, tr, yerr=er, fmt="s", ms=6, capsize=3,
                   color="#d62728", label=T("反射角", "reflected"))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax[0].set_ylabel(T("角度 (deg)", "angle (deg)"))
    ax[0].set_title(T("各组平均角度", "mean angle per group"))
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].bar(x, rho, color="#2ca02c", alpha=0.75)
    ax[1].axhline(1.0, ls=":", color="gray", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax[1].set_ylabel(r"$\rho=\tan\theta_r/\tan\theta_i$")
    ax[1].set_title(T("角度比", "angle ratio"))
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "fig_summary.png"))
    plt.close(fig)


def make_samples(root, groups=None, cfg=None, out_name=OUT_NAME):
    """One annotated representative frame per group -> <root>/sample/."""
    cfg = cfg or load_config(root)
    groups = groups or groups_of(root, cfg)
    sdir = os.path.join(root, "sample")
    os.makedirs(sdir, exist_ok=True)
    n = 0
    for g in groups:
        res = read_json(os.path.join(out_dir(g, out_name), "result.json"))
        if not res or not res.get("frames"):
            continue
        rows = [r for r in res["frames"] if r.get("valid")]
        if not rows:
            continue
        med = float(np.median([r["theta_r"] for r in rows]))
        pick = min(rows, key=lambda r: abs(r["theta_r"] - med))
        img = imread(os.path.join(g["dir"], pick["file"]))
        vis = draw_measurement(img, res.get("baseline"), pick)
        imwrite(os.path.join(sdir, "%s_overlay.png" % g["label"]), vis)
        n += 1
    print("samples -> %s  (%d images)" % (sdir, n))
    return n


# ---------------------------------------------------------------- review
def make_review(root, groups, cfg, out_name=OUT_NAME, tag=REVIEW_NAME):
    """Per-frame overlays + one sheet per group, for eyeballing the result."""
    import cv2
    sheets = []
    for g in groups:
        od = out_dir(g, out_name)
        res = read_json(os.path.join(od, "result.json"))
        if not res or not res.get("frames"):
            print("  [skip] %-22s no result (%s)" % (g["label"], od))
            continue
        rows = res["frames"]
        rdir = os.path.join(od, tag)
        os.makedirs(rdir, exist_ok=True)
        baseline = res.get("baseline")
        tiles = []
        for r in rows:
            img = imread(os.path.join(g["dir"], r["file"]))
            vis = draw_measurement(img, baseline, r)
            vis = cv2.resize(vis, None, fx=1.5, fy=1.5,
                             interpolation=cv2.INTER_LINEAR)
            _stamp(vis, g["label"], r)
            imwrite(os.path.join(rdir, "f%05d.png" % int(r["frame"])), vis)
            tiles.append(cv2.resize(vis, (440, 330)))
        while len(tiles) < 6:
            tiles.append(np.full((330, 440, 3), 30, np.uint8))
        sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
        sheet = np.vstack([_header(g, rows, sheet.shape[1]), sheet])
        p = os.path.join(od, "check_sheet.png")
        imwrite(p, sheet)
        sheets.append(p)
        print("  [ ok ] %-22s %s" % (g["label"], p))
    return sheets


def _stamp(vis, group, row):
    import cv2
    lines = [group, _row_caption(row),
             "rms %.2f/%.2f  n %d/%d  %s"
             % (row.get("rms_i") or 0.0, row.get("rms_r") or 0.0,
                row.get("n_i") or 0, row.get("n_r") or 0,
                row.get("why") or row.get("note") or row.get("status") or "")]
    h, w = vis.shape[:2]
    bar = vis.copy()
    cv2.rectangle(bar, (0, 0), (w, 18 + 22 * len(lines)), (0, 0, 0), -1)
    out = cv2.addWeighted(bar, 0.55, vis, 0.45, 0)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (90, 255, 90) if i == 0 else (225, 235, 225), 1,
                    cv2.LINE_AA)
    return out


def review(root, groups=None, cfg=None, canvas=(1000, 760)):
    """Show each group's QC montage and record an explicit Y/N sign-off."""
    import cv2
    cfg = cfg or load_config(root)
    groups = groups or groups_of(root, cfg)
    title = "jetreflect QC review"
    done = 0
    for g in groups:
        p = os.path.join(out_dir(g), "qc_montage.png")
        if not os.path.exists(p):
            print("  [skip] %-22s no montage (measure first)" % g["label"])
            continue
        img = _imread_color(p)
        h, w = img.shape[:2]
        scale = min(canvas[0] / float(w), (canvas[1] - 40) / float(h))
        disp = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, disp.shape[1], disp.shape[0] + 40)
        print("  review %-22s  Y=accept  N=reject  S=skip  Q=quit"
              % g["label"])
        ans = None
        while ans is None:
            cv2.imshow(title, disp)
            k = cv2.waitKeyEx(30) & 0xFF
            if k in (ord("y"), ord("Y")):
                ans = True
            elif k in (ord("n"), ord("N")):
                ans = False
            elif k in (ord("s"), ord("S"), 27, ord("q"), ord("Q")):
                ans = "skip"
        if ans == "skip":
            continue
        qc = {"schema": "jetreflect/qc-2", "approved": bool(ans),
              "operator": _operator(), "at": now_iso(),
              "montage": os.path.basename(p)}
        write_json(os.path.join(out_dir(g), "qc.json"), qc)
        done += 1
        print("    -> %s" % ("accepted" if ans else "rejected"))
    cv2.destroyAllWindows()
    return done


# ---------------------------------------------------------------- legacy
def import_legacy(root, legacy_json):
    """Convert the old config/mirror_lines.json into per-group baseline.json."""
    data = read_json(legacy_json) or {}
    cfg = load_config(root)
    n = 0
    for key, rec in data.items():
        parts = key.replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        ds, gname = parts[-2], parts[-1]
        if ds != os.path.basename(os.path.abspath(root)):
            continue
        gdir = os.path.join(root, gname)
        if not os.path.isdir(gdir):
            continue
        p1 = rec.get("ref_p1") or rec.get("p1")
        p2 = rec.get("ref_p2") or rec.get("p2")
        if not p1 or not p2:
            continue
        img = imread(list_images(gdir)[0])
        line = geom.Line.from_points(p1, p2)
        out = {
            "schema": "jetreflect/baseline-2",
            "dataset": ds, "group": gname,
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "p1": [round(float(p1[0]), 3), round(float(p1[1]), 3)],
            "p2": [round(float(p2[0]), 3), round(float(p2[1]), 3)],
            "line": line.to_dict(),
            "tilt_deg": round(line.tilt_deg(), 3),
            "source": "legacy-import",
            "created": now_iso(), "operator": "",
        }
        write_json(os.path.join(gdir, "baseline.json"), out)
        n += 1
    save_config(root, cfg)
    print("imported %d baselines from %s" % (n, legacy_json))
    return n
