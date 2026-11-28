#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 0 步：把射流速度登记到每个组的 baseline.json（测量之前就该填）。

    python scripts/07_set_velocity.py "D:\\data\\20261112_8group" --all
    python scripts/07_set_velocity.py "<组目录>" --v 2.5
    python scripts/07_set_velocity.py "<根目录>" --group 2.5 --v 2.5

写进 <组>/baseline.json：
    "velocity_mps": 2.5,
    "velocity_source": "folder" | "manual",
    "velocity_updated": "2026-11-25T..."

--all 时直接拿文件夹名当速度（0.9 / 1 / 1.5 / 2 / ...）；名字不是数字的组会跳过。
之后 08_jet_physics.py 会优先用这里的值算 We / Re。
"""
import argparse
import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from jetreflect import pipeline                                     # noqa: E402


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def as_float(s):
    try:
        return float(str(s).replace(",", "."))
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="00_set_velocity",
                                 description="把射流速度写进 baseline.json")
    ap.add_argument("root", help="数据集根目录，或单个组目录")
    ap.add_argument("--all", action="store_true",
                    help="所有子组都用文件夹名当速度")
    ap.add_argument("--group", default="", help="只处理名字包含这些词的组（逗号分隔）")
    ap.add_argument("--v", type=float, default=None,
                    help="指定速度 (m/s)，写入选中的组")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit("不是文件夹：%s" % root)

    # 单组目录（自身就有 baseline.json / 图片）
    if os.path.exists(os.path.join(root, "baseline.json")) and not args.all:
        groups = [dict(name=os.path.basename(root), dir=root,
                       label=os.path.basename(root))]
    else:
        cfg = pipeline.load_config(root)
        groups = pipeline.groups_of(root, cfg)
        if args.group:
            keys = [s.strip() for s in args.group.split(",") if s.strip()]
            groups = [g for g in groups if any(k in g["name"] for k in keys)]

    if not groups:
        raise SystemExit("没有找到组")

    n_ok = 0
    for g in groups:
        path = os.path.join(g["dir"], "baseline.json")
        base = read_json(path)
        if base is None:
            print("  [skip] %-22s 没有 baseline.json（先跑 01_mark_baseline.py）"
                  % g["label"])
            continue
        v = args.v
        src = "manual"
        if v is None:
            v = as_float(g["name"])
            src = "folder"
        if v is None:
            print("  [skip] %-22s 组名不是数字，请用 --v 指定" % g["label"])
            continue
        base["velocity_mps"] = round(float(v), 4)
        base["velocity_source"] = src
        base["velocity_updated"] = _dt.datetime.now().isoformat(timespec="seconds")
        if args.dry_run:
            print("  [dry ] %-22s V = %.3f m/s (%s)" % (g["label"], v, src))
            continue
        write_json(path, base)
        n_ok += 1
        print("  [ ok ] %-22s V = %.3f m/s  -> baseline.json" % (g["label"], v))

    print("\n登记完成：%d 组%s" % (n_ok, "（dry-run，未写入）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
