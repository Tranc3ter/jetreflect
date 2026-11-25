# -*- coding: utf-8 -*-
"""把一组测量结果渲染成带标注的 MP4（基准面 + 入射轴 + 出流轴 + 数值）。

    python 10_overlay_video.py <组目录> [--out 输出.mp4] [--fps 30] [--scale 1.5]

组目录里需要有 jet_angle/result.json（02_measure.py 的产物）和原始 tif。
"""
import argparse
import json
import math
import os
import sys

import cv2

try:
    cv2.setLogLevel(0)
except Exception:
    pass
import numpy as np
from PIL import Image, ImageDraw, ImageFont

C_BASE = (0, 190, 255)        # 基准面（青色）
C_INC = (255, 210, 0)         # 入射轴（琥珀）
C_REF = (60, 235, 110)        # 出流轴（绿）
C_NRM = (215, 215, 215)       # 法线（灰）
C_IMP = (255, 120, 20)        # 冲击点（橙）
C_OK = (60, 235, 110)
C_NONE = (255, 190, 60)
C_BAD = (255, 90, 90)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]


def font(size, bold=True):
    for p in ([FONT_CANDIDATES[0], FONT_CANDIDATES[1]] if bold
              else [FONT_CANDIDATES[1], FONT_CANDIDATES[2]]):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def line_pts(a, b, c, w, h):
    """直线 a*x+b*y+c=0 与图像边框的交点（返回两点或 None）。"""
    if abs(b) < 1e-9:                       # 竖直
        x = -c / a
        if 0 <= x <= w:
            return (x, 0.0), (x, float(h))
        return None
    pts = []
    for x in (0.0, float(w)):
        y = -(a * x + c) / b
        if 0 <= y <= h:
            pts.append((x, y))
    for y in (0.0, float(h)):
        x = -(b * y + c) / a
        if 0 <= x <= w:
            pts.append((x, y))
    if len(pts) < 2:
        return None
    pts.sort()
    return pts[0], pts[-1]


def unit(a, b):
    n = math.hypot(a, b) or 1.0
    return a / n, b / n


def dashed(draw, p0, p1, color, width=1, dash=9, gap=7):
    x0, y0 = p0
    x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-6:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    t = 0.0
    while t < L:
        t2 = min(t + dash, L)
        draw.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * t2, y0 + uy * t2)],
                  fill=color, width=width)
        t += dash + gap


def arrow(draw, p0, p1, color, width=2, size=9):
    draw.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (+1, -1):
        a = ang + math.pi + s * 0.42
        draw.line([p1, (p1[0] + math.cos(a) * size, p1[1] + math.sin(a) * size)],
                  fill=color, width=width)


def panel(draw, xy, lines, fs=17, pad=10, gap=6, alpha=150):
    """半透明信息面板。lines: [(text, color)]"""
    f = font(fs)
    wmax = max(draw.textlength(t, font=f) for t, _ in lines)
    htot = len(lines) * (fs + gap)
    x0, y0 = xy
    box = (x0, y0, x0 + wmax + 2 * pad, y0 + htot + 2 * pad - gap)
    draw.rounded_rectangle(box, radius=8, fill=(0, 0, 0, alpha),
                           outline=(255, 255, 255, 60), width=1)
    y = y0 + pad
    for t, col in lines:
        draw.text((x0 + pad, y), t, font=f, fill=col)
        y += fs + gap


def main(argv=None):
    ap = argparse.ArgumentParser(prog="10_overlay_video",
                                 description="测量结果叠加视频")
    ap.add_argument("group", help="组目录（含 jet_angle/result.json）")
    ap.add_argument("--out", default=None, help="输出 mp4（默认 <组目录>/overlay.mp4）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--no-panel", action="store_true",
                    help="不画左上角的帧号/角度信息框")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.group)
    res_path = os.path.join(root, "jet_angle", "result.json")
    if not os.path.exists(res_path):
        raise SystemExit("找不到 %s" % res_path)
    with open(res_path, encoding="utf-8") as f:
        res = json.load(f)
    frames = res["frames"]
    W, H = res["image_size"]
    bl = res["baseline"]["line"]
    _n = math.hypot(bl["a"], bl["b"]) or 1.0
    a_b, b_b, c_b = bl["a"] / _n, bl["b"] / _n, bl["c"] / _n

    out = args.out or os.path.join(root, "overlay.mp4")
    s = args.scale
    vw, vh = int(round(W * s)), int(round(H * s))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (vw, vh))
    if not writer.isOpened():
        raise SystemExit("无法创建视频：%s" % out)

    end = len(frames) if args.end < 0 else min(args.end, len(frames))
    idx = list(range(args.start, end, max(1, args.stride)))
    for k, i in enumerate(idx):
        fr = frames[i]
        path = os.path.join(root, fr["file"])
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = pil.resize((vw, vh), Image.LANCZOS).convert("RGBA")
        layer = Image.new("RGBA", (vw, vh), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        def S(p):
            return (p[0] * s, p[1] * s)

        # ---- 基准面 ----
        seg = line_pts(a_b, b_b, c_b, W, H)
        if seg:
            d.line([S(seg[0]), S(seg[1])], fill=C_BASE + (255,), width=2)

        x_imp, y_imp = fr.get("x_imp"), fr.get("y_imp")
        ok = fr.get("status") == "ok" and x_imp is not None
        if x_imp is not None:
            P = (x_imp, y_imp)
            # ---- 法线（虚线，指向液体侧，长度 110 px）----
            nx, ny = -a_b, -b_b
            if ny > 0:
                nx, ny = -nx, -ny
            seg_n = (S(P), S((x_imp + nx * 118, y_imp + ny * 118)))
            dashed(d, seg_n[0], seg_n[1], C_NRM + (235,), width=2,
                   dash=11, gap=7)

            # ---- 入射轴 ----
            inc = fr.get("inc_line")
            if inc:
                ia, ib = unit(inc["a"], inc["b"])
                ic = inc["c"]
                # 方向：取指向图像上方的那个
                dx, dy = ib, -ia
                if dy > 0:
                    dx, dy = -dx, -dy
                d.line([S((x_imp + dx * 30, y_imp + dy * 30)),
                        S((x_imp + dx * 330, y_imp + dy * 330))],
                       fill=C_INC + (255,), width=2)


            # ---- 出流轴（只画拟合段）----
            if ok and fr.get("ref_seg"):
                (x0, y0), (x1, y1) = fr["ref_seg"]
                d.line([S((x0, y0)), S((x1, y1))], fill=C_REF + (255,), width=3)
                for xx, yy in ((x0, y0), (x1, y1)):
                    r = 3
                    d.ellipse([S((xx - r, yy - r))[0], S((xx - r, yy - r))[1],
                               S((xx + r, yy + r))[0], S((xx + r, yy + r))[1]],
                              outline=C_REF + (255,), width=2)

            # ---- 冲击点 ----
            r = 5
            d.ellipse([S((x_imp - r, y_imp - r))[0], S((x_imp - r, y_imp - r))[1],
                       S((x_imp + r, y_imp + r))[0], S((x_imp + r, y_imp + r))[1]],
                      fill=C_IMP + (255,), outline=(20, 20, 20, 255), width=1)

        # ---- 信息面板 ----
        st = fr.get("status", "?")
        st_txt, st_col = {"ok": ("有效", C_OK), "none": ("无出流", C_NONE),
                          "invalid": ("无效", C_BAD)}.get(st, (st, C_NRM))
        lines = [("帧 %d / %d" % (i + 1, len(frames)), (255, 255, 255, 255))]
        if ok:
            lines.append(("θi = %.2f°    θr = %.2f°"
                          % (fr["theta_i"], fr["theta_r"]),
                          (255, 255, 255, 255)))
            lines.append(("ρ = %.2f     %s" % (fr["rho"], st_txt), st_col))
        else:
            lines.append(("θi = %.2f°" % fr["theta_i"]
                          if fr.get("theta_i") is not None else "θi = —",
                          (255, 255, 255, 255)))
            lines.append(("θr = —", (200, 200, 200, 255)))
            lines.append((st_txt, st_col))
        if not args.no_panel:
            panel(d, (14, 14), lines, fs=int(17 * s / 1.5))

        # ---- 底部图例 ----
        f = font(int(14 * s / 1.5), bold=False)
        items = [(C_BASE, "基准面"), (C_INC, "入射轴"),
                 (C_REF, "出流轴"), (C_NRM, "法线")]
        x = 16
        y = vh - int(26 * s / 1.5)
        for col, name in items:
            d.line([(x, y + 7), (x + 22, y + 7)], fill=col + (255,), width=3)
            d.text((x + 28, y), name, font=f, fill=(255, 255, 255, 235))
            x += 28 + d.textlength(name, font=f) + 20

        out_img = Image.alpha_composite(pil, layer).convert("RGB")
        writer.write(cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR))
        if k % 200 == 0:
            print("  %d/%d" % (k + 1, len(idx)), flush=True)

    writer.release()
    print("video ->", out)


if __name__ == "__main__":
    sys.exit(main())
