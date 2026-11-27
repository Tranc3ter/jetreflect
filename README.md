# jetreflect — 液体射流反射角测量工具

> **版本 1.4**
> * 新增 `00_set_velocity.py`：把**射流速度逐组写进 `baseline.json`**（`velocity_mps`），
>   `08_jet_physics.py` 优先用它，多速度数据集不再共用一个全局 V；
> * 新增 `09_plot_single.py`（θr–θi、ρ–θi 单张图）、`11_plot_velocity.py`（θr/ρ–V）、
>   `10_overlay_video.py`（带标注的 MP4）；
> * `08` 输出里补 `velocity_source`，便于追溯每组速度从哪来。

---

## 1. 功能

| 功能 | 说明 |
|---|---|
| 任意朝向的反射面 | 不假设水平/竖直，不假设射流竖直、出流在右下 |
| 贴板掠出的出流 | 出流与板面只差 10–20° 也能分出来（用上下缘中线） |
| 无出流帧 | 判 `none` 并单列计数，不混进平均 |
| 逐帧 QC | 每帧 `status / why / source / d_axis_env` 都留档 |
| 多数据集、多速度 | 每组一条基线、每组一个速度 |
| 出图与视频 | 汇总图、θr–θi、ρ–θi、θr/ρ–V、逐帧叠加 MP4 |


---
<img src="images\overlay_.png" width="45%">

## 2. 安装与运行

```bash
conda create -n jetreflect python=3.11 -y
conda activate jetreflect
pip install -r requirements.txt        # numpy scipy opencv-python matplotlib
```


依赖：`numpy`、`scipy`、`opencv-python`、`matplotlib`；GUI 另需 `tkinter`（Anaconda 自带）。

---

## 3. 数据组织约定

```
<数据集根目录>/
├── jet_angle_config.json      全局参数（首次运行自动生成）
├── <组 A>/                    一个组 = 一个文件夹
│   ├── xxx_00000000.tif       该组的全部图像（顺序即帧号）
│   ├── xxx_tif.txt            相机元数据
│   ├── baseline.json          反射面 + 速度 + 直径（人工）
│   └── jet_angle/             测量输出（自动生成）
├── <组 B>/
└── sample/                    每组一张汇总叠加图（自动生成）
```

* 组 = 根目录下的**一级子文件夹**；`skip_dirs`（默认 `sample, jet_angle, results, _summary`）里的不算组。
* 组内图像按**文件名排序**当帧号；少于 `min_frames`（默认 20）张的文件夹会被跳过。
* 图像格式：`.tif/.tiff/.bmp/.png`（8 bit 或 16 bit 均可，内部统一转 8 bit）。
* 路径可含**中文和空格**（读写走 `np.fromfile + cv2.imdecode`）。

---

## 4. 工作流总览

| 步 | 脚本 | 作用 | 交互 | 产出 |
|---|---|---|---|---|
| 0 | `00_set_velocity.py` | 登记每组射流速度 | 否 | `baseline.json.velocity_mps` |
| 1 | `01_mark_baseline.py` | 标反射面（一组一条线） | **是** | `baseline.json.line` |
| 2 | `02_measure.py` | 逐帧测量 θi/θr/ρ（可抽样） | 否 | `<组>/jet_angle/*` |
| 3 | `03_review.py` | 看图签核（Y/N） | **是** | `qc.json` |
| 4 | `04_report.py` | 汇总表 + 每组叠加图 + `fig_summary.png` | 否 | 根目录 `jet_angle_summary.*` |
| 5 | `05_calibrate.py` | px/mm 标定（可选，We 需要） | **是** | `config.px_per_mm` |
| 6 | `06_plot_rho.py` | θr / ρ / dθr/dθi ~ θi | 否 | `fig_rho_vs_theta_i.png` 等 |
| 7 | `07_mark_diameter.py` | 标初始射流直径 D（一组一次） | **是** | `baseline.json.jet_diameter_*` |
| 8 | `08_jet_physics.py` | We / We⊥ / Re | 否 | `<组>/jet_physics.json` |
| 9 | `09_plot_single.py` | θr–θi、ρ–θi 单张图 | 否 | `fig_single_*.png` |
| 10 | `10_overlay_video.py` | 逐帧叠加视频 | 否 | `overlay.mp4` |
| 11 | `11_plot_velocity.py` | θr / ρ ~ V | 否 | `fig_velocity_*.png` |

最小可用流程：**0 → 1 → 2 → 4**（只关心角度时）。
要 We/Re：**0 → 1 → 5 → 2 → 4 → 7 → 8**。

```bash
D="D:\data\20261119_11group"

python scripts/00_set_velocity.py "$D" --all            # 文件夹名就是速度时
python scripts/01_mark_baseline.py "$D"                 # 交互标线
python scripts/02_measure.py "$D" --sample 5 --out jet_angle_sample --review
python scripts/02_measure.py "$D"                       # 全量
python scripts/04_report.py "$D"

python scripts/05_calibrate.py "$D" --px-per-mm 41.063  # 或交互点标尺
python scripts/07_mark_diameter.py "$D"                 # 交互标直径
python scripts/08_jet_physics.py "$D" --csv

python scripts/06_plot_rho.py "$D"
python scripts/09_plot_single.py "$D"
python scripts/11_plot_velocity.py "$D"
python scripts/10_overlay_video.py "$D\<组名>" --fps 30
```

`run_all.py` = cli 的 `all`：测量 → 汇总 → 签核
（基线必须已存在；`--no-review` 跳过交互签核）。

---

## 5. 脚本详解

### 0. `00_set_velocity.py` — 登记射流速度

速度**逐组**存进 `<组>/baseline.json`，测量之前就该填。

```bash
python scripts/00_set_velocity.py "<根目录>" --all          # 文件夹名当速度（0.9 / 1 / 1.5 / ...）
python scripts/00_set_velocity.py "<根目录>" --group 2.5 --v 2.5
python scripts/00_set_velocity.py "<组目录>" --v 2.0       # 单组
```

| 选项 | 作用 |
|---|---|
| `--all` | 所有组的名字必须能转成数字，直接当速度 |
| `--group a,b` | 只处理名字里含这些词的组（子串匹配） |
| `--v 2.5` | 指定速度（不给就用文件夹名） |
| `--dry-run` | 只打印要写什么，不落盘 |

写入字段（**不覆盖**其他字段）：

```json
"velocity_mps": 2.5,
"velocity_source": "folder",              // 或 "manual"
"velocity_updated": "2026-11-25T13:03:17"
```

速度优先级（`08_jet_physics.py`）：`--velocity` 命令 > `baseline.json.velocity_mps` >
`config.physics.velocity_ms` > 默认 2.0。

---

### 1. `01_mark_baseline.py` — 标反射面

```bash
python scripts/01_mark_baseline.py "<根目录>"            # 全部组
python scripts/01_mark_baseline.py "<根目录>" --group 2.5
```

| 选项 | 作用 |
|---|---|
| `--group a,b` | 只标这些组 |
| `--frame-ratio 0.5` | 打开该组的第几帧（0–1） |

| 按键 | 作用 |
|---|---|
| 左键拖拽 / 左键点两下 | 拉一条**无限长**直线 |
| `U` / `Ctrl+Z` | 撤销 |
| `R` | 清空 |
| `N` / `P`、`→` / `←` | 下一帧 / 上一帧（只影响显示） |
| `J` / `K`、`↓` / `↑` | 下一组 / 上一组 |
| `S` / `Enter` | 保存并跳到下一组 |
| `Z` / `X`、滚轮 | 放大 / 缩小 |
| `F` | 全图 |
| 右键 / 中键拖拽 | 平移 |
| `Q` / `Esc` | 退出（已保存不丢） |

要点：画的是**反射面本身**（镜面过渡带中点 / 高光带中心），不是它的轮廓边。
线是**无向**的：`p1→p2` 与 `p2→p1` 等价；存盘同时写 `line: {a,b,c}`
（单位法线 `(a,b)`，`a·x + b·y + c = 0`），`tilt_deg` 折到 `(-90°, +90°]`。

`baseline.json` 全字段：

```json
{
  "schema": "jetreflect/baseline-2",
  "dataset": "20261112_8group",
  "group": "2.5",
  "image_size": [768, 576],
  "p1": [233.6, 0.0], "p2": [461.3, 575.0],
  "line": {"a": 0.9298, "b": -0.3681, "c": -217.23},
  "tilt_deg": 21.601,
  "frame": "MX-5 2058-1-M003-N000_00000100.tif",
  "source": "manual",
  "created": "2026-11-25T11:25:40+08:00",
  "operator": "31702",

  "velocity_mps": 2.5,                 // 00_set_velocity.py 写入
  "velocity_source": "folder",
  "velocity_updated": "2026-11-25T13:03:17",

  "jet_diameter_px": 25.0,             // 07_mark_diameter.py 写入
  "jet_diameter_mm": 0.6089,
  "diameter_p1": [372.1, 208.4],
  "diameter_p2": [397.1, 208.4],
  "diameter_frame": "..._00001000.tif",
  "diameter_px_per_mm": 41.063,
  "diameter_source": "manual"
}
```

`p1/p2` 只用于显示；**真正的几何是 `line`**。`image_size` 用来在换相机/裁图后直接报错。

---

### 2. `02_measure.py` — 逐帧测量

```bash
python scripts/02_measure.py "<根目录>"                      # 全部组、每帧
python scripts/02_measure.py "<根目录>" --sample 5 --out jet_angle_sample --review
python scripts/02_measure.py "<根目录>" --group 2,2.5 --stride 2 --force
```

| 选项 | 作用 |
|---|---|
| `--group a,b` | 只算这些组（子串匹配） |
| `--stride 5` | 隔 5 帧算一张 |
| `--sample 5` | 每组均匀抽 5 帧 |
| `--frames 0,500,1000` | 显式指定帧号 |
| `--out jet_angle_sample` | 输出到 `<组>/jet_angle_sample/`，不动正式结果 |
| `--review` | 额外输出逐帧叠加图 + 每组 `check_sheet.png` |
| `--engine v2\|legacy` | 覆盖 config 里的引擎（legacy 只用于复现老数据） |
| `--workers N` | 并行线程（默认 1；2–4 略有帮助） |
| `--force` | 已有 `result.json` 也重算 |

---

### 3. `03_review.py` — 人工签核

```bash
python scripts/03_review.py "<根目录>" [--group a,b]
```

逐组显示 `qc_montage.png`（4 帧叠加），按键：`Y` 通过 / `N` 拒绝 / `S` 或 `Esc` 跳过 / `Q` 退出。
结果写 `<组>/jet_angle/qc.json`：

```json
{"schema": "jetreflect/qc-2", "approved": true, "operator": "31702",
 "at": "...", "montage": "qc_montage.png"}
```

`04_report.py` 会把 `approved` 填进 `jet_angle_summary.csv`。

---

### 4. `04_report.py` — 汇总

```bash
python scripts/04_report.py "<根目录>" [--group a,b] [--out jet_angle]
```

* 写根目录 `jet_angle_summary.csv` / `.json`（**总是覆盖全部已测组**，不受 `--group` 截断）；
* 写根目录 `fig_summary.png`（各组 θi/θr/ρ 柱状+点图，横轴＝组名）；
* 写 `sample/<组>_overlay.png`（每组一张）。

`jet_angle_summary.csv` 列：

```
group, n_frames, n_measured, n_valid, valid_ratio,
theta_i_mean/std/median, theta_r_mean/std/median,
angle_between_mean/std/median, rho_mean/std/median,
jet_tilt_deg, px_per_mm, approved, operator, status
```

`status` 会自动标 `ok` 或 `low-validity(land/splash?)`（有效率 < 20% 时）。

---

### 5. `05_calibrate.py` — px/mm 标定

```bash
python scripts/05_calibrate.py "<根目录>" --px-per-mm 41.063     # 直接给值
python scripts/05_calibrate.py "<根目录>" "<标尺图>" --mm 10      # 点两点 + 真实长度
python scripts/05_calibrate.py "<根目录>" --p1 410.6,300 --p2 820.6,300 --mm 10
```

写入 `config.px_per_mm` 与 `config.calibration`（图、两点、像素距离、真实长度、时间）。
**角度不需要它**；只有 D(mm)、We、Re 需要。

---

### 6. `06_plot_rho.py` — θr / ρ / 灵敏度 ~ θi

```bash
python scripts/06_plot_rho.py "<根目录>" [--out 目录] [--fit quadratic] [--min-dx 1.0]
```

* 读 `jet_angle_summary.csv`，按 θi 升序；
* 三张图：θr、ρ、dθr/dθi；
* 相邻两组的离散斜率：θi 间隔 < `--min-dx` 时跳过（图中粉色带标出）；
* `--fit none|linear|quadratic|cubic` 拟合曲线及其导数（默认二次）。

产出：`fig_rho_vs_theta_i.png`、`rho_vs_theta_i.csv`
（含 `gap_to_next_deg`、`dtheta_r_dtheta_i`、`note`）。

---

### 7. `07_mark_diameter.py` — 标初始射流直径

```bash
python scripts/07_mark_diameter.py "<根目录>" [--group a,b] [--frame-ratio 0.5]
python scripts/07_mark_diameter.py "<根目录>" --diameter-px 25.0     # 批量直接给值
```

| 按键 | 作用 |
|---|---|
| 左键点两点 | 量喷嘴出口**收缩后的液柱直径**（垂直于射流轴线） |
| `U` | 撤销 |
| `N` / `P` | 下一帧 / 上一帧 |
| `S` / `Enter` | 保存并跳到下一组 |
| `Q` | 退出 |

写入**原有**的 `baseline.json`（不动 `p1/p2/line`）：
`jet_diameter_px`、`jet_diameter_mm`、`diameter_p1/p2`、`diameter_frame`、
`diameter_px_per_mm`、`diameter_source`。

---

### 8. `08_jet_physics.py` — We / We⊥ / Re

```bash
python scripts/08_jet_physics.py "<根目录>" [--group a,b] [--csv]
python scripts/08_jet_physics.py "<根目录>" --velocity 2.0 --rho 998 --gamma 0.072 --mu 0.001002
```

| 选项 | 作用 |
|---|---|
| `--velocity` | 覆盖所有组的速度（默认取 baseline.json，再退回 config） |
| `--rho/--gamma/--mu` | 覆盖流体物性（20 °C 水：998 / 0.072 / 0.001002） |
| `--diameter-px` | 覆盖直径（测试用） |
| `--csv` | 另在根目录写 `jet_physics_summary.csv` |

公式（写进输出，避免口径歧义）：

```
We      = ρ V² D / γ
We_perp = ρ R (V cosθi)² / γ ,  R = D/2
Re      = ρ V D / μ
r       = tanθr / tanθi
```

输出 `<组>/jet_physics.json`（`schema: jetreflect/physics-2`）：

```json
{
  "config":  {"velocity_ms": 2.5, "velocity_source": "baseline.json",
              "rho_kg_m3": 998, "gamma_N_m": 0.072, "mu_Pa_s": 0.001002,
              "temperature_C": 20, "px_per_mm": 40.75,
              "diameter_px": 25.0, "diameter_mm": 0.6135,
              "diameter_source": "manual", "diameter_frame": "..."},
  "angles":  {"theta_i": 68.8, "theta_r": 75.3, "r": 1.48, "n_valid": 201},
  "results": {"We": 45.2, "We_perp": 6.9, "Re": 1524},
  "formulas": {...}
}
```

---

### 9. `09_plot_single.py` — 单张图（θr–θi、ρ–θi）

```bash
python scripts/09_plot_single.py "<根目录>" [--out 目录] [--fit none|quadratic]
```

两张独立的 PNG（默认不画拟合，`--fit` 才加），**同画布尺寸（7.4×4.6 in）、同横轴范围（20–80°）**：

* `fig_single_theta_r.png`：θr vs θi，误差棒＝组内逐帧标准差；
  **两轴都取同一角度范围（20–80°）、都用 10° 主刻度 / 5° 副刻度**，
  并画 `θr = θi`（镜面反射）虚线作参考。不作等比例（横纵一格在纸面上不等长，但都是 10°）。
* `fig_single_rho.png`：ρ vs θi，横轴同样 20–80°、10° 主刻度；纵轴为 ρ（约 0.7–6.5，1.0 一格），
  含 ρ=1（镜面反射）虚线。

两张都是 **1628×1011 px（220 dpi）**，横轴刻度对齐，可直接并排或上下叠放。

> 注：`06_plot_rho.py` 的三联图共用一条 θi 横轴（三张图宽度必须一致），
> 所以那张里的 θr 面板**没有**做等比例；要看"一格＝同样度数"的 θr–θi 图，
> 用 `09` 的这张。

---

### 10. `10_overlay_video.py` — 带标注的 MP4

```bash
python scripts/10_overlay_video.py "<组目录>" [--out overlay.mp4] \
    [--fps 30] [--scale 1.5] [--stride 1] [--start 0] [--end -1] [--no-panel]
```

读 `<组>/jet_angle/result.json`（含每帧 `inc_line / ref_line / ref_seg / x_imp`），
在原始帧上叠：

| 颜色 | 元素 |
|---|---|
| 青色实线 | 基准面（反射面，贯穿整幅） |
| 琥珀色 | 入射轴 |
| 绿色 | 出流轴（只画拟合段 + 两端圆点） |
| 灰色虚线 | 法线 |
| 橙色圆点 | 冲击点 |
| 左下角 | 颜色图例 |
| 左上角 | 帧号 / θi / θr / ρ / 状态（`--no-panel` 可去掉） |

2001 帧、1.5×（960×720）、30 fps 约 2 分钟渲染，文件 30 MB 左右。
编码 `mp4v`（H.264 兼容容器），PowerPoint 可直接插入播放。

---

### 11. `11_plot_velocity.py` — θr / ρ ~ V

```bash
python scripts/11_plot_velocity.py "<根目录>" [--out 目录]
```

速度来源：`baseline.json.velocity_mps` → 组名（能转成数字时）。
有效率 < 20% 或有效帧 < 10 的组画成**空心灰标记**（如 0.9 m/s 的铺展组），不参与趋势解读。

产出：`fig_velocity_panels.png`（双联）、`fig_velocity_theta_r.png`、`fig_velocity_rho.png`。

θr 面板的**角度轴固定 20–80°、10° 主刻度**（与 `09` 的 θr–θi 图一致）；
超出 80° 的点（如 0.9 m/s 的 82.6°）在顶边画白色三角并标数值，不会静默剪掉。

---

### 其它

`python -m jetreflect <子命令>` 与脚本等价：

| 子命令 | 等价脚本 |
|---|---|
| `info` | 列出组、帧数、基线、有效率、是否签核 |
| `mark` | `01_mark_baseline.py` |
| `measure` | `02_measure.py` |
| `review` | `03_review.py` |
| `report` | `04_report.py` |
| `all` | `02 + 04 + 03` |
| `calibrate` | `05_calibrate.py` |
| `import-legacy` | 导入旧版 `mirror_lines.json` |

`run_all.py` = cli 的 `all`：measure → report → review（`--no-review` 跳过签核）。

---

## 6. 配置文件 `jet_angle_config.json`

首次运行自动生成；除 `params` 外都可以直接手改。

| 键 | 默认 | 含义 |
|---|---|---|
| `schema` | `jetreflect/dataset-2` | 配置版本 |
| `dataset` | 根目录名 | 数据集名（写进各输出） |
| `engine` | `v2` | 用哪个引擎（`v2` / `legacy`） |
| `params_engine` | 同 `engine` | `params` 属于哪个引擎（切引擎时自动丢弃旧参数） |
| `image_formats` | `.tif .tiff .bmp .png` | 认哪些扩展名 |
| `min_frames` | 20 | 少于这么多帧的文件夹不当组 |
| `skip_dirs` | `sample, jet_angle, results, _summary` | 不当作组的子目录 |
| `px_per_mm` | `null` | 标定值（角度不需要，We 需要） |
| `params` | v2 默认表 | 逐帧算法参数（见 §10） |
| `calibration` | — | `05_calibrate.py` 写入的两点与长度 |
| `physics` | — | `08_jet_physics.py` 读的物性块 |

`physics` 块示例（都可省，省了用内置默认）：

```json
"physics": {
  "velocity_ms": 2.0,
  "rho": 998.0, "gamma": 0.072, "mu": 0.001002, "temperature_C": 20.0
}
```

---

## 7. 输出文件

```
<数据集>/
├── jet_angle_config.json
├── jet_angle_summary.csv / .json        各组汇总（见 §4）
├── fig_summary.png                      各组 θi/θr/ρ
├── fig_rho_vs_theta_i.png               06 的三联图
├── rho_vs_theta_i.csv                   06 的排序表 + 斜率
├── fig_single_theta_r.png / _rho.png    09
├── fig_velocity_*.png                   11
├── jet_physics_summary.csv              08 --csv
├── sample/<组>_overlay.png              每组一张
└── <组>/
    ├── baseline.json
    ├── jet_physics.json                 08
    ├── overlay.mp4                      10（可选）
    └── jet_angle/
        ├── result.json                  逐帧全量 + 统计 + 参数（机器可读）
        ├── per_frame.csv                逐帧表（Excel 可读）
        ├── angles.csv                   精简逐帧表
        ├── summary.csv                  本组一行
        ├── fig_timeseries.png           角度-帧号 + 直方图
        ├── qc_montage.png               4 帧拼图
        └── qc.json                      签核结果
```

抽样（`--out jet_angle_sample --review`）额外得到
`jet_angle_sample/review/f#####.png` 与 `check_sheet.png`。

**`result.json` 结构**

```
schema, dataset, group, created, algorithm, engine,
baseline, params, px_per_mm, image_size,
frames_expected, frames_measured,
frames[]  → 每帧：
  file, frame, status, why, note, valid,
  theta_i, theta_r, rho, angle_between,
  x_imp, y_imp, rms_i, rms_r, n_i, n_r,
  inc_line {a,b,c}, ref_line {a,b,c}, ref_seg [[x0,y0],[x1,y1]],
  ref_source ("axis" | "envelope"), theta_r_env, d_axis_env
stats     → theta_i/theta_r/rho/angle_between 的 mean/std/median/min/max/n
n_valid, n_none, n_invalid
```

**`per_frame.csv` 列**

```
frame, file, status, valid, theta_i, theta_r, theta_r_env, d_axis_env,
source, angle_between, rho, x_imp, y_imp, rms_i, rms_r, n_i, n_r, why, note
```

`source` = `axis`（轴线，默认）或 `envelope`（回退外缘）。
`d_axis_env` = 轴线与上缘之差（QC 用，> 5° 说明中线跟错了边）。

---

## 8. 角度与无量纲数定义

| 字段 | 含义 |
|---|---|
| `theta_i` | 入射射流轴线与反射面**法线**的夹角（0° 正入射，90° 沿面） |
| `theta_r` | 出流轴线与法线的夹角（轴线不可用时回退外缘，见 `source`） |
| `angle_between` | 两轴线的**锐夹角** |
| `rho` | `tanθr / tanθi`；**ρ > 10 记 `none`**（θr→90° 发散，数值无意义） |
| `status` | `ok` / `none`（无出流）/ `invalid`（测得但不可信） |

We / Re 见 §5.8。

---

## 9. 算法（v2 引擎）

一句话：**先用入射射流自己定坐标系，再沿两条轴追踪；出流取上下缘中线（轴线），
只有中线算不出来时才回退外缘。**

1. **入射射流**：按"离反射面最远的暗边对"取候选种子，逐个沿其自身方向追踪
   （离面 > 18 px），MAD 稳健拟合 `x = k·y + c`；用拟合方向再旋转一次重追
   （收敛判据 1°）。**不用全局 Hough**——它会锁到反射面这条最长直边上。
2. **冲击点** = 入射轴线 ∩ 反射面（无向直线求交）。
3. **出流**：把图旋到"反射面竖直"，以冲击点为原点、沿面为 `u`、离面为 `v`，
   在 `v∈[3,140] px`、`u∈[0,340] px` **两侧**搜索。
   * **轴线**：逐列取"上缘"（离面最远的暗脊）与"下缘"（靠板侧暗脊，
     且与板之间必须有亮间隙，贴板段跳过），`c(u) = (上缘+下缘)/2` 后直线拟合。
     液柱两条轮廓关于轴线对称，取中线可把界面波与锥形张开一起消掉。
     轴线**就是结果**；`theta_r_env` / `d_axis_env` 只作 QC 写进 `result.json`。
   * **回退**：中线可用列 < 15（出流全程贴板、上下缘分不开）时退回外缘，
     `source` 写 `envelope`。外缘法用斜率标准误 `SE ≤ 0.02` 判方向可靠性；
     `|m| < tan8°` 视为"只是板边" → `none`；残差过大（破碎液滴）→ `invalid`。
4. **判定**：θi∈[5, 89.9]、θr ≤ 89.5、两轴夹角 ≥ 8°、残差达标 → `ok`；
   没有出流 → `none`；测得但被门限否掉 → `invalid`。后两类**都不计入统计**。

> QC 图里绿线**只画实际拟合段**（两端有小圆点），不向冲击点外推。
> 外推落点与冲击点相差 0.1–9.6 px（中位 2.9 px），画到全图会让人误以为
> "三条线交于一点"——那个点只是"入射轴 ∩ 反射面"的定义，不是测量结果。

---

## 10. 主要参数（`config.params`）

| 键 | 默认 | 含义 |
|---|---|---|
| `cmin` | 30 | 暗边相对背景的最小对比度（灰度级） |
| `dmin` / `dmax` | 6 / 45 | 双暗边间距（射流表观宽度）范围，px |
| `halfwin` | 22 | 逐行追踪的横向搜索半窗，px |
| `step` / `scan_step` | 2 / 4 | 追踪步长 / 种子扫描行步长 |
| `max_rows` / `max_miss` | 520 / 8 | 单次追踪最多行数 / 允许连续丢失行数 |
| `wall_margin` | 10 | 追踪时与反射面保持的最小距离，px |
| `seed_min_v` | 18 | 入射种子离反射面的最小距离，px |
| `min_inc_points` / `min_ref_points` | 25 / 12 | 拟合所需最少点数 |
| `theta_i_range` | [5, 89.9] | 入射角允许范围 |
| `theta_r_max` | 89.5 | 超过视为贴板走 |
| `rho_max` | 10 | ρ 上限，超过记 `none` |
| `ref_v` | [3, 140] | 出流搜索的离面距离带，px |
| `ref_u_max` / `ref_min_len` | 340 / 60 | 沿面最远 / 最短连续长度，px |
| `ref_v` / `env_thin` | — / 6 | 外缘搜索带宽 / 细化核 |
| `min_mid_points` | 15 | 拟合中线所需最少列数 |
| `min_axis_angle` | 8 | 两轴最小夹角 |
| `rms_i_max` / `rms_r_max` | 2.0 / 12 | 入射 / 出流拟合残差上限，px |
| `se_max` | 0.02 | 外缘方向斜率标准误上限 |

改参数后重跑该组要加 `--force`。

---

## 11. 判定与可信度

* `ok`：入射、出流都追到且过门限 → 写进统计。
* `none`：追不到出流（铺展成膜、射流未形成、出流贴板无法与板边区分）。单列计数。
* `invalid`：追到了但不可信（出流太短、外边界非直线＝破碎、入射拟合差）。原因写 `why`。

`summary.csv` 里 `n_measured / n_valid / n_none / n_invalid` 分列，`valid_ratio` 只算 `ok`。

---

## 12. 常见问题

**TIFF 私标签警告**：相机 TIFF 带非标准标签 0x7900，OpenCV 会打 `TIFF_Warning`；
工具已把 OpenCV 日志压到 SILENT，不影响读取。

**没有图形界面**：`measure / report / 06 / 08 / 09 / 10 / 11` 不需要窗口；
`mark / review / calibrate / 07` 需要。服务器上可先在本地标好 `baseline.json` 再拷过去。

**改了阈值想重判**：改 `params` 后对该组 `--force`。

**Excel 打开 CSV 乱码**：所有 CSV 用 UTF-8 with BOM 写出。

**多速度数据集**：每组用 `00_set_velocity.py` 写 `velocity_mps`，别只用 config 的全局 V。

**速度从哪来**：`--velocity` > `baseline.json.velocity_mps` > `config.physics.velocity_ms` > 2.0。

---

## 13. 测试

```bash
python tests/test_geom.py        # 无向直线几何
python tests/test_synthetic.py   # 合成图：已知 68.66/76.20，误差 < 0.1°
python tests/test_gui.py         # 模拟鼠标拖线 -> baseline.json
python tests/test_pipeline.py    # 端到端 measure -> report -> sample
# 或 pytest -q
```

---

## 14. 版本历史

| 版本 | 变更 |
|---|---|
| 1.0–1.1 | 初版：Hough + 固定右下搜索（对斜置板失效） |
| 1.2 | v2 引擎：方向自适应、两侧搜索、三态判定 |
| 1.3 | 轴线优先、去 0.5° 门限、绿线只画拟合段、无向直线基线、ρ>10 记 none |
| **1.4** | 逐组速度（`00_set_velocity.py` + `baseline.json`）、θr–θi/ρ–θi 单张图、θr/ρ–V 图、叠加视频、`velocity_source` |
