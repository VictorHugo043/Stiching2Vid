# 04_quality_improvement

## 阶段 4.1 目标
- 针对视频级拼接中的 `H jitter` 与 overlap 区域 `flicker` 做时序稳定增强。
- 保持 baseline 输出结构不变，新增诊断文件用于量化对比。

## 背景问题
- baseline 采用关键帧重估 + 非关键帧复用，关键帧切换时 `H` 会出现跳变。
- 跳变会直接放大到 `overlay` 和融合区域，形成视觉抖动和亮度闪烁。

## 新增参数
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--smooth_h` | `none` | 时序平滑策略：`none|ema|window` |
| `--smooth_alpha` | `0.8` | EMA 平滑系数，越大越平滑 |
| `--smooth_window` | `5` | `window` 模式滑窗长度 |
| `--snapshot_every` | `50` | 诊断快照步长，输出 raw/sm overlay 对比 |

## jitter 指标定义
- 令 `c_i(t)` 为第 `t` 帧（映射到画布后）第 `i` 个角点坐标（`i=1..4`）。
- 逐帧位移：`d_i(t) = ||c_i(t) - c_i(t-1)||_2`。
- 指标：
  - `jitter = mean_i(d_i)`（写入 `jitter_raw/jitter_sm`）
  - 额外记录 `max_i(d_i)`（`jitter_raw_max/jitter_sm_max`）
- `raw` 与 `smoothed` 各自独立计算，用于对比平滑前后稳定性。

## 输出文件说明（增量）
- `outputs/runs/<run_id>/transforms.csv`
  - 新增 `Hraw_00..Hraw_22`、`Hsm_00..Hsm_22`
  - 新增 `jitter_raw/jitter_sm` 与 `*_max`
  - 兼容保留 `H_00..H_22`（当前实际用于 warping 的 H）
- `outputs/runs/<run_id>/jitter_timeseries.csv`
  - `frame_idx, jitter_raw, jitter_sm, status`
- `outputs/runs/<run_id>/snapshots/`
  - 新增 `overlay_raw_<frame>.png` 与 `overlay_sm_<frame>.png`
- `outputs/runs/<run_id>/debug.json`
  - 新增 `smooth_h/smooth_alpha/smooth_window`
  - 新增 `jitter_summary`（均值与 95 分位）

## Ablation（4.1）
- 新增 `scripts/ablate_temporal.py`，同一 pair 自动跑两次：
  - A: baseline `--smooth_h none`
  - B: temporal `--smooth_h ema --smooth_alpha 0.8`
- 结果目录：`outputs/ablations/<pair_id>/`
  - `summary_temporal.csv`
  - `compare/`（两组 overlay 对比图）

## 已知限制
- 本阶段不解决视差导致的双影，只减少几何抖动与时序闪烁。
- 不包含 seam finding / multiband / mesh warp / 去畸变。

## 下一阶段预告
- 接入 seam + multiband，优先降低动态场景中的重影可见性。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| src/stitching/temporal.py | 新增时序平滑器与 jitter 计算 | Codex | 完成 |
| scripts/run_baseline_video.py | 集成 smooth_h、jitter 输出与 raw/sm overlay | Codex | 完成 |
| scripts/ablate_temporal.py | 新增 temporal ablation 一键对比脚本 | Codex | 完成 |
| ai-docs/current/04_quality_improvement/04_quality_improvement.md | 新增 4.1 阶段文档 | Codex | 完成 |
