# 03_baseline_video_pipeline

## 任务目标
- 将单帧拼接方法扩展为视频级流程，形成可运行、可播放、可诊断的基线输出。

## 验收标准(DoD)
- 至少 1 组 video 与 1 组 frames 数据可以产出可播放的 `stitched.mp4`。
- 失败不会导致全流程崩溃（除非首帧不可读）。
- 产出稳定的诊断文件：`transforms.csv`、`metrics_preview.json`、`debug.json`、`logs.txt`。

## Pipeline 设计（baseline）
- 数据读取：复用 `open_pair()` + `FrameSource.read_next()` 顺序解码，避免频繁 seek。
- 关键帧策略：第 0 帧强制估计 H；每隔 `keyframe_every` 帧重估一次；其余帧复用最近有效 H。
- 画布策略：以第一帧（或首个可用 H）估计的画布为固定画布，避免输出尺寸抖动。
- 融合策略：`blend=none|feather`，默认 feather（distance transform 权重）。

## FPS 决策规则（已实现）
- 优先：`input_type=video` 且 `source.fps()` 有效。
- 其次：`pairs.yaml` 的 `meta.fps`。
- 再次：命令行 `--fps`。
- 兜底：30，并在 `debug.json` 与日志中记录 warning。

## 失败兜底策略（关键帧）
- 首帧估计失败：允许 fallback 到单位矩阵（status=`FAIL_INIT`），流程继续并记录。
- 非首帧关键帧失败：复用上一帧有效 H（status=`FALLBACK`）。
- 任意单帧失败不应中断整个视频处理。

## 输出目录与文件含义
- 目录：`outputs/runs/<run_id>/`，默认 `<run_id>=YYYYMMDD-HHMM_<pair_id>_<feature>`。
- 主输出：`stitched.mp4`（固定尺寸，保证可播放）。
- 轨迹记录：`transforms.csv`（逐帧统计与 H 展平，供后续评估脚本读取）。
- 快速统计：`metrics_preview.json`（成功率、平均内点、平均耗时等）。
- 运行信息：`debug.json`（参数、fps 策略、长度信息、异常摘要）。
- 日志输出：`logs.txt`（INFO/WARNING/ERROR）。
- 诊断快照：`snapshots/`（每隔 K 帧存 left/right/stitched/overlay；关键帧额外存 matches/inliers）。

## transforms.csv 字段约定
- 必含列：`frame_idx`、`is_keyframe`、`status`、`n_kp_left/right`、`n_matches_raw/good`、`n_inliers`、`inlier_ratio`、`H_00..H_22`、`runtime_ms`、`note`。
- `status` 语义：`OK`、`FAIL_INIT`、`FAIL_EST`、`FALLBACK`。

## 已知问题（baseline 预期现象）
- 视差与动态物体会导致重影（全局单应性 + feather 的自然结果）。
- 首帧估计失败时固定画布可能偏小，后续会出现裁剪。
- 未做去畸变与柱面投影，广角场景可能拉伸明显。

## 运行示例
- Video 示例：
  - `python scripts/run_baseline_video.py --pair campus_sequences_campus4_c0_c1 --max_frames 300 --keyframe_every 5`
- Frames 示例：
  - `python scripts/run_baseline_video.py --pair dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right --max_frames 200 --keyframe_every 5 --fps 30`

## 下一步优先级建议
- P1：temporal smoothing（先稳住 H 抖动与闪烁）。
- P2：seam finding + multiband blending（显著降低重影观感）。
- P3：去畸变 / 柱面投影（提高几何一致性）。
- P4：局部网格形变（mesh/APAP）应对视差。

## P1 接入状态（已完成）
- 已在 `scripts/run_baseline_video.py` 接入 `--smooth_h none|ema|window`，默认 `none` 保持 baseline 行为。
- 已新增 `src/stitching/temporal.py`：对单应轨迹进行时序平滑，并输出 `raw/smoothed` 双轨诊断。
- 已新增 `jitter_timeseries.csv` 与 `overlay_raw/overlay_sm` 快照，用于量化和目视验证抖动改善。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| scripts/run_baseline_video.py | 新增视频级基线脚本（关键帧 + 兜底 + 输出规范） | Codex | 完成 |
| src/stitching/blending.py | 新增 `blend_none()` 诊断融合模式 | Codex | 完成 |
| ai-docs/current/03_baseline_video_pipeline/03_baseline_video_pipeline.md | 同步更新设计与输出约定 | Codex | 完成 |
| src/stitching/temporal.py | 新增时序平滑与 jitter 计算模块 | Codex | 完成 |
| scripts/run_baseline_video.py | 新增 smooth_h 参数、Hraw/Hsm 记录、jitter 诊断输出 | Codex | 完成 |
| scripts/ablate_temporal.py | 新增 temporal ablation 自动对比脚本 | Codex | 完成 |
| ai-docs/current/04_quality_improvement/04_quality_improvement.md | 新增阶段 4.1 文档 | Codex | 完成 |
