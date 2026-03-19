# 04_quality_improvement

## 任务目标
- 记录当前质量改进项的真实实现边界，区分“已完成能力”“当前不足”“后续 Dynamic seam/meaningful temporal evaluation 的实施边界”。

## 已实现能力（as-built）
### 1. 时序平滑
- 已实现对 homography 轨迹的轻量平滑：
  - `src/stitching/temporal.py`
  - `scripts/run_baseline_video.py`
- 当前支持：
  - `--smooth_h none|ema|window`
  - `--smooth_alpha`
  - `--smooth_window`
- 已有产物：
  - `jitter_timeseries.csv`
  - `overlay_raw_*.png`
  - `overlay_sm_*.png`
  - `debug.json::jitter_summary`

### 2. seam + crop + blend
- 已实现 OpenCV 风格 seam pipeline：
  - seam 计算位置：低分辨率 warped ROI
  - seam 形式：二值 seam masks
  - resize 与落回 full canvas 的流程已实现
- 已实现 crop-before-seam：
  - `src/stitching/cropper.py`
  - `--crop/--no_crop`
  - `--lir_method auto|lir|fallback`
- 已实现 blend：
  - `none`
  - `feather`
  - `multiband`

### 3. frame0 / re-init reuse
- 已实现 `VideoStitcher` 缓存执行：
  - `src/stitching/video_state.py`
  - `src/stitching/video_stitcher.py`
- 已支持：
  - `frame0_all`
  - `frame0_geom`
  - `frame0_seam`
  - `emaH`
- 但这些模式的真实行为需结合 `03_baseline_video_pipeline` 文档理解，名称不等于最终研究语义。

## 当前未实现但必须澄清的边界
- 当前没有真正的 dynamic seam policy：
  - 没有 `fixed seam / keyframe seam / trigger seam` 的显式策略层。
  - 没有 seam temporal smoothing。
  - 没有 foreground / object-aware seam penalty。
- 当前没有 object-centered seam backend：
  - 没有 graph-cut / MRF energy
  - 没有 duplication / omission / occlusion-aware term
  - 没有 seam-specific temporal consistency term
- 当前没有完整实验驱动：
  - `scripts/ablate_seam.py`、`scripts/ablate_temporal.py` 存在，但当前只作为 legacy exploratory helpers
  - 它们不是 Phase 0 / Phase 1 的正式工作流入口
  - `scripts/ablate_crop.py` 不存在
  - `scripts/ablate_video_reuse.py` 不存在

## 关键校正：文档与代码漂移
- 旧文档中将 `scripts/ablate_crop.py`、`scripts/ablate_video_reuse.py` 写成“已完成”，这与仓库现状不一致。
- 从本次文档更新开始：
  - 这两个脚本视为 Phase 0 / Phase 3 的 planned items
  - 不再在 `ai-docs` 中标记为已实现
- 对已存在的 `scripts/ablate_seam.py`、`scripts/ablate_temporal.py`：
  - 保留为历史参考与便捷实验入口
  - 但不再作为当前 baseline freeze 或 Phase 1 接口设计的依赖项

## 当前质量提升路径的真实上限
- 时序平滑当前主要作用于几何轨迹，不直接平滑 seam。
- 当前 seam 改进主要是：
  - 避免 overlap 外错误混合
  - 减少黑边干扰
  - 提供 seam visibility 的 debug artefacts
- 这意味着当前系统仍可能出现：
  - object cropping
  - object duplication
  - omission
  - seam flicker
  - fixed-geometry 下的“指标看起来稳定，但视觉 artefact 仍存在”

## 与后续 Dynamic seam 的接口边界
- 当前可复用：
  - `warp_to_roi()`、`compute_seam_masks_opencv()`、`resize_seam_to_compose()`
  - `VideoStitcher` 的缓存与 re-init 机制
  - `debug.json / metrics_preview.json / snapshots` bundle
- 当前需要新增：
  - `SeamPolicy`
  - `SeamUpdateTrigger`
  - `SeamMaskSmoother`
  - 可选 `ForegroundMaskProvider`

## 与后续 temporal evaluation 的边界
- 当前 `jitter` 只在几何更新模式下有强解释力。
- `video_mode=1` 的固定几何运行更应该关注：
  - seam visibility
  - flicker / temporal coherence
  - object artefacts
  - overlap diff

## 当前建议
- 将 `03_baseline_video_pipeline` 作为 baseline as-built 文档。
- 将 `09_dynamic_seam_and_temporal_eval` 作为 Dynamic seam 与 temporal evaluation 的主设计文档。
- 将 `05_evaluation` 作为指标和实验协议的唯一事实来源。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/04_quality_improvement/04_quality_improvement.md | 修正质量改进文档与代码漂移，明确已实现边界与未实现项 | Codex | 完成 |
