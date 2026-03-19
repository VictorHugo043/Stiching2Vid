# 09_dynamic_seam_and_temporal_eval

## 任务目标
- 给 Dynamic seam 与 meaningful temporal evaluation 提供可直接实施的专项设计文档。

## 当前实现位置（必须明确）
### fixed seam / frame0 reuse / fixed geometry 在哪里
- `src/stitching/video_stitcher.py`
  - `initialize_from_first_frame()`
    - 初始化并缓存 `H_or_cameras`
    - 缓存 low seam masks
    - 缓存 cropper 与 overlap metadata
  - `stitch_frame()`
    - 默认复用缓存 seam
    - `recompute_seam=True` 时才刷新当前帧 seam
- `scripts/run_baseline_video.py`
  - `video_mode=1`
    - 使用 `video_stitcher.state.H_or_cameras`
    - `frame0_geom` 当前只触发 seam refresh
    - 并不意味着 geometry per-frame update

## 为什么当前 `jitter` 会退化为 0
- `jitter` 当前定义为连续两帧变换后四角点位移。
- 当 `H` 长时间固定时：
  - `transform_corners(H)` 输出近似不变
  - `compute_jitter()` 返回接近 0
- 因此 `video_mode=1` 下大量 run 的 `mean_jitter_raw/sm = 0` 并不等价于“视觉更稳定”。

## 建议的显式运行模式
- `fixed_geometry`
  - 适合 frame0/reuse 路线
  - 不把 `jitter` 当主指标
- `keyframe_update`
  - 定期更新几何
  - 可用 `jitter`
- `adaptive_update`
  - 根据 trigger 自适应更新几何和 seam
  - 是 Phase 2 的目标模式

## Dynamic seam 两层方案
### 1. MVP 轻量版
#### 核心策略
- `fixed seam`
- `keyframe seam`
- `trigger seam`

#### 建议新增组件
- `SeamPolicy`
  - 决定当前帧是复用还是重算 seam
- `SeamUpdateTrigger`
  - 触发条件：
    - overlap area 明显变化
    - 几何 re-init / keyframe update
    - foreground ratio 变化
    - seam visibility / overlap diff 激增
- `SeamMaskSmoother`
  - 对 seam mask 做轻量时序平滑

#### 与现有 OpenCV seam finder 的兼容方式
- 保持 `compute_seam_masks_opencv()` 作为 seam backend。
- Dynamic seam 第一版只在它外面加：
  - 何时更新
  - 如何限制 seam 不穿过动态对象
  - 如何对 seam 结果做 temporal smoothing

#### foreground / object-aware penalty 的兼容式做法
- 第一优先级：
  - 利用 DynamicStereo 已有 `masks/`
- 第二优先级：
  - overlap ROI 内的前景变化检测 / absdiff / motion proxy
- 第一版不直接改 OpenCV seam energy，而是：
  - 对 object 区域做保护带
  - 对 seam 候选区域做限制 / fallback

#### previous seam template reuse
- 复用上一个 seam mask 或 seam centerline 模板。
- 仅在 trigger 命中时重新计算 seam。

#### seam temporal smoothing
- 第一版可选：
  - mask-level EMA
  - majority vote over recent seam masks
  - distance transform based seam blending band smoothing

### 2. Advanced 增强版
#### 目标
- 逐步脱离纯 OpenCV seam finder，进入自定义 seam backend。

#### 建议能力
- overlap ROI 内局部 graph-cut
- object-centered seam cost
- duplication / omission / occlusion-aware penalty
- 更强 temporal consistency

#### 与 ECCV 2018 的关系
- 吸收 object-centered seam 的思想：
  - cropping
  - duplication
  - occlusion
- 但不要求一步到位完整复现其 MRF 系统。

## 如何恢复 meaningful temporal evaluation
### 当前应避免的误用
- 不应在 `fixed_geometry` 模式下把 `jitter` 写成主要稳定性指标。

### 建议增加的 temporal 指标
- seam visibility over time
- overlap ROI flicker
- optical-flow compensated temporal coherence
- object-region temporal consistency

### 指标解释约束
- `jitter`
  - 只在 `keyframe_update / adaptive_update` 下为主指标
- `fixed_geometry`
  - 主要看 seam / flicker / temporal coherence / object artefacts

## 建议配置项
- `geometry_mode`
- `seam_policy`
- `seam_trigger_overlap_ratio`
- `seam_trigger_diff_threshold`
- `seam_smooth_method`
- `seam_smooth_window`
- `foreground_mode`
- `foreground_mask_source`

## 验收标准(DoD)
- 文档和实现层都能清楚区分：
  - `fixed_geometry`
  - `keyframe_update`
  - `adaptive_update`
- MVP 支持：
  - `fixed seam`
  - `keyframe seam`
  - `trigger seam`
- temporal evaluation 不再把 0-jitter 误解释为稳定性提升。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/09_dynamic_seam_and_temporal_eval/09_dynamic_seam_and_temporal_eval.md | 新增 Dynamic seam 与 temporal evaluation 专项设计文档 | Codex | 完成 |
