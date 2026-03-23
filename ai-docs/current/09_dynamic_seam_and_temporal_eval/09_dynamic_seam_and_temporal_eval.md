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
- 不应把 `fixed_geometry + seam_policy=keyframe/trigger` 下的 `jitter=0` 误判为 seam 没有更新。

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

## geometry mode 与 seam policy 的关系（2026-03-20 更新）
- `geometry_mode`
  - 决定几何更新路径：
    - `fixed_geometry`
    - `keyframe_update`
    - `adaptive_update`（MVP 已实现为 seam-driven geometry refresh）
- `seam_policy`
  - 决定 seam 更新路径：
    - `fixed`
    - `keyframe`
    - `trigger`
- 两者是正交关系，不应再混为一个 `video_mode` 概念。
- 当前推荐理解方式：
  - `geometry_mode=fixed_geometry + seam_policy=fixed`
    - 固定几何、固定 seam
  - `geometry_mode=fixed_geometry + seam_policy=keyframe`
    - 固定几何、按 cadence 更新 seam
  - `geometry_mode=fixed_geometry + seam_policy=trigger`
    - 固定几何、按触发器更新 seam
  - `geometry_mode=keyframe_update + seam_policy=keyframe`
    - 几何与 seam 都可按各自 cadence 更新
  - `geometry_mode=adaptive_update + seam_policy=keyframe/trigger`
    - 仍走 cached reuse
    - 但当 seam event 命中时，同帧刷新 geometry 并重建 compose 状态
- keyframe 的职责拆分：
  - `keyframe_every`
    - geometry keyframe cadence
  - `seam_keyframe_every`
    - seam keyframe cadence

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

## 当前 MVP 实现状态（2026-03-20）
### 已落地的最小控制面
- 新增 `src/stitching/seam_policy.py`
  - `resolve_seam_policy()`
  - `resolve_seam_keyframe_every()`
  - `decide_seam_update()`
- `scripts/run_baseline_video.py` 已新增 CLI：
  - `--seam_policy=auto|fixed|keyframe|trigger`
  - `--seam_keyframe_every`
  - `--seam_trigger_overlap_ratio`
  - `--seam_trigger_diff_threshold`
- `src/stitching/video_stitcher.py` 仍复用 OpenCV seam backend，只在外层新增：
  - 是否重算 seam
  - 重算原因
  - 重算前后 overlap diff 统计
  - seam mask 变化比例

### meaningful temporal evaluation 的当前实现
- `src/stitching/temporal.py` 已补：
  - `compute_mask_change_ratio()`
  - `compute_frame_absdiff_mean()`
- `run_baseline_video.py` 当前会导出：
  - `mean_overlap_diff_before`
  - `mean_overlap_diff_after`
  - `mean_seam_mask_change_ratio`
  - `mean_stitched_delta`
  - `temporal_primary_metric`
  - `temporal_primary_value`
  - `jitter_scope`
  - `geometry_update_count`
  - `geometry_update_events`
- 当前主指标解释规则：
  - `fixed_geometry` -> `mean_overlap_diff_after`
  - `keyframe_update` -> `mean_jitter_sm`
  - `adaptive_update` -> `mean_jitter_sm`
- 当前 `jitter_scope` 解释规则：
  - `fixed_geometry` -> `geometry_only`
  - `keyframe_update` -> `geometry_stream`
  - `adaptive_update` -> `geometry_stream`

### 当前验证结果
- `phase2_seam_fixed_smoke`
  - `geometry_mode=fixed_geometry`
  - `seam_policy=fixed`
  - `seam_recompute_count=1`
  - `temporal_primary_metric=mean_overlap_diff_after`
- `phase2_seam_keyframe_smoke`
  - `seam_policy=keyframe`
  - `seam_keyframe_every_effective=5`
  - `seam_recompute_count=4`
- `phase2_seam_trigger_smoke_v2`
  - `seam_policy=trigger`
  - `seam_recompute_count=8`
  - 已验证 `trigger_diff>=6.300` 可以触发多次 seam 重算
- `phase2_temporal_keyframeupdate_smoke`
  - `geometry_mode=keyframe_update`
  - `jitter_meaningful=1`
  - `temporal_primary_metric=mean_jitter_sm`
- `phase2_kitti0002_fixedgeom_fixed`
  - `geometry_mode=fixed_geometry`
  - `geometry_keyframe_every_effective=0`
  - `seam_policy=fixed`
  - `seam_recompute_count=1`
- `phase2_kitti0002_fixedgeom_keyframe`
  - `geometry_mode=fixed_geometry`
  - `seam_policy=keyframe`
  - `seam_keyframe_every_effective=10`
  - `seam_recompute_count=8`
- `phase2_kitti0002_fixedgeom_trigger`
  - `geometry_mode=fixed_geometry`
  - `seam_policy=trigger`
  - `seam_trigger_diff_threshold=21.0`
  - `seam_recompute_count=11`
- `phase2_kitti0002_adaptive_keyframe`
  - `geometry_mode=adaptive_update`
  - `seam_policy=keyframe`
  - `seam_recompute_count=8`
  - `geometry_update_count=7`
  - `mean_jitter_sm=0.6423`
- `phase2_kitti0002_adaptive_trigger`
  - `geometry_mode=adaptive_update`
  - `seam_policy=trigger`
  - `seam_trigger_diff_threshold=21.0`
  - `seam_recompute_count=3`
  - `geometry_update_count=2`
  - `mean_jitter_sm=0.3334`

### seam 重算事件图
- 新增开关：
  - `--seam_snapshot_on_recompute`
- 目的：
  - 当 seam 发生 keyframe/trigger 重算时，为对应帧留下证据图，不再只依赖 `snapshot_every`。
- 当前保存内容：
  - stitched frame snapshot
  - seam event low-res mask / overlay / overlap diff 图
- 在 `phase2_kitti0002_fixedgeom_keyframe_snap` 上：
  - `seam_recompute_count=8`
  - `seam_snapshot_count=7`
  - 首个 `seam_event_*` 文件对应 `frame_idx=10`
- 在 `phase2_kitti0002_fixedgeom_trigger_snap` 上：
  - `seam_recompute_count=11`
  - `seam_snapshot_count=10`
  - 首个 `seam_event_*` 文件对应 `frame_idx=21`

### 当前边界
- 本批实现只做 seam policy 外壳，不重写 `compute_seam_masks_opencv()`。
- `adaptive_update` 当前只是最小版：
  - seam event 驱动 geometry refresh
  - 不是完整的自适应 geometry controller
- 还没有 seam temporal smoothing。

## 当前 Phase 2 第三批落地（2026-03-23）
### 已实现的新增控制面
- `run_baseline_video.py` 已新增：
  - `--seam_trigger_foreground_ratio`
  - `--seam_trigger_cooldown_frames`
  - `--seam_trigger_hysteresis_ratio`
  - `--foreground_mode=off|disagreement`
  - `--foreground_diff_threshold`
  - `--foreground_dilate`
- 已新增 `src/stitching/foreground.py`
  - `compute_disagreement_mask()`
  - `compute_mask_ratio()`
  - `apply_protect_mask_assignment()`
- 当前 `foreground_mode=disagreement` 的语义：
  - 用 overlap ROI 的 cross-view absdiff 构造 protected region
  - protected ratio 同时参与 trigger 判断
  - final seam mask 再做一次兼容式 reassignment，尽量避免 seam 穿过 protected region

### 本轮没有执行的内容
- 没有接 detector / segmentation 模型。
- 没有重写 OpenCV seam backend。
- 没有直接复现 object-centered MRF seam。
- 原因：
  - 当前 `mine_source_*` pairs 没有现成 object masks
  - 本轮更适合先验证兼容式 foreground-aware MVP 是否值得继续推进

### 当前 calibration suite
- 入口：
  - `scripts/run_phase2_trigger_calibration.py`
- 正式 suite：
  - `outputs/video_calibration/phase2_trigger_adaptive_minesource_calib_v2`
- 覆盖 pairs：
  - `mine_source_mcd1_left_right`
  - `mine_source_mcd2_left_right`
  - `mine_source_square_left_right`
  - `mine_source_traffic1_left_right`
  - `mine_source_traffic2_left_right`
  - `mine_source_walking_left_right`
- preset 梯度：
  - `trigger_plain_d18`
  - `trigger_fused_d18_fg008`
  - `trigger_stable_d18_fg008_cd6_h075`
  - `adaptive_fused_d18_fg008`
  - `adaptive_stable_d18_fg008_cd6_h075`

### 当前 calibration 结论
- `trigger_plain_d18`
  - `mean_overlap_diff_after ≈ 6.29`
  - `seam_recompute_after_init_per_100f ≈ 0.42`
- `trigger_fused_d18_fg008`
  - `mean_overlap_diff_after ≈ 3.75`
  - `seam_recompute_after_init_per_100f ≈ 1.25`
  - `approx_fps ≈ 10.30`
- `trigger_stable_d18_fg008_cd6_h075`
  - `mean_overlap_diff_after ≈ 3.75`
  - 但 `seam_recompute_after_init_per_100f = 0`
  - 说明当前全局 `cooldown + hysteresis` 已把 trigger 压成近似一次性事件
- `adaptive_fused_d18_fg008`
  - `geometry_update_per_100f ≈ 1.25`
  - `approx_fps ≈ 7.63`
  - 当前没有显示出比 `trigger_fused` 更好的 `mean_overlap_diff_after`
- `adaptive_stable_d18_fg008_cd6_h075`
  - `geometry_update_per_100f = 0`
  - 在这批 `mine_source` 视频上过于保守，不适合作为默认 preset

### 当前推荐配置
- 当前 Phase 2 默认 seam preset，优先推荐：
  - `geometry_mode=fixed_geometry`
  - `seam_policy=trigger`
  - `seam_trigger_diff_threshold=18`
  - `foreground_mode=disagreement`
  - `seam_trigger_foreground_ratio=0.08`
- 原因：
  - 比 `trigger_plain_d18` 明显降低 `mean_overlap_diff_after`
  - 比 `adaptive_fused` 更快
  - 比 `stable` 版本更确实会在 init 后发生 seam 重算

### 当前暴露出的结构性问题
- 当前 `multi-trigger fusion` 是全局 OR 触发，但 `trigger_armed / hysteresis` 也是全局共享的。
- 在 `foreground_ratio` 长时间维持高位的真实视频上，这会导致：
  - 首次触发后长期无法 re-arm
  - `adaptive_update` 退化成“最多在开头补一次 geometry refresh”
- 因此本轮结论不是“adaptive_update 无效”，而是：
  - 当前全局 armed/hysteresis 设计对 sustained foreground 场景过于保守
  - 下一步应优先考虑 per-trigger rearm、foreground-specific cooldown 或更细的 trigger fusion 策略

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/09_dynamic_seam_and_temporal_eval/09_dynamic_seam_and_temporal_eval.md | 新增 Dynamic seam 与 temporal evaluation 专项设计文档 | Codex | 完成 |
