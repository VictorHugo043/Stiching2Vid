# 03_baseline_video_pipeline

## 任务目标
- 固化当前视频级 baseline pipeline 的 as-built 行为，作为 Phase 0 冻结基线与后续 Method B / Dynamic seam 改造的事实依据。

## 当前实现状态（代码对齐版）
- 主入口：`scripts/run_baseline_video.py`。
- 当前存在两条主路径：
  - `video_mode=0`：关键帧重估几何，非关键帧复用最近有效 `H`；支持 `smooth_h=none|ema|window`。
  - `video_mode=1`：通过 `src/stitching/video_stitcher.py` 做 frame0 / re-init 缓存执行，缓存几何、cropper、seam masks 和 metadata。
- 统一处理链路：
  - `geometry`: `src/stitching/geometry.py`
  - `temporal`: `src/stitching/temporal.py`
  - `seam`: `src/stitching/seam_opencv.py`
  - `crop`: `src/stitching/cropper.py`
  - `video reuse`: `src/stitching/video_state.py` + `src/stitching/video_stitcher.py`
  - frame-pair geometry adapter：
    - `src/stitching/frame_pair_pipeline.py`

## 当前 pipeline 真实行为
### 1. `video_mode=0`（baseline / keyframe update）
- 第 0 帧和每个 `keyframe_every` 帧重新做 `detect -> match -> findHomography`。
- 非关键帧复用最近一次有效 `H_raw`。
- `smooth_h` 在这一路径上是有意义的，因为输入的 `H_raw` 会随关键帧更新而变化。
- seam/crop 逻辑主要内联在 `scripts/run_baseline_video.py` 中，非关键帧复用 seam cache。

### 2. `video_mode=1`（frame0 reuse）
- 初始化或 re-init 时才做特征、匹配和单应估计，然后调用 `VideoStitcher.initialize_from_first_frame()`。
- 后续帧通过 `VideoStitcher.stitch_frame()` 复用缓存状态，几何主值来自 `video_stitcher.state.H_or_cameras`。
- re-init 触发条件当前只有三类：
  - 首帧未初始化
  - `reinit_every > 0`
  - `reinit_on_low_overlap_ratio > 0`
- 当显式使用 `geometry_mode=adaptive_update` 时：
  - 仍走 cached reuse 主路径
  - 但如果 `seam_policy=keyframe/trigger` 命中 seam event，会在同一帧追加一次 geometry refresh
  - 然后用新 `H` 调用 `VideoStitcher.initialize_from_first_frame()` 重建当前帧 compose 状态

## 当前推荐的用户入口（2026-03-20 更新）
- `geometry_mode` 现在是推荐使用的显式配置层：
  - `fixed_geometry`
  - `keyframe_update`
  - `adaptive_update`（最小版已实现为 seam-driven geometry refresh）
- `video_mode` 现在只保留为 legacy 兼容别名：
  - `video_mode=1` <-> `geometry_mode=fixed_geometry`
  - `video_mode=0` <-> `geometry_mode=keyframe_update`
- 当前最佳实践：
  - 新 run 优先传 `--geometry_mode`
  - 只有兼容旧脚本或历史命令时才继续显式传 `--video_mode`

## geometry keyframe 与 seam keyframe 的职责边界
- `geometry_mode`
  - 决定几何更新路径。
- `keyframe_every`
  - 只控制 geometry keyframe cadence。
  - 仅当 `geometry_mode=keyframe_update` 时有效。
  - 当 `geometry_mode=fixed_geometry` 时，`geometry_keyframe_every_effective=0`。
- `seam_policy`
  - 决定 seam 是固定复用、按 cadence 更新，还是按 trigger 更新。
- `seam_keyframe_every`
  - 只控制 seam 的 keyframe cadence。
  - 与 geometry keyframe 解耦。
  - 即使 `geometry_mode=fixed_geometry`，也可以使用 `seam_policy=keyframe`。

## 关键校正：`reuse_mode` 的真实语义
- `frame0_all`
  - 当前语义是固定几何 + 固定 crop + 固定 seam mask，直到触发 re-init。
- `frame0_geom`
  - 当前名字容易误导。
  - 实际行为是固定几何，但允许对当前帧重新计算 seam；并不是“每帧更新 geometry”。
- `frame0_seam`
  - 当前实现里没有独立于 `frame0_all` 的稳定专属路径。
  - 在默认流程下它与 `frame0_all` 几乎等价，通常仍复用缓存 seam mask。
- `emaH`
  - 当前对缓存 `H_or_cameras` 做平滑；如果几何本身不变化，平滑后的轨迹仍接近常量。

## 当前导出 artefacts
- `outputs/runs/<run_id>/`
  - `stitched.mp4`
  - `transforms.csv`
  - `metrics_preview.json`
  - `debug.json`
  - `jitter_timeseries.csv`
  - `logs.txt`
  - `snapshots/`
- `transforms.csv` 现有重点字段：
  - 匹配统计：`n_kp_left/right`、`n_matches_raw/good`、`n_inliers`、`inlier_ratio`
  - 几何轨迹：`H_*`、`Hraw_*`、`Hsm_*`
  - 时序项：`jitter_raw`、`jitter_sm`、`H_delta_norm`
  - 运行模式：`video_mode`、`reuse_mode`
  - seam/crop 相关：`overlap_area_current`、`crop_applied`、`crop_method`
  - Phase 2 MVP 新增：
    - `seam_policy`
    - `seam_recomputed`
    - `geometry_recomputed`
    - `geometry_update_reason`
    - `overlap_diff_before`
    - `overlap_diff_after`
    - `stitched_delta_mean`
    - `seam_mask_change_ratio`

## 已确认的关键限制与耦合点
- `scripts/run_baseline_video.py` 目前是大脚本，混合了：
  - CLI
  - 特征/匹配/几何 orchestration
  - seam/crop 细节
  - diagnostics 导出
  - video reuse 分支
- `video_mode=0` 与 `video_mode=1` 存在 seam/crop 逻辑重复。
- `VideoStitcher` 只负责 `warp -> crop -> seam -> blend`，不负责几何估计；这一点对 Method B 接入是利好。
- 当前 `run_baseline_video.py` 已通过 `frame_pair_pipeline` 接到结果对象层：
  - 关键帧几何估计不再直接依赖 legacy tuple/OpenCV 接口
  - `VideoStitcher` 仍只消费 `H / T / canvas_size`
  - 因此视频 compose/cache 语义保持不变，Method A / Method B 只替换前端的 frame-pair estimation
- 同一条 frame-level compose 路径现已通过 `src/stitching/frame_quality_preview.py` 复用到 `scripts/run_baseline_frame.py`，用于单帧静态质量预览。
- 当前 seam backend 是 OpenCV seam mask 风格，不是 object-centered energy / graph-cut 风格。

## Phase 1 新增状态（2026-03-20）
- `scripts/run_baseline_video.py` 现已支持：
  - `feature_backend`
  - `matcher_backend`
  - `geometry_backend`
  - `device`
  - `force_cpu`
  - `weights_dir`
  - `max_keypoints`
  - `resize_long_edge`
  - `depth_confidence`
  - `width_confidence`
  - `filter_threshold`
  - `feature_fallback_backend`
  - `matcher_fallback_backend`
- 已验证短视频 Method B smoke：
  - `phase1_video_adapter_methodb_mode0_smoke_v2`
    - `geometry_mode=keyframe_update`
    - `feature_backend_effective=superpoint`
    - `matcher_backend_effective=lightglue`
    - `geometry_backend_effective=opencv_usac_magsac`
  - `phase1_video_adapter_methodb_mode1_smoke_v2`
    - `geometry_mode=fixed_geometry`
    - `feature_backend_effective=superpoint`
    - `matcher_backend_effective=lightglue`
    - `geometry_backend_effective=opencv_usac_magsac`

## `jitter` 失真条件（必须冻结到文档）
- `jitter` 当前由 `src/stitching/temporal.py::compute_jitter()` 对连续两帧变换后四角点位移计算。
- 在 `video_mode=1` 且几何长期固定的场景下：
  - `raw_corners` 和 `sm_corners` 近似不变；
  - `jitter_raw`、`jitter_sm` 会系统性退化到 0 或接近 0。
- 因此：
  - `fixed geometry` 运行不应把 `jitter` 当作核心时序质量指标；
  - 后续必须显式区分 `fixed_geometry / keyframe_update / adaptive_update`。

## Phase 0 文档冻结项
- 冻结当前 as-built 行为，不再把 `frame0_geom` 解释成“几何更新模式”。
- 冻结当前导出 bundle 的核心字段，避免后续 Method B / Dynamic seam 改造时破坏已有实验可比性。
- 当前 `scripts/run_baseline_video.py` 已显式导出：
  - `geometry_mode`
  - `jitter_meaningful`
- 当前实现已可导出：
  - `fixed_geometry`
  - `keyframe_update`
  - `adaptive_update`
- 但要注意：
  - `adaptive_update` 当前仅表示“seam 事件驱动的 geometry refresh”
  - 还不是完整的自适应几何控制器
- 在文档层先引入三类运行模式：
  - `fixed_geometry`
  - `keyframe_update`
  - `adaptive_update`

## 已知问题
- 视差和动态目标仍会导致 ghosting、cropping、duplication 等 artefacts。
- `video_mode=1` 下的时序指标存在解释风险。
- 当前没有 GUI，也没有统一 experiments driver。

## Phase 2 MVP 新增状态（2026-03-20）
- 当前没有重写 `src/stitching/seam_opencv.py` backend。
- 只新增了 seam 更新控制壳层：
  - `src/stitching/seam_policy.py`
  - `--seam_policy=auto|fixed|keyframe|trigger`
  - `--seam_keyframe_every`
  - `--seam_trigger_overlap_ratio`
  - `--seam_trigger_diff_threshold`
- `video_mode=0` 与 `video_mode=1` 现在都能显式导出：
  - seam 是否重算
  - 重算原因
  - seam mask 变化比例
  - overlap diff 的 before/after
- 当前 `metrics_preview.json` 已新增：
  - `mean_overlap_diff_before`
  - `mean_overlap_diff_after`
  - `mean_seam_mask_change_ratio`
  - `mean_stitched_delta`
  - `temporal_primary_metric`
  - `temporal_primary_value`
  - `jitter_scope`
  - `seam_policy`
  - `seam_keyframe_every_effective`
  - `seam_recompute_count`
  - `seam_snapshot_count`
  - `geometry_update_count`
  - `adaptive_update_strategy`
- 当前解释约束：
  - `fixed_geometry` 下主 temporal 指标改为 `mean_overlap_diff_after`
  - `keyframe_update` 下主 temporal 指标仍为 `mean_jitter_sm`
  - `adaptive_update` 下主 temporal 指标当前也为 `mean_jitter_sm`
  - `fixed_geometry` 下 `jitter_scope=geometry_only`
    - seam 的重算不会改变 `jitter`
    - 应结合 `seam_recompute_count / seam_mask_change_ratio / overlap_diff_after` 解读
  - `adaptive_update` 下 `jitter_scope=geometry_stream`
    - 应结合 `geometry_update_count / geometry_update_events / transforms.csv::geometry_recomputed` 解读

## Phase 2 模式语义验证（KITTI 0002, 80 frames, fps=10）
- pair：
  - `kitti_raw_data_2011_09_26_drive_0002_image_02_image_03`
- 统一运行前提：
  - `geometry_mode=fixed_geometry`
  - Method B：`superpoint + lightglue + opencv_usac_magsac`
- 验证结果：
  - `seam_policy=fixed`
    - `video_mode=1`
    - `geometry_keyframe_every_effective=0`
    - `seam_recompute_count=1`
  - `seam_policy=keyframe`
    - `seam_keyframe_every_effective=10`
    - `seam_recompute_count=8`
    - 首个 keyframe seam event 出现在 `frame_idx=10`
  - `seam_policy=trigger`
    - `seam_trigger_diff_threshold=21.0`
    - `seam_recompute_count=11`
    - 首个 trigger 事件出现在 `frame_idx=21`
  - `geometry_mode=adaptive_update + seam_policy=keyframe`
    - `seam_recompute_count=8`
    - `geometry_update_count=7`
    - `mean_jitter_sm=0.6423`
  - `geometry_mode=adaptive_update + seam_policy=trigger`
    - `seam_recompute_count=3`
    - `geometry_update_count=2`
    - `mean_jitter_sm=0.3334`
- 结论：
  - 当前代码已经能清楚证明：
    - `geometry_mode` 决定几何路径
    - `seam_policy` 决定 seam 更新路径
    - 两者已从配置语义上正交化
    - `fixed_geometry` 下 `keyframe/trigger` 的 `jitter=0` 并不表示 seam 没有更新，只表示几何没有更新
    - `adaptive_update` 已能把 seam event 转成同帧 geometry refresh，但当前仍属于 MVP

## seam 事件 snapshot（2026-03-20 更新）
- 新增开关：
  - `--seam_snapshot_on_recompute {0,1}`
- 当前行为：
  - 当 seam 重算时，保存对应帧 stitched snapshot：
    - `frame_<idx>_stitched.png`
  - 在 `video_mode=1 / fixed_geometry` 路径上，额外保存：
    - `seam_event_<idx>_mask_left_roi.png`
    - `seam_event_<idx>_mask_right_roi.png`
    - `seam_event_<idx>_seam_mask_left.png`
    - `seam_event_<idx>_seam_mask_right.png`
    - `seam_event_<idx>_seam_overlay.png`
    - `seam_event_<idx>_overlap_diff.png`
- 这套图与 `snapshot_every` 解耦，不再依赖碰巧命中定时 snapshot。

## trigger / foreground / adaptive 校准更新（2026-03-23）
- 当前 video pipeline 已新增：
  - `--seam_trigger_foreground_ratio`
  - `--seam_trigger_cooldown_frames`
  - `--seam_trigger_hysteresis_ratio`
  - `--foreground_mode=off|disagreement`
  - `--foreground_diff_threshold`
  - `--foreground_dilate`
- `foreground_mode=disagreement` 当前不引入新 detector：
  - 只在 overlap ROI 上用 cross-view absdiff 构造 protected region
  - 同时影响 trigger 判定与 final seam mask reassignment
- 当前校准结果表明：
  - `trigger_fused_d18_fg008` 是现阶段更合适的默认 seam preset
  - `adaptive_fused_d18_fg008` 已能触发 geometry refresh，但当前主要发生在开头，且 runtime 代价更高
  - `trigger_stable_d18_fg008_cd6_h075` 与 `adaptive_stable_d18_fg008_cd6_h075` 在真实 `mine_source` 视频上过于保守
- 当前结构性问题：
  - `trigger_armed / hysteresis` 仍是全局共享状态
  - 当 `foreground_ratio` 长时间高位时，trigger 容易退化成“一次触发后长期不上膛”

## 下一步
- 总路线见 `ai-docs/current/08_project_status_and_master_plan/08_project_status_and_master_plan.md`。
- Dynamic seam 与 temporal evaluation 方案见 `ai-docs/current/09_dynamic_seam_and_temporal_eval/09_dynamic_seam_and_temporal_eval.md`。
- 评测协议见 `ai-docs/current/05_evaluation/05_evaluation.md`。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/03_baseline_video_pipeline/03_baseline_video_pipeline.md | 按当前代码重写 video pipeline 文档并修正 `reuse_mode` / `jitter` 语义 | Codex | 完成 |
