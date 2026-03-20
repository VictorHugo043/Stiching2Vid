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
- 当前实现实际只会导出：
  - `fixed_geometry`
  - `keyframe_update`
- `adaptive_update` 目前仍是文档保留模式，不代表已经在代码中实现。
- 在文档层先引入三类运行模式：
  - `fixed_geometry`
  - `keyframe_update`
  - `adaptive_update`

## 已知问题
- 视差和动态目标仍会导致 ghosting、cropping、duplication 等 artefacts。
- `video_mode=1` 下的时序指标存在解释风险。
- 当前没有 GUI，也没有统一 experiments driver。

## 下一步
- 总路线见 `ai-docs/current/08_project_status_and_master_plan/08_project_status_and_master_plan.md`。
- Dynamic seam 与 temporal evaluation 方案见 `ai-docs/current/09_dynamic_seam_and_temporal_eval/09_dynamic_seam_and_temporal_eval.md`。
- 评测协议见 `ai-docs/current/05_evaluation/05_evaluation.md`。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/03_baseline_video_pipeline/03_baseline_video_pipeline.md | 按当前代码重写 video pipeline 文档并修正 `reuse_mode` / `jitter` 语义 | Codex | 完成 |
