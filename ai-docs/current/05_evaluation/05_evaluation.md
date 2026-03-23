# 05_evaluation

## 任务目标
- 固定后续 final project 的评测协议，确保 Method A / Method B、不同 seam policy、不同 temporal 模式和 GUI 封装都沿用同一套输入、输出和比较维度。

## 实验主轴
### 1. 方法主轴
- `Method A`
  - ORB / SIFT
  - KNN + ratio test
  - RANSAC homography
- `Method B`
  - SuperPoint
  - LightGlue
  - MAGSAC++

### 2. seam 主轴
- `fixed seam`
- `keyframe seam`
- `trigger seam`

### 3. temporal 主轴
- `no smoothing`
- `EMA`
- `window`

### 4. geometry mode 主轴
- `fixed_geometry`
- `keyframe_update`
- `adaptive_update`

## 当前可直接复用的输出与指标
### 现有输出 bundle
- `metrics_preview.json`
- `debug.json`
- `transforms.csv`
- `jitter_timeseries.csv`
- `snapshots/`
- 当前 `run_baseline_video.py` 已补充导出：
  - `geometry_mode`
  - `jitter_meaningful`
  - `reuse_mode`（在 `metrics_preview.json` 中显式写出）
  - `seam_policy`
  - `seam_keyframe_every_effective`
  - `seam_recompute_count`
  - `geometry_update_count`
  - `adaptive_update_strategy`
  - `transforms.csv::geometry_recomputed`
  - `transforms.csv::geometry_update_reason`

### 现有可复用指标
- runtime / FPS
  - `avg_runtime_ms`
  - `approx_fps`
- 匹配统计
  - `n_matches_raw`
  - `n_matches_good`
  - `n_inliers`
  - `inlier_ratio`
- overlap 相关
  - `overlap_area_current`
  - `overlap_diff_mean_before`
  - `overlap_diff_mean_after`
  - `mean_overlap_diff_before`
  - `mean_overlap_diff_after`
- 时序项
  - `jitter_raw`
  - `jitter_sm`
  - `p95_jitter_raw`
  - `p95_jitter_sm`
  - `jitter_meaningful`
  - `jitter_scope`
  - `seam_mask_change_ratio`
  - `stitched_delta_mean`
  - `temporal_primary_metric`
  - `temporal_primary_value`

## 必须新增的指标
### 几何质量
- reprojection error
  - 基于最终内点集的 mean / median reprojection error
- boundary misalignment
  - overlap / seam band 内边界错位统计

### seam / blending 质量
- seam visibility
  - seam band 内颜色差 / 梯度差
- flicker / temporal coherence
  - overlap ROI 或 seam band 的 frame-to-frame coherence

### object artefacts
- cropping / omission / duplication
  - 优先用 DynamicStereo masks 或后续 detector 支持
- object-count based omission / duplication
  - 作为 object-aware secondary metrics
- object-region MS-SSIM 或 template matching
  - 作为局部质量补充，不作为唯一主指标
- optical-flow compensated temporal coherence
  - 作为 advanced temporal metric，优先在 seam band 或 overlap ROI 内计算

## 使用规则：哪些模式下哪些指标有效
- `jitter`
  - 仅用于 `keyframe_update`、`adaptive_update`
  - 不作为 `fixed_geometry` 的主指标
- `temporal_primary_metric`
  - `fixed_geometry` 当前默认为 `mean_overlap_diff_after`
  - `keyframe_update` 当前默认为 `mean_jitter_sm`
  - `adaptive_update` 当前也默认为 `mean_jitter_sm`
- `jitter_scope`
  - `geometry_only` 表示当前 `jitter` 只反映 geometry 变化，不反映 seam 更新
  - `geometry_stream` 表示当前 `jitter` 反映几何流本身，适用于 `keyframe_update / adaptive_update`
- seam visibility / flicker / temporal coherence
  - 在所有 seam 模式下都可比较
- object artefacts
  - 优先在 DynamicStereo 或有 object mask / detector 支持的数据上报告

## 变量固定原则
- 必须固定：
  - pair 列表
  - `start / max_frames / stride`
  - 输出分辨率 / resize 策略
  - crop on/off
  - blend mode
  - `keyframe_every`
  - `geometry_mode`
  - hardware / device
  - Method B 的 keypoint 上限和 robust estimator 参数
- 只允许单轴变化：
  - 方法变化时，其余 seam / temporal / geometry mode 固定
  - seam 变化时，方法和 geometry mode 固定
  - smoothing 变化时，方法、seam、geometry mode 固定

## 推荐实验矩阵
### Phase 0 / Phase 1
- `Method A vs Method B`
- 固定：
  - 正式视频级 compare 入口：`scripts/run_video_compare_suite.py`
  - 方法集合：
    - `method_a_orb`
    - `method_a_sift`
    - `method_b`
  - 固定运行参数：
    - `video_mode=1`
    - `reuse_mode=frame0_all`
    - `max_frames=6000`
    - 不依赖 keyframe 更新
  - 解读边界：
    - 这是 Phase 1 的 fixed-geometry compare preset，用于验证 Method B 已完整接到视频质量链路
    - `jitter` 在这一 preset 下不作为主比较指标

### Phase 2
- 在最佳方法下比较：
  - `fixed seam vs keyframe seam vs trigger seam`
  - `no smoothing vs EMA vs window`
  - `fixed_geometry vs keyframe_update vs adaptive_update`
- 当前最小验证已完成：
  - `phase2_seam_fixed_smoke`
  - `phase2_seam_keyframe_smoke`
  - `phase2_seam_trigger_smoke_v2`
  - `phase2_temporal_keyframeupdate_smoke`
- 当前已验证的最小结论：
  - `fixed seam` 只在初始化时重算 seam
  - `keyframe seam` 可按 cadence 重算
  - `trigger seam` 可按 `overlap_diff_before` 阈值触发
  - `fixed_geometry` 下可用 `mean_overlap_diff_after` 代替退化 `jitter` 作为主 temporal 指标
  - `fixed_geometry + seam_policy=keyframe/trigger` 下即使 `jitter=0` 也不代表 seam 没更新，应结合 `jitter_scope / seam_recompute_count / seam_snapshot_count` 解读
  - `adaptive_update` 当前已最小落地为 seam-driven geometry refresh，应结合 `geometry_update_count / geometry_update_events / transforms.csv::geometry_recomputed` 解读

### Phase 3
- 在代表性 pair 上做系统矩阵：
  - DynamicStereo 动态样例
  - `mine_source` 真实视频
  - 至少 1 组较稳定的静态样例

## 可视化与 final report 保留项
- 必保留图：
  - 匹配图与 inlier 图
  - seam overlay
  - overlap diff
  - `overlay_raw` vs `overlay_sm`
  - 代表性 stitched frame 对比
- 必保留表：
  - 方法对比总表
  - seam ablation 表
  - temporal ablation 表
  - runtime / quality trade-off 表

## 当前脚本与后续建议
- 现有：
  - `scripts/ablate_temporal.py`
  - `scripts/ablate_seam.py`
- 当前定位：
  - 这两个脚本保留为 legacy exploratory helpers
  - 不作为 Phase 0 / Phase 1 的正式实验入口或验收前提
  - 当前正式入口仍以 `scripts/run_baseline_video.py` 的统一 bundle 为准
- 后续建议新增：
  - 统一 experiment driver
  - 独立 metrics 模块
  - summary CSV / plots 生成脚本

## 运行参数解释约束（2026-03-20 更新）
- 新实验应优先使用：
  - `--geometry_mode`
  - `--seam_policy`
- `--video_mode`
  - 仅保留为 legacy 兼容入口，不再建议作为新实验的主配置项。
- `--keyframe_every`
  - 只表示 geometry keyframe cadence。
- `--seam_keyframe_every`
  - 只表示 seam keyframe cadence。
- 因此：
  - `geometry_mode=fixed_geometry + seam_policy=keyframe`
    - 是合法且有意义的组合
    - 表示固定几何，但按 cadence 刷新 seam

## 当前 Phase 1 正式 compare 产物（2026-03-20）
- compare driver：
  - `scripts/run_video_compare_suite.py`
- 当前正式 suite：
  - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/summary.csv`
  - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/summary.json`
  - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/pair_compare.csv`
- 当前代表性结论：
  - 3 个 pair、3 个方法预设共 9 条 full-length run 全部成功完成
  - `fallback_frames=0`、`errors_count=0`
  - ORB / SIFT / Method B 在不同数据上各有优势，因此 Phase 1 不应提前删掉 ORB 或 SIFT 这两个 Method A 预设

## 验收标准(DoD)
- 为每个实验输出统一 bundle。
- 至少固定一套 final report 可复用的表格与图像清单。
- 文档中明确哪些指标当前可直接复用，哪些需要新增实现。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/05_evaluation/05_evaluation.md | 将骨架文档升级为正式评测协议与实验矩阵 | Codex | 完成 |
