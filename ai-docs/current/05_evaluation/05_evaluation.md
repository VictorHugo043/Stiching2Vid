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
  - `mean_foreground_ratio`
  - `foreground_triggered_count`
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
  - Method B 的 `SuperPoint / LightGlue` preset
- 只允许单轴变化：
  - 方法变化时，其余 seam / temporal / geometry mode 固定
  - seam 变化时，方法和 geometry mode 固定
  - smoothing 变化时，方法、seam、geometry mode 固定

## 推荐实验矩阵
### Phase 0 / Phase 1
- `Method A vs Method B`
- 固定：
  - 正式视频级 compare 入口：`scripts/eval_method_compare_matrix.py`
  - 方法集合：
    - `method_a_orb`
    - `method_a_sift`
    - `method_b`
  - 固定运行参数：
    - `video_mode=1`
    - `reuse_mode=frame0_all`
    - `max_frames=6000`
    - 不依赖 keyframe 更新
  - Method B 当前推荐 accuracy preset（2026-03-23 复盘后）：
    - `feature_backend=superpoint`
    - `matcher_backend=lightglue`
    - `geometry_backend=opencv_usac_magsac`
    - `max_keypoints=4096`
    - `resize_long_edge=1536`
    - `depth_confidence=-1`
    - `width_confidence=-1`
    - `filter_threshold=0.1`
  - 解读边界：
    - 这是 Phase 1 的 fixed-geometry compare preset，用于验证 Method B 已完整接到视频质量链路
    - `jitter` 在这一 preset 下不作为主比较指标
    - 2026-03-23 之前的 Phase 3 Method B 结果使用的是旧 implicit preset：
      - `max_keypoints=2048`
      - `SuperPoint` package default resize `1024`
      - `LightGlue` adaptive defaults（`depth_confidence=0.95`, `width_confidence=0.99`）
    - 因此旧 Phase 3 方法表目前只能作为“旧 preset 下的实验事实”，不能再直接当 final report 的最终 Method B 结论

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

### Phase 2 正式 dynamic seam compare（2026-03-23 收尾）
- 正式入口：
  - `scripts/eval_dynamic_compare.py`
- 正式 suite：
  - `outputs/video_compare/phase2_dynamic_compare_full_v1/summary.csv`
  - `outputs/video_compare/phase2_dynamic_compare_full_v1/preset_summary.csv`
  - `outputs/video_compare/phase2_dynamic_compare_full_v1/pair_compare.csv`
  - `outputs/video_compare/phase2_dynamic_compare_full_v1/visual_manifest.csv`
  - `outputs/video_compare/phase2_dynamic_compare_full_v1/visual_summary.md`
- 正式代表性 pairs：
  - `mine_source_square_left_right`
  - `mine_source_mcd1_left_right`
  - `mine_source_traffic2_left_right`
  - `mine_source_walking_left_right`
- 正式 presets：
  - `baseline_fixed`
  - `keyframe_seam10`
  - `trigger_fused_d18_fg008`
  - `adaptive_trigger_fused_d18_fg008`
- 聚合结果：
  - `baseline_fixed`
    - `mean_overlap_diff_after ≈ 5.244`
    - `mean_stitched_delta ≈ 4.740`
    - `approx_fps ≈ 12.65`
  - `keyframe_seam10`
    - `mean_overlap_diff_after ≈ 4.662`
    - `mean_stitched_delta ≈ 4.743`
    - `approx_fps ≈ 9.94`
  - `trigger_fused_d18_fg008`
    - `mean_overlap_diff_after ≈ 3.261`
    - `mean_stitched_delta ≈ 4.746`
    - `approx_fps ≈ 8.69`
  - `adaptive_trigger_fused_d18_fg008`
    - `mean_overlap_diff_after ≈ 3.862`
    - `mean_stitched_delta ≈ 4.772`
    - `geometry_update_count ≈ 50`
    - `approx_fps ≈ 1.76`
- 当前 Phase 2 主结论：
  - `trigger_fused_d18_fg008` 是当前最合适的默认 dynamic seam preset。
  - `keyframe_seam10` 是可解释、简单的中间方案，但质量弱于 `trigger_fused`。
  - `adaptive_trigger_fused_d18_fg008` 只保留为实验 preset：
    - 在动态场景能触发 geometry refresh
    - 但当前速度代价过大
    - 且在稳定场景 `square` 上会退化

### Phase 2 smoothing ablation（2026-03-23 更新）
- 当前 smoothing 对比入口：
  - `scripts/legacy/run_phase2_seam_smoothing_suite.py`
- 固定前提：
  - `geometry_mode=fixed_geometry`
  - `seam_policy=trigger`
  - `seam_trigger_diff_threshold=18`
  - `foreground_mode=disagreement`
  - `seam_trigger_foreground_ratio=0.08`
  - Method B：`superpoint + lightglue + opencv_usac_magsac`
- full-length suite：
  - `outputs/video_smoothing/phase2_seam_smoothing_full_v1/smooth_summary.csv`
- 当前聚合结果：
  - `smooth_none`
    - `approx_fps ≈ 10.645`
    - `mean_stitched_delta ≈ 6.14523`
    - `mean_seam_mask_change_ratio ≈ 0.00641`
  - `smooth_ema_a080`
    - `approx_fps ≈ 10.713`
    - `mean_stitched_delta ≈ 6.14661`
    - `mean_seam_mask_change_ratio ≈ 0.0000436`
  - `smooth_window_5`
    - `approx_fps ≈ 10.112`
    - `mean_stitched_delta ≈ 6.14661`
    - `mean_seam_mask_change_ratio ≈ 0.0000437`
- 当前结论：
  - smoothing 的主要效果是明显降低 seam mask 抖动。
  - 但在当前推荐 preset 上，没有带来 `mean_stitched_delta` 改善。
  - 因此：
    - 当前默认值仍应保持 `seam_smooth=none`
    - `ema/window` 继续保留为实验 preset
    - 若后续要强调 flicker，需要补更贴近 seam band / object region 的 temporal metric

### smoothing 指标解释约束（2026-03-23 更新）
- `mean_overlap_diff_after`
  - 仍适合作为 fixed-geometry seam policy 的一般主指标
  - 但不适合作为 smoothing 的主比较项
  - 原因：当前 smoothed mask 会把 assignment 压成更严格的分区，导致该值可退化到 `0.0`
- smoothing 当前优先比较：
  - `mean_seam_mask_change_ratio`
  - `mean_stitched_delta`
  - `approx_fps`

### Phase 3
- 在代表性 pair 上做系统矩阵：
  - DynamicStereo 动态样例
  - `mine_source` 真实视频
  - 至少 1 组较稳定的静态样例

### 固定几何下的方法 compare 指标扩展建议（2026-03-24）
- 当前三项主指标：
  - `mean_inliers`
  - `mean_inlier_ratio`
  - `approx_fps`
- 当前问题：
  - 这三项不足以解释 `fixed_geometry + frame0_all` 下的真实 trade-off。
  - 尤其是：
    - `approx_fps` 会把初始化代价和 steady-state compose 代价混在一起
    - `inlier_ratio` 会把“保留更多 matches 带来的 recall 提升”和“几何更差”混在一起
    - 完全没有覆盖 seam/blending/temporal artefacts

#### 建议改成四类指标
- 1. 初始化与运行代价
  - 直接复用 / 可低成本导出：
    - `init_ms_mean`
    - `reuse_per_frame_ms_mean` 或 `time_breakdown_summary.per_frame_ms_mean`
    - `avg_runtime_ms`
    - `approx_fps`
  - 解读方式：
    - `init_ms_mean` 表示首帧建图 / 匹配 / 建立 geometry 的启动延迟
    - `per_frame_ms_mean` 表示固定几何下的 steady-state 视频合成代价
    - `approx_fps` 只作为 full-run amortized throughput，不单独作为“算法快慢”的唯一结论

- 2. 几何质量
  - 当前可直接复用：
    - `mean_inliers`
    - `mean_inlier_ratio`
  - 建议新增：
    - `mean_reprojection_error`
    - `inlier spatial coverage`
      - 例如 inlier 凸包面积 / 图像面积，或网格占据率
    - `match_confidence summary`
      - LightGlue score 均值 / 中位数 / p10
  - 理由：
    - `mean_inliers` 更像 recall
    - `inlier_ratio` 更像 purity
    - `reprojection_error` 更能直接描述几何拟合质量
    - `coverage` 可以避免 “inliers 很多但只集中在局部” 的误判

- 3. Blending / seam 质量
  - 当前可直接复用：
    - `mean_overlap_diff_before`
    - `mean_overlap_diff_after`
  - 建议新增：
    - `seam-band illuminance difference`
      - seam 附近窄带区域的亮度均值差 / 低频亮度差
    - `seam-band gradient disagreement`
      - 近 seam 边缘的梯度差，用于近似 ghosting / bleeding
    - `crop / valid-area ratio`
      - 输出有效区域占比、bbox 面积、黑边比例
  - 论文映射：
    - `A Metric for Video Blending Quality Assessment` 强调同时考虑：
      - `illuminance consistency`
      - artefact：`bleeding / ghosting`
  - 对当前工程的落地方式：
    - 不需要完整复现论文总分
    - 先做 seam-band 局部代理指标即可

- 4. Temporal coherence
  - 当前可直接复用：
    - `mean_stitched_delta`
    - `mean_seam_mask_change_ratio`
    - `mean_jitter_sm`（仅 geometry 非固定时）
  - fixed-geometry 下建议新增：
    - `flow-compensated temporal residual`
      - 对 stitched frame 做 optical flow warp 后再算 SSD / MAE
    - `seam-band flicker`
      - seam band 区域跨帧亮度变化
  - 论文映射：
    - Video Blending Quality 论文把 `temporal coherence` 作为独立质量维度，而不是附属指标
  - 当前建议：
    - fixed-geometry 下不要再把 `jitter` 当主时序指标
    - 优先看 `mean_stitched_delta` 和 flow-compensated temporal residual

#### 2026-03-24 当前落地状态
- 已落地到 `metrics_preview.json` / `transforms.csv`：
  - 运行代价：
    - `init_ms_mean`
    - `per_frame_ms_mean`
    - `avg_runtime_ms`
    - `avg_feature_runtime_ms_left`
    - `avg_feature_runtime_ms_right`
    - `avg_matching_runtime_ms`
    - `avg_geometry_runtime_ms`
  - 几何质量：
    - `mean_reprojection_error`
    - `mean_inlier_spatial_coverage`
  - blending / seam 质量：
    - `mean_seam_band_illuminance_diff`
    - `mean_seam_band_gradient_disagreement`
  - temporal artefact：
    - `mean_seam_band_flicker`
    - `mean_stitched_delta`
- 当前仍延期：
  - `flow-compensated temporal residual`
  - `match_confidence summary`
  - 完整论文式 blended video 总分
- 当前选择：
  - fixed-geometry 的时序 artefact MVP 先用 `seam-band flicker`
  - 不在本轮引入 optical flow，以避免把评测层复杂度拉得过高
  - `avg_feature_runtime_ms_left/right`、`avg_matching_runtime_ms`、`avg_geometry_runtime_ms`
    - 当前口径是“geometry-update event”的阶段耗时均值
    - 在 `fixed_geometry + frame0_all` 下，它们主要反映初始化阶段成本
    - 不应与 `per_frame_ms_mean` 混读为 steady-state 逐帧成本

#### 2026-03-24 Method B candidate sweep 结果
- 代表性 sweep 输出：
  - `outputs/analysis/methodb_preset_sweep_v2/summary.csv`
  - `outputs/analysis/methodb_preset_sweep_v2/preset_summary.csv`
- 当前 candidate preset：
  - `accuracy_v1`
  - `kp3072_v1`
- 当前最合理的下一轮 candidate：
  - `kp3072_v1`
- 原因：
  - 相比 `accuracy_v1`，它在代表性 sweep 上把 `approx_fps` 从约 `3.68` 提高到约 `5.21`
  - 同时总体 `mean_inliers` 只从约 `588` 降到约 `540`
  - `mean_reprojection_error` 反而从约 `1.585` 改善到约 `1.492`
  - 但它在 `mine_source_walking_left_right` 上有明显回退，因此当前仍只能作为 candidate，不替换正式 baseline
- 当前不建议直接替换正式 baseline 的原因：
  - `kp3072_v1` 仍未做 full-length 多数据域复验

#### 2026-03-24 `kp3072_v1` full-length 多数据域复验结果
- 输出目录：
  - `outputs/phase3/phase3_kitti_methods_kp3072_v1/`
  - `outputs/phase3/phase3_dynamicstereo_methods_kp3072_v1/`
  - `outputs/phase3/phase3_minesource_methodb_kp3072_v1/`
  - `outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/overall_method_compare.csv`
  - `outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/by_dataset_method_compare.csv`
  - `outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/method_b_accuracy_vs_kp3072_delta.csv`
- overall 对比：
  - `method_b_accuracy_v1`
    - `mean_inliers ≈ 748.88`
    - `mean_inlier_ratio ≈ 0.5558`
    - `approx_fps ≈ 7.355`
    - `mean_reprojection_error ≈ 1.4309`
  - `method_b_kp3072_v1`
    - `mean_inliers ≈ 609.58`
    - `mean_inlier_ratio ≈ 0.5634`
    - `approx_fps ≈ 7.453`
    - `mean_reprojection_error ≈ 1.3832`
- 数据域差异：
  - KITTI：
    - `kp3072_v1` 与 `accuracy_v1` 的 `mean_inliers` 基本一致
    - `fps` 略有提升
  - DynamicStereo：
    - `kp3072_v1` 的 `inliers / inlier_ratio / reprojection` 略优
    - `fps` 略低
  - `mine_source`：
    - `accuracy_v1 mean_inliers ≈ 842.59`
    - `kp3072_v1 mean_inliers ≈ 628.65`
    - 出现明显负优化
- 当前正式结论：
  - `kp3072_v1` 不应升格为正式默认
  - `accuracy_v1` 继续作为正式 Method B baseline
  - `kp3072_v1` 仅作为候选复验与方法讨论材料保留
  - 其余 exploratory candidate 已移出当前主框架

### Phase 3 正式 KITTI color stereo full-length suite（2026-03-23）
- 正式入口：
  - `scripts/legacy/run_phase3_kitti_compare_suite.py`
  - `scripts/internal/summarize_method_compare_dataset.py`
- 当前正式 suite：
  - `outputs/phase3/phase3_kitti_full_v1/phase3_kitti_summary.md`
  - `outputs/phase3/phase3_kitti_full_v1/method_summary.csv`
  - `outputs/phase3/phase3_kitti_full_v1/method_pair_compare.csv`
  - `outputs/phase3/phase3_kitti_full_v1/dynamic_preset_summary.csv`
  - `outputs/phase3/phase3_kitti_full_v1/dynamic_pair_compare.csv`
  - `outputs/phase3/phase3_kitti_full_v1/pair_coverage.csv`
- 固定配置：
  - pair 只使用 `kitti_raw_data_2011_09_xx_drive_0xxx_image_02_image_03`
  - `fps=10`
  - `max_frames=6000`
  - 方法对比仍沿用：
    - `video_mode=1`
    - `reuse_mode=frame0_all`
  - dynamic seam 对比当前只纳入正式 preset：
    - `baseline_fixed`
    - `keyframe_seam10`
    - `trigger_fused_d18_fg008`
- 当前正式 KITTI pairs：
  - `kitti_raw_data_2011_09_26_drive_0001_image_02_image_03`
  - `kitti_raw_data_2011_09_26_drive_0002_image_02_image_03`
  - `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
  - `kitti_raw_data_2011_09_26_drive_0019_image_02_image_03`
  - `kitti_raw_data_2011_09_28_drive_0016_image_02_image_03`
  - `kitti_raw_data_2011_09_28_drive_0021_image_02_image_03`
- full-length 覆盖结果：
  - 6 个 pair 全部跑到结尾
  - `pair_coverage.csv` 中 `method_processed_frames_mean == dynamic_processed_frames_mean == 实际总帧数`
  - `0001=108`
  - `0002=77`
  - `0005=154`
  - `0019=481`
  - `0016=186`
  - `0021=209`
  - 总计 `1215` 帧 / 子 suite
- 方法对比聚合结果：
  - 2026-03-24 起，正式方法表改以 richer-metrics full-length suite 为准：
    - `outputs/phase3/phase3_kitti_methods_rich_v3/method_summary.csv`
    - 旧 `phase3_kitti_methods_acc_v2/method_summary.csv` 降级为“仅含较少指标的旧正式表”
    - `phase3_kitti_full_v1/method_summary.csv` 继续仅保留为旧 implicit preset 历史结果
  - `method_a_orb`
    - `mean_inliers ≈ 435.17`
    - `mean_inlier_ratio ≈ 0.610`
    - `approx_fps ≈ 33.79`
  - `method_a_sift`
    - `mean_inliers ≈ 529.17`
    - `mean_inlier_ratio ≈ 0.487`
    - `approx_fps ≈ 38.75`
  - `method_b`
    - `mean_inliers ≈ 594.00`
    - `mean_inlier_ratio ≈ 0.393`
    - `approx_fps ≈ 17.66`
- 当前 KITTI 方法结论：
  - 在刷新后的正式 KITTI 方法 compare 上，`Method B` 的 `mean_inliers` 已高于 ORB/SIFT。
  - 但 `Method B` 的 `mean_inlier_ratio` 和 `approx_fps` 仍弱于 ORB/SIFT。
  - 因此 final report 中不应再把 KITTI 结论写成“Method B 整体偏弱”，而应写成：
    - Method B 能换来更高的内点数量
    - 但代价是更低的内点率与更慢的速度
- KITTI dynamic seam 聚合结果：
  - `baseline_fixed`
    - `mean_overlap_diff_after ≈ 4.690`
    - `mean_stitched_delta ≈ 16.510`
    - `approx_fps ≈ 19.99`
  - `keyframe_seam10`
    - `mean_overlap_diff_after ≈ 4.476`
    - `mean_stitched_delta ≈ 16.515`
    - `approx_fps ≈ 15.09`
  - `trigger_fused_d18_fg008`
    - `mean_overlap_diff_after ≈ 1.150`
    - `mean_stitched_delta ≈ 16.520`
    - `approx_fps ≈ 13.57`
- 当前 KITTI dynamic seam 结论：
  - `trigger_fused_d18_fg008` 仍是当前正式推荐 preset。
  - 但它在 KITTI 上的收益并不均匀：
    - `0001 / 0002` 上改善显著
    - `0005 / 0019 / 0016 / 0021` 上与 baseline 的 `overlap_diff_after` 差异接近 0
  - 因此 final report 应同时展示：
    - preset 级平均值
    - pair 级 `dynamic_pair_compare.csv`

### Phase 3 DynamicStereo full-length suite（2026-03-23）
- 正式 suite：
  - `outputs/phase3/phase3_dynamicstereo_full_v1`
- 正式 pairs：
  - `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`
  - `dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right`
  - `dynamicstereo_real_000_teddy_static_test_frames_rect_left_right`
- 固定配置：
  - `fps=10`
  - `max_frames=6000`
  - 方法对比：
    - `method_a_orb / method_a_sift / method_b`
  - dynamic seam preset：
    - `baseline_fixed / keyframe_seam10 / trigger_fused_d18_fg008`
- full-length 覆盖结果：
  - `ignacio=99`
  - `nikita=83`
  - `teddy=99`
  - 合计 `281` 帧 / 子 suite
- 方法对比聚合结果：
  - 2026-03-24 起，正式方法表改以 richer-metrics full-length suite 为准：
    - `outputs/phase3/phase3_dynamicstereo_methods_rich_v3/method_summary.csv`
    - 旧 `phase3_dynamicstereo_methods_acc_v2/method_summary.csv` 降级为“仅含较少指标的旧正式表”
    - `phase3_dynamicstereo_full_v1/method_summary.csv` 继续仅保留为旧 implicit preset 历史结果
  - `method_a_orb`
    - `mean_inliers ≈ 532.67`
    - `mean_inlier_ratio ≈ 0.628`
    - `approx_fps ≈ 18.01`
  - `method_a_sift`
    - `mean_inliers ≈ 257.00`
    - `mean_inlier_ratio ≈ 0.482`
    - `approx_fps ≈ 17.04`
  - `method_b`
    - `mean_inliers ≈ 527.67`
    - `mean_inlier_ratio ≈ 0.396`
    - `approx_fps ≈ 7.80`
- dynamic seam 聚合结果：
  - `baseline_fixed`
    - `mean_overlap_diff_after ≈ 6.530`
    - `approx_fps ≈ 5.85`
  - `keyframe_seam10`
    - `mean_overlap_diff_after ≈ 6.169`
    - `approx_fps ≈ 4.94`
  - `trigger_fused_d18_fg008`
    - `mean_overlap_diff_after ≈ 2.439`
    - `approx_fps ≈ 4.87`
- 当前 DynamicStereo 结论：
  - `trigger_fused_d18_fg008` 继续显著优于 `baseline_fixed / keyframe_seam10`。
  - 在这组更强动态样例上，Method B 仍未整体超过 ORB，但其结论与 KITTI 不同，不应只用单一数据域下定论。

### Phase 3 mine_source full-length suite（2026-03-23）
- 正式 suite：
  - `outputs/phase3/phase3_minesource_full_v1`
- 当前工作区可运行的正式 pairs：
  - `mine_source_bow1_left_right`
  - `mine_source_bow2_left_right`
  - `mine_source_lake_left_right`
  - `mine_source_robot_left_right`
  - `mine_source_church_left_right`
  - `mine_source_park1_left_right`
  - `mine_source_pujiang1_left_right`
  - `mine_source_pujiang2_left_right`
  - `mine_source_pujiang3_left_right`
  - `mine_source_indoor_left_right`
  - `mine_source_indoor2_left_right`
  - `mine_source_mcd1_left_right`
  - `mine_source_mcd2_left_right`
  - `mine_source_square_left_right`
  - `mine_source_traffic1_left_right`
  - `mine_source_traffic2_left_right`
  - `mine_source_walking_left_right`
- 未纳入：
  - `mine_source_leaves_left_right`
    - 当前本地源文件缺失，无法打开
- 固定配置：
  - 调用层按 `fps=30` 运行本套件
  - 但 run bundle 中对视频输入仍记录 `video_source` 的原始 fps
  - 因此大多数 pair 为约 `30fps`，`bow1 / bow2` 仍显示为 `25fps`
  - `max_frames=6000`
- full-length 覆盖结果：
  - 17 个可用 pair 全部跑完
  - 合计 `5873` 帧 / 子 suite
- 方法对比聚合结果：
  - 2026-03-24 起，正式方法表改以 richer-metrics full-length suite 为准：
    - `outputs/phase3/phase3_minesource_methods_rich_v3/method_summary.csv`
    - 旧 `phase3_minesource_methods_acc_v2/method_summary.csv` 降级为“仅含较少指标的旧正式表”
    - `phase3_minesource_full_v1/method_summary.csv` 继续仅保留为旧 implicit preset 历史结果
  - `method_a_orb`
    - `mean_inliers ≈ 468.76`
    - `mean_inlier_ratio ≈ 0.847`
    - `approx_fps ≈ 12.58`
  - `method_a_sift`
    - `mean_inliers ≈ 754.35`
    - `mean_inlier_ratio ≈ 0.788`
    - `approx_fps ≈ 11.40`
  - `method_b`
    - `mean_inliers ≈ 842.59`
    - `mean_inlier_ratio ≈ 0.641`
    - `approx_fps ≈ 7.69`
- dynamic seam 聚合结果：
  - `baseline_fixed`
    - `mean_overlap_diff_after ≈ 6.439`
    - `approx_fps ≈ 11.39`
  - `keyframe_seam10`
    - `mean_overlap_diff_after ≈ 5.591`
    - `approx_fps ≈ 8.34`
  - `trigger_fused_d18_fg008`
    - `mean_overlap_diff_after ≈ 2.887`
    - `approx_fps ≈ 7.64`
- 当前 mine_source 结论：
  - `trigger_fused_d18_fg008` 在自采视频上继续显著优于 `baseline_fixed / keyframe_seam10`。
  - 在刷新后的正式方法 compare 上，Method B 的 `mean_inliers` 已高于 ORB/SIFT。
  - 但 `mean_inlier_ratio` 与 `approx_fps` 仍明显落后于 Method A。

### Phase 3 统一总表（2026-03-24，richer metrics 全量重跑）
- 总表入口：
  - `scripts/internal/summarize_method_compare_overall.py`
- 当前正式方法总表：
  - `outputs/phase3/phase3_overall_methods_rich_v3/overall_method_summary.csv`
  - `outputs/phase3/phase3_overall_methods_rich_v3/overall_method_by_dataset.csv`
  - `outputs/phase3/phase3_overall_methods_rich_v3/overall_pair_coverage.csv`
  - `outputs/phase3/phase3_overall_methods_rich_v3/phase3_overall_summary.md`
  - `outputs/phase3/phase3_overall_methods_rich_v3/figures/figure_manifest.csv`
  - `outputs/phase3/phase3_overall_methods_rich_v3/figures/figures.md`
- 当前正式 dynamic seam 总表：
  - `outputs/phase3/phase3_overall_full_v1/overall_dynamic_preset_summary.csv`
  - `outputs/phase3/phase3_overall_full_v1/overall_dynamic_by_dataset.csv`
- 说明：
  - 方法总表与 dynamic seam 总表当前来自不同 suite：
    - 方法 compare 用显式 Method B accuracy preset 的 `*_methods_rich_v3`
    - dynamic seam compare 继续沿用 `*_full_v1`
- 当前整体覆盖：
  - `26` 个可运行 pair
  - richer-metrics 方法 compare：`78` 条 full-length run
  - 正式 dynamic seam compare：`78` 条 full-length run
  - 共覆盖 `7369` 帧 / 子 suite 维度
- 当前 overall 结论：
  - 方法主轴：
    - `method_b` 当前总体 `mean_inliers` 最高：`≈748.88`
    - `method_a_orb` 当前总体 `mean_inlier_ratio` 最高：`≈0.767`
    - `method_a_sift` 当前总体速度最高：`approx_fps≈12.63`
    - `method_b` 当前总体 `mean_inlier_ratio≈0.556`、`approx_fps≈7.36`
    - `method_b` 当前总体 `mean_seam_band_flicker≈7.44` 最低
    - `method_b` 当前总体 `mean_overlap_diff_after≈6.08` 优于 Method A
    - `method_a_sift` 当前总体 `mean_reprojection_error≈0.843` 最低
  - dynamic seam 主轴：
    - `trigger_fused_d18_fg008` 当前总体 `mean_overlap_diff_after` 最优
    - 但相较 `baseline_fixed` 会带来明显速度下降
  - 因此 final report 中当前最稳的表述应是：
    - `trigger_fused_d18_fg008` 是当前 OpenCV seam backend 路线下最有效的 dynamic seam preset
    - `Method B` 已完整接入并稳定运行；在显式 accuracy preset 下，它提供更高的内点数量和更好的部分 seam/blending 代理指标，但仍以更低的内点率、更高 reprojection error 和更慢的速度为代价
    - stage runtime breakdown 当前应解释为“初始化 geometry-update event 的阶段耗时”，不是 steady-state 逐帧成本

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
  - `scripts/legacy/ablate_temporal.py`
  - `scripts/legacy/ablate_seam.py`
- 当前定位：
  - 这两个脚本保留为 legacy exploratory helpers
  - 不作为 Phase 0 / Phase 1 的正式实验入口或验收前提
  - 当前正式入口仍以 `scripts/run_baseline_video.py` 的统一 bundle 为准
- 后续建议新增：
  - 统一 experiment driver
  - 独立 metrics 模块
  - summary CSV / plots 生成脚本

## 当前 Phase 2 calibration 入口（2026-03-23）
- calibration driver：
  - `scripts/legacy/run_phase2_trigger_calibration.py`
- 当前正式 suite：
  - `outputs/video_calibration/phase2_trigger_adaptive_minesource_calib_v2/summary.csv`
  - `outputs/video_calibration/phase2_trigger_adaptive_minesource_calib_v2/preset_summary.csv`
- 当前 suite 结论摘要：
  - `trigger_fused_d18_fg008`
    - 把 `mean_overlap_diff_after` 从约 `6.29` 降到约 `3.75`
    - `seam_recompute_after_init_per_100f ≈ 1.25`
    - `approx_fps ≈ 10.30`
  - `adaptive_fused_d18_fg008`
    - 已能产生 `geometry_update_per_100f ≈ 1.25`
    - 但当前比 `trigger_fused` 更慢，且没有显示出更好的 `mean_overlap_diff_after`
  - `trigger_stable_d18_fg008_cd6_h075` 与 `adaptive_stable_d18_fg008_cd6_h075`
    - 在当前 `mine_source` 视频上过于保守，init 后重算次数趋近于 0
- 当前推荐默认值：
  - `geometry_mode=fixed_geometry`
  - `seam_policy=trigger`
  - `seam_trigger_diff_threshold=18`
  - `foreground_mode=disagreement`
  - `seam_trigger_foreground_ratio=0.08`

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
  - `scripts/eval_method_compare_matrix.py`
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
