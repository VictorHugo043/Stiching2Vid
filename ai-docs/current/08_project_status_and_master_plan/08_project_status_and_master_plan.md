# 08_project_status_and_master_plan

## 任务目标
- 形成后续实施的 master plan，统一当前项目现状、优先级、Phase 0 ~ Phase 4 路线和关键约束。

## 当前项目状态
### 1. Method A 已完成能力
- 特征与匹配：
  - ORB / SIFT
  - KNN + ratio test
- 几何：
  - RANSAC homography
  - canvas / warp
- seam / crop / blend：
  - crop-before-seam
  - OpenCV seam finder
  - feather / multiband
- video stitching：
  - `VideoStitcher` cached execution
  - frame0 / re-init reuse
  - temporal smoothing
- diagnostics：
  - `metrics_preview.json`
  - `debug.json`
  - `transforms.csv`
  - `jitter_timeseries.csv`
  - `snapshots/`
- data management：
  - `pairs.yaml` 驱动
  - 统一 `FrameSource` I/O 层

### 2. 当前实验与输出 artefacts 现状
- `outputs/runs/` 中已有大量 baseline、seam、reuse、temporal 相关 run bundle。
- `outputs/ablations/` 中已有 seam、temporal 等部分对比目录。
- 现有 bundle 已足够支撑：
  - runtime / FPS
  - match / inlier 统计
  - overlap 统计
  - seam debug
  - 部分 temporal debug

### 3. 当前最关键的限制点与耦合点
- `scripts/run_baseline_video.py` 为单体 orchestrator，承载过多职责。
- `features / matching / geometry` 概念解耦，但接口仍绑定 OpenCV 类型。
- `video_mode=1` 的 `reuse_mode` 命名容易误导真实行为。
- 当前 `jitter` 在固定几何路径下会系统性失真。
- 当前 seam backend 是 OpenCV seam mask 风格，不支持 object-centered seam energy。

## 推荐优先级顺序
1. `Phase 0` 冻结基线 / 统一运行模式 / 固定评测协议 / 固定导出 artefacts
2. `Phase 1` Method B 落地
3. `Phase 2` Dynamic seam + meaningful temporal smoothing / evaluation
4. `Phase 3` 系统实验与比较
5. `Phase 4` GUI thin wrapper

## 为什么这样排序
- Method B 应先于 GUI：
  - Method B 会改变接口、配置、诊断字段和实验矩阵。
  - GUI 属于包装层，过早实现会造成参数面和日志展示返工。
- Dynamic seam 应放在 Method B 后：
  - 更稳的匹配与几何会改变 overlap ROI 和 seam 难点。
  - 先换匹配再调 seam，变量更少。
- 不建议直接完整复现 object-centered MRF seam：
  - 当前 seam 模块结构差异太大。
  - 第一版更适合做兼容 OpenCV seam finder 的轻量 dynamic seam。
- 核心研究任务：
  - Method B
  - Dynamic seam
  - meaningful temporal evaluation
  - experiments
- 包装层：
  - GUI
  - 更丰富的结果展示与交互

## Phase 0
### 目标
- 冻结当前 Method A 基线。
- 修正文档与代码漂移。
- 在文档与配置层显式拆分运行模式：
  - `fixed_geometry`
  - `keyframe_update`
  - `adaptive_update`
- 固定评测协议与导出 artefacts。

### 前置依赖
- 无新增算法依赖。

### 重点模块
- `scripts/run_baseline_video.py`
- `src/stitching/video_stitcher.py`
- `src/stitching/temporal.py`
- `ai-docs/current/03~05`

### 建议新增项
- 运行模式文档语义层
- `config.json` 导出
- 统一 run bundle 说明

### 验收标准
- baseline 行为、字段和术语不再自相矛盾。
- `jitter` 的适用范围在文档中固定。
- 文档与代码漂移清理完毕。

### 风险与规避
- 风险：旧 run bundle 名称与新术语不一致。
  - 规避：文档先解释映射，不立即重命名历史 run。

### 当前完成判断（2026-03-19）
- 对进入 Phase 1 所必需的最小冻结已完成：
  - 当前 baseline video pipeline 的 as-built 行为已对齐到文档。
  - `run_baseline_video.py` 已显式导出 `geometry_mode` 与 `jitter_meaningful`。
  - `fixed_geometry / keyframe_update / adaptive_update` 的语义边界已固定。
  - `scripts/ablate_temporal.py` 与 `scripts/ablate_seam.py` 已降级为 legacy exploratory helpers，不再作为当前 Phase 0 闭环前提。
- 暂未完成但不阻塞 Phase 1：
  - 显式 `config.json` 导出
  - 统一 experiment driver
  - 更完整的 summary / plotting pipeline

## Phase 1
### 目标
- 接入 `SuperPoint + LightGlue + MAGSAC++`。
- 让 Method A / Method B 共用同一条 stitching、diagnostics、evaluation 路线。

### 前置依赖
- Phase 0 已冻结输出与运行模式。

### 重点模块
- `src/stitching/features.py`
- `src/stitching/matching.py`
- `src/stitching/geometry.py`
- `scripts/run_baseline_frame.py`
- `scripts/run_baseline_video.py`

### 建议新增项
- `FeatureResult / MatchResult / GeometryResult`
- `FeatureBackend / MatcherBackend / GeometryBackend`
- Method B 专属 config 与 runtime logging
- 可选支持子任务：`frame_quality_preview`
  - 目标：给单帧入口补齐 seam / crop / blend 质量链路，尽量贴近视频输出观感
  - 定位：支持 Method B qualitative check，不是 Phase 1 主线阻塞项
  - 推荐插入位置：在单帧 Method B backend loader 稳定之后、视频路径迁移之前
  - 当前状态：最小版已完成，通过共享 helper 复用 `VideoStitcher` 的单帧 compose 路径

### 验收标准
- 单帧先跑通，再扩展到视频。
- Method A / Method B 能共用 `VideoStitcher`、crop、seam、blend、diagnostics。

### 当前 Phase 1 进度（2026-03-20）
- 已完成：
  - `FeatureResult / MatchResult / GeometryResult`
  - 单帧 `frame_quality_preview`
  - 单帧 Method B dependency probe / lazy import / failure diagnostics / fallback 骨架
  - `.venv-methodb` 环境下真实 `SuperPoint + LightGlue + OpenCV USAC_MAGSAC` 单帧成功 smoke
  - 正式环境入口：
    - `requirements.txt`
    - `requirements-methodb.txt`
    - `docs/environment.md`
  - 默认多 pair 单帧 smoke suite：
    - `scripts/run_frame_smoke_suite.py`
    - 当前已验证 Method B 默认 suite 跑通：
      - `mine_source_indoor2_left_right`
      - `kitti_raw_data_2011_09_28_drive_0119_image_02_image_03`
      - `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
      - `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`
  - 两组多帧 Method B 抽样回归：
    - `kitti_raw_data_2011_09_28_drive_0119_image_02_image_03`
    - `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`
    - 抽样帧：`0 / 20 / 40 / 60`
  - 最小视频路径 adapter：
    - 新增 `src/stitching/frame_pair_pipeline.py`
    - `run_baseline_video.py` 已接到结果对象层
    - 短视频 Method B smoke 已覆盖：
      - `keyframe_update`
      - `fixed_geometry`
  - 3 条 60 帧 Method B 视频回归已完成：
    - `mine_source_indoor2_left_right`
    - `mine_source_pujiang1_left_right`
    - `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
  - 正式视频级 compare 入口：
    - `scripts/run_video_compare_suite.py`
  - 正式 Phase 1 compare suite：
    - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/summary.csv`
    - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/pair_compare.csv`
  - 正式 compare 预设：
    - `method_a_orb`
    - `method_a_sift`
    - `method_b`
    - `video_mode=1`
    - `reuse_mode=frame0_all`
    - `max_frames=6000`
  - 正式 compare 结果：
    - 3 个 pair、9 条 full-length run 全部通过
    - `fallback_frames=0`
    - `errors_count=0`
    - Phase 1 没有发现新的结构性接入问题
- Phase 1 完成判断：
  - Phase 1 的目标可以视为已完成。
  - Method B 已从单帧 loader、单帧质量预览、视频 adapter、短视频 smoke、长视频回归一直闭环到正式视频级比较入口与统计表。
  - 当前剩余 open issues 均属于 Phase 2/Phase 3，而不是 Phase 1 阻塞项。
- 后续主线：
  - Phase 2 的 dynamic seam / meaningful temporal evaluation

### 风险与规避
- 风险：环境缺依赖。
  - 规避：optional dependency、lazy import、fallback。
- 风险：本地已有 `.venv` / `.venv-methodb` 与正式 requirements 漂移。
  - 规避：把 root requirements 和 `docs/environment.md` 作为唯一正式环境入口；必要时重建虚拟环境。
- 风险：为了补单帧 seam/crop parity，直接复制 `run_baseline_video.py` 的内联质量链路逻辑。
  - 规避：若要做 `frame_quality_preview`，应优先抽共享 helper / adapter，而不是复制大段视频脚本。

## Phase 2
### 目标
- 实现 Dynamic seam MVP。
- 恢复有意义的 temporal smoothing / temporal evaluation。

### 前置依赖
- Method B 已稳定接通。

### 重点模块
- `src/stitching/seam_opencv.py`
- `src/stitching/video_stitcher.py`
- `src/stitching/temporal.py`
- `scripts/run_baseline_video.py`

### 建议新增项
- `SeamPolicy`
- `SeamUpdateTrigger`
- `SeamMaskSmoother`
- 可选 `ForegroundMaskProvider`

### 验收标准
- 支持 `fixed seam / keyframe seam / trigger seam`
- 在非固定几何模式下 `jitter` 不再系统性退化为 0
- seam / temporal 相关 artefacts 能写入 diagnostics

### 风险与规避
- 风险：OpenCV seam finder 无法直接表达 object-centered cost。
  - 规避：先做兼容式 MVP，再单独做新 seam backend。

### 当前 Phase 2 进度（2026-03-20）
- 已完成第一批最小落地：
  - 新增 `src/stitching/seam_policy.py`
  - 在 `run_baseline_video.py` 中接入：
    - `fixed seam`
    - `keyframe seam`
    - `trigger seam`
    - `auto`
  - 在 `video_mode=0/1` 两条路径中统一 seam 更新决策壳层
  - 在 `VideoStitcher.stitch_frame()` 与 video loop 中导出：
    - `overlap_diff_before`
    - `overlap_diff_after`
    - `seam_recomputed`
    - `seam_trigger_reason`
    - `seam_mask_change_ratio`
    - `stitched_delta_mean`
  - meaningful temporal evaluation 的最小恢复：
    - `fixed_geometry` 主指标切到 `mean_overlap_diff_after`
    - `keyframe_update` 主指标保持 `mean_jitter_sm`
  - 已完成第二批最小落地：
    - `adaptive_update` 作为 seam-driven geometry refresh MVP 接入
    - 新增 `geometry_update_count`
    - 新增 `geometry_update_events`
    - `transforms.csv` 新增 `geometry_recomputed / geometry_update_reason`
- 已验证的 smoke run：
  - `outputs/runs/phase2_seam_fixed_smoke`
  - `outputs/runs/phase2_seam_keyframe_smoke`
  - `outputs/runs/phase2_seam_trigger_smoke_v2`
  - `outputs/runs/phase2_temporal_keyframeupdate_smoke`
  - `outputs/runs/phase2_kitti0002_adaptive_keyframe`
  - `outputs/runs/phase2_kitti0002_adaptive_trigger`
- 当前尚未完成：
  - 新 seam backend
  - 更强的 smoothing-specific temporal metric
- 本轮进一步收敛的配置结论：
  - `geometry_mode` 现在应视为正式用户入口。
  - `video_mode` 仅保留为 legacy 兼容别名。
  - `keyframe_every` 只控制 geometry keyframe。
  - `seam_keyframe_every` 只控制 seam keyframe。
  - `fixed_geometry` 下 `jitter` 只跟 geometry 相关，不跟 seam 更新相关。
  - `adaptive_update` 当前只表示 seam event 触发的 geometry refresh，不表示完整自适应几何系统。
  - 后续调试 dynamic seam 时应优先查看：
    - `jitter_scope`
    - `seam_recompute_count`
    - `seam_snapshot_count`
    - `mean_overlap_diff_after`
    - `geometry_update_count`
    - `geometry_update_events`
  - 当前已新增兼容式 foreground-aware MVP：
    - `foreground_mode=disagreement`
    - `seam_trigger_foreground_ratio`
    - `seam_trigger_cooldown_frames`
    - `seam_trigger_hysteresis_ratio`
  - 当前 Phase 2 calibration suite：
    - `outputs/video_calibration/phase2_trigger_adaptive_minesource_calib_v2`
  - 当前推荐默认 seam preset：
    - `geometry_mode=fixed_geometry`
    - `seam_policy=trigger`
    - `seam_trigger_diff_threshold=18`
    - `foreground_mode=disagreement`
    - `seam_trigger_foreground_ratio=0.08`
  - 当前 `adaptive_fused` 仅作为实验 preset 保留：
    - 已能触发 geometry refresh
    - 但当前比 `trigger_fused` 更慢，且没有显示出更好的 `mean_overlap_diff_after`
  - 当前已完成 per-trigger rearm：
    - `trigger_armed / hysteresis` 已从单一全局状态细化为 `overlap / diff / foreground` 三路状态
    - `phase2_adaptive_fused_mcd1_rearm_smoke_v1` 中 `geometry_update_count=5`
    - 说明 sustained foreground 下的“一次性事件”问题已明显缓解
  - 当前已完成 seam temporal smoothing：
    - `none / ema / window`
    - full-length suite：`outputs/video_smoothing/phase2_seam_smoothing_full_v1`
  - 当前 smoothing 结论：
    - `ema/window` 明显降低 `mean_seam_mask_change_ratio`
    - 但对 `mean_stitched_delta` 没有带来收益
    - 因此默认值仍保持 `seam_smooth=none`
  - 当前已完成正式 Phase 2 compare matrix：
    - `outputs/video_compare/phase2_dynamic_compare_full_v1`
    - 4 个代表性 pair × 4 个正式 preset
  - 当前已完成代表性可视化汇总：
    - `outputs/video_compare/phase2_dynamic_compare_full_v1/visual_manifest.csv`
    - `outputs/video_compare/phase2_dynamic_compare_full_v1/visual_summary.md`
  - 当前正式 Phase 2 主结论：
    - `trigger_fused_d18_fg008` 是默认推荐 preset
    - `keyframe_seam10` 是简单但较弱的中间方案
    - `adaptive_trigger_fused_d18_fg008` 只保留为实验 preset
      - 当前速度代价过高
      - 在稳定场景上会退化
  - 当前 Phase 2 完成判断：
    - 以 MVP 范围计，Phase 2 可以视为已完成
    - 剩余 open issues 均不再阻塞进入 Phase 3

## Phase 3
### 目标
- 完成系统实验、对比表和最终素材。

### 前置依赖
- Method B 与 Dynamic seam 至少有 MVP 可跑版本。

### 重点模块
- experiment driver
- metrics 模块
- `ai-docs/current/05` 与 `07`

### 建议新增项
- summary CSV
- plot / report export pipeline

### 验收标准
- 主要对比都能一次性生成 bundle、表格和图片。

### 风险与规避
- 风险：实验矩阵过大。
  - 规避：先筛选最佳方法，再局部展开。

### 当前 Phase 3 进度（2026-03-23）
- 已完成第一个正式 Phase 3 block：
  - 新增 Phase 3 KITTI color stereo full-length 正式入口：
    - `scripts/run_phase3_kitti_compare_suite.py`
    - `scripts/build_phase3_kitti_summary.py`
  - 当前正式 suite：
    - `outputs/phase3/phase3_kitti_full_v1`
  - 当前正式 KITTI pairs：
    - `kitti_raw_data_2011_09_26_drive_0001_image_02_image_03`
    - `kitti_raw_data_2011_09_26_drive_0002_image_02_image_03`
    - `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
    - `kitti_raw_data_2011_09_26_drive_0019_image_02_image_03`
    - `kitti_raw_data_2011_09_28_drive_0016_image_02_image_03`
    - `kitti_raw_data_2011_09_28_drive_0021_image_02_image_03`
  - 当前正式固定条件：
    - `fps=10`
    - `max_frames=6000`
    - 方法对比保持 `fixed_geometry`
    - dynamic seam 采用 `baseline_fixed / keyframe_seam10 / trigger_fused_d18_fg008`
- 当前结果摘要：
  - 方法对比聚合：
    - `method_a_orb`
      - `mean_inliers ≈ 435.17`
      - `mean_inlier_ratio ≈ 0.610`
      - `approx_fps ≈ 24.32`
    - `method_a_sift`
      - `mean_inliers ≈ 529.17`
      - `mean_inlier_ratio ≈ 0.487`
      - `approx_fps ≈ 27.43`
    - `method_b`
      - `mean_inliers ≈ 345.50`
      - `mean_inlier_ratio ≈ 0.389`
      - `approx_fps ≈ 18.82`
  - dynamic seam 聚合：
    - `baseline_fixed`
      - `mean_overlap_diff_after ≈ 4.690`
      - `approx_fps ≈ 19.99`
    - `keyframe_seam10`
      - `mean_overlap_diff_after ≈ 4.476`
      - `approx_fps ≈ 15.09`
    - `trigger_fused_d18_fg008`
      - `mean_overlap_diff_after ≈ 1.150`
      - `approx_fps ≈ 13.57`
- 当前 Phase 3 结论：
  - 正式 KITTI full-length suite 已经可复跑，正式表格和 summary 已形成。
  - 在这组 KITTI color stereo full-length fixed-geometry compare 上：
    - Method B 没有在匹配质量或速度上整体超过 Method A
    - 这应作为实验结论记录，而不是接入 bug
  - `trigger_fused_d18_fg008` 仍是当前推荐 dynamic seam preset，但其收益在 KITTI 上是 pair-dependent 的。
- 当前未完成但不阻塞：
  - 将 DynamicStereo / `mine_source` 的正式结果与 KITTI 一并并入最终 report 表格
  - 统一 final report 的 plot/export 脚本

### 当前 Phase 3 进度补充（2026-03-23，多数据域 full-length）
- 已完成第二个正式 Phase 3 block：
  - DynamicStereo full-length suite：
    - `outputs/phase3/phase3_dynamicstereo_full_v1`
    - 3 个 pair
    - 合计 `281` 帧 / 子 suite
  - `mine_source` full-length suite：
    - `outputs/phase3/phase3_minesource_full_v1`
    - 17 个当前可运行 pair
    - 合计 `5873` 帧 / 子 suite
  - 统一总表：
    - `outputs/phase3/phase3_overall_full_v1`
    - 26 个可运行 pair
    - 156 条 full-length run
- 当前多数据域结论：
  - `trigger_fused_d18_fg008` 在 KITTI / DynamicStereo / `mine_source` 三个数据域上都保持当前最优 `mean_overlap_diff_after`。
  - Method B 已稳定接通并完成大规模 full-length 回归，但在当前三数据域 fixed-geometry compare 上都没有整体超过 Method A。
- 当前剩余问题：
  - `mine_source_leaves_left_right` 本地源缺失，未能纳入正式 suite。
  - DynamicStereo 的当前可访问帧数与 manifest 元数据存在偏差，应按实际 `pair_coverage.csv` 解释。
  - final report 的 plot/export pipeline 仍待补齐。

### 当前 Phase 3 进度补充（2026-03-23，Method B 复盘）
- 已完成一次针对 `SuperPoint / LightGlue / MAGSAC` 层的复盘与最小修复。
- 当前确认的关键问题：
  - 旧 Phase 3 compare 使用的是偏速度的 Method B implicit preset，而不是显式 accuracy preset。
  - 旧 `resize_long_edge` 并没有真正控制 `SuperPoint.extract()` 的 preprocess resize，导致高分辨率样例上的 Method B 能力被压低。
- 本次已完成的修复：
  - Method B 的 `resize_long_edge` 现已按官方 `SuperPoint.extract(..., resize=...)` 语义生效。
  - `max_keypoints <= 0` 现表示“不设上限”。
  - `matching.py` 现额外记录 `LightGlue stop_layer / prune stats`，便于后续解释性能与匹配质量。
- 当前代表性复查结果：
  - `kitti_raw_data_2011_09_26_drive_0002_image_02_image_03`
    - 77 帧 fixed-geometry 短视频回归：
      - 旧：`mean_inliers=365`, `approx_fps=4.06`
      - 新：`mean_inliers=647`, `approx_fps=3.41`
  - `mine_source_walking_left_right`
    - 120 帧 fixed-geometry 短视频回归：
      - 旧：`mean_inliers=148`, `mean_inlier_ratio=0.180`, `approx_fps=3.70`
      - 新：`mean_inliers=432`, `mean_inlier_ratio=0.271`, `approx_fps=2.54`
- 当前含义：
  - 旧 Phase 3 方法对比中的 “Method B 整体弱于 Method A” 结论现在应视为“旧 preset 下的实验事实”。
  - 在写 final report 之前，应使用显式 Method B accuracy preset 重跑正式方法 compare。
- 当前推荐 Method B accuracy preset：
  - `max_keypoints=4096`
  - `resize_long_edge=1536`
  - `depth_confidence=-1`
  - `width_confidence=-1`
  - `filter_threshold=0.1`

### 当前 Phase 3 进度补充（2026-03-24，正式方法 compare 刷新）
- 已完成显式 Method B accuracy preset 下的正式 full-length 方法 compare 刷新：
  - KITTI：
    - `outputs/phase3/phase3_kitti_methods_acc_v2`
  - DynamicStereo：
    - `outputs/phase3/phase3_dynamicstereo_methods_acc_v2`
  - `mine_source`：
    - `outputs/phase3/phase3_minesource_methods_acc_v2`
  - unified overall：
    - `outputs/phase3/phase3_overall_methods_acc_v2`
- 当前正式方法结论已经替换为：
  - KITTI：
    - `method_b mean_inliers≈594.00`，高于 ORB/SIFT
    - 但 `mean_inlier_ratio≈0.393`、`fps≈17.66` 仍低于 ORB/SIFT
  - DynamicStereo：
    - `method_b mean_inliers≈527.67`，与 ORB 接近，明显高于 SIFT
    - 但 `mean_inlier_ratio≈0.396`、`fps≈7.80` 仍明显偏弱
  - `mine_source`：
    - `method_b mean_inliers≈842.59`，高于 ORB/SIFT
    - 但 `mean_inlier_ratio≈0.641`、`fps≈7.69` 仍低于 Method A
  - overall：
    - `method_b mean_inliers≈748.88` 最高
    - `method_a_orb mean_inlier_ratio≈0.767` 最高
    - `method_a_sift fps≈18.36` 最高
- 当前含义：
  - 旧的 “Method B 整体偏弱” 结论已不再成立。
  - 当前更准确的表述是：
    - Method B 在 accuracy preset 下能换来更高的内点数量
    - 但当前 CPU 路线下仍以更低的内点率和更慢的速度为代价
- 当前正式结果引用边界：
  - 方法 compare：
    - 以 `phase3_*_methods_acc_v2` 和 `phase3_overall_methods_acc_v2` 为准
  - dynamic seam：
    - 仍以 `phase3_*_full_v1` 和 `phase3_overall_full_v1` 为准

## Phase 4
### 目标
- 做 GUI thin wrapper。

### 前置依赖
- 核心 pipeline、config、outputs、错误语义稳定。

### 重点模块
- GUI 新目录
- `src/stitching/io.py`
- `scripts/preprocess/split_sbs_stereo.py`

### 建议新增项
- pair 选择
- 左右视频上传
- pairs.yaml 安全写入
- 参数配置
- 运行日志与 artefact 展示

### 验收标准
- GUI 不引入新的算法分支与数据格式。
- 输出仍严格落在既有 `outputs/` 结构内。

### 风险与规避
- 风险：GUI 提前侵入核心 pipeline。
  - 规避：只做 CLI / bundle 的薄封装。

## 相关文档
- baseline 行为：`03_baseline_video_pipeline`
- quality 边界：`04_quality_improvement`
- 评测协议：`05_evaluation`
- Method B 设计：`06_method2_strong_matching`
- Dynamic seam 设计：`09_dynamic_seam_and_temporal_eval`
- 执行规则：`10_execution_workflow`

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/08_project_status_and_master_plan/08_project_status_and_master_plan.md | 新增 master plan 文档，统一项目现状、优先级与 Phase 0~4 | Codex | 完成 |
