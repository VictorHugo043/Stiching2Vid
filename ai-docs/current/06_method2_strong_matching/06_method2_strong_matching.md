# 06_method2_strong_matching

## 任务目标
- 落地 Method B：`SuperPoint + LightGlue + MAGSAC++`，并确保它与当前 Method A 共用同一条 video stitching、diagnostics、evaluation 路线。

## 当前代码结构诊断
- 概念上已经解耦：
  - `src/stitching/features.py`
  - `src/stitching/matching.py`
  - `src/stitching/geometry.py`
- 但接口仍过于绑定 OpenCV 类型：
  - `features.py` 直接返回 `cv2.KeyPoint` + descriptors
  - `matching.py` 直接返回 `cv2.DMatch`
  - `geometry.py` 直接消费 `KeyPoint/DMatch`
- 结论：
  - 模块边界是对的
  - 数据边界需要升级

## 推荐统一接口
### 建议新增数据结构
- `FeatureResult`
  - `keypoints_xy`
  - `descriptors`
  - `scores`
  - `image_size`
  - `backend_name`
  - `runtime_ms`
  - `meta`
- `MatchResult`
  - `matches_lr`
  - `match_scores`
  - `tentative_count`
  - `good_count`
  - `backend_name`
  - `runtime_ms`
  - `meta`
- `GeometryResult`
  - `H`
  - `inlier_mask`
  - `inlier_count`
  - `inlier_ratio`
  - `reprojection_error`
  - `backend_name`
  - `runtime_ms`
  - `status`
  - `meta`

### 建议新增后端抽象
- `FeatureBackend`
- `MatcherBackend`
- `GeometryBackend`

## 推荐实现顺序
### 1. 先单帧
- 先接 `scripts/run_baseline_frame.py`
- 验证：
  - 模型加载
  - 特征输出
  - 匹配输出
  - MAGSAC++ 求解
  - diagnostics 字段

### 当前已落地的接口骨架（Phase 1 子任务 1）
- 兼容策略：
  - 保留旧 tuple/OpenCV 接口给 `run_baseline_video.py`
  - 新增结果对象接口给 `run_baseline_frame.py`
- 当前已落地的新入口：
  - `features.detect_and_describe_result()`
  - `matching.match_feature_results()`
  - `geometry.estimate_homography_result()`
- 当前已落地的数据结构：
  - `FeatureResult`
  - `MatchResult`
  - `GeometryResult`
- 当前已验证可用的 backend：
  - `opencv_orb`
  - `opencv_sift`
  - `opencv_bf_ratio`
  - `opencv_ransac`
  - `opencv_usac_magsac`
- 当前已落地但受环境依赖约束的 backend：
  - `superpoint`
    - optional dependency probe
    - lazy import
    - device resolution
    - explicit weights path resolution
    - fail-fast diagnostics
  - `lightglue`
    - optional dependency probe
    - lazy import
    - device resolution
    - explicit weights path resolution
    - fail-fast diagnostics
- 当前验证边界：
  - `run_baseline_frame.py` 只验证单帧 feature / matching / geometry / warp 接口层
  - 当前已经补齐单帧的 seam / crop / blend 静态质量预览
  - 不验证 temporal smoothing parity
  - 不应用它来判断视频质量链路是否与 baseline video 完全一致
  - 已验证：
    - 缺依赖 fail-fast
    - fallback 路径
    - `.venv-methodb` 环境下的真实 `SuperPoint / LightGlue` 单帧成功路径

### 2. 再视频
- 视频层继续复用：
  - `scripts/run_baseline_video.py`
  - `src/stitching/video_stitcher.py`
  - `src/stitching/seam_opencv.py`
  - `src/stitching/cropper.py`
- 当前最小视频 adapter 已落地：
  - 新增 `src/stitching/frame_pair_pipeline.py`
  - `run_baseline_video.py` 现在通过该 adapter 调用：
    - `FeatureResult`
    - `MatchResult`
    - `GeometryResult`
  - `VideoStitcher` 本身不改几何职责，仍只消费 `H / T / canvas_size`
  - 当前已完成的真实 Method B 视频回归：
    - `mine_source_indoor2_left_right`
    - `mine_source_pujiang1_left_right`
    - `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
  - 当前视频回归统一配置：
    - `max_frames=60`
    - `keyframe_every=10`
    - `video_mode=0`
    - `feature_backend=superpoint`
    - `matcher_backend=lightglue`
    - `geometry_backend=opencv_usac_magsac`
    - `device=cpu`
    - `force_cpu=true`
  - 当前结果摘要：
    - `mine_source_indoor2_left_right`
      - `mean_inliers=126.17`
      - `mean_inlier_ratio=0.262`
      - `approx_fps=2.79`
    - `mine_source_pujiang1_left_right`
      - `mean_inliers=401.83`
      - `mean_inlier_ratio=0.787`
      - `approx_fps=1.63`
    - `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
      - `mean_inliers=352.00`
      - `mean_inlier_ratio=0.392`
      - `approx_fps=5.00`
  - 当前结论：
    - Phase 1 已不再只停留在单帧与短视频 smoke，Method B 已在 3 条 60 帧视频回归上跑通。
    - Phase 1 后续重点应从“是否能跑”转向“正式视频级比较结果如何解释”和“何时切入 Phase 2”。
  - 当前正式视频级 compare 入口已落地：
    - `scripts/run_video_compare_suite.py`
    - 正式 preset：
      - `method_a_orb`
      - `method_a_sift`
      - `method_b`
      - `video_mode=1`
      - `reuse_mode=frame0_all`
      - `max_frames=6000`
  - 当前正式 compare suite：
    - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/summary.csv`
    - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/pair_compare.csv`
  - 当前正式 compare 结论：
    - 3 个 pair、9 条 full-length run 全部通过
    - `fallback_frames=0`
    - `errors_count=0`
    - 没有发现 Phase 1 层面的结构性接入问题
    - ORB / SIFT / Method B 在不同数据上没有单一统治者，因此三条正式预设都应保留到 Phase 3

### 介于单帧与视频之间的可选支持子任务
- 名称建议：`frame_quality_preview`
- 目标：
  - 在单帧入口中补齐 seam / crop / blend，尽量接近视频 baseline 的质量链路
  - 便于在 Method B 早期做更可信的 qualitative 对比
- 适合插入的位置：
  - 单帧 backend loader 和 diagnostics 稳定之后
  - 视频 orchestrator 迁移之前
- 不推荐的实现方式：
  - 直接把 `run_baseline_video.py` 中大量 seam/crop 内联逻辑复制到 `run_baseline_frame.py`
- 推荐方向：
  - 抽出共享 frame-level compose helper
  - 或设计单独 `frame_quality_preview` 入口
  - 保持它是支持任务，不与 Method B 主线 backend 接入混做
- 当前已落地的最小实现：
  - 新增 `src/stitching/frame_quality_preview.py`
  - `run_baseline_frame.py` 现通过该 helper 复用 `VideoStitcher.initialize_from_first_frame()`
  - 已对齐的质量参数：
    - `blend`
    - `mb_levels`
    - `seam`
    - `seam_megapix`
    - `seam_dilate`
    - `crop`
    - `lir_method`
    - `lir_erode`
    - `crop_debug`
  - 当前仍未覆盖：
    - temporal smoothing
    - cached execution across frames
    - 完整视频 run bundle 指标

## Method B 的关键决策
- 不训练新模型。
- 默认使用预训练 SuperPoint + LightGlue。
- robust estimator 默认使用 OpenCV `USAC_MAGSAC`。
- Method B 失败时允许配置化 fallback 到 Method A。

## 依赖与运行时设计
### 依赖现状
- 当前环境缺少：
  - `torch`
  - `kornia`
  - `lightglue`
- 当前环境可用：
  - OpenCV
  - `cv2.USAC_MAGSAC`

### 工程约束
- 必须 `lazy import`
- 必须 `GPU optional`
- 必须记录 weights / device / runtime / fallback
- 必须在依赖缺失时给出显式错误，而不是静默退化

### 当前单帧 loader 落地状态（Phase 1 子任务 2）
- 新增共享 runtime helper：
  - `src/stitching/method_b_runtime.py`
- `features.py`
  - `superpoint` 分支已接入：
    - dependency probe
    - device resolution
    - optional weights loading
    - lazy import
    - structured diagnostics
- `matching.py`
  - `lightglue` 分支已接入：
    - dependency probe
    - device resolution
    - optional weights loading
    - lazy import
    - structured diagnostics
- `run_baseline_frame.py`
  - 已新增最小 Method B 配置项：
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
  - 已新增 diagnostics：
    - requested/effective backend
    - fallback events
    - dependency diagnostics
    - device / weights config echo
  - 已验证单帧真实 smoke：
    - `superpoint + lightglue + opencv_usac_magsac`
    - 当前成功样例：
      - `outputs/runs/phase1_methodb_real_smoke_fix1`
    - 当前样例统计：
      - `n_kp_left = 1756`
      - `n_matches_good = 1016`
      - `n_inliers = 415`
      - `inlier_ratio = 0.408`
      - `reprojection_error = 1.660`
  - 新增兼容点：
    - 兼容官方 LightGlue compact 输出中 `matches / scores` 为 batch-wise list 的情况
  - 新增正式环境入口：
    - root `requirements.txt`
    - root `requirements-methodb.txt`
    - `docs/environment.md`
  - 新增多 pair 单帧回归入口：
    - `scripts/run_frame_smoke_suite.py`
    - 默认 smoke pair：
      - `mine_source_indoor2_left_right`
      - `kitti_raw_data_2011_09_28_drive_0119_image_02_image_03`
      - `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
      - `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`
    - 用户别名：
      - `mysourceindoor2 -> mine_source_indoor2_left_right`
    - 当前定位：
      - 只是快速 regression utility
      - 不单独作为 Phase 1 研究结论本身
  - 当前多 pair Method B 回归结果：
    - `outputs/frame_smoke/phase1_methodb_multi_pair_smoke`
    - 4/4 pair 通过真实 `superpoint + lightglue + opencv_usac_magsac`
    - 当前关键统计：
      - `mine_source_indoor2_left_right`：`good=766`，`inliers=150`，`inlier_ratio=0.196`
      - `kitti_raw_data_2011_09_28_drive_0119_image_02_image_03`：`good=1229`，`inliers=582`，`inlier_ratio=0.474`
      - `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`：`good=944`，`inliers=426`，`inlier_ratio=0.451`
      - `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`：`good=827`，`inliers=371`，`inlier_ratio=0.449`
  - 当前多帧抽样回归结果：
    - `outputs/frame_smoke/phase1_methodb_multiframe_sample`
    - 选取 pair：
      - `kitti_raw_data_2011_09_28_drive_0119_image_02_image_03`
      - `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`
    - 抽样帧：
      - `0 / 20 / 40 / 60`
    - 汇总：
      - KITTI：`mean_good=1218.0`，`mean_inliers=598.25`，`mean_inlier_ratio=0.491`，`mean_reprojection_error=1.422`
      - DynamicStereo：`mean_good=830.5`，`mean_inliers=365.25`，`mean_inlier_ratio=0.440`，`mean_reprojection_error=1.780`
  - 当前短视频 smoke：
    - `outputs/runs/phase1_video_adapter_methodb_mode0_smoke_v2`
    - `outputs/runs/phase1_video_adapter_methodb_mode1_smoke_v2`
    - 两条路径均已验证：
      - `feature_backend_effective=superpoint`
      - `matcher_backend_effective=lightglue`
      - `geometry_backend_effective=opencv_usac_magsac`
      - `fallback_frames=0`

## 建议配置项
- `feature_backend`
- `matcher_backend`
- `geometry_backend`
- `device`
- `weights_dir`
- `force_cpu`
- `max_keypoints`
- `resize_long_edge`
- `depth_confidence`
- `width_confidence`
- `filter_threshold`
- `robust_estimator`
- `magsac_thresh`
- `magsac_confidence`
- `magsac_max_iters`
- `fallback_backend`

## 诊断字段建议
- 特征阶段：
  - `feature_backend`
  - `n_keypoints`
  - `feature_runtime_ms`
- 匹配阶段：
  - `matcher_backend`
  - `tentative_matches`
  - `good_matches`
  - `match_runtime_ms`
- 几何阶段：
  - `geometry_backend`
  - `inlier_count`
  - `inlier_ratio`
  - `reprojection_error`
  - `geometry_runtime_ms`
- 运行时：
  - `device`
  - `weights`
  - `fallback_used`
  - `failure_stage`

## 风险与规避
- 风险：依赖复杂、环境不可用
  - 规避：optional dependency + fail-fast diagnostics
- 风险：Method A/B 路线分叉
  - 规避：先抽象统一结果结构，再接到既有 video / diagnostics 路线
- 风险：Method B 引入新的输出字段破坏旧实验
  - 规避：兼容保留旧 bundle 字段，在 `debug.json` 里新增 Method B 专属字段

## 验收标准(DoD)
- 已有正式环境文档，能清楚区分 baseline env 与 Method B env 的安装和使用方式。
- root requirements 已固定：
  - `requirements.txt`
  - `requirements-methodb.txt`
- 单帧 smoke / regression 已有正式入口：
  - `scripts/run_baseline_frame.py`
  - `scripts/run_frame_smoke_suite.py`
- 单帧能在 Method A / Method B 间切换。
- 视频路径能复用同一套 seam/crop/blend/diagnostics。
- 依赖缺失、权重缺失、GPU 不可用时有明确错误或 fallback 记录。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/06_method2_strong_matching/06_method2_strong_matching.md | 将骨架文档升级为 Method B 专项设计说明 | Codex | 完成 |
