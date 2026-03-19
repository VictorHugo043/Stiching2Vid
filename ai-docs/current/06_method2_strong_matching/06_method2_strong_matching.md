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
- 当前仅做 fail-fast 占位的 backend：
  - `superpoint`
  - `lightglue`
- 当前验证边界：
  - `run_baseline_frame.py` 只验证单帧 feature / matching / geometry / warp 接口层
  - 不验证 seam / crop / temporal smoothing parity
  - 不应用它来判断视频质量链路是否与 baseline video 完全一致

### 2. 再视频
- 视频层继续复用：
  - `scripts/run_baseline_video.py`
  - `src/stitching/video_stitcher.py`
  - `src/stitching/seam_opencv.py`
  - `src/stitching/cropper.py`

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
- 单帧能在 Method A / Method B 间切换。
- 视频路径能复用同一套 seam/crop/blend/diagnostics。
- 依赖缺失、权重缺失、GPU 不可用时有明确错误或 fallback 记录。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/06_method2_strong_matching/06_method2_strong_matching.md | 将骨架文档升级为 Method B 专项设计说明 | Codex | 完成 |
