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
    - 当前统一正式环境下的真实 `SuperPoint / LightGlue` 单帧成功路径

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
    - `scripts/eval_method_compare_matrix.py`
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
    - root `requirements-methodb.txt`（兼容 alias）
    - `docs/environment.md`
  - 新增多 pair 单帧回归入口：
    - `scripts/legacy/run_frame_smoke_suite.py`
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

## 2026-03-23 复盘：为什么当前 Method B 会显得偏弱
### 复盘结论
- 当前 Phase 3 中 `method_b` 的整体偏弱，不应直接解释为 `SuperPoint + LightGlue` 本身不适合项目。
- 本次复查确认了两个更像工程接入 / preset 选择的问题：
  - 旧正式 compare 实际跑在偏速度的 LightGlue 默认配置上，而不是官方 README 中偏准确率的配置。
  - 旧 `resize_long_edge` 对 Method B 没有真正按官方方式作用到 `SuperPoint.extract()` 的 preprocess，导致想要提高 SuperPoint 输入分辨率时，实际仍常常停留在 package default `1024`。

### 具体问题 1：旧 Method B compare 使用的是 speed-oriented implicit preset
- 旧默认配置实质上接近：
  - `max_keypoints=2048`
  - `SuperPoint.extract()` 默认 resize `1024`
  - `LightGlue depth_confidence=0.95`
  - `LightGlue width_confidence=0.99`
  - `filter_threshold=0.1`
- 这更接近官方 README 所说的 “good trade-off between speed and accuracy”，而不是 accuracy mode。
- LightGlue 官方给出的 accuracy 建议是：
  - 更多 keypoints
  - `depth_confidence=-1`
  - `width_confidence=-1`
- 因此旧 compare 更像 “balanced/speed preset 对比”，不适合作为最终 Method B 能力上限。

### 具体问题 2：旧 `resize_long_edge` 语义对 Method B 不正确
- 旧实现先在外层把图像 resize，再调用 `SuperPoint.extract(tensor)`。
- 但官方 `Extractor.extract()` 自身还会按 preprocess conf 再做一次内部 resize。
- 结果是：
  - `resize_long_edge` 无法真正把 SuperPoint 提升到更高输入分辨率
  - 对于高分辨率 `mine_source` 这类视频，Method B 的 keypoints 和匹配上限被压低
- 本次修复后，Method B 的 `resize_long_edge` 语义改为：
  - `None`：沿用 package default `1024`
  - `> 0`：显式传给 `SuperPoint.extract(..., resize=...)`
  - `<= 0`：禁用 auto-resize

### 本次模块级修复
- `src/stitching/features.py`
  - 修复 Method B 的 `resize_long_edge` 语义，使其真正作用到 `SuperPoint.extract()`
  - `max_keypoints <= 0` 现在表示 “不设上限”
  - 在 `FeatureResult.meta` 中新增：
    - `superpoint_preprocess_resize`
    - `payload_image_size`
- `src/stitching/matching.py`
  - 在 `MatchResult.meta` 中新增 LightGlue 运行细节：
    - `stop_layer`
    - `prune0/prune1` 统计
- `scripts/run_baseline_frame.py`
  - CLI help 更新为新的 Method B 语义
- `scripts/run_baseline_video.py`
  - CLI help 更新为新的 Method B 语义

### 当前推荐 Method B accuracy preset
- 用于后续正式 compare 的推荐参数：
  - `feature_backend=superpoint`
  - `matcher_backend=lightglue`
  - `geometry_backend=opencv_usac_magsac`
  - `max_keypoints=4096`
  - `resize_long_edge=1536`
  - `depth_confidence=-1`
  - `width_confidence=-1`
  - `filter_threshold=0.1`
- 选择理由：
  - 相比“完全不限制 keypoints + 完全不 resize”，该 preset 更适合当前 CPU 环境
  - 相比旧 implicit preset，它能明显抬高 Method B 的匹配数量和内点数

### 本次代表性验证
- 单帧复查（frame 0）：
  - `kitti_raw_data_2011_09_26_drive_0002_image_02_image_03`
    - 旧：`1187 kp / 874 matches / 365 inliers`
    - 新：`2134 kp / 1566 matches / 647 inliers`
  - `dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right`
    - 旧：`1756 kp / 1016 matches / 415 inliers`
    - 新：`3108 kp / 1566 matches / 685 inliers`
  - `mine_source_walking_left_right`
    - 旧：`2048 kp / 821 matches / 148 inliers`
    - 新：`4096 kp / 1596 matches / 432 inliers`
- 短视频复查：
  - `outputs/analysis/methodb_recheck_video_summary.json`
  - `kitti_raw_data_2011_09_26_drive_0002_image_02_image_03`
    - 旧：`mean_inliers=365`, `approx_fps=4.06`
    - 新：`mean_inliers=647`, `approx_fps=3.41`
  - `mine_source_walking_left_right`
    - 旧：`mean_inliers=148`, `mean_inlier_ratio=0.180`, `approx_fps=3.70`
    - 新：`mean_inliers=432`, `mean_inlier_ratio=0.271`, `approx_fps=2.54`

### 当前结论
- Method B 之前“显得很差”，至少部分是 preset 和 SuperPoint preprocess 接入问题，而不是纯算法结论。
- 本次修复后，Method B 的匹配和 inlier 质量已经明显改善。
- 2026-03-24 已完成正式 full-length 方法 compare 刷新：
  - `outputs/phase3/phase3_kitti_methods_acc_v2/method_summary.csv`
  - `outputs/phase3/phase3_dynamicstereo_methods_acc_v2/method_summary.csv`
  - `outputs/phase3/phase3_minesource_methods_acc_v2/method_summary.csv`
  - `outputs/phase3/phase3_overall_methods_acc_v2/overall_method_summary.csv`
- 刷新后的正式结论：
  - KITTI：
    - `method_b mean_inliers≈594.00`，已高于 ORB/SIFT
    - 但 `mean_inlier_ratio≈0.393`、`fps≈17.66` 仍低于 ORB/SIFT
  - DynamicStereo：
    - `method_b mean_inliers≈527.67`，与 ORB 接近，明显高于 SIFT
    - 但 `mean_inlier_ratio≈0.396`、`fps≈7.80` 仍明显偏弱
  - `mine_source`：
    - `method_b mean_inliers≈842.59`，已高于 ORB/SIFT
    - 但 `mean_inlier_ratio≈0.641`、`fps≈7.69` 仍低于 Method A
  - overall：
    - `method_b mean_inliers≈748.88` 最高
    - `method_a_orb mean_inlier_ratio≈0.767` 最高
    - `method_a_sift fps≈18.36` 最高
- 因此 Method B 当前应被表述为：
  - 在 accuracy preset 下能够稳定换来更高的内点数量
  - 但当前 CPU 路线下仍显著更慢，且内点率并不占优

## 2026-03-24 复盘：当前还能怎么继续优化 Method B
### 原则
- 不直接覆盖当前正式 compare 使用的 `method_b_accuracy_v1`。
- 后续优化必须作为新增候选 preset 并存验证，而不是直接把正式基线改掉。
- 当前正式基线保持：
  - `max_keypoints=4096`
  - `resize_long_edge=1536`
  - `depth_confidence=-1`
  - `width_confidence=-1`
  - `filter_threshold=0.1`

### 当前 trade-off 的更细解释
- 当前 `fixed_geometry + frame0_all` 正式 compare 里，Method B 的“慢”主要包含两层：
  - 初始化阶段更慢：
    - `SuperPoint + LightGlue` 首帧 feature/matching 代价显著高于 ORB/SIFT
  - steady-state 也偏慢：
    - 不同方法得到的几何和 output bbox/crop 可能不同，导致后续每帧 compose 代价并不完全相同
- 当前 `inlier_ratio` 偏低也不能直接解读为“几何更差”：
  - LightGlue 在 accuracy preset 下会保留更多中低置信度 matches
  - 这会抬高 `mean_inliers`，但也会拉低 `inlier_ratio`
  - 因此后续需要同时看：
    - `inlier_count`
    - `inlier_ratio`
    - `reprojection_error`
    - `inlier spatial coverage`

### 当前看到的工程信号
- 在代表性 run 中，accuracy preset 下：
  - `stop_layer=9`
  - `prune0_mean≈9`, `prune1_mean≈9`
- 这表明在当前 accuracy preset 下，LightGlue 基本跑满层数、几乎不提前停止，也没有真正利用 pruning。
- 同时当前正式数据域的源分辨率主要是：
  - KITTI：`1242x375`
  - DynamicStereo / `mine_source`：`1280x720`
- 因而 `resize_long_edge=1536` 对这些数据实际上是上采样，而不是单纯“保留更多原始细节”。
- 这解释了为什么当前 preset 能把内点数抬得很高，但也可能拖慢速度、引入更多低质量 keypoints/matches，从而压低 `inlier_ratio`。

### 历史候选项复盘
- 已探索过但未保留在当前主框架中的方向：
  - `no_upsample`
  - 更严格的 `filter_threshold=0.15`
  - moderate `LightGlue` adaptivity
- 当前保留的唯一活跃候选：
  - `kp3072_v1`
    - 核心想法：
      - 在保持 accuracy 路线的前提下，将 `max_keypoints` 从 `4096` 下调到 `3072`
    - 预期收益：
      - matching runtime 降低
      - 低置信度尾部 matches 减少
    - 风险：
      - 极纹理场景和 `mine_source` 真实视频上可能出现 `mean_inliers` 明显下降

### 历史候选项为何未继续保留
- `no_upsample`
  - 风险是 `mean_inliers` 下滑幅度不可控，且当前主框架目标是保留稳定的正式 compare baseline。
- `filter_threshold=0.15`
  - 更容易直接丢掉有效 matches，当前没有形成比 `accuracy_v1` 更稳的跨数据域收益。
- moderate `LightGlue` adaptivity
  - 速度收益有吸引力，但最容易把当前高内点数优势压回去，因此不作为当前主框架内的活跃候选。

### 当前唯一保留的安全候选项
- 候选：`max_keypoints=3072`
  - 核心想法：
    - 保持较高召回，但减少 LightGlue 输入规模
  - 预期收益：
    - matching runtime 降低
    - 低置信度尾部 matches 减少
  - 风险：
    - 极纹理场景的 `mean_inliers` 可能下降

### 当前推荐的执行顺序
- 当前主框架内只保留：
  - `kp3072_v1`
- 其余探索项：
  - 已移出主框架，仅保留为历史讨论材料
- 不建议当前优先做：
  - 重新训练 SuperPoint / LightGlue
  - 重写 matcher 边界
  - 替换 MAGSAC 实现
  - 直接改正式 compare 默认 preset

### 2026-03-24 已实现的安全优化机制
- 已新增：
  - `src/stitching/method_b_presets.py`
  - `scripts/legacy/run_method_b_preset_sweep.py`
- 当前 active preset 命名固定为：
  - `accuracy_v1`
  - `kp3072_v1`
- 当前已移出主框架的 exploratory preset：
  - `no_upsample_v1`
  - `filter015_v1`
- 当前 active preset 的目标是：
  - 只做并行验证
  - 不覆盖正式 compare 默认值
  - 先筛选“值得做 full-length 复验的候选项”

### 2026-03-24 `kp3072_v1` full-length 复验结论
- 已完成：
  - KITTI full-length：`outputs/phase3/phase3_kitti_methods_kp3072_v1/`
  - DynamicStereo full-length：`outputs/phase3/phase3_dynamicstereo_methods_kp3072_v1/`
  - `mine_source` full-length：`outputs/phase3/phase3_minesource_methodb_kp3072_v1/`
  - 四方法对照：`outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/`
- 结果：
  - `kp3072_v1` 在 overall 上只换来了轻微的 `inlier_ratio / fps / reprojection` 改善
  - 但 `mean_inliers` 从约 `748.88` 明显降到约 `609.58`
  - 且在 `mine_source` 上从约 `842.59` 明显回退到约 `628.65`
- 当前结论：
  - `kp3072_v1` 不能替换正式 `accuracy_v1`
  - 但它仍可作为一个“轻量降 keypoints 的对照候选”保留在讨论中

### 2026-03-24 richer metrics 已落地
- `geometry.py`
  - 新增 `inlier_spatial_coverage`
- `video_stitcher.py`
  - 新增 seam-band 指标：
    - `seam_band_illuminance_diff`
    - `seam_band_gradient_disagreement`
    - `seam_band_mask`
- `temporal.py`
  - `compute_frame_absdiff_mean()` 已支持 mask-aware 计算
- `run_baseline_video.py`
  - 新增视频级导出：
    - `init_ms_mean`
    - `per_frame_ms_mean`
    - `avg_feature_runtime_ms_left`
    - `avg_feature_runtime_ms_right`
    - `avg_matching_runtime_ms`
    - `avg_geometry_runtime_ms`
    - `mean_reprojection_error`
    - `mean_inlier_spatial_coverage`
    - `mean_seam_band_illuminance_diff`
    - `mean_seam_band_gradient_disagreement`
    - `mean_seam_band_flicker`

### 2026-03-24 candidate sweep 结果
- 结果目录：
  - `outputs/analysis/methodb_preset_sweep_v2/preset_summary.csv`
  - `outputs/analysis/methodb_preset_sweep_v2/summary.csv`
- representative pairs：
  - `kitti_raw_data_2011_09_26_drive_0002_image_02_image_03`
  - `dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right`
  - `mine_source_walking_left_right`
- 聚合结果：
  - `accuracy_v1`
    - `mean_inliers ≈ 588.0`
    - `mean_inlier_ratio ≈ 0.374`
    - `approx_fps ≈ 3.68`
    - `mean_reprojection_error ≈ 1.585`
  - `kp3072_v1`
    - `mean_inliers ≈ 539.7`
    - `mean_inlier_ratio ≈ 0.366`
    - `approx_fps ≈ 5.21`
    - `mean_reprojection_error ≈ 1.492`

### 当前结论
- 当前正式 baseline 仍保持 `accuracy_v1`。
- 当前最值得继续做 full-length 复验的 candidate 是 `kp3072_v1`。
- 原因：
  - 它在 KITTI 上几乎保住了 `accuracy_v1` 的内点数，同时明显更快
  - 在 DynamicStereo 上甚至略有提升
  - overall `reprojection_error` 更低
- 但暂不直接升格为正式 baseline，因为：
  - 在 `mine_source_walking_left_right` 上，它的 `mean_inliers` 从 `432` 降到 `272`
  - 说明该 candidate 仍具有明显 data-dependent 风险
- 其余 exploratory candidate 已移出当前主框架：
  - 没有形成足够清晰的收益
  - 且会扩大当前 Method B 配置面

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
  - `requirements-methodb.txt`（兼容 alias）
- 单帧 smoke / regression 已有正式入口：
  - `scripts/run_baseline_frame.py`
  - `scripts/legacy/run_frame_smoke_suite.py`
- 单帧能在 Method A / Method B 间切换。
- 视频路径能复用同一套 seam/crop/blend/diagnostics。
- 依赖缺失、权重缺失、GPU 不可用时有明确错误或 fallback 记录。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/06_method2_strong_matching/06_method2_strong_matching.md | 将骨架文档升级为 Method B 专项设计说明 | Codex | 完成 |
