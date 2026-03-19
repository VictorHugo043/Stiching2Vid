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
- 时序项
  - `jitter_raw`
  - `jitter_sm`
  - `p95_jitter_raw`
  - `p95_jitter_sm`
  - `jitter_meaningful`

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
  - `keyframe_update`
  - `fixed seam`
  - `no smoothing`

### Phase 2
- 在最佳方法下比较：
  - `fixed seam vs keyframe seam vs trigger seam`
  - `no smoothing vs EMA vs window`
  - `fixed_geometry vs keyframe_update vs adaptive_update`

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
- 后续建议新增：
  - 统一 experiment driver
  - 独立 metrics 模块
  - summary CSV / plots 生成脚本

## 验收标准(DoD)
- 为每个实验输出统一 bundle。
- 至少固定一套 final report 可复用的表格与图像清单。
- 文档中明确哪些指标当前可直接复用，哪些需要新增实现。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/05_evaluation/05_evaluation.md | 将骨架文档升级为正式评测协议与实验矩阵 | Codex | 完成 |
