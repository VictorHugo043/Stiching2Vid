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
  - `fixed_geometry / keyframe_update / adaptive_update` 的语义边界已固定，且当前实现不会误导性导出 `adaptive_update`。
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

### 验收标准
- 单帧先跑通，再扩展到视频。
- Method A / Method B 能共用 `VideoStitcher`、crop、seam、blend、diagnostics。

### 风险与规避
- 风险：环境缺依赖。
  - 规避：optional dependency、lazy import、fallback。

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
