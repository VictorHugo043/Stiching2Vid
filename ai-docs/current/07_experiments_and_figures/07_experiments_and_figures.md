# 07_experiments_and_figures

## 任务目标
- 固化 final report / final demo 的实验清单、图表清单和素材导出规则，避免后期重复跑实验或缺关键截图。

## 推荐保留的实验结果
### 1. 方法对比
- `Method A vs Method B`
- 至少保留：
  - 1 组动态场景
  - 1 组较稳定场景
  - 1 组真实自采视频

### 2. seam 对比
- `fixed seam`
- `keyframe seam`
- `trigger seam`

### 3. temporal 对比
- `no smoothing`
- `EMA`
- `window`

### 4. geometry mode 对比
- `fixed_geometry`
- `keyframe_update`
- `adaptive_update`

## final report 必保留表格
- 表 1：Method A vs Method B 总表
- 表 2：seam policy ablation
- 表 3：temporal smoothing ablation
- 表 4：runtime / quality trade-off

## final report 必保留图像
- 匹配图与 inlier 图
- stitched frame qualitative comparison
- seam overlay
- overlap diff
- `overlay_raw` vs `overlay_sm`
- jitter / temporal coherence 曲线

## 建议的图表来源
- `outputs/runs/<run_id>/snapshots/`
- `outputs/video_compare/<suite_id>/summary.csv`
- `outputs/video_compare/<suite_id>/pair_compare.csv`
- `outputs/ablations/<pair_id>/seam/`
- `outputs/ablations/<pair_id>/...`
- 后续统一 summary CSV / plot 脚本

## 结果组织建议
- 所有正式图表必须能追溯到具体 `run_id`。
- 正式表格只引用冻结协议下生成的 run bundle。
- 每张图保留：
  - pair id
  - frame index
  - method / seam / temporal / geometry mode
  - 对应 `run_id`

## 与评测协议的关系
- 本文档只规定“保留什么素材”。
- 指标和变量控制以 `05_evaluation` 为准。
- Dynamic seam 的特殊 qualitative case 以 `09_dynamic_seam_and_temporal_eval` 为准。

## 验收标准(DoD)
- 明确 final report 必保留的表格与图片。
- 每类图表都能映射到可复现的输出目录和 `run_id`。
- 后续脚本只需要按本清单导出素材，不再重新决定图表范围。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/07_experiments_and_figures/07_experiments_and_figures.md | 将骨架文档升级为 final report 图表与素材清单 | Codex | 完成 |
