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
- `outputs/phase3/<suite_id>/method_summary.csv`
- `outputs/phase3/<suite_id>/dynamic_preset_summary.csv`
- `outputs/phase3/<suite_id>/pair_coverage.csv`
- `outputs/ablations/<pair_id>/seam/`
- `outputs/ablations/<pair_id>/...`
- 正式导出入口：
  - `scripts/export_dynamic_visuals.py`
  - `scripts/export_report_figures.py`
  - `scripts/internal/summarize_method_compare_dataset.py`
  - `scripts/internal/summarize_method_compare_overall.py`

## 当前命名口径（2026-03-25 更新）
- formal compare/export 脚本新生成的 suite id、summary filename 和 markdown/figure 标题不再使用 `phase2 / phase3` 前缀。
- 当前文档里仍引用的大量 `outputs/video_compare/phase2_*` 与 `outputs/phase3/phase3_*` 路径，都是已经冻结的历史正式结果目录。
- 新生成 artefact 的默认 summary 命名：
  - dataset-level：`dataset_summary.md`
  - overall：`overall_summary.md`
  - dynamic visuals：`visual_summary.md`
  - report figures：`figures.md`

## 当前正式推荐素材（2026-03-23）
- Phase 1 方法对比正式入口：
  - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/summary.csv`
  - `outputs/video_compare/phase1_video_compare_fixedgeom_full_v1/pair_compare.csv`
- Phase 2 dynamic seam 正式入口：
  - `outputs/video_compare/phase2_dynamic_compare_full_v1/preset_summary.csv`
  - `outputs/video_compare/phase2_dynamic_compare_full_v1/pair_compare.csv`
  - `outputs/video_compare/phase2_dynamic_compare_full_v1/visual_summary.md`
- Phase 3 KITTI full-length 正式入口：
  - 方法 compare richer-metrics 刷新版：
    - `outputs/phase3/phase3_kitti_methods_rich_v3/method_summary.csv`
    - `outputs/phase3/phase3_kitti_methods_rich_v3/method_pair_compare.csv`
  - Dynamic seam 正式表：
  - `outputs/phase3/phase3_kitti_full_v1/dynamic_preset_summary.csv`
  - `outputs/phase3/phase3_kitti_full_v1/dynamic_pair_compare.csv`
  - `outputs/phase3/phase3_kitti_full_v1/pair_coverage.csv`
  - `outputs/phase3/phase3_kitti_methods_rich_v3/phase3_kitti_summary.md`
  - `outputs/phase3/phase3_dynamicstereo_methods_rich_v3/method_summary.csv`
  - `outputs/phase3/phase3_dynamicstereo_methods_rich_v3/method_pair_compare.csv`
  - `outputs/phase3/phase3_dynamicstereo_full_v1/dynamic_preset_summary.csv`
  - `outputs/phase3/phase3_dynamicstereo_full_v1/pair_coverage.csv`
  - `outputs/phase3/phase3_minesource_methods_rich_v3/method_summary.csv`
  - `outputs/phase3/phase3_minesource_methods_rich_v3/method_pair_compare.csv`
  - `outputs/phase3/phase3_minesource_full_v1/dynamic_preset_summary.csv`
  - `outputs/phase3/phase3_minesource_full_v1/pair_coverage.csv`
  - `outputs/phase3/phase3_overall_methods_rich_v3/overall_method_summary.csv`
  - `outputs/phase3/phase3_overall_methods_rich_v3/overall_method_by_dataset.csv`
  - `outputs/phase3/phase3_overall_full_v1/overall_dynamic_preset_summary.csv`
  - `outputs/phase3/phase3_overall_methods_rich_v3/phase3_overall_summary.md`
  - `outputs/phase3/phase3_overall_methods_rich_v3/figures/figure_manifest.csv`
  - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_core_metrics.png`
  - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_runtime_metrics.png`
  - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_quality_metrics.png`
  - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_temporal_metrics.png`
  - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_by_dataset.png`
  - Method B candidate 对照：
    - `outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/overall_method_compare.csv`
    - `outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/by_dataset_method_compare.csv`
    - `outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/method_b_accuracy_vs_kp3072_delta.csv`
    - `outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/summary.md`
- 当前 final report 最值得保留的 KITTI 图表：
  - 方法总表：
    - `method_summary.csv`
  - 方法 pair 级对比：
    - `method_pair_compare.csv`
  - dynamic seam preset 总表：
    - `dynamic_preset_summary.csv`
  - dynamic seam pair 级对比：
    - `dynamic_pair_compare.csv`
  - dynamic seam 代表性截图：
    - `outputs/video_compare/phase3_kitti_full_v1__dynamic/visual_summary.md`
    - `outputs/video_compare/phase3_dynamicstereo_full_v1__dynamic/visual_summary.md`
    - `outputs/video_compare/phase3_minesource_full_v1__dynamic/visual_summary.md`
  - richer metrics 方法图表：
    - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_core_metrics.png`
    - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_runtime_metrics.png`
    - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_quality_metrics.png`
    - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_temporal_metrics.png`
    - `outputs/phase3/phase3_overall_methods_rich_v3/figures/method_by_dataset.png`

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
