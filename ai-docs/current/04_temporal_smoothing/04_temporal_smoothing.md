# 04_temporal_smoothing

## 任务目标
- 保留 temporal smoothing 的阶段定位，并明确当前已实现内容与后续需要补齐的 temporal evaluation 边界。

## 当前已实现内容
- 当前已实现的是几何轨迹平滑，而不是 seam mask 平滑。
- 主要模块：
  - `src/stitching/temporal.py`
  - `scripts/run_baseline_video.py`
- 当前方法：
  - `none`
  - `ema`
  - `window`
- 当前输出：
  - `jitter_timeseries.csv`
  - `overlay_raw_*.png`
  - `overlay_sm_*.png`
  - `metrics_preview.json` 中的 `mean_jitter_* / p95_jitter_*`

## 当前限制
- 在 `fixed geometry` 或长期复用缓存 `H` 的路径下，`jitter` 可能退化为 0。
- 当前尚未实现：
  - seam temporal smoothing
  - optical-flow compensated temporal coherence
  - object-aware temporal metrics

## 文档归属
- baseline 与运行模式语义见：
  - `ai-docs/current/03_baseline_video_pipeline/03_baseline_video_pipeline.md`
- Dynamic seam 与 meaningful temporal evaluation 设计见：
  - `ai-docs/current/09_dynamic_seam_and_temporal_eval/09_dynamic_seam_and_temporal_eval.md`
- 实验协议见：
  - `ai-docs/current/05_evaluation/05_evaluation.md`

## 下一步
- 先完成 Phase 0 的运行模式拆分与评测协议冻结，再推进 Phase 2 的 seam temporal smoothing。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/04_temporal_smoothing/04_temporal_smoothing.md | 将旧骨架文档更新为 temporal smoothing 现状与边界说明 | Codex | 完成 |
