# 11_decision_log

## 使用规则
- append-only，不回写历史结论。
- 每条记录都必须带唯一 ID。
- 如后续推翻旧决策，新增一条新决策并引用旧 ID，而不是修改旧条目。

## 决策记录
| ID | 日期 | 决策 | 原因 | 影响 | 替代方案 | 关联文件 |
| --- | --- | --- | --- | --- | --- | --- |
| DEC-20260319-01 | 2026-03-19 | 沿用现有 `ai-docs/current/00~07` 编号体系，不新建平行文档树 | 现有结构已经稳定，且按阶段编号组织清晰 | 后续所有新文档采用 `08+` 编号追加 | 新建平行 wiki 式目录，未采用 | `ai-docs/current/*` |
| DEC-20260319-02 | 2026-03-19 | 总体优先级固定为 `Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4` | Method B、Dynamic seam、evaluation 是核心研究任务，GUI 是包装层 | 后续排期必须先完成基线冻结与 Method B 设计，再做 GUI | 先做 GUI，未采用 | `08_project_status_and_master_plan` |
| DEC-20260319-03 | 2026-03-19 | Method B 默认采用预训练 `SuperPoint + LightGlue + OpenCV USAC_MAGSAC` | 当前项目更适合接入预训练与现成 robust estimator，而非训练新模型 | Method B 设计将围绕 optional dependency、lazy import、fallback 展开 | 训练新模型；引入独立 MAGSAC++ 实现，未采用 | `06_method2_strong_matching` |
| DEC-20260319-04 | 2026-03-19 | Dynamic seam 采用“两层路线”：先兼容 OpenCV seam finder 的 MVP，再做 advanced 新 seam backend | 当前 seam 模块是 mask 风格，直接复现 object-centered MRF 风险过高 | Phase 2 优先做 `keyframe/trigger seam policy + object-aware penalty + seam smoothing` | 一步到位完整复现 ECCV 2018 MRF，未采用 | `09_dynamic_seam_and_temporal_eval` |
| DEC-20260319-05 | 2026-03-19 | 运行模式文档层显式拆分为 `fixed_geometry / keyframe_update / adaptive_update` | 当前 `reuse_mode` 名称与真实行为不完全一致，且 `jitter` 解释依赖模式 | Phase 0 必须先修正文档与评测语义 | 继续沿用旧术语且不解释差异，未采用 | `03_baseline_video_pipeline`, `05_evaluation`, `09_dynamic_seam_and_temporal_eval` |
| DEC-20260319-06 | 2026-03-19 | GUI 只做 thin wrapper，不重写核心 stitching pipeline | 当前核心 pipeline 与 artefacts 已可复用，过早侵入核心会增加返工 | GUI 只复用 CLI、I/O 和 outputs bundle | 在 GUI 中重写运行逻辑，未采用 | `08_project_status_and_master_plan` |
| DEC-20260319-07 | 2026-03-19 | Phase 0 先保留 `run_baseline_video.py` 的 legacy CLI 参数，但在导出 artefacts 中显式冻结 `geometry_mode` 和 `jitter_meaningful` | 当前首要目标是稳定语义与评测解释，而不是提前破坏 CLI 兼容性 | `transforms.csv`、`metrics_preview.json`、`debug.json` 以派生字段承载模式语义；当前仅导出 `fixed_geometry` 或 `keyframe_update`，`adaptive_update` 保留给后续实现 | 立即重命名 CLI 参数并切断旧字段，未采用 | `scripts/run_baseline_video.py`, `03_baseline_video_pipeline`, `05_evaluation` |
| DEC-20260319-08 | 2026-03-19 | `scripts/ablate_temporal.py` 与 `scripts/ablate_seam.py` 保留为 legacy exploratory helpers，不再作为当前 Phase 0 / Phase 1 的正式入口或验收依赖 | 这两个脚本只是早期便捷封装，继续把它们当成 schema 冻结的一部分会误导排期 | Phase 0 可在不改这两个脚本消费逻辑的前提下闭环；未来若需要正式实验驱动，另建统一 experiment driver | 立即删除两个脚本；继续把它们当主线入口，均未采用 | `05_evaluation`, `08_project_status_and_master_plan`, `scripts/ablate_temporal.py`, `scripts/ablate_seam.py` |

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/11_decision_log/11_decision_log.md | 新增 append-only 决策日志并写入首批全局决策 | Codex | 完成 |
