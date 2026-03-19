# 14_open_issues_and_next_steps

## 使用规则
- 统一记录当前 blocker、未决问题和下一步优先队列。
- 每次实施结束后都必须更新本文件。

## 当前 open issues
| ID | 状态 | 问题 | 影响 | 处理建议 |
| --- | --- | --- | --- | --- |
| ISSUE-20260319-01 | open | 根仓库缺少正式依赖文件（无 `requirements.txt / pyproject.toml / environment.yml`） | Method B 与 GUI 的环境复现成本高 | 在 Phase 0 或 Phase 1 早期补环境说明，必要时新增最小依赖文档 |
| ISSUE-20260319-02 | open | 当前环境缺少 `torch / kornia / lightglue` | Method B 无法直接落地 | 在 Phase 1 设计中采用 optional dependency、lazy import 与 fallback |
| ISSUE-20260319-03 | open | `video_mode=1` 下 `jitter` 容易退化为 0 | temporal evaluation 容易被误解 | 已通过 `geometry_mode / jitter_meaningful` 导出降低误读风险；后续仍需在 `adaptive_update` 与 dynamic seam 中恢复更有意义的 temporal evaluation |
| ISSUE-20260319-06 | open | 下游实验脚本尚未统一消费 `geometry_mode / jitter_meaningful` 新字段 | Phase 0 的 schema 冻结还未完全闭环 | 下一步检查 `scripts/ablate_temporal.py`、`scripts/ablate_seam.py` 及后续 summary 逻辑对新字段的兼容性 |
| ISSUE-20260319-04 | open | 当前 seam 模块是 OpenCV mask 风格，无法直接承载 object-centered MRF seam | Dynamic seam advanced 实现难度高 | 先做兼容式 MVP，再单独设计新 seam backend |
| ISSUE-20260319-05 | open | `scripts/ablate_crop.py`、`scripts/ablate_video_reuse.py` 在旧文档中出现，但仓库中不存在 | 文档与代码曾有漂移 | 后续若需要，再按 Phase 0 / Phase 3 新增，不再视作已完成 |

## 接下来最先做的 3 件事
1. 继续完成 Phase 0 的 schema 冻结，检查下游实验脚本与汇总逻辑对 `geometry_mode / jitter_meaningful` 的兼容性。
2. 在不改算法能力的前提下，统一 `metrics_preview/debug/transforms` 的字段说明和消费方式。
3. 进入 Phase 1 前，先开一条 Method B 的最小子任务计划，明确单帧接入、依赖、weights、fallback 和 diagnostics 字段。

## 当前建议的下一步实施入口
- 先读：
  - `08_project_status_and_master_plan`
  - `05_evaluation`
  - `06_method2_strong_matching`
  - `09_dynamic_seam_and_temporal_eval`
  - `10_execution_workflow`
- 再以 `IMP-*` 的形式写下一步最小实施计划。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/14_open_issues_and_next_steps/14_open_issues_and_next_steps.md | 新增 open issues 与 next steps 文档 | Codex | 完成 |
