# 14_open_issues_and_next_steps

## 使用规则
- 统一记录当前 blocker、未决问题和下一步优先队列。
- 每次实施结束后都必须更新本文件。

## 当前 open issues
| ID | 状态 | 问题 | 影响 | 处理建议 |
| --- | --- | --- | --- | --- |
| ISSUE-20260319-01 | closed | 根仓库缺少正式依赖文件（无 `requirements.txt / pyproject.toml / environment.yml`） | Method B 与 GUI 的环境复现成本高 | 已补 root `requirements.txt`、`requirements-methodb.txt` 与 `docs/environment.md` |
| ISSUE-20260319-02 | closed | 当前 Method B 正式环境已经按 `requirements-methodb.txt` 安装并验证通过；`.venv-methodb` 已在多 pair 单帧、多帧抽样和短视频 smoke 上跑通 | Method B 环境不再是当前主 blocker | 后续默认沿用当前 `.venv-methodb`；如环境再漂移，再按 `docs/environment.md` 重建 |
| ISSUE-20260319-03 | open | `video_mode=1` 下 `jitter` 容易退化为 0 | temporal evaluation 容易被误解 | 已通过 `geometry_mode / jitter_meaningful` 导出降低误读风险；后续仍需在 `adaptive_update` 与 dynamic seam 中恢复更有意义的 temporal evaluation |
| ISSUE-20260319-06 | deferred | 旧 ablation 脚本尚未统一消费 `geometry_mode / jitter_meaningful` 新字段 | 若继续依赖旧脚本会造成 Phase 0 收尾不必要拖延 | 已将 `scripts/ablate_temporal.py`、`scripts/ablate_seam.py` 降级为 legacy helpers；正式 experiment driver 在 Phase 3 单独建设 |
| ISSUE-20260319-04 | open | 当前 seam 模块是 OpenCV mask 风格，无法直接承载 object-centered MRF seam | Dynamic seam advanced 实现难度高 | 先做兼容式 MVP，再单独设计新 seam backend |
| ISSUE-20260319-05 | closed | `scripts/ablate_crop.py`、`scripts/ablate_video_reuse.py` 的历史文档漂移已不再构成当前主线问题；相关旧描述已移出当前工作流 | 不再影响 Phase 1 / Phase 2 推进 | 保留为历史背景，不再作为活跃 issue 跟踪 |
| ISSUE-20260319-07 | closed | `run_baseline_video.py` 已通过 `frame_pair_pipeline` 接到结果对象层，视频入口已可使用 Method B backend 配置 | Method B 现在可以复用视频 orchestrator 的现有 seam/crop/blend/cache 路线 | 后续若要继续演进，只需在当前 adapter 基础上补更长时长回归和实验，不再把“未接到结果对象层”视为 issue |
| ISSUE-20260319-08 | partial | `run_baseline_frame.py` 与 `run_baseline_video.py` 在质量链路上仍有边界差距：单帧入口现已补齐 seam / crop / blend 静态路径，但仍没有 temporal / cache / 完整 run bundle | 用户若忽略边界，仍可能把单帧 smoke 输出误认为完整视频行为 | 已通过 `frame_quality_preview` 缩小静态质量差距；后续仅在需要时再补 diagnostics parity，不把 temporal/cache 强塞进单帧入口 |

## 接下来最先做的 3 件事
1. 按 Phase 2 主线开始 dynamic seam MVP，先做 `fixed seam / keyframe seam / trigger seam` policy 外壳，而不是直接重写 seam backend。
2. 恢复 meaningful temporal evaluation：明确 `fixed_geometry / keyframe_update / adaptive_update` 的指标边界，并在非固定几何模式下重新引入有意义的 temporal 指标。
3. 在 Phase 2 MVP 稳定后，再进入更系统的 Phase 3 experiments/plots automation。

## 当前建议的下一步实施入口
- 先读：
  - `08_project_status_and_master_plan`
  - `05_evaluation`
  - `09_dynamic_seam_and_temporal_eval`
  - `10_execution_workflow`
- 再以 `IMP-*` 的形式写下一步最小实施计划。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/14_open_issues_and_next_steps/14_open_issues_and_next_steps.md | 新增 open issues 与 next steps 文档 | Codex | 完成 |
