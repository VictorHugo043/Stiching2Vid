# 14_open_issues_and_next_steps

## 使用规则
- 统一记录当前 blocker、未决问题和下一步优先队列。
- 每次实施结束后都必须更新本文件。

## 当前 open issues
| ID | 状态 | 问题 | 影响 | 处理建议 |
| --- | --- | --- | --- | --- |
| ISSUE-20260319-01 | closed | 根仓库缺少正式依赖文件（无 `requirements.txt / pyproject.toml / environment.yml`） | Method B 与 GUI 的环境复现成本高 | 已补 root `requirements.txt`、`requirements-methodb.txt` 与 `docs/environment.md` |
| ISSUE-20260319-02 | closed | 当前 Method B 正式环境已经按 `requirements-methodb.txt` 安装并验证通过；`.venv-methodb` 已在多 pair 单帧、多帧抽样和短视频 smoke 上跑通 | Method B 环境不再是当前主 blocker | 后续默认沿用当前 `.venv-methodb`；如环境再漂移，再按 `docs/environment.md` 重建 |
| ISSUE-20260319-03 | partial | `fixed_geometry` 下 `jitter` 容易退化为 0，仍可能被误读为“seam 没更新” | temporal evaluation 容易被误解 | 已新增 `mean_overlap_diff_after / temporal_primary_metric / jitter_scope / seam_snapshot_on_recompute`，且 `adaptive_update` 已能提供非零几何流；后续仍需在 seam smoothing 与更强 temporal metrics 中继续完善 |
| ISSUE-20260319-06 | deferred | 旧 ablation 脚本尚未统一消费 `geometry_mode / jitter_meaningful` 新字段 | 若继续依赖旧脚本会造成 Phase 0 收尾不必要拖延 | 已将 `scripts/ablate_temporal.py`、`scripts/ablate_seam.py` 降级为 legacy helpers；正式 experiment driver 在 Phase 3 单独建设 |
| ISSUE-20260319-04 | open | 当前 seam 模块是 OpenCV mask 风格，无法直接承载 object-centered MRF seam | Dynamic seam advanced 实现难度高 | 先做兼容式 MVP，再单独设计新 seam backend |
| ISSUE-20260319-05 | closed | `scripts/ablate_crop.py`、`scripts/ablate_video_reuse.py` 的历史文档漂移已不再构成当前主线问题；相关旧描述已移出当前工作流 | 不再影响 Phase 1 / Phase 2 推进 | 保留为历史背景，不再作为活跃 issue 跟踪 |
| ISSUE-20260319-07 | closed | `run_baseline_video.py` 已通过 `frame_pair_pipeline` 接到结果对象层，视频入口已可使用 Method B backend 配置 | Method B 现在可以复用视频 orchestrator 的现有 seam/crop/blend/cache 路线 | 后续若要继续演进，只需在当前 adapter 基础上补更长时长回归和实验，不再把“未接到结果对象层”视为 issue |
| ISSUE-20260319-08 | partial | `run_baseline_frame.py` 与 `run_baseline_video.py` 在质量链路上仍有边界差距：单帧入口现已补齐 seam / crop / blend 静态路径，但仍没有 temporal / cache / 完整 run bundle | 用户若忽略边界，仍可能把单帧 smoke 输出误认为完整视频行为 | 已通过 `frame_quality_preview` 缩小静态质量差距；后续仅在需要时再补 diagnostics parity，不把 temporal/cache 强塞进单帧入口 |
| ISSUE-20260320-01 | open | `trigger seam` 当前对阈值较敏感，不同 pair 上可能出现“不触发”或“过于频繁触发” | 若直接拿默认阈值做实验，容易得到不稳定或难解释的结果 | 先在代表性 pair 上校准 `seam_trigger_diff_threshold / seam_trigger_overlap_ratio`，必要时再加入更稳的 trigger 条件或 cooldown |
| ISSUE-20260320-02 | open | `adaptive_update` 当前只是 seam-driven geometry refresh MVP，尚未加入 cooldown / hysteresis / 多 trigger 融合 | 若直接把当前实现当作最终版 adaptive geometry，可能在某些动态样例上过于频繁刷新或引入额外抖动 | 先在代表性 pair 上记录 `geometry_update_count / mean_jitter_sm / runtime` 的变化，再决定是否引入 cooldown 或更稳的自适应触发器 |

## 接下来最先做的 3 件事
1. 继续 Phase 2，先做 `trigger seam` 与 `adaptive_update` 的阈值 / cadence 校准，固定 DynamicStereo 与代表性静态样例上的推荐参数。
2. 在现有 OpenCV seam backend 外层加入 foreground / object-aware penalty 的兼容式做法，再评估对 seam artefacts 的改善。
3. 然后再补 seam temporal smoothing，并扩展 Phase 2 的对比实验矩阵。

## 当前配置使用建议（2026-03-20 更新）
- 新 run 优先使用：
  - `--geometry_mode`
  - `--seam_policy`
- 仅在兼容旧脚本或历史命令时继续使用：
  - `--video_mode`
- 当前应避免的误用：
  - 把 `keyframe_every` 理解成 seam cadence
  - 把 `video_mode` 继续当作“几何 + seam + temporal”的总开关
  - 在 `fixed_geometry` 下把 `jitter=0` 理解成 seam 没有更新

## 当前调试建议（2026-03-20 更新）
- 若你在 `fixed_geometry` 下观察 dynamic seam：
  - 先看 `jitter_scope`
  - 再看 `seam_recompute_count`
  - 再看 `seam_snapshot_count`
  - 再看 `snapshots/seam_event_*`
- 推荐开启：
  - `--seam_snapshot_on_recompute 1`

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
