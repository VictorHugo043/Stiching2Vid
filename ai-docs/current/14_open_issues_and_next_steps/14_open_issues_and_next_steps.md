# 14_open_issues_and_next_steps

## 使用规则
- 统一记录当前 blocker、未决问题和下一步优先队列。
- 每次实施结束后都必须更新本文件。

## 当前 open issues
| ID | 状态 | 问题 | 影响 | 处理建议 |
| --- | --- | --- | --- | --- |
| ISSUE-20260319-01 | closed | 根仓库缺少正式依赖文件（无 `requirements.txt / pyproject.toml / environment.yml`） | Method B 与 GUI 的环境复现成本高 | 已补 root `requirements.txt`、`requirements-methodb.txt` 与 `docs/environment.md` |
| ISSUE-20260319-02 | closed | 当前 Method B 正式环境已经按 `requirements-methodb.txt` 安装并验证通过；`.venv-methodb` 已在多 pair 单帧、多帧抽样和短视频 smoke 上跑通 | Method B 环境不再是当前主 blocker | 后续默认沿用当前 `.venv-methodb`；如环境再漂移，再按 `docs/environment.md` 重建 |
| ISSUE-20260319-03 | closed | `fixed_geometry` 下 `jitter` 容易退化为 0，仍可能被误读为“seam 没更新” | 该解释风险已不再阻塞当前主线 | 已通过 `mean_overlap_diff_after / temporal_primary_metric / jitter_scope / seam_snapshot_on_recompute`、正式 Phase 2 compare matrix 与 `visual_summary.md` 固定解释边界 |
| ISSUE-20260319-06 | deferred | 旧 ablation 脚本尚未统一消费 `geometry_mode / jitter_meaningful` 新字段 | 若继续依赖旧脚本会造成 Phase 0 收尾不必要拖延 | 已将 `scripts/ablate_temporal.py`、`scripts/ablate_seam.py` 降级为 legacy helpers；正式 experiment driver 在 Phase 3 单独建设 |
| ISSUE-20260319-04 | deferred | 当前 seam 模块是 OpenCV mask 风格，无法直接承载 object-centered MRF seam | 属于 advanced 路线，不再阻塞当前 Phase 2 收尾 | 保持当前 OpenCV seam backend 路线；若后续需要 advanced seam，再单独立项 |
| ISSUE-20260319-05 | closed | `scripts/ablate_crop.py`、`scripts/ablate_video_reuse.py` 的历史文档漂移已不再构成当前主线问题；相关旧描述已移出当前工作流 | 不再影响 Phase 1 / Phase 2 推进 | 保留为历史背景，不再作为活跃 issue 跟踪 |
| ISSUE-20260319-07 | closed | `run_baseline_video.py` 已通过 `frame_pair_pipeline` 接到结果对象层，视频入口已可使用 Method B backend 配置 | Method B 现在可以复用视频 orchestrator 的现有 seam/crop/blend/cache 路线 | 后续若要继续演进，只需在当前 adapter 基础上补更长时长回归和实验，不再把“未接到结果对象层”视为 issue |
| ISSUE-20260319-08 | partial | `run_baseline_frame.py` 与 `run_baseline_video.py` 在质量链路上仍有边界差距：单帧入口现已补齐 seam / crop / blend 静态路径，但仍没有 temporal / cache / 完整 run bundle | 用户若忽略边界，仍可能把单帧 smoke 输出误认为完整视频行为 | 已通过 `frame_quality_preview` 缩小静态质量差距；后续仅在需要时再补 diagnostics parity，不把 temporal/cache 强塞进单帧入口 |
| ISSUE-20260320-01 | partial | `trigger seam` 的参数已完成一轮 mine_source calibration，但默认阈值仍对场景分布敏感 | 当前已有 `phase2_trigger_adaptive_minesource_calib_v2` 可供选默认值，但换数据域后仍可能需要重标定 | 当前默认先用 `trigger_fused_d18_fg008`；后续在 DynamicStereo 或更强动态样例上继续补校准 |
| ISSUE-20260320-02 | partial | `adaptive_update` 的 controller 已从全局 armed 细化为 per-trigger rearm，但带 cooldown/hysteresis 的 stable preset 在 sustained foreground 场景下仍偏保守 | `adaptive_update` 已不再系统性退化为“一次性 geometry refresh”，但若直接把 stable preset 当默认值，仍可能低估 adaptive geometry 的作用 | 当前默认仍使用 `trigger_fused_d18_fg008`；若后续继续研究 adaptive geometry，再考虑 foreground-specific cooldown 或更细的 trigger fusion |
| ISSUE-20260323-01 | partial | `foreground_mode=disagreement` 长时间高位的问题已通过 per-trigger rearm 明显缓解，但当前 `adaptive_stable` 仍会在部分 `mine_source_*` 视频上几乎不再重算 | 会继续限制 adaptive geometry 的默认可用性，并使 stable preset 难以作为正式推荐值 | 当前将其降级为实验 preset；后续若继续优化 adaptive controller，再专门围绕 stable preset 调参 |
| ISSUE-20260323-02 | deferred | seam temporal smoothing 在当前实现下会把 `mean_overlap_diff_after` 压到 `0.0`，导致该指标不再适合拿来比较 smoothing 质量 | 当前已知解释边界清楚，但若未来要把 smoothing 升级为正式主轴，仍需要新 temporal metric | 当前 smoothing 评估优先看 `mean_seam_mask_change_ratio / mean_stitched_delta / approx_fps`；若 Phase 3 需要更深入 temporal 结论，再单独补 smoothing-specific metric |
| ISSUE-20260323-03 | partial | 在刷新后的 Phase 3 正式方法 compare 中，Method B 已不再“整体偏弱”，但它呈现出清晰 trade-off：`mean_inliers` 更高，而 `mean_inlier_ratio / approx_fps` 仍整体弱于 ORB/SIFT；dynamic seam 的收益也仍具明显 pair-dependence | 会直接影响 final report 的表述方式：不能把 “Method B 更强” 或 “Method B 更弱” 写成单一句子，也不能把 dynamic seam 的平均收益误读为所有 pair 都有效 | 当前应把 Method B 表述为“高内点数量、低内点率/低速度”的替代路线；后续若要进一步解释，可补按数据域的可视化与 plot/export 脚本 |
| ISSUE-20260323-04 | partial | 多数据域 full-length suite 暴露出本地数据与 manifest 的轻微漂移：`mine_source_leaves_left_right` 当前源文件缺失；DynamicStereo `ignacio / teddy` 的当前可访问帧数分别为 `99 / 99`，与 manifest 元数据 `189 / 218` 不一致 | 影响“全量”和“全帧”的解释边界，也会影响 pair coverage 统计；但不阻塞当前 Phase 3 正式总表，因为 suite 已按当前可访问 source 长度完整跑完 | 当前正式口径统一以 `outputs/phase3/*/pair_coverage.csv` 的实际长度为准；后续若要修复，可补 manifest 校正或恢复缺失源文件 |
| ISSUE-20260323-05 | closed | 旧 Phase 3 Method B compare 使用了旧 implicit preset，且此前 `SuperPoint` preprocess resize 语义存在接入偏差，导致当前 Method B 总表可能低估其能力 | 该问题已不再阻塞 final report 的正式方法结论 | 已用显式 Method B accuracy preset 重跑 `phase3_kitti_methods_acc_v2`、`phase3_dynamicstereo_methods_acc_v2`、`phase3_minesource_methods_acc_v2`，并生成 `phase3_overall_methods_acc_v2` 作为新的正式方法总表 |

## 接下来最先做的 3 件事
1. 基于 `phase3_overall_methods_acc_v2` 与 `phase3_overall_full_v1` 补统一 plot/export 脚本，把方法主表和 dynamic seam 主表转成最终论文图表。
2. 从刷新后的正式方法总表中挑选代表性 pair，补 Method A vs Method B 的可视化案例，解释“高内点数量但较低内点率/较慢速度”的 trade-off。
3. 若 Phase 3 图表导出完成且结论稳定，再决定是否进入 Phase 4 的 GUI thin wrapper。

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

## 当前 Method B 推荐配置（2026-03-23 更新）
- 正式 accuracy preset：
  - `feature_backend=superpoint`
  - `matcher_backend=lightglue`
  - `geometry_backend=opencv_usac_magsac`
  - `max_keypoints=4096`
  - `resize_long_edge=1536`
  - `depth_confidence=-1`
  - `width_confidence=-1`
  - `filter_threshold=0.1`
- 当前不建议再直接沿用：
  - 旧 implicit preset：
    - `max_keypoints=2048`
    - `SuperPoint` package default resize `1024`
    - `LightGlue` adaptive defaults

## 当前调试建议（2026-03-20 更新）
- 若你在 `fixed_geometry` 下观察 dynamic seam：
  - 先看 `jitter_scope`
  - 再看 `seam_recompute_count`
  - 再看 `seam_snapshot_count`
  - 再看 `snapshots/seam_event_*`
- 推荐开启：
  - `--seam_snapshot_on_recompute 1`

## 当前 Phase 2 推荐配置（2026-03-23 更新）
- 默认 seam preset：
  - `geometry_mode=fixed_geometry`
  - `seam_policy=trigger`
  - `seam_trigger_diff_threshold=18`
  - `foreground_mode=disagreement`
  - `seam_trigger_foreground_ratio=0.08`
- 默认 seam smoothing：
  - `seam_smooth=none`
- 当前仅作实验 preset：
  - `geometry_mode=adaptive_update`
  - `seam_policy=trigger`
  - `seam_trigger_diff_threshold=18`
  - `foreground_mode=disagreement`
  - `seam_trigger_foreground_ratio=0.08`
  - `seam_smooth=ema`
  - `seam_smooth=window`
- 当前不建议作为默认值：
  - 任何依赖全局 `cooldown + hysteresis` 的 stable preset

## 当前建议的下一步实施入口
- 先读：
  - `08_project_status_and_master_plan`
  - `05_evaluation`
  - `09_dynamic_seam_and_temporal_eval`
  - `10_execution_workflow`
- 再以 `IMP-*` 的形式写下一步最小实施计划。
- 当前建议直接从“补 Phase 3 统一 plot/export 脚本，并把刷新后的方法总表与 Phase 2/3 dynamic seam 总表转成 final report 图表”开始。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/14_open_issues_and_next_steps/14_open_issues_and_next_steps.md | 新增 open issues 与 next steps 文档 | Codex | 完成 |
