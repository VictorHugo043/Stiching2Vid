# 14_open_issues_and_next_steps

## 使用规则
- 统一记录当前 blocker、未决问题和下一步优先队列。
- 每次实施结束后都必须更新本文件。

## 当前 open issues
| ID | 状态 | 问题 | 影响 | 处理建议 |
| --- | --- | --- | --- | --- |
| ISSUE-20260319-01 | closed | 根仓库缺少正式依赖文件（无 `requirements.txt / pyproject.toml / environment.yml`） | Method B 与 GUI 的环境复现成本高 | 已补 root `requirements.txt` 与 `docs/environment.md` |
| ISSUE-20260319-02 | closed | 当前 Method B 已并入统一正式环境，历史 `.venv-methodb` 仅作为本机兼容目录名保留 | Method B 环境不再是当前主 blocker | 后续统一按 `.venv + requirements.txt + docs/environment.md` 安装与重建 |
| ISSUE-20260319-03 | closed | `fixed_geometry` 下 `jitter` 容易退化为 0，仍可能被误读为“seam 没更新” | 该解释风险已不再阻塞当前主线 | 已通过 `mean_overlap_diff_after / temporal_primary_metric / jitter_scope / seam_snapshot_on_recompute`、正式 Phase 2 compare matrix 与 `visual_summary.md` 固定解释边界 |
| ISSUE-20260319-06 | closed | 旧 ablation 脚本尚未统一消费 `geometry_mode / jitter_meaningful` 新字段 | 该问题已不再影响当前主框架 | 已将 `scripts/legacy/ablate_temporal.py`、`scripts/legacy/ablate_seam.py` 移出顶层正式入口，保留为历史探索脚本 |
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
| ISSUE-20260323-05 | closed | 旧 Phase 3 Method B compare 使用了旧 implicit preset，且此前 `SuperPoint` preprocess resize 语义存在接入偏差，导致当前 Method B 总表可能低估其能力 | 该问题已不再阻塞 final report 的正式方法结论 | 已先用显式 Method B accuracy preset 重跑 `phase3_*_methods_acc_v2`，随后进一步完成 richer-metrics full-length 重跑并以 `phase3_*_methods_rich_v3` / `phase3_overall_methods_rich_v3` 取代旧方法主表 |
| ISSUE-20260324-01 | partial | fixed-geometry richer metrics 已完成 full-length 正式重跑与 plot/export，但 temporal coherence 仍缺少更强的 motion-compensated 指标；当前只实现了 `seam-band flicker` 这一层 | 现在已经能用 full-length richer-metrics 正式表更完整地解释 Method B 的 trade-off，但如果 final report 需要更强的时序论证，单靠 `mean_stitched_delta + seam-band flicker` 仍有限 | 当前先保留 `seam-band flicker` 作为 MVP temporal artefact 指标；若后续需要更强时序论证，再单独补 `flow-compensated temporal residual` |
| ISSUE-20260324-02 | closed | `kp3072_v1` 的 full-length 多数据域复验已完成；它虽然略微改善了 overall `inlier_ratio / fps / reprojection`，但 `mean_inliers` 从约 `748.88` 降到约 `609.58`，且在 `mine_source` 上明显回退，因此不能替换正式 `accuracy_v1` | 该问题已不再阻塞当前正式 baseline 选择 | 继续保持 `accuracy_v1` 为正式默认；把 `kp3072_v1` 仅作为候选复验与方法讨论材料保留在 `outputs/phase3/phase3_methodb_accuracy_vs_kp3072_v1/` |
| ISSUE-20260324-03 | partial | Phase 4 GUI thin wrapper 已补齐 existing pair 首帧预览、条件显示且已重排的 keyframe 参数、注册弹窗和 run 目录打开能力，但范围仍只覆盖桌面 `tkinter` 单 run 流程：新 pair 注册仅支持左右视频文件，不支持 frame directory / batch compare / figure export | 当前可用性已明显提升，但仍不应被误解为完整实验工作台 | 当前先由用户在本机完成一次真实交互确认；后续如继续做 GUI，只做错误提示和 artefact 预览等 polish，不回头侵入核心 pipeline |
| ISSUE-20260325-01 | closed | 正式 compare/export 脚本与部分结果文件名仍保留 `phase2 / phase3` 命名残留 | 该问题已不再影响当前 formal 入口与新生成 artefact 的命名口径 | 已收敛 formal compare/export 的默认 suite id、summary filename 和 markdown/figure 标题；历史 `phase*` 目录继续仅作为冻结结果保留 |
| ISSUE-20260401-01 | closed | 当前 Windows 工作区缺失的 formal `outputs/phase3` artefacts 已迁入：`phase3_overall_methods_rich_v3`、`overall_method_compare_rich_v3_mps_real_accuracy_v2`、`method_b_accuracy_v1_cpu_vs_mps_real_v2`、`method_b_accuracy_v1_vs_native_res_mps_v1` | 该问题已不再阻塞本机回读历史正式总表、设备对照和 Method B 高分辨率对照结果 | 已在当前 Windows 工作区恢复这 4 个 formal artefact 目录，并基于它们继续生成新的 CUDA full-length suite |
| ISSUE-20260401-02 | closed | 当前 Windows 工作区缺失的 `.git/` 已恢复，本机重新具备完整 Git 仓库语义 | 该问题已不再阻塞 `git status / diff / commit` 类工作流 | 后续继续按正式 Git 工作流推进，不再把当前工作区视为纯复制树 |
| ISSUE-20260401-03 | partial | Windows + CUDA 上的 authoritative CUDA formal suite 已切到新的 `*_cuda_real_accuracy_v2` 口径：它修正了旧 subprocess artefact 的重复冷启动问题，但与 CPU 的 overall 对照仍不是 same-code device-isolated compare | 若直接把 `cpu_vs_cuda_real_v2` 读成“纯 device 差异”，仍会夸大 preserved CPU baseline 与当前 CUDA rerun 之间的代码/口径差异 | 当前对外应优先引用 `*_cuda_real_accuracy_v2`；若要回答最严格的 CPU vs CUDA 设备问题，下一步仍需补 same-code CPU full-length rich-v3 suite |
| ISSUE-20260401-04 | closed | 旧 subprocess 口径下的 CUDA formal artefact 已被新的 `auto/inprocess` 全量重跑结果替代 | 该问题已不再阻塞 authoritative CUDA summary 的更新 | 新 authoritative CUDA artefact 已切换到：`kitti/dynamicstereo/minesource/overall_method_compare_rich_v3_cuda_real_accuracy_v2` 与 `method_b_accuracy_v1_cpu_vs_cuda_real_v2`；旧 `v1` 仅保留为历史 artefact |

## 接下来最先做的 3 件事
1. 若需要最严格回答“CPU vs CUDA 是否只体现 device 差异”，在当前代码版本上补 same-code CPU `method_b_accuracy_v1` full-length suite，再与新的 `overall_method_compare_rich_v3_cuda_real_accuracy_v2` 做对照。
2. 若继续追求更高 steady-state fps，下一层优化重点应转向 `VideoStitcher` 的 fixed-geometry CPU compose / warp / crop path；Method B 的 CUDA geometry 本体在 same-code 与同进程 warm benchmark 中并未显示为当前主瓶颈。
3. 若后续要把这轮 Windows CUDA 结果迁回 Mac，优先迁移新的 `*_cuda_real_accuracy_v2` 与 `method_b_accuracy_v1_cpu_vs_cuda_real_v2`，不要再优先带旧 `v1`。

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
- 当前 high-resolution 对照结论（2026-03-26 更新）：
  - `resize_long_edge<=0 + mps` 已完成 full-length 多数据域复验：
    - `outputs/phase3/overall_method_compare_rich_v3_mps_real_native_v1/`
    - `outputs/phase3/method_b_accuracy_v1_vs_native_res_mps_v1/`
  - 当前不建议把它升格为正式默认：
    - overall `mean_inliers` 与 `mean_inlier_ratio` 均低于 `accuracy_v1_mps`
  - 当前更适合的定位：
    - 作为“原始分辨率提特征”的 full-length 对照变体
    - 若后续继续优化高分辨率 `mine_source`，再考虑只针对 `1920x1080` pair 做分数据域 preset，而不是全局替换默认值

## 当前 Method B CPU / MPS / CUDA 结果口径（2026-04-01 更新）
- 历史 `outputs/phase3/overall_method_compare_rich_v3_mps_accuracy_v1/`
  - 当前只保留为历史 artefact。
  - 原因：该 suite 运行在 sandbox 内，实际发生了 `requested_device=mps -> resolved_device=cpu` fallback。
- 当前 authoritative 的 real-MPS formal suite：
  - `outputs/phase3/overall_method_compare_rich_v3_mps_real_accuracy_v2/`
- 当前 preserved CPU vs real MPS 对照 artefact：
  - `outputs/phase3/method_b_accuracy_v1_cpu_vs_mps_real_v2/`
- 当前 authoritative 的 Windows CUDA formal suite：
  - `outputs/phase3/overall_method_compare_rich_v3_cuda_real_accuracy_v2/`
- 当前 authoritative 的 preserved CPU vs CUDA 对照 artefact：
  - `outputs/phase3/method_b_accuracy_v1_cpu_vs_cuda_real_v2/`
- 旧 subprocess CUDA artefact（仅历史保留）：
  - `outputs/phase3/overall_method_compare_rich_v3_cuda_real_accuracy_v1/`
  - `outputs/phase3/method_b_accuracy_v1_cpu_vs_cuda_real_v1/`
- 当前结论：
  - Method A 没有跑 GPU；当前设备对照只重跑了 Method B。
  - preserved CPU vs real MPS 的 overall 对照显示：
    - `mean_inliers`：`748.88 -> 737.54`
    - `mean_inlier_ratio`：`0.5558 -> 0.5498`
    - `approx_fps`：`7.355 -> 10.810`
    - `mean_reprojection_error`：`1.4309 -> 1.4215`
  - same-code 代表性回归显示 CPU / MPS 质量一致而 MPS 更快，因此当前仍应把 MPS 视作运行时部署选项，而不是新的算法方法。
  - preserved CPU vs CUDA `v2` 的 overall 对照显示：
    - `mean_inliers`：`748.88 -> 733.46`
    - `mean_inlier_ratio`：`0.5558 -> 0.5506`
    - `approx_fps`：`7.355 -> 6.091`
    - `mean_reprojection_error`：`1.4309 -> 1.4335`
  - 新的 CUDA `v2` 相比旧 subprocess `v1`：
    - `approx_fps`：`4.409 -> 6.091`
    - `avg_runtime_ms`：`308.31 -> 264.78`
    - `init_ms_mean`：`2332.45 -> 535.16`
    - 质量指标保持不变，说明提升主要来自 orchestration 与冷启动成本修正，而不是算法质量变化。
  - 在当前 Windows + RTX 3060 Laptop GPU 上，full-length `accuracy_v1` 的 CUDA rerun 已经稳定跑通且 `resolved_device=cuda`；在新的 authoritative `v2` 口径下，它仍未整体超过 preserved CPU formal baseline，但性能差距已显著缩小，因此当前更合理的表述是：CUDA 已经是可用且更接近 CPU baseline 的正式 device artefact，而不是“显著更慢”的失败路径。
  - 2026-04-01 的复盘补充结论：
    - same-code `nikita` 代表性回归里，当前代码版本的 CUDA 并没有复现“质量更差”；相反 `mean_inliers / inlier_ratio / reprojection` 均优于 current-code CPU。
    - 同进程 `dynamicstereo` probe 表明，batch compare 若改用 `execution_mode=inprocess`，dataset `avg_runtime_ms` 可从约 `225.84` 降到约 `163.02`，`approx_fps` 从约 `4.57` 升到约 `7.15`。
    - 因此旧 preserved CPU vs CUDA artefact 更适合保留为“旧口径历史结果”，不应继续被读成“3060 上 CUDA 本体更慢 / 更差”的充分证据。
- 当前已知限制：
  - preserved CPU formal suite 早于这轮 `SuperPoint` 预处理优化与 device 字段导出，因此上述 overall delta 是“历史 CPU baseline vs 当前 real MPS”对照，不是纯 device-isolated 对照。
  - 若后续需要最严格的 same-code full-length CPU vs MPS 表，应在当前代码上补一轮 CPU rich-v3 复跑。
  - preserved CPU vs CUDA 也存在同样限制；若要做严格 same-code device compare，应在当前代码上补一轮 CPU rich-v3 复跑，而不是直接拿 preserved CPU suite 下结论。
  - 当前新的 `steady_frame_ms_mean / steady_approx_fps` 只会出现在 2026-04-01 之后的新 run / 新 summary 中；旧 formal artefact 不会自动回填这些字段。

## 当前环境入口（2026-03-25 更新）
- 正式推荐：
  - `.venv`
  - `requirements.txt`
- 当前口径：
  - `.venv + requirements.txt + docs/environment.md` 是唯一正式环境入口。
  - 若本机仍保留历史 `.venv-methodb` 目录，只视为旧本地目录名，不再对应单独的依赖文件。
  - 若 Windows 上的 `python` 指向 `WindowsApps` stub，或本机没有 `py` launcher，可先用真实解释器的绝对路径创建 `.venv`；之后统一切回 `.venv\Scripts\python.exe`
- 当前 Method B 设备支持（2026-03-26 更新）：
  - 正式支持 `cpu / cuda / mps`
  - `auto` 当前按 `cuda -> mps -> cpu` 顺序自动解析
  - GUI 已提供 `Device (Method B / GPU)` 下拉框；若用户显式选择非 `cpu` 设备，界面会自动取消 `Force CPU`
  - 实际能否使用 `mps` 仍取决于本机 `torch.backends.mps.is_available()`

## 当前对外文档入口（2026-03-25 更新）
- `README.md`
  - 面向使用者，只保留项目介绍、功能、安装、GUI/CLI 使用和正式评测/图表导出入口
- `docs/environment.md`
  - 只负责正式环境安装与运行时说明
- `ai-docs/current/*`
  - 继续承载设计、阶段记录、决策、实验和工作流，不再把这些开发过程信息堆进 README

## 当前脚本入口边界（2026-03-25 更新）
- 正式工作流优先使用：
  - `scripts/run_baseline_video.py`
  - `scripts/run_baseline_frame.py`
  - `scripts/run_stitching_gui.py`
  - `scripts/eval_method_compare_matrix.py`
  - `scripts/eval_method_compare.py`
  - `scripts/eval_dynamic_compare.py`
  - `scripts/export_dynamic_visuals.py`
  - `scripts/export_report_figures.py`
- 辅助 / 调试工具：
  - `scripts/inspect_pair.py`
  - `scripts/preprocess/split_sbs_stereo.py`
- 内部脚本：
  - `scripts/internal/summarize_method_compare_dataset.py`
  - `scripts/internal/summarize_method_compare_overall.py`
- 历史 / 探索性工具：
  - `scripts/legacy/ablate_temporal.py`
  - `scripts/legacy/ablate_seam.py`
  - `scripts/legacy/run_method_b_preset_sweep.py`
  - `scripts/legacy/run_frame_smoke_suite.py`
  - `scripts/legacy/run_phase2_trigger_calibration.py`
  - `scripts/legacy/run_phase2_seam_smoothing_suite.py`
  - `scripts/legacy/run_phase3_kitti_compare_suite.py`
- 当前 Method B active preset：
  - `accuracy_v1`
  - `kp3072_v1`

## 当前 GUI 使用约束（2026-03-24 更新）
- `Register Pair... (Upload New Videos)` 仅支持左右视频文件注册，不支持 frame directory。
- 注册新 pair 时必须显式填写唯一 `pair_id`：
  - 为空会被拒绝
  - 清洗后为空会被拒绝
  - 与现有 pair 重名会被拒绝
- 注册成功写入 `pairs.yaml` 时，当前默认采用文本级局部插入，尽量只追加新增 block，不重排整份 manifest。
- `Run Config` 里只有 `Snapshot Every / Force CPU` 是稳定显示字段。
- `Run Config` 里当前还稳定显示：
  - `Device (Method B / GPU)`
- `Keyframe Every / Seam Keyframe Every / Trigger Diff / FG Ratio` 都属于动态参数区，会根据 `geometry_mode / seam_policy` 条件显示。
- 动态参数区当前应与主表单共用 `grid` 风格对齐；若后续仍出现视觉错位，应优先检查 widget 父容器和布局管理器是否一致。
- `run_id` 仍允许 GUI 自动生成默认值，这与 `pair_id` 的强制显式填写是两套不同规则。

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
- 当前脚本命名与入口体系已经收尾完成。
- 若后续继续：
  - 优先做算法或评测层面的新增工作，而不是继续改入口命名。
  - 仅在确有必要时做 GUI polish 或 report/export polish。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/14_open_issues_and_next_steps/14_open_issues_and_next_steps.md | 新增 open issues 与 next steps 文档 | Codex | 完成 |
