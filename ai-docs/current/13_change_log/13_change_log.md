# 13_change_log

## 使用规则
- append-only。
- 每次实施结束后至少新增一条 `CHG-*`。
- 记录内容至少包括：
  - 变更范围
  - 影响面
  - 是否涉及配置 / 文档 / 脚本 / 核心代码

## 变更记录
| ID | 日期 | 范围 | 变更摘要 | 影响面 | 备注 |
| --- | --- | --- | --- | --- | --- |
| CHG-20260319-01 | 2026-03-19 | docs | 更新 `03~07`，新增 `08~14`，建立 master plan、专项设计与日志机制 | 仅影响 `ai-docs/`，不影响核心 pipeline 运行 | 本次为文档体系补全，不含核心代码改动 |
| CHG-20260319-02 | 2026-03-19 | code+docs | 在 `run_baseline_video.py` 中补充 `geometry_mode / jitter_meaningful` 导出字段，保留 legacy CLI 兼容；同步更新 `03`、`05` 与实施日志 | 影响 video run bundle schema 与 ai-docs，对算法行为无影响 | 本次为 Phase 0 语义冻结，不包含 Method B、dynamic seam 或 GUI 功能新增 |
| CHG-20260319-03 | 2026-03-19 | docs+scripts | 将 `ablate_temporal.py`、`ablate_seam.py` 降级为 legacy exploratory helpers，并在 ai-docs 中明确 Phase 0 已完成的冻结边界 | 影响实验入口的推荐路径与 Phase 0 完成定义，不影响核心 stitching 行为 | 本次不删除旧脚本，不引入新的 experiment driver |
| CHG-20260319-04 | 2026-03-19 | code+docs | 为 `features / matching / geometry` 增加 `FeatureResult / MatchResult / GeometryResult` 与结果对象接口；`run_baseline_frame.py` 改走新接口并补充 backend 骨架 | 影响单帧路径与 Phase 1 接口层，不影响现有视频路径；新增单帧 smoke run bundle | 本次保留 legacy tuple/OpenCV 接口，Method B backend 仅落 fail-fast 占位 |
| CHG-20260319-05 | 2026-03-19 | docs | 记录并澄清 `run_baseline_frame.py` 与 `run_baseline_video.py` 的职责差异：单帧入口不包含 seam/crop/temporal 质量链路，不能直接对比视频最终视觉效果 | 影响后续对 smoke 输出的解释与 ai-docs 的使用边界，不影响任何算法行为 | 本次为偏差检查与文档澄清，不包含新的功能实现 |
| CHG-20260319-06 | 2026-03-19 | code+docs | 新增 `frame_quality_preview` 共享 helper，并让 `run_baseline_frame.py` 复用 `VideoStitcher` 的 `crop -> seam -> blend` 链路；同步更新 ai-docs 与后续建议 | 影响单帧入口输出、debug 字段和 snapshots，使其更接近视频静态质量链路；不改视频主流程 | 本次仍不覆盖 temporal/cache/multi-frame bundle parity，仅补齐单帧静态质量预览 |
| CHG-20260319-07 | 2026-03-19 | code+docs | 为单帧 Method B 接入真实 optional dependency probe、lazy import、weights/device 配置、structured diagnostics 与 fallback；新增 `method_b_runtime.py`，并更新 ai-docs | 影响 `features.py`、`matching.py`、`run_baseline_frame.py` 的 Method B 工程边界与 debug 输出；不影响视频主流程 | 当前环境仍缺 `torch / lightglue`，因此本次验证集中在 fail-fast 和 fallback 路径 |
| CHG-20260320-01 | 2026-03-20 | code+docs | 修复 LightGlue compact list 输出兼容问题，使真实 `superpoint + lightglue + opencv_usac_magsac` 单帧 smoke 在 `.venv-methodb` 下跑通；同步更新 ai-docs | 影响 `matching.py` 的 LightGlue 结果解析层与 Phase 1 当前完成定义；不影响视频主流程 | 本次只修输出解析，不改变视频路径，也不改官方 LightGlue 源码 |

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/13_change_log/13_change_log.md | 新增变更日志并记录首批文档体系更新 | Codex | 完成 |
