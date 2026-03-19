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

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/13_change_log/13_change_log.md | 新增变更日志并记录首批文档体系更新 | Codex | 完成 |
