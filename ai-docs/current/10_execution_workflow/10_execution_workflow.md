# 10_execution_workflow

## 任务目标
- 固化后续每一步实施时必须遵守的“先回读、先记录、再实施、再回填”的强制工作流。

## 总规则
- 所有后续实施都必须先更新 `ai-docs`，再改代码。
- 不允许只改代码不记过程。
- 不允许在未回读已有记录的情况下重新从零判断。

## 记录文件职责
- `08_project_status_and_master_plan`
  - 全局路线、当前状态、阶段边界
- `11_decision_log`
  - append-only 决策记录
- `12_implementation_log`
  - append-only 实施记录
- `13_change_log`
  - 变更条目与影响面
- `14_open_issues_and_next_steps`
  - blocker、待办和下一步优先队列

## ID 命名规则
- 决策记录：`DEC-YYYYMMDD-NN`
- 实施记录：`IMP-YYYYMMDD-NN`
- 变更记录：`CHG-YYYYMMDD-NN`
- 问题项：`ISSUE-YYYYMMDD-NN`

## Step A. 先回读
- 每次开始实施前，必须先读：
  - `08_project_status_and_master_plan`
  - 当前相关 phase 文档
  - `11_decision_log`
  - `12_implementation_log`
  - `14_open_issues_and_next_steps`
- 同时读取本步相关代码文件。
- 如本步涉及论文结论，必须回看对应论文的关键段落或已提炼文档。

## Step B. 先写本步实施计划
- 在 `12_implementation_log` 新增一条 `planned` 记录，至少包含：
  - 本步目标
  - 关联的上一步结论
  - 准备修改哪些文件
  - 为什么改这些文件
  - 风险点
  - 验收标准
  - 替代方案与为什么不选

## Step C. 再实施最小改动
- 只做本步最小可验证改动。
- 禁止同时混入无关重构。
- 必须保持以下路径可并存：
  - Method A / Method B
  - baseline seam / new seam
  - `fixed_geometry / keyframe_update / adaptive_update`

## Step D. 实施后回填文档
- 本步完成后，必须更新同一条 `implementation log` 记录为：
  - `done`
  - `blocked`
  - `partial`
- 必填内容：
  - 实际修改了哪些文件
  - 新增了哪些配置项 / 类 / 函数 / 脚本
  - 运行结果与验证结果
  - 遇到的错误和修复
  - 与原计划相比的偏差
  - 下一步建议
- 同时补充：
  - `11_decision_log`
  - `13_change_log`
  - `14_open_issues_and_next_steps`

## Step E. 保持连续性
- 下一步开始前，必须以前一步的文档记录为前提。
- 不允许绕开 `11/12/13/14` 重新开始。

## 最小验收要求
- 每个实施步骤都必须留下：
  - 一条 `IMP-*`
  - 至少一条 `CHG-*`
  - 如发生新决策，则新增 `DEC-*`
  - 如出现 blocker，则新增或更新 `ISSUE-*`

## 不允许的行为
- 只改代码不记文档
- 一步里混入多个无关主题
- 未完成验证就把状态写成完成
- 在旧问题未关闭时重复开新的同类 issue 而不建立关联

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| ai-docs/current/10_execution_workflow/10_execution_workflow.md | 新增后续实施的强制工作流文档 | Codex | 完成 |
