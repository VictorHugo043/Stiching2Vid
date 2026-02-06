# 00_repo_bootstrap

## 任务目标
- 固化项目文档/计划结构，明确数据位置与目录约定，建立后续脚本与实验的统一入口与命名规则。

## 验收标准(DoD)
- `ai-docs/templates` 与 `ai-docs/current` 结构就绪，任务文档齐全。
- 形成仓库级约定：数据根目录为 `data/raw/Videos`，清洗与产物目录分别为 `data/processed` 与 `outputs`。
- 确认后续脚本的统一入口规划（仅文档说明，不写业务代码）。

## 下一步
- 梳理数据清单与视频对（见 `01_data_manifest`）。

## 技术方案
- 以文档为先，先固定目录与产物命名，再开展实现与实验。
- 后续脚本统一放置在 `scripts/`，按“编号_用途”命名（例如 `00_env_check.py`）。

## I/O 抽象
- 统一 `FrameSource` 接口：`read(i)`、`read_next()`、`length()`、`fps()`、`resolution()`、`close()`。
- `frames` 类型优先使用 index CSV 保证顺序稳定，其次使用 `frame_pattern` glob 并按数字序号排序。
- `video` 类型通过 `cv2.VideoCapture` 读取，`read(i)` 依赖 `CAP_PROP_POS_FRAMES`，随机 seek 可能不精确。
- 若左右帧数不一致，使用最小长度并记录 warning（避免越界）。

## 关键决策
- 先只做“文档与计划”，不写业务代码。
- 数据以 `data/raw/Videos` 为唯一事实来源，其他目录均由脚本生成。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| src/stitching/io.py | 新增统一 I/O 抽象与数据读取层 | Codex | 完成 |
| scripts/inspect_pair.py | 新增 pair 检查脚本 | Codex | 完成 |
