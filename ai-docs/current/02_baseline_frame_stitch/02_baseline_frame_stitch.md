# 02_baseline_frame_stitch

## 任务目标
- 搭建经典特征法的单帧拼接基线，用于后续视频拼接。

## 验收标准(DoD)
- 对 1-2 组样例帧产出可视化拼接结果。
- 输出基础中间产物：关键点匹配图、单应估计结果、拼接边界示意。

## 下一步
- 将单帧方法串成视频处理流程（见 `03_baseline_video_pipeline`）。

## 技术方案
- ORB 或 SIFT + KNN + Ratio Test + RANSAC 估计单应。
- 透视/柱面投影 + 简单融合，先保证稳定可运行。

## Pipeline（单帧）
1) 读取数据：`open_pair()` + `read(frame_index)`，兼容 video/frames  
2) 特征检测：ORB（默认）或 SIFT  
3) 特征匹配：KNN + ratio test  
4) 单应估计：RANSAC（cv2.findHomography）  
5) 透视 warp：自动估计画布尺寸与平移补偿  
6) 融合：feather blending（预留 multiband 接口）

## 参数默认值
- feature: `orb`
- nfeatures: `2000`
- ratio: `0.75`
- min_matches: `30`
- ransac_thresh: `3.0`
- frame_index: `0`

## 输出说明（outputs/runs/<run_id>/）
- `left.png` / `right.png`: 原始输入帧
- `matches.png`: ratio test 后匹配可视化
- `inliers.png`: RANSAC 内点可视化
- `warp_overlay.png`: 透视对齐后的叠加对比
- `stitched_frame.png`: feather 融合输出
- `debug.json`: 参数与关键统计（含失败阶段与原因）
- `logs.txt`: 运行日志（可选，已默认输出）

## 已知问题 / 限制
- 视频 seek 依赖 `CAP_PROP_POS_FRAMES`，GOP 结构会导致轻微不精确。
- 帧序列排序优先数字序号；若命名不规范，时序可能不稳定。
- 画布估计以包围盒为准，视差较大时可能产生明显黑边。
- Feather 融合对曝光差异敏感，后续可切换 multiband。

## 调试建议
- 若读取失败，请先运行 `scripts/inspect_pair.py` 验证 I/O。
- 匹配不足时检查 ratio/min_matches 或尝试 SIFT。

## 关键决策
- 基线优先简洁可靠，避免过早引入复杂变形。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| src/stitching/features.py | 新增特征检测与描述接口 | Codex | 完成 |
| src/stitching/matching.py | 新增匹配与可视化接口 | Codex | 完成 |
| src/stitching/geometry.py | 新增单应与 warp 工具 | Codex | 完成 |
| src/stitching/blending.py | 新增 feather 融合 | Codex | 完成 |
| src/stitching/viz.py | 新增可视化保存工具 | Codex | 完成 |
| scripts/run_baseline_frame.py | 新增单帧拼接脚本 | Codex | 完成 |
