# 04_quality_improvement

## 阶段 4.1 目标
- 针对视频级拼接中的 `H jitter` 与 overlap 区域 `flicker` 做时序稳定增强。
- 保持 baseline 输出结构不变，新增诊断文件用于量化对比。

## 背景问题
- baseline 采用关键帧重估 + 非关键帧复用，关键帧切换时 `H` 会出现跳变。
- 跳变会直接放大到 `overlay` 和融合区域，形成视觉抖动和亮度闪烁。

## 新增参数
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--smooth_h` | `none` | 时序平滑策略：`none|ema|window` |
| `--smooth_alpha` | `0.8` | EMA 平滑系数，越大越平滑 |
| `--smooth_window` | `5` | `window` 模式滑窗长度 |
| `--snapshot_every` | `50` | 诊断快照步长，输出 raw/sm overlay 对比 |

## jitter 指标定义
- 令 `c_i(t)` 为第 `t` 帧（映射到画布后）第 `i` 个角点坐标（`i=1..4`）。
- 逐帧位移：`d_i(t) = ||c_i(t) - c_i(t-1)||_2`。
- 指标：
  - `jitter = mean_i(d_i)`（写入 `jitter_raw/jitter_sm`）
  - 额外记录 `max_i(d_i)`（`jitter_raw_max/jitter_sm_max`）
- `raw` 与 `smoothed` 各自独立计算，用于对比平滑前后稳定性。

## 输出文件说明（增量）
- `outputs/runs/<run_id>/transforms.csv`
  - 新增 `Hraw_00..Hraw_22`、`Hsm_00..Hsm_22`
  - 新增 `jitter_raw/jitter_sm` 与 `*_max`
  - 兼容保留 `H_00..H_22`（当前实际用于 warping 的 H）
- `outputs/runs/<run_id>/jitter_timeseries.csv`
  - `frame_idx, jitter_raw, jitter_sm, status`
- `outputs/runs/<run_id>/snapshots/`
  - 新增 `overlay_raw_<frame>.png` 与 `overlay_sm_<frame>.png`
- `outputs/runs/<run_id>/debug.json`
  - 新增 `smooth_h/smooth_alpha/smooth_window`
  - 新增 `jitter_summary`（均值与 95 分位）

## Ablation（4.1）
- 新增 `scripts/ablate_temporal.py`，同一 pair 自动跑两次：
  - A: baseline `--smooth_h none`
  - B: temporal `--smooth_h ema --smooth_alpha 0.8`
- 结果目录：`outputs/ablations/<pair_id>/`
  - `summary_temporal.csv`
  - `compare/`（两组 overlay 对比图）

## 已知限制
- 本阶段不解决视差导致的双影，只减少几何抖动与时序闪烁。
- 不包含 seam finding / multiband / mesh warp / 去畸变。

## 下一阶段预告
- 接入 seam + multiband，优先降低动态场景中的重影可见性。

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| src/stitching/temporal.py | 新增时序平滑器与 jitter 计算 | Codex | 完成 |
| scripts/run_baseline_video.py | 集成 smooth_h、jitter 输出与 raw/sm overlay | Codex | 完成 |
| scripts/ablate_temporal.py | 新增 temporal ablation 一键对比脚本 | Codex | 完成 |
| ai-docs/current/04_quality_improvement/04_quality_improvement.md | 新增 4.1 阶段文档 | Codex | 完成 |

## 阶段 4.2 目标（Seam Transition Line）
- 目标：用 OpenCV 风格 seam finder 在 overlap 内自动寻找 transition line，替代硬直线切割。
- 核心约束：seam 只在有效 warped mask 内生效，最终 compositing mask 必须与 final warp mask 做 `bitwise_and`。
- 与 4.1 兼容：关键帧估计 seam，非关键帧复用 seam cache；`smooth_h` 逻辑不变。

## 4.2 关键流程（OpenCV 风格）
- Step1：在低分辨率 warped ROI 上找 seam（`seam_megapix` 控制尺度）。
  - 使用缩放变换：`H_s = S * H * S^-1`，`S=diag(scale, scale, 1)`。
  - 两路图像都通过 `warp_to_roi()` 得到 `(corner, roi_img, roi_mask)`。
- Step2：调用 seam finder（默认 `dp_color`）输出低分辨率 seam masks。
  - 方法支持：`dp_color`、`dp_colorgrad`、`voronoi`、`none`。
- Step3：将 seam mask resize 到全分辨率 ROI mask，再执行 `bitwise_and`。
  - `resize -> (optional dilate) -> threshold -> AND target_warp_mask`。
- Step4：把 ROI seam mask 按 corner 贴回 full canvas，构造最终 compositing masks。
  - `left_only` 和 `right_only` 必须保持原始取值。
  - overlap 区域使用 seam 决策，禁止 overlap 外半透明混合。

## 新增参数（4.2）
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seam` | `opencv_dp_color` | seam 方式：`none|opencv_dp_color|opencv_dp_colorgrad|opencv_voronoi` |
| `--seam_megapix` | `0.1` | seam 求解分辨率预算（MP） |
| `--seam_dilate` | `1` | seam mask resize 前膨胀次数 |
| `--blend` | `feather` | 兼容 `none|feather|multiband` |
| `--mb_levels` | `5` | multiband 金字塔层数 |

## 诊断输出（4.2）
- 关键帧输出：
  - `snapshots/warp_left_roi*.png`、`snapshots/warp_right_roi*.png`
  - `snapshots/mask_left_roi*.png`、`snapshots/mask_right_roi*.png`
  - `snapshots/seam_mask_left*.png`、`snapshots/seam_mask_right*.png`
  - `snapshots/seam_overlay*.png`
  - `snapshots/overlap_diff*.png`
- 运行统计：
  - `debug.json` 中 `seam_keyframe_stats`（overlap area、mask ratio、diff before/after、seam runtime）
  - `debug.json` 中 `seam_summary`

## Ablation（4.2）
- 脚本：`scripts/ablate_seam.py`
- 输出目录：`outputs/ablations/<pair_id>/seam/`
- 产物：
  - `summary_seam.csv`
  - `compare/`（固定帧 0/20/50 的 stitched + seam_overlay + overlap_diff）
- 对比组：
  - A: no seam + feather
  - B: seam(dp_color) + hard (`blend=none`)
  - C: seam(dp_color) + feather
  - D: seam(dp_color) + multiband

## 推荐调试流程
- 先跑 `blend=none`（看 transition line 是否正确）。
- 再跑 `blend=feather`（看 seam+blend 是否自然）。
- 最后跑 `blend=multiband`（确认没有全局半透明覆盖层）。

## 常见故障与排查
- 故障1：整幅大矩形半透明重影。
  - 检查点：是否把 ROI seam mask 直接与 full-canvas mask 运算（尺寸错位）。
  - 修复：必须先按 corner 把 seam ROI mask 贴回 full canvas，再参与 compositing。
- 故障2：seam 位置明显错误或漂移。
  - 检查点：`H_s = S*H*S^-1` 是否正确；`corner` 是否在同一坐标系。
  - 修复：低分辨率 seam 全链路（warp/corner/mask）保持统一 scale 与坐标系。

## 4.2.1 Crop Before Seam（LIR）
- 目标：在 seam 之前先做 `Crop(Largest Interior Rectangle)`，减少黑边干扰，让 transition line 只在有效重叠区决策，避免“大矩形重影/整块叠加”。
- 流程图（关键帧）：
  - `warp_low_roi -> panorama_mask_low -> LIR -> crop_low + crop_final -> seam(low) -> resize_to_final -> blend`
- 与 4.1/4.2 兼容性：
  - 不修改相机几何估计与 warp 的 `H`；
  - 仅修改 seam/blend 阶段使用的 ROI 与坐标系；
  - 非关键帧继续复用 seam cache（包含 crop rectangles）。

## 4.2.2 参数
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--crop/--no_crop` | `--crop` | 是否启用 seam 前 crop |
| `--lir_method` | `auto` | `auto|lir|fallback` |
| `--lir_erode` | `2` | fallback 路径腐蚀迭代次数 |
| `--crop_debug` | `1` | 输出 crop 中间可视化 |

## 4.2.3 Debug 输出
- `snapshots/panorama_mask_low.png`
- `snapshots/lir_on_mask_low.png`
- `snapshots/cropped_low_masks.png`
- `snapshots/cropped_final_frame_as_is_XXXXXX.png`
- `debug.json`：
  - `crop_enabled`、`crop_fallback_to_no_crop`
  - `crop_keyframe_stats[*].lir_method_used / lir_rect / crop_time_ms`
  - `crop_keyframe_stats[*].mask_area_before/after`

## 4.2.4 DoD
- `--no_crop`：可稳定运行，输出几何与 4.1/4.2 原流程一致（允许轻微像素差）。
- `--crop`：黑边比例下降（`black_border_ratio_low`），seam 输入确实来自 cropped ROIs（日志含 shape/corner before/after）。
- smoke test 必须产出 `mask/lir/cropped_final_as_is` 三类 snapshots。

## 4.2.5 Ablation（Crop）
- 脚本：`scripts/ablate_crop.py`
- A/B 两组：
  - A：`warp -> seam -> blend`（`--no_crop`）
  - B：`warp -> crop -> seam -> blend`（`--crop`）
- 输出：`outputs/ablations/<pair_id>/crop/summary_crop.csv`
  - `black_border_ratio`：越低越好
  - `seam_visible_proxy`：基于 overlap diff 的 seam 可见性代理
  - `avg_runtime_ms` / `crop_time_ms`：额外开销评估

## OpenStitching 对齐表（4.2.x/4.3）
| OpenStitching（0.6.1） | 本项目对应实现 |
| --- | --- |
| `Stitcher.stitch()` | `scripts/run_baseline_video.py` 主循环 |
| `find_features()` / `match_features()` | `stitching.features.detect_and_describe()` + `stitching.matching.match_descriptors()` |
| `estimate_camera_parameters()` | `stitching.geometry.estimate_homography()`（两视角 H） |
| `warp_low_resolution()` | `stitching.seam_opencv.scale_homography()` + `warp_to_roi()`（低分辨率 seam ROI） |
| `warp_final_resolution()` | `warp_to_roi()`（full compose ROI） |
| `prepare_cropper()` | `stitching.cropper.Cropper.prepare()` |
| `crop_low_resolution()` / `crop_final_resolution()` | `Cropper.crop_images()` + `Cropper.crop_rois()`（low/final 一致裁剪） |
| `find_seam_masks()` | `stitching.seam_opencv.compute_seam_masks_opencv()` |
| `resize_seam_masks()` | `stitching.seam_opencv.resize_seam_to_compose()` |
| `blend_images()` | `scripts/run_baseline_video.py::_blend_frames()` / `stitching.video_stitcher.VideoStitcher._blend()` |
| `create_final_panorama()` | `stitched` frame 写入 `stitched.mp4`，并同步 `transforms.csv/debug.json/snapshots` |

## 4.3 Video Stitching（Frame0 Reuse）
- 目标：首帧初始化几何/裁剪/接缝，后续帧复用，降低 frame-to-frame warp jitter 并提速。
- 新增模块：
  - `src/stitching/video_state.py`：`VideoStitchState`（缓存 H/crop/seam/metadata）
  - `src/stitching/video_stitcher.py`：`VideoStitcher.initialize_from_first_frame()` 与 `stitch_frame()`
- 新增 CLI（`scripts/run_baseline_video.py`）：
  - `--video_mode 0|1`
  - `--reuse_mode frame0_all|frame0_geom|frame0_seam|emaH`
  - `--reinit_every N`
  - `--reinit_on_low_overlap_ratio T`
- 默认推荐：`--video_mode 1 --reuse_mode frame0_all`。

## 4.3 Debug 与指标
- `debug.json`：
  - `video_mode`、`reuse_mode`
  - `init_frame_index`
  - `overlap_area_init`、`overlap_area_samples`
  - `reinit_count`
  - `time_breakdown_ms.init_ms/per_frame_ms`
- `snapshots`（初始化帧）：
  - `frame0_warp_low.png`
  - `frame0_mask_low.png`
  - `frame0_panorama_mask.png`
  - `frame0_lir.png`
  - `frame0_seam_mask_left.png` / `frame0_seam_mask_right.png`
  - `frame0_seam_overlay.png`
- `transforms.csv` 新增：
  - `H_delta_norm`
  - `overlap_area_current`

## 4.3 Ablation（Video Reuse）
- 脚本：`scripts/ablate_video_reuse.py`
- 组别：
  - A：baseline（`video_mode=0`）
  - B：`frame0_geom`（固定几何，seam 每帧可更新）
  - C：`frame0_all`（固定几何 + crop + seam）
- 输出：`outputs/ablations/<pair_id>/video_reuse/summary.csv`
  - `avg_per_frame_ms`
  - `avg_H_delta_norm`
  - `avg_overlap_area`
  - `reinit_count`

## 4.2 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| src/stitching/seam_opencv.py | 新增 OpenCV seam 工具（ROI warp + seam finder + resize/AND） | Codex | 完成 |
| src/stitching/blending.py | 支持显式 mask 的 none/feather，新增 multiband | Codex | 完成 |
| scripts/run_baseline_video.py | 接入 4.2 seam 参数、关键帧 seam cache、seam debug 输出 | Codex | 完成 |
| scripts/ablate_seam.py | 新增 4 组 seam ablation（A/B/C/D） | Codex | 完成 |
| src/stitching/cropper.py | 新增 LIR cropper（含 fallback） | Codex | 完成 |
| scripts/run_baseline_video.py | seam 前接入 crop、新增 crop debug/统计 | Codex | 完成 |
| scripts/ablate_crop.py | 新增 crop ablation（A/B） | Codex | 完成 |
| src/stitching/video_state.py | 新增视频复用状态对象 | Codex | 完成 |
| src/stitching/video_stitcher.py | 新增 frame0 初始化 + 复用执行器 | Codex | 完成 |
| scripts/run_baseline_video.py | 新增 `video_mode/reuse_mode/reinit` 接口与分支 | Codex | 完成 |
| scripts/ablate_video_reuse.py | 新增视频复用 ablation（A/B/C） | Codex | 完成 |
