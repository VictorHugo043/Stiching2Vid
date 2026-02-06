# 01_data_manifest

## 总览
- Basketball sequence: types=video, scenes=1, cameras=c0, c1, c2, c3, pairs=6 (required=1, optional=5)
- Campus sequences: types=video, scenes=1, cameras=c0, c1, c2, pairs=3 (required=1, optional=2)
- DynamicStereo: types=frames, scenes=3, cameras=left, right, pairs=3 (required=3, optional=0)
- EPFL-RLC_dataset: types=frames, scenes=1, cameras=cam0, cam1, cam2, pairs=3 (required=1, optional=2)
- Karlsruhe Dataset: types=frames, scenes=2, cameras=i1, i2, pairs=2 (required=2, optional=0)
- Laboratory sequences1: types=video, scenes=1, cameras=c0, c1, c2, c3, pairs=6 (required=1, optional=5)
- MultiScene360 Dataset: types=video, scenes=5, cameras=cam01, cam02, cam03, cam04, pairs=30 (required=5, optional=25)
- Terrace sequences1: types=video, scenes=1, cameras=c0, c1, c2, c3, pairs=6 (required=1, optional=5)

## Karlsruhe Dataset（新规则解析结果）
- 场景目录：`2010_03_04_drive_0033`、`2010_03_17_drive_0046`
- 相机命名：`I1_*.png` 与 `I2_*.png`（同目录双视角帧序列）
- 主配对：I1 + I2（两组均为 required）
- 标定文件：`2010_03_04_calib.txt`、`2010_03_17_calib.txt`

## Video 统计说明
- 当前文档保留 `fps/length/resolution=null`，原因是命令行环境未稳定提供 `cv2/ffprobe`。
- 如需补齐，可在 `.venv` 中用 OpenCV 读取 `CAP_PROP_*` 后再回填。

## Frames 统计
- DynamicStereo:
  - `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`: 189/189, resolution=(1280, 720), pattern=000*-left.jpg / 000*-right.jpg
  - `dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right`: 83/83, resolution=(1280, 720), pattern=000*-left.jpg / 000*-right.jpg
  - `dynamicstereo_real_000_teddy_static_test_frames_rect_left_right`: 218/218, resolution=(1280, 720), pattern=000*-left.jpg / 000*-right.jpg
- EPFL-RLC_dataset:
  - cam0/cam1/cam2: 8001 frames each, resolution=(480, 270), pattern=RLCAFTCONF-C*_10*.jpeg
  - 使用 index：`data/manifests/epfl_frames_index.csv`
- Karlsruhe Dataset:
  - `karlsruhe_dataset_2010_03_04_drive_0033_i1_i2`: 400/400, resolution=(1348, 374), pattern=I1_*.png / I2_*.png
  - `karlsruhe_dataset_2010_03_17_drive_0046_i1_i2`: 967/967, resolution=(1365, 369), pattern=I1_*.png / I2_*.png

## 新增变化
- 已解析并纳入：Karlsruhe Dataset（2 个 required pairs）
- 代表性新增 pair_id：
  - `karlsruhe_dataset_2010_03_04_drive_0033_i1_i2`
  - `karlsruhe_dataset_2010_03_17_drive_0046_i1_i2`

## 潜在问题
- MultiScene360 Dataset: 未发现明确标定文件（`calib=null`）
- DynamicStereo: 存在 `masks/`，提示前景动态区域需要后续 seam/mask 处理
- 视频类数据的 fps/分辨率/帧数尚未在 manifest 中写实值（不影响当前 pipeline 运行）

## Ignore/Unrelated Files（存在但未纳入配对）
- `data/raw/Videos/2010_03_04_drive_0033.zip`
- `data/raw/Videos/Karlsruhe Dataset/*/insdata.txt`
- `data/raw/Videos/DynamicStereo/**/masks/*`
- `data/raw/Videos/EPFL-RLC_dataset/mv_examples/*`

## 下一步动作（smoke tests）
- `python scripts/inspect_pair.py --pair basketball_sequence_match5_c0_c1 --frame_index 0`
- `python scripts/inspect_pair.py --pair dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right --frame_index 0`
- `python scripts/inspect_pair.py --pair karlsruhe_dataset_2010_03_04_drive_0033_i1_i2 --frame_index 0`

## 变更文件清单
| 文件 | 变更说明 | 负责人 | 状态 |
| --- | --- | --- | --- |
| scripts/update_data_manifest.py | 按要求删除（不再使用脚本扫描目录） | Codex | 完成 |
| data/manifests/pairs.yaml | 手动重建并纳入 Karlsruhe Dataset | Codex | 完成 |
| ai-docs/current/01_data_manifest/01_data_manifest.md | 同步更新数据清单文档 | Codex | 完成 |
