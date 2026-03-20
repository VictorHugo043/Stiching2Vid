# Project Title

Stiching2Vid: Two-View Overlapping Video Stitching Baseline

## Overview

This repository implements a practical two-view stitching pipeline for videos and frame sequences.  
The focus is stable panorama generation from overlapping left/right streams, with diagnostics for geometry quality, temporal jitter, seam behavior, and runtime.

## Features

- Unified input layer for both `video` and `frames` datasets (`pairs.yaml` driven).
- Baseline feature pipeline: ORB/SIFT -> KNN ratio test -> RANSAC homography.
- Fixed-canvas two-view warping (`right -> left`) with robust fallback behavior.
- Temporal stabilization options for homography stream (`none`, `ema`, `window`).
- Seam pipeline on low resolution with OpenCV seam finders:
  - `opencv_dp_color`, `opencv_dp_colorgrad`, `opencv_voronoi`
- Crop-before-seam (Largest Interior Rectangle, LIR) with safe fallback when LIR package is unavailable.
- Blending modes: `none`, `feather`, `multiband`.
- Video reuse mode (`frame0` initialization + reuse) to reduce frame-to-frame warp jitter and improve speed.
- Rich run artifacts: stitched video, transforms, debug JSON, jitter time series, snapshots.

## Repository Structure

```text
.
├─ scripts/
│  ├─ run_baseline_video.py      # Main video pipeline (video/frames inputs)
│  ├─ run_baseline_frame.py      # Single-frame baseline
│  ├─ inspect_pair.py            # I/O sanity check for one pair + frame
│  ├─ ablate_temporal.py         # Temporal smoothing A/B runs
│  └─ ablate_seam.py             # Seam/blend A/B/C/D runs
├─ src/stitching/
│  ├─ io.py                      # Manifest parsing + FrameSource abstractions
│  ├─ features.py                # ORB/SIFT detection
│  ├─ matching.py                # Descriptor matching + match visualization
│  ├─ geometry.py                # Homography + warp canvas helpers
│  ├─ temporal.py                # Homography smoothing + jitter metrics
│  ├─ seam_opencv.py             # Seam-scale warp, seam finders, mask resize
│  ├─ cropper.py                 # LIR cropper and safe fallback
│  ├─ blending.py                # none/feather/multiband blending
│  ├─ video_state.py             # Cached state for reuse mode
│  ├─ video_stitcher.py          # Frame0 initialize + reuse execution
│  └─ viz.py                     # Snapshot helpers
├─ data/
│  ├─ manifests/pairs.yaml       # Pair registry and per-pair metadata
│  └─ raw/Videos/                # Dataset root expected by manifest paths
├─ outputs/
│  ├─ runs/                      # Per-run outputs
│  └─ ablations/                 # Ablation summaries and comparison images
├─ ai-docs/current/              # Internal design/quality/evaluation docs
└─ stitching-0.6.1/              # Local OpenStitching reference source
```

## Installation

Formal environment documentation:
- [docs/environment.md](docs/environment.md)

Recommended Python:
- baseline / preprocess: `3.10` to `3.14`
- Method B: `3.10` to `3.14`

Baseline environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Method B environment:

```bash
python3 -m venv .venv-methodb
source .venv-methodb/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-methodb.txt
git clone https://github.com/cvg/LightGlue.git external/LightGlue
python -m pip install -e external/LightGlue
```

OpenCV note:
- This project uses stitching/detail APIs (for seam and multiband), so `opencv-contrib-python` is required.

Dependency files:
- `requirements.txt`: Method A + preprocess baseline
- `requirements-methodb.txt`: Method B extras on top of baseline

Important:
- `largestinteriorrectangle` is optional in practice because the project has a conservative crop fallback path.
- existing local `.venv` / `.venv-methodb` may drift; if behavior looks inconsistent, reinstall from the requirements files above.

## Quick Start

From repository root.

1) Validate I/O for a pair:

```bash
python3 scripts/inspect_pair.py --pair campus_sequences_campus4_c0_c1 --frame_index 0
```

2) Video-input pair (manifest `input_type: video`):

```bash
python3 scripts/run_baseline_video.py \
  --pair campus_sequences_campus4_c0_c1 \
  --max_frames 120 \
  --blend multiband
```

3) Frames-input pair (manifest `input_type: frames`):

```bash
python3 scripts/run_baseline_video.py \
  --pair dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right \
  --max_frames 120 \
  --fps 30 \
  --blend multiband
```

4) Single-frame debug run:

```bash
python3 scripts/run_baseline_frame.py \
  --pair campus_sequences_campus4_c0_c1 \
  --frame_index 0
```

5) Multi-pair single-frame smoke suite:

```bash
python3 scripts/run_frame_smoke_suite.py --method method_a
```

Method B smoke suite:

```bash
source .venv-methodb/bin/activate
python scripts/run_frame_smoke_suite.py --method method_b --device cpu --force_cpu
```

## Side-by-Side Preprocess (mine_source)

Use this script to split side-by-side stereo videos (expected `2560x720`) into
left/right streams and append pair entries to `data/manifests/pairs.yaml`.

```bash
python3 scripts/preprocess/split_sbs_stereo.py
```

Defaults:
- Input scan dir: `data/preprocess` (`.avi/.mp4/.mov`)
- Output root: `data/raw/Videos/mine_source`
- Manifest: `data/manifests/pairs.yaml`
- Output naming: `data/raw/Videos/mine_source/<stem>/left|right.<ext>`
- Output ext: inferred from existing `mine_source` outputs (currently `mp4`)

Common options:

```bash
python3 scripts/preprocess/split_sbs_stereo.py \
  --dry_run \
  --ext mp4 \
  --overwrite
```

Notes:
- Prefers `ffmpeg` crop; falls back to OpenCV if `ffmpeg` is unavailable.
- Checks input resolution strictly (`2560x720`), otherwise exits with error.
- Backs up manifest before write to `data/manifests/pairs.yaml.bak`.
- Rejects pair id conflicts (`mine_source_<stem>_left_right`) to avoid duplicates.

## Data Format

Pair registration is defined in `data/manifests/pairs.yaml` under top-level `pairs`.

Core fields per pair:
- `id`: unique pair identifier used by CLI `--pair`.
- `dataset`: dataset name.
- `input_type`: `video` or `frames`.
- `left`, `right`: repo-relative paths to video files or frame directories.
- `calib`: optional calibration file path (metadata only in current pipeline).
- `meta`: optional fields such as:
  - `frame_pattern`, `frame_pattern_right`
  - `index_csv` (used for stable frame ordering)
  - `fps`, `scene`, `cameras`, frame counts, etc.

Frame ordering for `input_type: frames`:
- Uses `index_csv` when available (preferred).
- Falls back to glob + numeric filename sort.

Calibration handling:
- Calibration paths are stored in manifest.
- Current baseline pipeline does **not** perform undistortion/rectification from those files (`TODO`).

## Configuration / CLI Arguments

Main script: `scripts/run_baseline_video.py`

Important flags:
- Data/run control:
  - `--pair`, `--manifest`, `--start`, `--max_frames`, `--stride`
- Geometry/matching:
  - `--keyframe_every`, `--feature`, `--nfeatures`, `--ratio`, `--min_matches`, `--ransac_thresh`
- Blending:
  - `--blend none|feather|multiband`, `--mb_levels`
- Seam:
  - `--seam none|opencv_dp_color|opencv_dp_colorgrad|opencv_voronoi`
  - `--seam_megapix`, `--seam_dilate`
- Crop before seam:
  - `--crop` / `--no_crop`
  - `--lir_method auto|lir|fallback`
  - `--lir_erode`, `--crop_debug`
- Temporal smoothing:
  - `--smooth_h none|ema|window`
  - `--smooth_alpha`, `--smooth_window`
- Video reuse mode:
  - `--video_mode 0|1`
  - `--reuse_mode frame0_all|frame0_geom|frame0_seam|emaH`
  - `--reinit_every`, `--reinit_on_low_overlap_ratio`
- Output/fps:
  - `--fps`, `--snapshot_every`, `--out_dir`, `--run_id`

Example: stable frame0 reuse mode

```bash
python3 scripts/run_baseline_video.py \
  --pair mine_source_lake_left_right \
  --max_frames 120 \
  --video_mode 1 \
  --reuse_mode frame0_all \
  --blend multiband
```

## Outputs

Default output path:
- `outputs/runs/<run_id>/`

Main artifacts:
- `stitched.mp4`
- `transforms.csv`
- `metrics_preview.json`
- `debug.json`
- `jitter_timeseries.csv`
- `logs.txt`
- `snapshots/`

Common snapshot files:
- periodic frame snapshots: left/right/stitched/overlay
- keyframe seam diagnostics: seam masks, overlays, overlap diff
- crop diagnostics: panorama mask, LIR overlay, cropped previews
- video reuse init snapshots: `frame0_*` seam/crop/warp visualization

## Experiments / Ablations

Implemented:

1) Temporal smoothing ablation

```bash
python3 scripts/ablate_temporal.py \
  --pair mine_source_autumn_left_right \
  --max_frames 120
```

Output:
- `outputs/ablations/<pair_id>/summary_temporal.csv`
- `outputs/ablations/<pair_id>/compare/`

2) Seam/blend ablation

```bash
python3 scripts/ablate_seam.py \
  --pair mine_source_lake_left_right \
  --max_frames 60
```

Output:
- `outputs/ablations/<pair_id>/seam/summary_seam.csv`
- `outputs/ablations/<pair_id>/seam/compare/`

Planned:
- `TODO`: dedicated crop ablation script.
- `TODO`: dedicated video-reuse ablation script (`ablate_video_reuse.py`).

## Frame Smoke Suite

The repository now includes:
- `scripts/run_frame_smoke_suite.py`

Default smoke-suite pairs:
- `mine_source_indoor2_left_right`
- `kitti_raw_data_2011_09_28_drive_0119_image_02_image_03`
- `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
- `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`

Alias:
- `mysourceindoor2` -> `mine_source_indoor2_left_right`

## Development Notes

- Keep pipeline modules under `src/stitching/` focused and composable (I/O, geometry, seam, crop, blend, temporal).
- When adding a new dataset/pair, update `data/manifests/pairs.yaml` and verify with `scripts/inspect_pair.py`.
- For frame datasets, prefer adding `meta.index_csv` for deterministic ordering.
- Keep docs synchronized in:
  - `ai-docs/current/03_baseline_video_pipeline/03_baseline_video_pipeline.md`
  - `ai-docs/current/04_quality_improvement/04_quality_improvement.md`

## Troubleshooting

- Not enough matches / homography fails:
  - Increase texture content, try `--feature sift`, tune `--ratio`/`--min_matches`.
- Severe black borders or seam artifacts:
  - Ensure crop is enabled (`--crop`), inspect crop/seam snapshots in `snapshots/`.
- Large rectangle ghosting:
  - Check seam mode is not `none`; inspect seam mask and overlap debug images.
- FPS or timing mismatch:
  - Verify `meta.fps` or pass `--fps` explicitly for frame sequences.
- OpenCV API errors for seam/blender:
  - Use `opencv-contrib-python` and verify installed OpenCV includes `cv2.detail_*`.
- I/O issues (missing files, frame mismatch):
  - Run `scripts/inspect_pair.py` first; confirm manifest paths and frame patterns.

## Roadmap

- Integrate calibration-aware preprocessing (undistortion/rectification).
- Add dedicated crop and video-reuse ablation scripts with unified summaries.
- Expand stronger matching / robust geometry options for challenging overlap.
- Improve evaluation metrics and reporting automation.
