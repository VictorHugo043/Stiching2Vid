# Stiching2Vid

Two-view overlapping video stitching project with:
- `Method A`: ORB / SIFT + ratio test + RANSAC homography
- `Method B`: SuperPoint + LightGlue + OpenCV USAC_MAGSAC
- OpenCV seam-based composition with crop, blending, diagnostics, evaluation, and a desktop GUI thin wrapper

## Overview

This project stitches overlapping left/right videos or frame sequences into a single panorama-style video.

The current pipeline supports:
- pair management through `data/manifests/pairs.yaml`
- video and frame-sequence inputs
- fixed-geometry, keyframe-update, and adaptive-update execution modes
- fixed / keyframe / trigger seam policies
- crop-before-seam, OpenCV seam estimation, feather/multiband blending
- per-run diagnostics and experiment export

## Features

### Method A
- ORB / SIFT local features
- KNN matching with ratio test
- homography estimation with OpenCV RANSAC

### Method B
- SuperPoint features
- LightGlue matching
- homography estimation with OpenCV `USAC_MAGSAC`
- formal baseline preset: `accuracy_v1`

### Video composition
- cached `VideoStitcher` execution
- crop-before-seam
- OpenCV seam finders
- `none / feather / multiband` blending
- optional homography smoothing
- seam policy shell on top of the existing OpenCV seam backend

### Outputs
- stitched video
- `metrics_preview.json`
- `debug.json`
- `transforms.csv`
- `jitter_timeseries.csv`
- snapshots and seam/crop diagnostics

### Evaluation
- Method A vs Method B comparison
- dynamic seam comparison
- figure export for final report

### GUI
- desktop `tkinter` GUI
- select existing pairs
- preview left/right first frame
- upload/register new left/right videos into the manifest
- start one stitching run and open the resulting run folder

## Installation

Formal environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
git clone https://github.com/cvg/LightGlue.git external/LightGlue
python -m pip install -e external/LightGlue
```

Environment details:
- the formal environment is `.venv + requirements.txt`
- `opencv-contrib-python` is required because the project uses stitching/detail APIs
- on Apple Silicon, the current project code should be treated as `cpu` first; use `--force_cpu` when in doubt
- `largestinteriorrectangle` is optional in practice because the cropper already has a fallback path

Method B weights:
- if `--weights_dir` is omitted, SuperPoint / LightGlue may download package-default weights into `~/.cache/torch/hub/checkpoints/`
- if SSL download fails on macOS:

```bash
python -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
```

More environment notes:
- [docs/environment.md](docs/environment.md)

## Use the GUI First

Launch:

```bash
source .venv/bin/activate
python scripts/run_stitching_gui.py
```

The GUI currently supports:
- selecting an existing pair from `pairs.yaml`
- previewing the first left/right frame
- registering a new left/right video pair
- choosing a stitching method and a thin subset of run parameters
- launching `scripts/run_baseline_video.py`
- opening the resulting `outputs/runs/<run_id>/` folder

Recommended GUI use:
- quick qualitative runs
- choosing an existing pair and testing `method_a_orb`, `method_a_sift`, or `method_b_accuracy_v1`
- registering your own left/right videos without editing the manifest manually

## CLI Usage

### 1. Check a pair

```bash
python scripts/inspect_pair.py \
  --pair kitti_raw_data_2011_09_26_drive_0002_image_02_image_03 \
  --frame_index 0
```

Use this when you want to confirm:
- the pair id exists
- left/right inputs can be opened
- frame ordering and resolution look correct

### 2. Run one video stitching job

Method A example:

```bash
python scripts/run_baseline_video.py \
  --pair mine_source_indoor2_left_right \
  --feature orb \
  --feature_backend opencv_orb \
  --matcher_backend opencv_bf_ratio \
  --geometry_backend opencv_ransac \
  --geometry_mode fixed_geometry \
  --seam_policy fixed \
  --blend multiband \
  --max_frames 120 \
  --run_id demo_method_a_orb
```

Formal Method B baseline example (`accuracy_v1`):

```bash
python scripts/run_baseline_video.py \
  --pair kitti_raw_data_2011_09_26_drive_0002_image_02_image_03 \
  --feature_backend superpoint \
  --matcher_backend lightglue \
  --geometry_backend opencv_usac_magsac \
  --geometry_mode fixed_geometry \
  --seam_policy fixed \
  --max_keypoints 4096 \
  --resize_long_edge 1536 \
  --depth_confidence -1 \
  --width_confidence -1 \
  --filter_threshold 0.1 \
  --blend multiband \
  --max_frames 120 \
  --force_cpu \
  --run_id demo_method_b_accuracy_v1
```

Frames-input example with explicit fps:

```bash
python scripts/run_baseline_video.py \
  --pair dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right \
  --fps 10 \
  --max_frames 120 \
  --blend multiband \
  --run_id demo_dynamicstereo
```

Important video CLI arguments:
- `--pair`: pair id from `data/manifests/pairs.yaml`
- `--start`, `--max_frames`, `--stride`: frame range control
- `--fps`: fps override, mainly useful for frame-directory datasets
- `--geometry_mode`:
  - `fixed_geometry`
  - `keyframe_update`
  - `adaptive_update`
- `--keyframe_every`: geometry refresh cadence for `keyframe_update`
- `--seam_policy`:
  - `fixed`
  - `keyframe`
  - `trigger`
- `--seam_keyframe_every`: seam cadence for `seam_policy=keyframe`
- `--seam_trigger_diff_threshold`: main trigger threshold for `seam_policy=trigger`
- `--seam_trigger_foreground_ratio`: extra foreground-aware trigger threshold
- `--blend`: `none | feather | multiband`
- `--feature_backend`, `--matcher_backend`, `--geometry_backend`: choose Method A or Method B route explicitly
- `--force_cpu`, `--device`, `--weights_dir`: Method B runtime control
- `--snapshot_every`: snapshot interval
- `--run_id`: output folder name under `outputs/runs/`

Notes:
- `--video_mode` is still accepted for legacy compatibility, but new runs should prefer `--geometry_mode`
- the project still uses the same OpenCV seam backend; seam policy controls when seam is refreshed, not a different seam solver

### 3. Run one single-frame debug job

```bash
python scripts/run_baseline_frame.py \
  --pair mine_source_indoor2_left_right \
  --frame_index 0 \
  --feature_backend superpoint \
  --matcher_backend lightglue \
  --geometry_backend opencv_usac_magsac \
  --max_keypoints 4096 \
  --resize_long_edge 1536 \
  --depth_confidence -1 \
  --width_confidence -1 \
  --filter_threshold 0.1 \
  --force_cpu \
  --run_id frame_debug_method_b
```

Use this when you want:
- a quick one-frame geometry check
- match / inlier visualization
- a static compose preview without running a full video

## Formal Evaluation and Export

### Method comparison matrix

Use this for flexible multi-pair Method A / Method B comparison:

```bash
python scripts/eval_method_compare_matrix.py \
  --python_bin .venv/bin/python \
  --video_mode 1 \
  --max_frames 6000 \
  --force_cpu
```

This driver runs:
- `method_a_orb`
- `method_a_sift`
- `method_b`

and writes:
- `outputs/video_compare/<suite_id>/summary.csv`
- `outputs/video_compare/<suite_id>/summary.json`
- `outputs/video_compare/<suite_id>/pair_compare.csv`

### Full-length formal method suite

```bash
python scripts/eval_method_compare.py \
  --python_bin .venv/bin/python \
  --force_cpu \
  --max_frames 6000 \
  --snapshot_every 1000
```

This runs the formal full-length method comparison across:
- KITTI
- DynamicStereo
- mine_source

and then builds:
- dataset summaries under `outputs/phase3/<suite_id>/`
- overall summary tables under `outputs/phase3/<overall_suite_id>/`
- final-report figures

### Dynamic seam comparison

```bash
python scripts/eval_dynamic_compare.py \
  --python_bin .venv/bin/python \
  --force_cpu \
  --max_frames 6000
```

This compares the formal dynamic seam presets, including:
- `baseline_fixed`
- `keyframe_seam10`
- `trigger_fused_d18_fg008`
- `adaptive_trigger_fused_d18_fg008`

### Export dynamic visuals

```bash
python scripts/export_dynamic_visuals.py \
  --suite_id <dynamic_suite_id>
```

### Export report figures

```bash
python scripts/export_report_figures.py \
  --suite_id <overall_method_suite_id>
```

## Outputs

Single-run outputs:
- `outputs/runs/<run_id>/`

Typical files:
- `stitched.mp4`
- `metrics_preview.json`
- `debug.json`
- `transforms.csv`
- `jitter_timeseries.csv`
- `snapshots/`

Evaluation outputs:
- `outputs/video_compare/<suite_id>/...`
- `outputs/phase3/<suite_id>/...`

Pair registry:
- `data/manifests/pairs.yaml`

## Current Formal Baselines

### Method A
- `method_a_orb`
- `method_a_sift`

### Method B
- formal baseline: `accuracy_v1`
- retained candidate: `kp3072_v1`

Current Method B formal baseline parameters:
- `max_keypoints=4096`
- `resize_long_edge=1536`
- `depth_confidence=-1`
- `width_confidence=-1`
- `filter_threshold=0.1`
