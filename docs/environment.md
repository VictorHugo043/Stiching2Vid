# Environment Setup

## Scope
- This document formalizes the install and usage paths for the environments currently used in this repository.
- It covers:
  - baseline `Method A` development and execution
  - `Method B` single-frame development and validation
  - preprocess helpers such as `split_sbs_stereo.py`
  - the auxiliary frame smoke suite added in Phase 1

## Environment Matrix
| Environment | Purpose | Recommended Python | Install Source | Current Notes |
| --- | --- | --- | --- | --- |
| system `python3` | quick local `Method A` runs on this machine | current local interpreter | ad-hoc / machine-local | works for baseline inspection on this machine, but not reproducible and missing Method B deps |
| `.venv` | recommended reproducible baseline env | `3.10` to `3.14` | `requirements.txt` | use for `run_baseline_video.py`, `run_baseline_frame.py`, `inspect_pair.py`, preprocess |
| `.venv-methodb` | `Method B` env for `SuperPoint + LightGlue + USAC_MAGSAC` | `3.10` to `3.14` | `requirements-methodb.txt` + LightGlue install | use for single-frame Method B validation first; current code path should be run with `--device cpu` / `--force_cpu` on Apple Silicon until `mps` support is added |

## Important Reality Check
- Do not assume the existing local `.venv` or `.venv-methodb` directories are clean or complete.
- If behavior looks inconsistent, recreate or reinstall from the requirements files below.
- `largestinteriorrectangle` is optional in practice because `src/stitching/cropper.py` already contains a conservative fallback path.

## Baseline Environment (`.venv`)
Create a clean baseline environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Recommended use cases:
- `python scripts/inspect_pair.py --pair <pair_id> --frame_index 0`
- `python scripts/run_baseline_frame.py --pair <pair_id> --frame_index 0`
- `python scripts/run_baseline_video.py --pair <pair_id> --max_frames 120`
- `python scripts/preprocess/split_sbs_stereo.py --dry_run`

## Method B Environment (`.venv-methodb`)
Create a clean Method B environment:

```bash
python3 -m venv .venv-methodb
source .venv-methodb/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-methodb.txt
```

Install LightGlue after the requirements file.

Option A: local editable clone under this repo

```bash
git clone https://github.com/cvg/LightGlue.git external/LightGlue
python -m pip install -e external/LightGlue
```

Option B: direct install from GitHub

```bash
python -m pip install git+https://github.com/cvg/LightGlue.git
```

Recommended use cases:
- single-frame Method B validation
- future Phase 1 video-path adapter work

Current runtime guidance on Apple Silicon:
- use `--device cpu` or `--force_cpu`
- do not assume `mps` is supported by the current project code yet

## Weights
Default behavior:
- if `--weights_dir` is not provided, `SuperPoint` / `LightGlue` will use their package-default initialization path
- this may trigger automatic downloads into:
  - `~/.cache/torch/hub/checkpoints/`

If auto-download fails or you want local control:

```text
weights/method_b/
  superpoint_v1.pth
  superpoint_lightglue.pth
```

Then run with:

```bash
--weights_dir weights/method_b
```

Supported filename patterns in the current loader include:
- `superpoint.pth`
- `superpoint.pt`
- `superpoint_v1.pth`
- `superpoint_v1.pt`
- `lightglue_superpoint.pth`
- `lightglue_superpoint.pt`
- `superpoint_lightglue.pth`
- `superpoint_lightglue.pt`
- `lightglue.pth`
- `lightglue.pt`

If you hit SSL certificate failures during weight download on macOS:

```bash
python -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
```

## Quick Verification Commands
Baseline verification:

```bash
source .venv/bin/activate
python scripts/inspect_pair.py --pair dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right --frame_index 0
python scripts/run_baseline_frame.py --pair mine_source_indoor2_left_right --frame_index 0 --run_id envcheck_methoda_indoor2
```

Method B single-frame verification:

```bash
source .venv-methodb/bin/activate
python scripts/run_baseline_frame.py \
  --pair dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right \
  --frame_index 0 \
  --feature_backend superpoint \
  --matcher_backend lightglue \
  --geometry_backend opencv_usac_magsac \
  --device cpu \
  --force_cpu \
  --run_id envcheck_methodb_nikita
```

## Frame Smoke Suite
The repo includes a multi-pair frame smoke suite as an auxiliary debug tool:
- `scripts/run_frame_smoke_suite.py`

Default pairs:
- `mine_source_indoor2_left_right`
- `kitti_raw_data_2011_09_28_drive_0119_image_02_image_03`
- `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
- `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`

Notes:
- user shorthand `mysourceindoor2` maps to the real manifest id `mine_source_indoor2_left_right`
- the suite launches `scripts/run_baseline_frame.py` once per pair and writes a summary bundle under `outputs/frame_smoke/<suite_id>/`
- the suite is not part of the current formal Phase 2 / Phase 3 experiment workflow

Run Method A smoke suite:

```bash
source .venv/bin/activate
python scripts/run_frame_smoke_suite.py --method method_a
```

Run Method B smoke suite:

```bash
source .venv-methodb/bin/activate
python scripts/run_frame_smoke_suite.py --method method_b --device cpu --force_cpu
```

## Which Environment Should I Use?
- If you only need baseline video/frame stitching now, use `.venv`.
- If you are touching `superpoint`, `lightglue`, or Method B diagnostics, use `.venv-methodb`.
- If you are just checking a pair quickly on this machine and do not want to recreate environments yet, system `python3` can still be used for limited Method A inspection, but it is not the documented reproducible path.
