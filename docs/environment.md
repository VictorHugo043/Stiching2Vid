# Environment Setup

## Scope
- This document defines the current formal install and usage path for the repository.
- It covers:
  - the single formal environment for `Method A` and `Method B`
  - preprocess helpers such as `split_sbs_stereo.py`
  - legacy smoke helpers retained for debugging

## Environment Matrix
| Environment | Purpose | Recommended Python | Install Source | Current Notes |
| --- | --- | --- | --- | --- |
| system `python3` | quick local inspection on this machine | current local interpreter | ad-hoc / machine-local | not reproducible; do not treat as the formal project environment |
| `.venv` | recommended formal environment for the whole project | `3.10` to `3.14` | `requirements.txt` + LightGlue install | use for `Method A`, `Method B`, compare/export scripts, GUI, preprocess |

## Important Reality Check
- Do not assume the existing local `.venv` directory is clean or complete.
- If behavior looks inconsistent, recreate or reinstall from the formal steps below.
- If you still have a historical local `.venv-methodb` directory from earlier phases, treat it only as an old local environment name, not as a separate documented install path.
- `largestinteriorrectangle` is optional in practice because `src/stitching/cropper.py` already contains a conservative fallback path.

## Unified Formal Environment (`.venv`)
Create a clean formal environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
git clone https://github.com/cvg/LightGlue.git external/LightGlue
python -m pip install -e external/LightGlue
```

Recommended use cases:
- `python scripts/inspect_pair.py --pair <pair_id> --frame_index 0`
- `python scripts/run_baseline_frame.py --pair <pair_id> --frame_index 0`
- `python scripts/run_baseline_video.py --pair <pair_id> --max_frames 120`
- `python scripts/eval_method_compare_matrix.py ...`
- `python scripts/eval_method_compare.py ...`
- `python scripts/eval_dynamic_compare.py ...`
- `python scripts/export_dynamic_visuals.py ...`
- `python scripts/export_report_figures.py ...`
- `python scripts/run_stitching_gui.py`
- `python scripts/preprocess/split_sbs_stereo.py --dry_run`

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
source .venv/bin/activate
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

## Legacy Frame Smoke Suite
The repo includes a multi-pair frame smoke suite as an auxiliary debug tool:
- `scripts/legacy/run_frame_smoke_suite.py`

Default pairs:
- `mine_source_indoor2_left_right`
- `kitti_raw_data_2011_09_28_drive_0119_image_02_image_03`
- `kitti_raw_data_2011_09_26_drive_0005_image_02_image_03`
- `dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right`

Notes:
- user shorthand `mysourceindoor2` maps to the real manifest id `mine_source_indoor2_left_right`
- the suite launches `scripts/run_baseline_frame.py` once per pair and writes a summary bundle under `outputs/frame_smoke/<suite_id>/`
- the suite is not part of the current formal experiment workflow

Run Method A smoke suite:

```bash
source .venv/bin/activate
python scripts/legacy/run_frame_smoke_suite.py --method method_a
```

Run Method B smoke suite:

```bash
source .venv/bin/activate
python scripts/legacy/run_frame_smoke_suite.py --method method_b --device cpu --force_cpu
```

## Desktop GUI Thin Wrapper
Current GUI entry:
- `scripts/run_stitching_gui.py`

Launch:

```bash
source .venv/bin/activate
python scripts/run_stitching_gui.py
```

Notes:
- the GUI is a desktop `tkinter` wrapper, not a web UI
- it launches `scripts/run_baseline_video.py` as a subprocess and writes a `gui_request.json` into the selected run directory
- it previews the first left/right frame of the selected pair by reusing the existing I/O layer
- it keeps `keyframe_every` / `seam_keyframe_every` visible only when the selected geometry mode or seam policy actually uses them
- it is safest to launch it from the same environment you use for the target method
- on headless shells or remote terminals without a display server, the GUI may not launch even though the script imports correctly
- current upload path only supports registering left/right video files; uploads are copied into `data/raw/Videos/gui_uploads/` and appended to `data/manifests/pairs.yaml` with repo-relative paths
- output remains restricted to `outputs/runs/<run_id>/`
- after a run finishes, the GUI can auto-open that run directory in the system file manager

## Which Environment Should I Use?
- Formal recommendation: use `.venv` for the whole project.
- Historical local `.venv-methodb` directories, if they still exist on your machine, should be treated as leftover compatibility artifacts rather than a separate documented environment.
- If you are just checking a pair quickly on this machine and do not want to recreate environments yet, system `python3` can still be used for limited inspection, but it is not the documented reproducible path.
