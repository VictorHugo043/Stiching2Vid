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
- Method B 现在正式支持 `mps`
- `--device auto` 会按 `cuda -> mps -> cpu` 的顺序自动选择
- 若你想强制使用 Mac GPU，可显式传 `--device mps`
- 若你想显式回到 CPU，可传 `--device cpu` 或 `--force_cpu`
- 实际是否能走 `mps` 仍取决于本机 `torch.backends.mps.is_available()`

## Windows + NVIDIA CUDA Migration
### Important expectation
- Do not treat the VS Code Codex chat thread as a reliable cross-machine migration channel.
- The closest official OpenAI statement is that Codex works across surfaces connected by your ChatGPT account, and the Codex app can pick up session history/configuration from the CLI and IDE extension.
- That is not the same as an explicit guarantee that a VS Code Codex thread on Mac will reopen with full editable context on a different Windows machine.
- In practice on this project, the actionable continuity source is:
  - the repo
  - `ai-docs/current/`
  - formal output artefacts under `outputs/phase3/`
  - a carry-over prompt

### Must-copy project files
Copy these from the Mac repo to the Windows repo:
- full repository working tree
- `ai-docs/current/`
- `docs/environment.md`
- `requirements.txt`
- `data/manifests/pairs.yaml`
- the actual datasets needed for the pairs you plan to test under `data/raw/Videos/`

If you want Windows to compare directly against existing formal results without re-running CPU/MPS history, also copy:
- `outputs/phase3/phase3_overall_methods_rich_v3/`
- `outputs/phase3/overall_method_compare_rich_v3_mps_real_accuracy_v2/`
- `outputs/phase3/method_b_accuracy_v1_cpu_vs_mps_real_v2/`
- `outputs/phase3/method_b_accuracy_v1_vs_native_res_mps_v1/`

### Optional cached weights
To avoid re-downloading Method B weights, copy this cache directory if present:

```text
~/.cache/torch/hub/checkpoints/
  superpoint_v1.pth
  superpoint_lightglue_v0-1_arxiv.pth
```

This is optional. If omitted, the Windows machine can download the weights again.

### Do not rely on copying local Codex state
Local Codex state on this Mac currently lives under `~/.codex/`, including files such as:
- `session_index.jsonl`
- `state_5.sqlite`
- `logs_1.sqlite`
- `config.toml`

These are useful for local inspection, but they are not the recommended migration path for this project.
Do not copy `auth.json` between machines.

### Windows CUDA setup steps
1. Install Git and a supported Python version.
2. Copy or clone the repo to the Windows machine.
3. Create the formal environment:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip setuptools wheel
```

If `python` on Windows resolves to the Microsoft Store / `WindowsApps` stub, or if the `py`
launcher is unavailable, use the full path of a real installed interpreter for this one-time
bootstrap step instead, for example:

```bash
C:\\Path\\To\\Python\\python.exe -m venv .venv
```

After `.venv` exists, switch to `.venv\\Scripts\\python.exe` / `.venv\\Scripts\\activate`.

4. Install a CUDA-enabled PyTorch build that matches the Windows machine's NVIDIA driver and CUDA runtime.
5. Then install the rest of the formal project dependencies:

```bash
python -m pip install -r requirements.txt
git clone https://github.com/cvg/LightGlue.git external/LightGlue
python -m pip install -e external/LightGlue
```

6. Verify CUDA:

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

7. Run Method B with:
- `--device cuda`
- or `--device auto`

### Recommended handoff prompt for a new Codex session on the Windows machine
Use this as the first prompt in the new VS Code Codex session:

```text
You are continuing the /Users/fallenwind/Desktop/mine/Stiching2Vid project on a new Windows + NVIDIA CUDA machine.

Before doing anything, strictly follow the ai-docs workflow:
1. Read:
- ai-docs/current/08_project_status_and_master_plan/08_project_status_and_master_plan.md
- ai-docs/current/10_execution_workflow/10_execution_workflow.md
- ai-docs/current/11_decision_log/11_decision_log.md
- ai-docs/current/12_implementation_log/12_implementation_log.md
- ai-docs/current/13_change_log/13_change_log.md
- ai-docs/current/14_open_issues_and_next_steps/14_open_issues_and_next_steps.md
- ai-docs/current/05_evaluation/05_evaluation.md
- ai-docs/current/06_method2_strong_matching/06_method2_strong_matching.md
- docs/environment.md

2. Treat these as current formal Method B artefacts:
- outputs/phase3/phase3_overall_methods_rich_v3/
- outputs/phase3/overall_method_compare_rich_v3_mps_real_accuracy_v2/
- outputs/phase3/method_b_accuracy_v1_cpu_vs_mps_real_v2/
- outputs/phase3/method_b_accuracy_v1_vs_native_res_mps_v1/

3. Current formal Method B baseline:
- accuracy_v1
- max_keypoints=4096
- resize_long_edge=1536
- depth_confidence=-1
- width_confidence=-1
- filter_threshold=0.1

4. Do not assume previous Mac chat history is available.
Use ai-docs and existing outputs as the source of truth.

5. Before changing code, add a new IMP planned record in ai-docs/current/12_implementation_log/12_implementation_log.md.
```

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
  --device auto \
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
- it exposes `Device (Method B / GPU)` as a GUI selector with `auto / cpu / mps / cuda`
- it is safest to launch it from the same environment you use for the target method
- on headless shells or remote terminals without a display server, the GUI may not launch even though the script imports correctly
- current upload path only supports registering left/right video files; uploads are copied into `data/raw/Videos/gui_uploads/` and appended to `data/manifests/pairs.yaml` with repo-relative paths
- output remains restricted to `outputs/runs/<run_id>/`
- after a run finishes, the GUI can auto-open that run directory in the system file manager

## Which Environment Should I Use?
- Formal recommendation: use `.venv` for the whole project.
- Historical local `.venv-methodb` directories, if they still exist on your machine, should be treated as leftover compatibility artifacts rather than a separate documented environment.
- If you are just checking a pair quickly on this machine and do not want to recreate environments yet, system `python3` can still be used for limited inspection, but it is not the documented reproducible path.
