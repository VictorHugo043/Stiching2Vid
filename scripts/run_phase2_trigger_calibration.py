#!/usr/bin/env python3
"""Run Phase 2 trigger seam/adaptive calibration suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Sequence


DEFAULT_PAIRS: List[str] = [
    "mine_source_mcd1_left_right",
    "mine_source_mcd2_left_right",
    "mine_source_square_left_right",
    "mine_source_traffic1_left_right",
    "mine_source_traffic2_left_right",
    "mine_source_walking_left_right",
]

PRESETS: List[Dict[str, object]] = [
    {
        "preset_id": "trigger_plain_d18",
        "geometry_mode": "fixed_geometry",
        "seam_policy": "trigger",
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.0,
        "seam_trigger_cooldown_frames": 0,
        "seam_trigger_hysteresis_ratio": 1.0,
        "foreground_mode": "off",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
    },
    {
        "preset_id": "trigger_fused_d18_fg008",
        "geometry_mode": "fixed_geometry",
        "seam_policy": "trigger",
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.08,
        "seam_trigger_cooldown_frames": 0,
        "seam_trigger_hysteresis_ratio": 1.0,
        "foreground_mode": "disagreement",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
    },
    {
        "preset_id": "trigger_stable_d18_fg008_cd6_h075",
        "geometry_mode": "fixed_geometry",
        "seam_policy": "trigger",
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.08,
        "seam_trigger_cooldown_frames": 6,
        "seam_trigger_hysteresis_ratio": 0.75,
        "foreground_mode": "disagreement",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
    },
    {
        "preset_id": "adaptive_fused_d18_fg008",
        "geometry_mode": "adaptive_update",
        "seam_policy": "trigger",
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.08,
        "seam_trigger_cooldown_frames": 0,
        "seam_trigger_hysteresis_ratio": 1.0,
        "foreground_mode": "disagreement",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
    },
    {
        "preset_id": "adaptive_stable_d18_fg008_cd6_h075",
        "geometry_mode": "adaptive_update",
        "seam_policy": "trigger",
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.08,
        "seam_trigger_cooldown_frames": 6,
        "seam_trigger_hysteresis_ratio": 0.75,
        "foreground_mode": "disagreement",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
    },
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight Phase 2 calibration suite for trigger seam, "
            "adaptive geometry refresh, cooldown/hysteresis, and compatible "
            "foreground-aware protection."
        )
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional pair ids; default uses the mine_source dynamic pairs",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable used to invoke scripts/run_baseline_video.py",
    )
    parser.add_argument(
        "--suite_id",
        default=None,
        help="Optional suite id; default is timestamped",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running remaining cases after a failure",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=80,
        help="Maximum number of frames per run",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride",
    )
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=1000,
        help="Snapshot interval passed through to run_baseline_video.py",
    )
    parser.add_argument(
        "--seam_snapshot_on_recompute",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to save seam-event snapshots for calibration runs",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Method B device override",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force Method B backends to use CPU",
    )
    parser.add_argument(
        "--weights_dir",
        default=None,
        help="Optional Method B weights dir passed through to run_baseline_video.py",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=None,
        help="Optional Method B max_keypoints override",
    )
    parser.add_argument(
        "--resize_long_edge",
        type=int,
        default=None,
        help="Optional Method B resize_long_edge override",
    )
    return parser


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_case_command(
    args: argparse.Namespace,
    pair_id: str,
    preset: Dict[str, object],
    run_id: str,
) -> List[str]:
    cmd = [
        str(args.python_bin),
        "scripts/run_baseline_video.py",
        "--pair",
        pair_id,
        "--manifest",
        str(args.manifest),
        "--start",
        str(args.start),
        "--max_frames",
        str(args.max_frames),
        "--stride",
        str(args.stride),
        "--geometry_mode",
        str(preset["geometry_mode"]),
        "--reuse_mode",
        "frame0_all",
        "--feature_backend",
        "superpoint",
        "--matcher_backend",
        "lightglue",
        "--geometry_backend",
        "opencv_usac_magsac",
        "--seam_policy",
        str(preset["seam_policy"]),
        "--seam_trigger_diff_threshold",
        str(preset["seam_trigger_diff_threshold"]),
        "--seam_trigger_overlap_ratio",
        str(preset["seam_trigger_overlap_ratio"]),
        "--seam_trigger_foreground_ratio",
        str(preset["seam_trigger_foreground_ratio"]),
        "--seam_trigger_cooldown_frames",
        str(preset["seam_trigger_cooldown_frames"]),
        "--seam_trigger_hysteresis_ratio",
        str(preset["seam_trigger_hysteresis_ratio"]),
        "--foreground_mode",
        str(preset["foreground_mode"]),
        "--foreground_diff_threshold",
        str(preset["foreground_diff_threshold"]),
        "--foreground_dilate",
        str(preset["foreground_dilate"]),
        "--snapshot_every",
        str(args.snapshot_every),
        "--seam_snapshot_on_recompute",
        str(args.seam_snapshot_on_recompute),
        "--run_id",
        run_id,
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.force_cpu:
        cmd.append("--force_cpu")
    if args.weights_dir:
        cmd.extend(["--weights_dir", str(args.weights_dir)])
    if args.max_keypoints is not None:
        cmd.extend(["--max_keypoints", str(args.max_keypoints)])
    if args.resize_long_edge is not None:
        cmd.extend(["--resize_long_edge", str(args.resize_long_edge)])
    return cmd


def _extract_result_row(repo_root: Path, suite_id: str, pair_id: str, preset: Dict[str, object], run_id: str) -> Dict[str, object]:
    run_dir = repo_root / "outputs" / "runs" / run_id
    metrics = _load_json(run_dir / "metrics_preview.json")
    debug = _load_json(run_dir / "debug.json")
    processed_frames = int(metrics.get("processed_frames", 0) or 0)
    seam_recompute_count = int(metrics.get("seam_recompute_count", 0) or 0)
    geometry_update_count = int(metrics.get("geometry_update_count", 0) or 0)
    seam_recompute_after_init = max(0, seam_recompute_count - 1)
    return {
        "suite_id": suite_id,
        "pair_id": pair_id,
        "preset_id": str(preset["preset_id"]),
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(repo_root)) if run_dir.exists() else str(run_dir),
        "geometry_mode": metrics.get("geometry_mode"),
        "seam_policy": metrics.get("seam_policy"),
        "seam_trigger_diff_threshold": preset["seam_trigger_diff_threshold"],
        "seam_trigger_overlap_ratio": preset["seam_trigger_overlap_ratio"],
        "seam_trigger_foreground_ratio": preset["seam_trigger_foreground_ratio"],
        "seam_trigger_cooldown_frames": preset["seam_trigger_cooldown_frames"],
        "seam_trigger_hysteresis_ratio": preset["seam_trigger_hysteresis_ratio"],
        "foreground_mode": preset["foreground_mode"],
        "foreground_diff_threshold": preset["foreground_diff_threshold"],
        "foreground_dilate": preset["foreground_dilate"],
        "processed_frames": processed_frames,
        "success_frames": metrics.get("success_frames"),
        "fallback_frames": metrics.get("fallback_frames"),
        "seam_recompute_count": seam_recompute_count,
        "seam_recompute_after_init": seam_recompute_after_init,
        "seam_recompute_per_100f": (
            100.0 * float(seam_recompute_count) / float(max(1, processed_frames))
        ),
        "seam_recompute_after_init_per_100f": (
            100.0 * float(seam_recompute_after_init) / float(max(1, processed_frames))
        ),
        "geometry_update_count": geometry_update_count,
        "geometry_update_per_100f": (
            100.0 * float(geometry_update_count) / float(max(1, processed_frames))
        ),
        "mean_inliers": metrics.get("mean_inliers"),
        "mean_inlier_ratio": metrics.get("mean_inlier_ratio"),
        "mean_overlap_diff_after": metrics.get("mean_overlap_diff_after"),
        "mean_seam_mask_change_ratio": metrics.get("mean_seam_mask_change_ratio"),
        "mean_stitched_delta": metrics.get("mean_stitched_delta"),
        "mean_foreground_ratio": metrics.get("mean_foreground_ratio"),
        "foreground_triggered_count": metrics.get("foreground_triggered_count"),
        "mean_jitter_sm": metrics.get("mean_jitter_sm"),
        "temporal_primary_metric": metrics.get("temporal_primary_metric"),
        "temporal_primary_value": metrics.get("temporal_primary_value"),
        "approx_fps": metrics.get("approx_fps"),
        "feature_backend_effective": metrics.get("feature_backend_effective"),
        "matcher_backend_effective": metrics.get("matcher_backend_effective"),
        "geometry_backend_effective": metrics.get("geometry_backend_effective"),
        "warnings_count": len(debug.get("warnings", []) if isinstance(debug.get("warnings"), list) else []),
        "errors_count": len(debug.get("errors", []) if isinstance(debug.get("errors"), list) else []),
    }


def _aggregate_preset_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["preset_id"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    numeric_fields = [
        "processed_frames",
        "success_frames",
        "fallback_frames",
        "seam_recompute_count",
        "seam_recompute_after_init",
        "seam_recompute_per_100f",
        "seam_recompute_after_init_per_100f",
        "geometry_update_count",
        "geometry_update_per_100f",
        "mean_inliers",
        "mean_inlier_ratio",
        "mean_overlap_diff_after",
        "mean_seam_mask_change_ratio",
        "mean_stitched_delta",
        "mean_foreground_ratio",
        "foreground_triggered_count",
        "mean_jitter_sm",
        "temporal_primary_value",
        "approx_fps",
        "warnings_count",
        "errors_count",
    ]
    for preset_id, preset_rows in grouped.items():
        base = dict(preset_rows[0])
        summary = {
            "suite_id": base["suite_id"],
            "preset_id": preset_id,
            "geometry_mode": base["geometry_mode"],
            "seam_policy": base["seam_policy"],
            "pair_count": len(preset_rows),
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in preset_rows if row.get(field) is not None]
            summary[field] = float(sum(values) / len(values)) if values else 0.0
        summary_rows.append(summary)
    return summary_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pairs = list(args.pairs) if args.pairs else list(DEFAULT_PAIRS)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_id = args.suite_id or f"{timestamp}_phase2_trigger_calibration"
    suite_dir = repo_root / "outputs" / "video_calibration" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    overall_rc = 0

    for pair_index, pair_id in enumerate(pairs, start=1):
        for preset_index, preset in enumerate(PRESETS, start=1):
            safe_pair = pair_id.replace("/", "_").replace(" ", "_")
            run_id = f"{suite_id}__{pair_index:02d}_{preset_index:02d}__{safe_pair}__{preset['preset_id']}"
            cmd = _build_case_command(args, pair_id, preset, run_id)
            stdout_path = suite_dir / f"{pair_index:02d}_{preset_index:02d}_{safe_pair}_{preset['preset_id']}.stdout.txt"
            stderr_path = suite_dir / f"{pair_index:02d}_{preset_index:02d}_{safe_pair}_{preset['preset_id']}.stderr.txt"

            row: Dict[str, object] = {
                "suite_id": suite_id,
                "pair_id": pair_id,
                "preset_id": str(preset["preset_id"]),
                "run_id": run_id,
                "status": "dry_run" if args.dry_run else "planned",
                "returncode": None,
                "command": cmd,
            }
            if args.dry_run:
                results.append(row)
                continue

            completed = subprocess.run(
                cmd,
                cwd=str(repo_root),
                text=True,
                capture_output=True,
            )
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            row["returncode"] = int(completed.returncode)
            row["status"] = "passed" if completed.returncode == 0 else "failed"
            row["stdout_path"] = str(stdout_path.relative_to(repo_root))
            row["stderr_path"] = str(stderr_path.relative_to(repo_root))

            if completed.returncode == 0:
                row.update(_extract_result_row(repo_root, suite_id, pair_id, preset, run_id))
            else:
                overall_rc = completed.returncode or 1
                if not args.continue_on_error:
                    results.append(row)
                    _write_csv(suite_dir / "summary.csv", results)
                    (suite_dir / "summary.json").write_text(
                        json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    return overall_rc
            results.append(row)

    preset_summary = _aggregate_preset_rows([row for row in results if row.get("status") == "passed"])
    _write_csv(suite_dir / "summary.csv", results)
    _write_csv(suite_dir / "preset_summary.csv", preset_summary)
    (suite_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (suite_dir / "preset_summary.json").write_text(
        json.dumps(preset_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Calibration suite completed: {suite_dir}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
