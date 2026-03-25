#!/usr/bin/env python3
"""Run the formal dynamic seam comparison suite."""

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
    "mine_source_square_left_right",
    "mine_source_mcd1_left_right",
    "mine_source_traffic2_left_right",
    "mine_source_walking_left_right",
]

PAIR_ALIASES: Dict[str, str] = {
    "square": "mine_source_square_left_right",
    "mcd1": "mine_source_mcd1_left_right",
    "traffic2": "mine_source_traffic2_left_right",
    "walking": "mine_source_walking_left_right",
}

PRESETS: List[Dict[str, object]] = [
    {
        "preset_id": "baseline_fixed",
        "geometry_mode": "fixed_geometry",
        "seam_policy": "fixed",
        "seam_keyframe_every": 10,
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.0,
        "seam_trigger_cooldown_frames": 0,
        "seam_trigger_hysteresis_ratio": 1.0,
        "foreground_mode": "off",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
        "seam_smooth": "none",
        "seam_smooth_alpha": 0.8,
        "seam_smooth_window": 5,
    },
    {
        "preset_id": "keyframe_seam10",
        "geometry_mode": "fixed_geometry",
        "seam_policy": "keyframe",
        "seam_keyframe_every": 10,
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.0,
        "seam_trigger_cooldown_frames": 0,
        "seam_trigger_hysteresis_ratio": 1.0,
        "foreground_mode": "off",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
        "seam_smooth": "none",
        "seam_smooth_alpha": 0.8,
        "seam_smooth_window": 5,
    },
    {
        "preset_id": "trigger_fused_d18_fg008",
        "geometry_mode": "fixed_geometry",
        "seam_policy": "trigger",
        "seam_keyframe_every": 10,
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.08,
        "seam_trigger_cooldown_frames": 0,
        "seam_trigger_hysteresis_ratio": 1.0,
        "foreground_mode": "disagreement",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
        "seam_smooth": "none",
        "seam_smooth_alpha": 0.8,
        "seam_smooth_window": 5,
    },
    {
        "preset_id": "adaptive_trigger_fused_d18_fg008",
        "geometry_mode": "adaptive_update",
        "seam_policy": "trigger",
        "seam_keyframe_every": 10,
        "seam_trigger_diff_threshold": 18.0,
        "seam_trigger_overlap_ratio": 0.0,
        "seam_trigger_foreground_ratio": 0.08,
        "seam_trigger_cooldown_frames": 0,
        "seam_trigger_hysteresis_ratio": 1.0,
        "foreground_mode": "disagreement",
        "foreground_diff_threshold": 24.0,
        "foreground_dilate": 5,
        "seam_smooth": "none",
        "seam_smooth_alpha": 0.8,
        "seam_smooth_window": 5,
    },
]
PRESET_BY_ID: Dict[str, Dict[str, object]] = {str(preset["preset_id"]): preset for preset in PRESETS}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal dynamic seam comparison suite "
            "by invoking scripts/run_baseline_video.py on representative pairs."
        )
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional pair ids or aliases; default uses the built-in dynamic compare pairs",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        choices=sorted(PRESET_BY_ID),
        help="Optional preset ids; default uses the built-in dynamic compare preset set",
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
        default=6000,
        help="Maximum number of frames to process; use a large value to traverse full videos",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=100,
        help="Snapshot interval passed through to run_baseline_video.py",
    )
    parser.add_argument(
        "--seam_snapshot_on_recompute",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to save seam event snapshots",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional fps override passed through to run_baseline_video.py; useful for frames datasets with missing manifest fps",
    )
    parser.add_argument("--device", default=None, help="Optional Method B device override")
    parser.add_argument("--force_cpu", action="store_true", help="Force Method B to use CPU")
    parser.add_argument("--weights_dir", default=None, help="Optional Method B weights dir")
    parser.add_argument("--max_keypoints", type=int, default=None, help="Optional max_keypoints override")
    parser.add_argument("--resize_long_edge", type=int, default=None, help="Optional resize_long_edge override")
    return parser


def _resolve_pairs(requested_pairs: Sequence[str] | None) -> List[str]:
    if not requested_pairs:
        return list(DEFAULT_PAIRS)
    return [PAIR_ALIASES.get(raw_name, raw_name) for raw_name in requested_pairs]


def _resolve_presets(requested_presets: Sequence[str] | None) -> List[Dict[str, object]]:
    if not requested_presets:
        return list(PRESETS)
    return [dict(PRESET_BY_ID[preset_id]) for preset_id in requested_presets]


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
        "--seam_keyframe_every",
        str(preset["seam_keyframe_every"]),
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
        "--seam_smooth",
        str(preset["seam_smooth"]),
        "--seam_smooth_alpha",
        str(preset["seam_smooth_alpha"]),
        "--seam_smooth_window",
        str(preset["seam_smooth_window"]),
        "--snapshot_every",
        str(args.snapshot_every),
        "--seam_snapshot_on_recompute",
        str(args.seam_snapshot_on_recompute),
        "--run_id",
        run_id,
    ]
    if args.fps is not None:
        cmd.extend(["--fps", str(args.fps)])
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


def _extract_result_row(
    repo_root: Path,
    suite_id: str,
    pair_id: str,
    preset: Dict[str, object],
    run_id: str,
) -> Dict[str, object]:
    run_dir = repo_root / "outputs" / "runs" / run_id
    metrics = _load_json(run_dir / "metrics_preview.json")
    debug = _load_json(run_dir / "debug.json")
    warnings = debug.get("warnings") if isinstance(debug.get("warnings"), list) else []
    errors = debug.get("errors") if isinstance(debug.get("errors"), list) else []
    return {
        "suite_id": suite_id,
        "pair_id": pair_id,
        "preset_id": str(preset["preset_id"]),
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(repo_root)) if run_dir.exists() else str(run_dir),
        "fps": debug.get("fps"),
        "fps_source": debug.get("fps_source"),
        "geometry_mode": metrics.get("geometry_mode"),
        "seam_policy": metrics.get("seam_policy"),
        "seam_keyframe_every_effective": metrics.get("seam_keyframe_every_effective"),
        "seam_trigger_diff_threshold": preset["seam_trigger_diff_threshold"],
        "seam_trigger_foreground_ratio": preset["seam_trigger_foreground_ratio"],
        "foreground_mode": preset["foreground_mode"],
        "seam_smooth": preset["seam_smooth"],
        "processed_frames": metrics.get("processed_frames"),
        "success_frames": metrics.get("success_frames"),
        "fallback_frames": metrics.get("fallback_frames"),
        "seam_recompute_count": metrics.get("seam_recompute_count"),
        "geometry_update_count": metrics.get("geometry_update_count"),
        "mean_inliers": metrics.get("mean_inliers"),
        "mean_inlier_ratio": metrics.get("mean_inlier_ratio"),
        "mean_overlap_diff_after": metrics.get("mean_overlap_diff_after"),
        "mean_seam_mask_change_ratio": metrics.get("mean_seam_mask_change_ratio"),
        "mean_stitched_delta": metrics.get("mean_stitched_delta"),
        "mean_foreground_ratio": metrics.get("mean_foreground_ratio"),
        "mean_jitter_sm": metrics.get("mean_jitter_sm"),
        "temporal_primary_metric": metrics.get("temporal_primary_metric"),
        "temporal_primary_value": metrics.get("temporal_primary_value"),
        "approx_fps": metrics.get("approx_fps"),
        "feature_backend_effective": metrics.get("feature_backend_effective"),
        "matcher_backend_effective": metrics.get("matcher_backend_effective"),
        "geometry_backend_effective": metrics.get("geometry_backend_effective"),
        "seam_snapshot_count": metrics.get("seam_snapshot_count"),
        "warnings_count": len(warnings),
        "errors_count": len(errors),
    }


def _aggregate_preset_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["preset_id"]), []).append(row)

    numeric_fields = [
        "processed_frames",
        "success_frames",
        "fallback_frames",
        "seam_recompute_count",
        "geometry_update_count",
        "mean_inliers",
        "mean_inlier_ratio",
        "mean_overlap_diff_after",
        "mean_seam_mask_change_ratio",
        "mean_stitched_delta",
        "mean_foreground_ratio",
        "mean_jitter_sm",
        "temporal_primary_value",
        "approx_fps",
        "seam_snapshot_count",
        "warnings_count",
        "errors_count",
    ]
    out: List[Dict[str, object]] = []
    for preset_id, preset_rows in grouped.items():
        base = preset_rows[0]
        summary = {
            "suite_id": base["suite_id"],
            "preset_id": preset_id,
            "pair_count": len(preset_rows),
            "geometry_mode": base["geometry_mode"],
            "seam_policy": base["seam_policy"],
            "foreground_mode": base["foreground_mode"],
            "seam_smooth": base["seam_smooth"],
            "temporal_primary_metric": base["temporal_primary_metric"],
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in preset_rows if row.get(field) is not None]
            summary[field] = float(sum(values) / len(values)) if values else 0.0
        out.append(summary)
    return out


def _safe_delta(new_value, old_value):
    try:
        if new_value is None or old_value is None:
            return None
        return float(new_value) - float(old_value)
    except (TypeError, ValueError):
        return None


def _build_pair_compare(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_pair: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        if str(row.get("status")) != "passed":
            continue
        by_pair.setdefault(str(row["pair_id"]), {})[str(row["preset_id"])] = row

    compare_rows: List[Dict[str, object]] = []
    baseline_id = "baseline_fixed"
    compare_order = [
        "keyframe_seam10",
        "trigger_fused_d18_fg008",
        "adaptive_trigger_fused_d18_fg008",
    ]
    for pair_id, preset_rows in sorted(by_pair.items()):
        baseline = preset_rows.get(baseline_id)
        if not baseline:
            continue
        for preset_id in compare_order:
            current = preset_rows.get(preset_id)
            if not current:
                continue
            compare_rows.append(
                {
                    "pair_id": pair_id,
                    "baseline_preset": baseline_id,
                    "preset_id": preset_id,
                    "baseline_run_id": baseline.get("run_id"),
                    "run_id": current.get("run_id"),
                    "baseline_mean_overlap_diff_after": baseline.get("mean_overlap_diff_after"),
                    "mean_overlap_diff_after": current.get("mean_overlap_diff_after"),
                    "delta_mean_overlap_diff_after": _safe_delta(
                        current.get("mean_overlap_diff_after"),
                        baseline.get("mean_overlap_diff_after"),
                    ),
                    "baseline_mean_seam_mask_change_ratio": baseline.get("mean_seam_mask_change_ratio"),
                    "mean_seam_mask_change_ratio": current.get("mean_seam_mask_change_ratio"),
                    "delta_mean_seam_mask_change_ratio": _safe_delta(
                        current.get("mean_seam_mask_change_ratio"),
                        baseline.get("mean_seam_mask_change_ratio"),
                    ),
                    "baseline_mean_stitched_delta": baseline.get("mean_stitched_delta"),
                    "mean_stitched_delta": current.get("mean_stitched_delta"),
                    "delta_mean_stitched_delta": _safe_delta(
                        current.get("mean_stitched_delta"),
                        baseline.get("mean_stitched_delta"),
                    ),
                    "baseline_seam_recompute_count": baseline.get("seam_recompute_count"),
                    "seam_recompute_count": current.get("seam_recompute_count"),
                    "delta_seam_recompute_count": _safe_delta(
                        current.get("seam_recompute_count"),
                        baseline.get("seam_recompute_count"),
                    ),
                    "baseline_geometry_update_count": baseline.get("geometry_update_count"),
                    "geometry_update_count": current.get("geometry_update_count"),
                    "delta_geometry_update_count": _safe_delta(
                        current.get("geometry_update_count"),
                        baseline.get("geometry_update_count"),
                    ),
                    "baseline_approx_fps": baseline.get("approx_fps"),
                    "approx_fps": current.get("approx_fps"),
                    "delta_approx_fps": _safe_delta(
                        current.get("approx_fps"),
                        baseline.get("approx_fps"),
                    ),
                }
            )
    return compare_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pairs = _resolve_pairs(args.pairs)
    presets = _resolve_presets(args.presets)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_id = args.suite_id or f"{timestamp}_dynamic_compare"
    suite_dir = repo_root / "outputs" / "video_compare" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    overall_rc = 0

    for pair_index, pair_id in enumerate(pairs, start=1):
        for preset_index, preset in enumerate(presets, start=1):
            safe_pair = pair_id.replace("/", "_").replace(" ", "_")
            run_id = (
                f"{suite_id}__{pair_index:02d}_{preset_index:02d}__"
                f"{safe_pair}__{preset['preset_id']}"
            )
            cmd = _build_case_command(args, pair_id, preset, run_id)
            stdout_path = suite_dir / (
                f"{pair_index:02d}_{preset_index:02d}_{safe_pair}_{preset['preset_id']}.stdout.txt"
            )
            stderr_path = suite_dir / (
                f"{pair_index:02d}_{preset_index:02d}_{safe_pair}_{preset['preset_id']}.stderr.txt"
            )
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

    passed_rows = [row for row in results if row.get("status") == "passed"]
    preset_summary = _aggregate_preset_rows(passed_rows)
    pair_compare = _build_pair_compare(results)
    _write_csv(suite_dir / "summary.csv", results)
    _write_csv(suite_dir / "preset_summary.csv", preset_summary)
    _write_csv(suite_dir / "pair_compare.csv", pair_compare)
    (suite_dir / "summary.json").write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "pairs": pairs,
                "presets": [str(preset["preset_id"]) for preset in presets],
                "max_frames": args.max_frames,
                "fps": args.fps,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (suite_dir / "preset_summary.json").write_text(
        json.dumps(preset_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dynamic compare suite completed: {suite_dir}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
