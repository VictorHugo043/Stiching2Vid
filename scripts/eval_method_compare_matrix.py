#!/usr/bin/env python3
"""Run a flexible multi-pair Method A(ORB/SIFT) vs Method B video comparison suite."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Sequence


DEFAULT_PAIRS: List[str] = [
    "mine_source_indoor2_left_right",
    "mine_source_pujiang1_left_right",
    "kitti_raw_data_2011_09_26_drive_0005_image_02_image_03",
]

PAIR_ALIASES: Dict[str, str] = {
    "mysourceindoor2": "mine_source_indoor2_left_right",
    "mysourcepujiang": "mine_source_pujiang1_left_right",
    "pujiang": "mine_source_pujiang1_left_right",
    "kitti0005": "kitti_raw_data_2011_09_26_drive_0005_image_02_image_03",
}

DEFAULT_METHOD_B_MAX_KEYPOINTS = 4096
DEFAULT_METHOD_B_RESIZE_LONG_EDGE = 1536
DEFAULT_METHOD_B_DEPTH_CONFIDENCE = -1.0
DEFAULT_METHOD_B_WIDTH_CONFIDENCE = -1.0
DEFAULT_METHOD_B_FILTER_THRESHOLD = 0.1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Method A(ORB/SIFT) vs Method B video comparison suite "
            "by invoking scripts/run_baseline_video.py once per pair and method."
        )
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional pair ids or aliases; default uses the built-in formal compare pairs",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["method_a_orb", "method_a_sift", "method_b"],
        choices=["method_a_orb", "method_a_sift", "method_b"],
        help="Methods to include in the suite",
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
        "--video_mode",
        type=int,
        default=1,
        choices=[0, 1],
        help="Video mode preset for the suite; formal fixed-geometry compare uses 1",
    )
    parser.add_argument(
        "--reuse_mode",
        default="frame0_all",
        choices=["frame0_all", "frame0_geom", "frame0_seam", "emaH"],
        help="Reuse mode passed through to run_baseline_video.py",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=6000,
        help="Maximum number of frames to process; formal full-length compare uses 6000",
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
        default=200,
        help="Snapshot interval passed through to run_baseline_video.py",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Method B device override",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional fps override passed through to run_baseline_video.py; useful for frames datasets with missing manifest fps",
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
        default=DEFAULT_METHOD_B_MAX_KEYPOINTS,
        help="Method B max_keypoints override; formal compare defaults to the accuracy preset",
    )
    parser.add_argument(
        "--resize_long_edge",
        type=int,
        default=DEFAULT_METHOD_B_RESIZE_LONG_EDGE,
        help="Method B resize_long_edge override; formal compare defaults to the accuracy preset",
    )
    parser.add_argument(
        "--depth_confidence",
        type=float,
        default=DEFAULT_METHOD_B_DEPTH_CONFIDENCE,
        help="Method B LightGlue depth_confidence override; formal compare defaults to the accuracy preset",
    )
    parser.add_argument(
        "--width_confidence",
        type=float,
        default=DEFAULT_METHOD_B_WIDTH_CONFIDENCE,
        help="Method B LightGlue width_confidence override; formal compare defaults to the accuracy preset",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=DEFAULT_METHOD_B_FILTER_THRESHOLD,
        help="Method B LightGlue filter_threshold override; formal compare defaults to the accuracy preset",
    )
    parser.add_argument(
        "--method_a_feature",
        default="orb",
        choices=["orb", "sift"],
        help="Deprecated compatibility option; use --methods method_a_orb/method_a_sift instead",
    )
    return parser


def _resolve_pairs(requested_pairs: Sequence[str] | None) -> List[str]:
    if not requested_pairs:
        return list(DEFAULT_PAIRS)
    return [PAIR_ALIASES.get(raw_name, raw_name) for raw_name in requested_pairs]


def _build_case_command(
    args: argparse.Namespace,
    method: str,
    pair_id: str,
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
        "--video_mode",
        str(args.video_mode),
        "--reuse_mode",
        str(args.reuse_mode),
        "--snapshot_every",
        str(args.snapshot_every),
        "--run_id",
        run_id,
    ]
    if method in {"method_a", "method_a_orb", "method_a_sift"}:
        feature_name = "sift" if method == "method_a_sift" else "orb"
        if method == "method_a":
            feature_name = str(args.method_a_feature).strip().lower()
        feature_backend = "opencv_sift" if feature_name == "sift" else "opencv_orb"
        cmd.extend(
            [
                "--feature",
                feature_name,
                "--feature_backend",
                feature_backend,
                "--matcher_backend",
                "opencv_bf_ratio",
                "--geometry_backend",
                "opencv_ransac",
            ]
        )
        if args.fps is not None:
            cmd.extend(["--fps", str(args.fps)])
        return cmd

    cmd.extend(
        [
            "--feature_backend",
            "superpoint",
            "--matcher_backend",
            "lightglue",
            "--geometry_backend",
            "opencv_usac_magsac",
        ]
    )
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.fps is not None:
        cmd.extend(["--fps", str(args.fps)])
    if args.force_cpu:
        cmd.append("--force_cpu")
    if args.weights_dir:
        cmd.extend(["--weights_dir", str(args.weights_dir)])
    if args.max_keypoints is not None:
        cmd.extend(["--max_keypoints", str(args.max_keypoints)])
    if args.resize_long_edge is not None:
        cmd.extend(["--resize_long_edge", str(args.resize_long_edge)])
    if args.depth_confidence is not None:
        cmd.extend(["--depth_confidence", str(args.depth_confidence)])
    if args.width_confidence is not None:
        cmd.extend(["--width_confidence", str(args.width_confidence)])
    if args.filter_threshold is not None:
        cmd.extend(["--filter_threshold", str(args.filter_threshold)])
    return cmd


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_result_row(repo_root: Path, suite_id: str, method: str, pair_id: str, run_id: str) -> Dict[str, object]:
    run_dir = repo_root / "outputs" / "runs" / run_id
    metrics = _load_json(run_dir / "metrics_preview.json")
    debug = _load_json(run_dir / "debug.json")

    warnings = debug.get("warnings") if isinstance(debug.get("warnings"), list) else []
    errors = debug.get("errors") if isinstance(debug.get("errors"), list) else []
    pair_meta = {
        "suite_id": suite_id,
        "pair_id": pair_id,
        "method": method,
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(repo_root)) if run_dir.exists() else str(run_dir),
        "dataset": debug.get("dataset"),
        "input_type": debug.get("input_type"),
        "fps": debug.get("fps"),
        "fps_source": debug.get("fps_source"),
        "video_mode": metrics.get("video_mode"),
        "reuse_mode": metrics.get("reuse_mode"),
        "geometry_mode": metrics.get("geometry_mode"),
        "jitter_meaningful": metrics.get("jitter_meaningful"),
        "total_frames": metrics.get("total_frames"),
        "processed_frames": metrics.get("processed_frames"),
        "success_frames": metrics.get("success_frames"),
        "fallback_frames": metrics.get("fallback_frames"),
        "mean_inliers": metrics.get("mean_inliers"),
        "mean_inlier_ratio": metrics.get("mean_inlier_ratio"),
        "avg_runtime_ms": metrics.get("avg_runtime_ms"),
        "approx_fps": metrics.get("approx_fps"),
        "mean_reprojection_error": metrics.get("mean_reprojection_error"),
        "mean_inlier_spatial_coverage": metrics.get("mean_inlier_spatial_coverage"),
        "mean_overlap_diff_after": metrics.get("mean_overlap_diff_after"),
        "mean_seam_band_illuminance_diff": metrics.get("mean_seam_band_illuminance_diff"),
        "mean_seam_band_gradient_disagreement": metrics.get("mean_seam_band_gradient_disagreement"),
        "mean_seam_band_flicker": metrics.get("mean_seam_band_flicker"),
        "mean_stitched_delta": metrics.get("mean_stitched_delta"),
        "mean_jitter_raw": metrics.get("mean_jitter_raw"),
        "mean_jitter_sm": metrics.get("mean_jitter_sm"),
        "p95_jitter_raw": metrics.get("p95_jitter_raw"),
        "p95_jitter_sm": metrics.get("p95_jitter_sm"),
        "seam_keyframe_count": metrics.get("seam_keyframe_count"),
        "seam_runtime_ms_mean": metrics.get("seam_runtime_ms_mean"),
        "crop_keyframe_count": metrics.get("crop_keyframe_count"),
        "crop_black_border_ratio_low_mean": metrics.get("crop_black_border_ratio_low_mean"),
        "reinit_count": metrics.get("reinit_count"),
        "init_ms_mean": metrics.get("init_ms_mean"),
        "per_frame_ms_mean": metrics.get("per_frame_ms_mean"),
        "reuse_per_frame_ms_mean": metrics.get("reuse_per_frame_ms_mean"),
        "avg_feature_runtime_ms_left": metrics.get("avg_feature_runtime_ms_left"),
        "avg_feature_runtime_ms_right": metrics.get("avg_feature_runtime_ms_right"),
        "avg_matching_runtime_ms": metrics.get("avg_matching_runtime_ms"),
        "avg_geometry_runtime_ms": metrics.get("avg_geometry_runtime_ms"),
        "feature_backend_effective": metrics.get("feature_backend_effective"),
        "matcher_backend_effective": metrics.get("matcher_backend_effective"),
        "geometry_backend_effective": metrics.get("geometry_backend_effective"),
        "warnings_count": len(warnings),
        "errors_count": len(errors),
        "stitched_exists": int((run_dir / "stitched.mp4").exists()),
        "transforms_exists": int((run_dir / "transforms.csv").exists()),
        "debug_exists": int((run_dir / "debug.json").exists()),
        "metrics_exists": int((run_dir / "metrics_preview.json").exists()),
        "jitter_exists": int((run_dir / "jitter_timeseries.csv").exists()),
    }
    return pair_meta


def _write_summary_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_pair_compare(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_pair: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), {})[str(row["method"])] = row

    compare_rows: List[Dict[str, object]] = []
    for pair_id in sorted(by_pair):
        method_rows = by_pair[pair_id]
        for lhs_method, rhs_method in combinations(sorted(method_rows), 2):
            lhs = method_rows.get(lhs_method, {})
            rhs = method_rows.get(rhs_method, {})
            compare_rows.append(
                {
                    "pair_id": pair_id,
                    "compare": f"{lhs_method}__vs__{rhs_method}",
                    "lhs_method": lhs_method,
                    "rhs_method": rhs_method,
                    "lhs_run_id": lhs.get("run_id"),
                    "rhs_run_id": rhs.get("run_id"),
                    "lhs_processed_frames": lhs.get("processed_frames"),
                    "rhs_processed_frames": rhs.get("processed_frames"),
                    "lhs_success_frames": lhs.get("success_frames"),
                    "rhs_success_frames": rhs.get("success_frames"),
                    "lhs_fallback_frames": lhs.get("fallback_frames"),
                    "rhs_fallback_frames": rhs.get("fallback_frames"),
                    "lhs_mean_inliers": lhs.get("mean_inliers"),
                    "rhs_mean_inliers": rhs.get("mean_inliers"),
                    "delta_mean_inliers": _safe_delta(rhs.get("mean_inliers"), lhs.get("mean_inliers")),
                    "lhs_mean_inlier_ratio": lhs.get("mean_inlier_ratio"),
                    "rhs_mean_inlier_ratio": rhs.get("mean_inlier_ratio"),
                    "delta_mean_inlier_ratio": _safe_delta(
                        rhs.get("mean_inlier_ratio"),
                        lhs.get("mean_inlier_ratio"),
                    ),
                    "lhs_avg_runtime_ms": lhs.get("avg_runtime_ms"),
                    "rhs_avg_runtime_ms": rhs.get("avg_runtime_ms"),
                    "delta_avg_runtime_ms": _safe_delta(rhs.get("avg_runtime_ms"), lhs.get("avg_runtime_ms")),
                    "lhs_approx_fps": lhs.get("approx_fps"),
                    "rhs_approx_fps": rhs.get("approx_fps"),
                    "delta_approx_fps": _safe_delta(rhs.get("approx_fps"), lhs.get("approx_fps")),
                    "lhs_mean_reprojection_error": lhs.get("mean_reprojection_error"),
                    "rhs_mean_reprojection_error": rhs.get("mean_reprojection_error"),
                    "delta_mean_reprojection_error": _safe_delta(
                        rhs.get("mean_reprojection_error"),
                        lhs.get("mean_reprojection_error"),
                    ),
                    "lhs_mean_stitched_delta": lhs.get("mean_stitched_delta"),
                    "rhs_mean_stitched_delta": rhs.get("mean_stitched_delta"),
                    "delta_mean_stitched_delta": _safe_delta(
                        rhs.get("mean_stitched_delta"),
                        lhs.get("mean_stitched_delta"),
                    ),
                    "lhs_warnings_count": lhs.get("warnings_count"),
                    "rhs_warnings_count": rhs.get("warnings_count"),
                    "lhs_feature_backend": lhs.get("feature_backend_effective"),
                    "rhs_feature_backend": rhs.get("feature_backend_effective"),
                    "lhs_matcher_backend": lhs.get("matcher_backend_effective"),
                    "rhs_matcher_backend": rhs.get("matcher_backend_effective"),
                    "lhs_geometry_backend": lhs.get("geometry_backend_effective"),
                    "rhs_geometry_backend": rhs.get("geometry_backend_effective"),
                    "geometry_mode": rhs.get("geometry_mode") or lhs.get("geometry_mode"),
                    "jitter_meaningful": rhs.get("jitter_meaningful") or lhs.get("jitter_meaningful"),
                }
            )
    return compare_rows


def _safe_delta(new_value, old_value):
    try:
        if new_value is None or old_value is None:
            return None
        return float(new_value) - float(old_value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pairs = _resolve_pairs(args.pairs)
    methods = list(dict.fromkeys(args.methods))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_id = args.suite_id or f"{timestamp}_video_compare"
    suite_dir = repo_root / "outputs" / "video_compare" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    overall_rc = 0

    for pair_index, pair_id in enumerate(pairs, start=1):
        for method in methods:
            safe_pair = pair_id.replace("/", "_").replace(" ", "_")
            run_id = f"{suite_id}__{pair_index:02d}__{method}__{safe_pair}"
            cmd = _build_case_command(args, method, pair_id, run_id)
            stdout_path = suite_dir / f"{pair_index:02d}_{method}_{safe_pair}.stdout.txt"
            stderr_path = suite_dir / f"{pair_index:02d}_{method}_{safe_pair}.stderr.txt"

            result: Dict[str, object] = {
                "suite_id": suite_id,
                "pair_id": pair_id,
                "method": method,
                "run_id": run_id,
                "command": cmd,
                "status": "dry_run" if args.dry_run else "planned",
                "returncode": None,
                "stdout_path": str(stdout_path.relative_to(repo_root)),
                "stderr_path": str(stderr_path.relative_to(repo_root)),
            }

            if args.dry_run:
                results.append(result)
                continue

            completed = subprocess.run(
                cmd,
                cwd=str(repo_root),
                text=True,
                capture_output=True,
            )
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")

            result["returncode"] = int(completed.returncode)
            result["status"] = "passed" if completed.returncode == 0 else "failed"
            result.update(_extract_result_row(repo_root, suite_id, method, pair_id, run_id))
            results.append(result)

            if completed.returncode != 0:
                overall_rc = completed.returncode
                if not args.continue_on_error:
                    break
        if overall_rc != 0 and not args.continue_on_error:
            break

    summary_json = suite_dir / "summary.json"
    summary_csv = suite_dir / "summary.csv"
    pair_compare_csv = suite_dir / "pair_compare.csv"

    summary_json.write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "pairs": pairs,
                "methods": methods,
                "video_mode": args.video_mode,
                "reuse_mode": args.reuse_mode,
                "max_frames": args.max_frames,
                "method_b_preset": {
                    "max_keypoints": args.max_keypoints,
                    "resize_long_edge": args.resize_long_edge,
                    "depth_confidence": args.depth_confidence,
                    "width_confidence": args.width_confidence,
                    "filter_threshold": args.filter_threshold,
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_fieldnames = [
        "suite_id",
        "pair_id",
        "method",
        "run_id",
        "status",
        "returncode",
        "dataset",
        "input_type",
        "fps",
        "fps_source",
        "video_mode",
        "reuse_mode",
        "geometry_mode",
        "jitter_meaningful",
        "total_frames",
        "processed_frames",
        "success_frames",
        "fallback_frames",
        "mean_inliers",
        "mean_inlier_ratio",
        "avg_runtime_ms",
        "approx_fps",
        "mean_reprojection_error",
        "mean_inlier_spatial_coverage",
        "mean_overlap_diff_after",
        "mean_seam_band_illuminance_diff",
        "mean_seam_band_gradient_disagreement",
        "mean_seam_band_flicker",
        "mean_stitched_delta",
        "mean_jitter_raw",
        "mean_jitter_sm",
        "p95_jitter_raw",
        "p95_jitter_sm",
        "seam_keyframe_count",
        "seam_runtime_ms_mean",
        "crop_keyframe_count",
        "crop_black_border_ratio_low_mean",
        "reinit_count",
        "init_ms_mean",
        "per_frame_ms_mean",
        "reuse_per_frame_ms_mean",
        "avg_feature_runtime_ms_left",
        "avg_feature_runtime_ms_right",
        "avg_matching_runtime_ms",
        "avg_geometry_runtime_ms",
        "feature_backend_effective",
        "matcher_backend_effective",
        "geometry_backend_effective",
        "warnings_count",
        "errors_count",
        "stitched_exists",
        "transforms_exists",
        "debug_exists",
        "metrics_exists",
        "jitter_exists",
        "run_dir",
        "stdout_path",
        "stderr_path",
    ]
    _write_summary_csv(summary_csv, results, summary_fieldnames)

    pair_compare_rows = _build_pair_compare(results)
    pair_compare_fieldnames = [
        "pair_id",
        "compare",
        "lhs_method",
        "rhs_method",
        "lhs_run_id",
        "rhs_run_id",
        "lhs_processed_frames",
        "rhs_processed_frames",
        "lhs_success_frames",
        "rhs_success_frames",
        "lhs_fallback_frames",
        "rhs_fallback_frames",
        "lhs_mean_inliers",
        "rhs_mean_inliers",
        "delta_mean_inliers",
        "lhs_mean_inlier_ratio",
        "rhs_mean_inlier_ratio",
        "delta_mean_inlier_ratio",
        "lhs_avg_runtime_ms",
        "rhs_avg_runtime_ms",
        "delta_avg_runtime_ms",
        "lhs_approx_fps",
        "rhs_approx_fps",
        "delta_approx_fps",
        "lhs_mean_reprojection_error",
        "rhs_mean_reprojection_error",
        "delta_mean_reprojection_error",
        "lhs_mean_stitched_delta",
        "rhs_mean_stitched_delta",
        "delta_mean_stitched_delta",
        "lhs_warnings_count",
        "rhs_warnings_count",
        "lhs_feature_backend",
        "rhs_feature_backend",
        "lhs_matcher_backend",
        "rhs_matcher_backend",
        "lhs_geometry_backend",
        "rhs_geometry_backend",
        "geometry_mode",
        "jitter_meaningful",
    ]
    _write_summary_csv(pair_compare_csv, pair_compare_rows, pair_compare_fieldnames)

    print(f"suite_id={suite_id}")
    print(f"summary_json={summary_json.relative_to(repo_root)}")
    print(f"summary_csv={summary_csv.relative_to(repo_root)}")
    print(f"pair_compare_csv={pair_compare_csv.relative_to(repo_root)}")
    for row in results:
        print(
            f"{row['status']}: method={row['method']} pair={row['pair_id']} "
            f"run_id={row['run_id']} rc={row['returncode']}"
        )

    return int(overall_rc)


if __name__ == "__main__":
    raise SystemExit(main())
