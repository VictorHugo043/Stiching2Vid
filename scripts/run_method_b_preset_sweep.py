#!/usr/bin/env python3
"""Run a small Method B preset sweep without modifying the formal baseline."""

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
    "kitti_raw_data_2011_09_26_drive_0002_image_02_image_03",
    "dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right",
    "mine_source_walking_left_right",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a controlled Method B preset sweep on representative pairs."
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional pair override")
    parser.add_argument("--presets", nargs="*", default=None, help="Optional preset subset")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable")
    parser.add_argument("--manifest", default="data/manifests/pairs.yaml", help="Path to pairs manifest")
    parser.add_argument("--max_frames", type=int, default=120, help="Frames per run")
    parser.add_argument("--snapshot_every", type=int, default=1000, help="Snapshot interval")
    parser.add_argument("--suite_id", default=None, help="Optional suite id")
    parser.add_argument("--weights_dir", default=None, help="Optional Method B weights dir")
    parser.add_argument("--device", default=None, help="Optional device override")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU backend")
    return parser


def _default_fps_for_pair(pair_id: str) -> float | None:
    pair_key = str(pair_id).lower()
    if pair_key.startswith("kitti_") or pair_key.startswith("dynamicstereo_"):
        return 10.0
    if pair_key.startswith("mine_source_"):
        return 30.0
    return None


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from stitching.method_b_presets import get_method_b_preset, list_method_b_presets

    pairs = list(args.pairs or DEFAULT_PAIRS)
    presets = (
        [get_method_b_preset(name) for name in args.presets]
        if args.presets
        else list_method_b_presets()
    )
    suite_id = args.suite_id or f"methodb_preset_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    suite_dir = repo_root / "outputs" / "analysis" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for pair_id in pairs:
        fps_value = _default_fps_for_pair(pair_id)
        for preset in presets:
            run_id = f"{suite_id}__{preset.name}__{pair_id}"
            cmd = [
                str(args.python_bin),
                "scripts/run_baseline_video.py",
                "--pair",
                pair_id,
                "--manifest",
                str(args.manifest),
                "--geometry_mode",
                "fixed_geometry",
                "--reuse_mode",
                "frame0_all",
                "--feature_backend",
                "superpoint",
                "--matcher_backend",
                "lightglue",
                "--geometry_backend",
                "opencv_usac_magsac",
                "--max_frames",
                str(args.max_frames),
                "--snapshot_every",
                str(args.snapshot_every),
                "--max_keypoints",
                str(preset.max_keypoints),
                "--run_id",
                run_id,
            ]
            if preset.resize_long_edge is not None:
                cmd.extend(["--resize_long_edge", str(preset.resize_long_edge)])
            if preset.depth_confidence is not None:
                cmd.extend(["--depth_confidence", str(preset.depth_confidence)])
            if preset.width_confidence is not None:
                cmd.extend(["--width_confidence", str(preset.width_confidence)])
            if preset.filter_threshold is not None:
                cmd.extend(["--filter_threshold", str(preset.filter_threshold)])
            if args.device:
                cmd.extend(["--device", str(args.device)])
            if args.weights_dir:
                cmd.extend(["--weights_dir", str(args.weights_dir)])
            if args.force_cpu:
                cmd.append("--force_cpu")
            if fps_value is not None:
                cmd.extend(["--fps", str(fps_value)])

            result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
            metrics_path = repo_root / "outputs" / "runs" / run_id / "metrics_preview.json"
            metrics = {}
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            rows.append(
                {
                    "pair_id": pair_id,
                    "preset": preset.name,
                    "description": preset.description,
                    "returncode": int(result.returncode),
                    "processed_frames": int(metrics.get("processed_frames", 0)),
                    "mean_inliers": float(metrics.get("mean_inliers", 0.0)),
                    "mean_inlier_ratio": float(metrics.get("mean_inlier_ratio", 0.0)),
                    "mean_reprojection_error": float(metrics.get("mean_reprojection_error", 0.0)),
                    "mean_inlier_spatial_coverage": float(
                        metrics.get("mean_inlier_spatial_coverage", 0.0)
                    ),
                    "avg_runtime_ms": float(metrics.get("avg_runtime_ms", 0.0)),
                    "approx_fps": float(metrics.get("approx_fps", 0.0)),
                    "init_ms_mean": float(metrics.get("init_ms_mean", 0.0)),
                    "per_frame_ms_mean": float(metrics.get("per_frame_ms_mean", 0.0)),
                    "avg_feature_runtime_ms_left": float(
                        metrics.get("avg_feature_runtime_ms_left", 0.0)
                    ),
                    "avg_feature_runtime_ms_right": float(
                        metrics.get("avg_feature_runtime_ms_right", 0.0)
                    ),
                    "avg_matching_runtime_ms": float(
                        metrics.get("avg_matching_runtime_ms", 0.0)
                    ),
                    "avg_geometry_runtime_ms": float(
                        metrics.get("avg_geometry_runtime_ms", 0.0)
                    ),
                    "mean_overlap_diff_after": float(metrics.get("mean_overlap_diff_after", 0.0)),
                    "mean_seam_band_illuminance_diff": float(
                        metrics.get("mean_seam_band_illuminance_diff", 0.0)
                    ),
                    "mean_seam_band_gradient_disagreement": float(
                        metrics.get("mean_seam_band_gradient_disagreement", 0.0)
                    ),
                    "mean_seam_band_flicker": float(
                        metrics.get("mean_seam_band_flicker", 0.0)
                    ),
                    "mean_stitched_delta": float(metrics.get("mean_stitched_delta", 0.0)),
                    "run_id": run_id,
                }
            )

    summary_path = suite_dir / "summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    preset_rows: List[Dict[str, object]] = []
    for preset in presets:
        preset_items = [row for row in rows if row["preset"] == preset.name and int(row["returncode"]) == 0]
        preset_rows.append(
            {
                "preset": preset.name,
                "description": preset.description,
                "runs": int(len(preset_items)),
                "mean_inliers": _mean([float(row["mean_inliers"]) for row in preset_items]),
                "mean_inlier_ratio": _mean([float(row["mean_inlier_ratio"]) for row in preset_items]),
                "mean_reprojection_error": _mean(
                    [float(row["mean_reprojection_error"]) for row in preset_items]
                ),
                "mean_inlier_spatial_coverage": _mean(
                    [float(row["mean_inlier_spatial_coverage"]) for row in preset_items]
                ),
                "avg_runtime_ms": _mean([float(row["avg_runtime_ms"]) for row in preset_items]),
                "approx_fps": _mean([float(row["approx_fps"]) for row in preset_items]),
                "init_ms_mean": _mean([float(row["init_ms_mean"]) for row in preset_items]),
                "per_frame_ms_mean": _mean([float(row["per_frame_ms_mean"]) for row in preset_items]),
                "mean_overlap_diff_after": _mean(
                    [float(row["mean_overlap_diff_after"]) for row in preset_items]
                ),
                "mean_seam_band_illuminance_diff": _mean(
                    [float(row["mean_seam_band_illuminance_diff"]) for row in preset_items]
                ),
                "mean_seam_band_gradient_disagreement": _mean(
                    [float(row["mean_seam_band_gradient_disagreement"]) for row in preset_items]
                ),
                "mean_seam_band_flicker": _mean(
                    [float(row["mean_seam_band_flicker"]) for row in preset_items]
                ),
                "mean_stitched_delta": _mean(
                    [float(row["mean_stitched_delta"]) for row in preset_items]
                ),
            }
        )

    preset_summary_path = suite_dir / "preset_summary.csv"
    if preset_rows:
        with preset_summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(preset_rows[0].keys()))
            writer.writeheader()
            writer.writerows(preset_rows)

    (suite_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "preset_summary": preset_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote sweep outputs to {suite_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
