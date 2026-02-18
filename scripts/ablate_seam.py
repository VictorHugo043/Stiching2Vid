#!/usr/bin/env python3
"""Run seam ablation cases and summarize seam quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seam ablation for one pair.")
    parser.add_argument("--pair", required=True, help="Pair id from pairs manifest")
    parser.add_argument("--manifest", default="data/manifests/pairs.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--keyframe_every", type=int, default=5)
    parser.add_argument("--feature", default="orb")
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min_matches", type=int, default=30)
    parser.add_argument("--ransac_thresh", type=float, default=3.0)
    parser.add_argument("--smooth_h", default="ema", choices=["none", "ema", "window"])
    parser.add_argument("--smooth_alpha", type=float, default=0.8)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--seam_megapix", type=float, default=0.1)
    parser.add_argument("--seam_dilate", type=int, default=1)
    parser.add_argument("--mb_levels", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--snapshot_every", type=int, default=10)
    return parser


def _run_case(repo_root: Path, args, case_name: str, seam: str, blend: str) -> Path:
    out_dir = repo_root / "outputs" / "ablations" / args.pair / "seam" / "runs" / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = repo_root / "scripts" / "run_baseline_video.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--pair",
        args.pair,
        "--manifest",
        args.manifest,
        "--start",
        str(args.start),
        "--max_frames",
        str(args.max_frames),
        "--stride",
        str(args.stride),
        "--keyframe_every",
        str(args.keyframe_every),
        "--feature",
        args.feature,
        "--nfeatures",
        str(args.nfeatures),
        "--ratio",
        str(args.ratio),
        "--min_matches",
        str(args.min_matches),
        "--ransac_thresh",
        str(args.ransac_thresh),
        "--smooth_h",
        args.smooth_h,
        "--smooth_alpha",
        str(args.smooth_alpha),
        "--smooth_window",
        str(args.smooth_window),
        "--seam",
        seam,
        "--seam_megapix",
        str(args.seam_megapix),
        "--seam_dilate",
        str(args.seam_dilate),
        "--blend",
        blend,
        "--mb_levels",
        str(args.mb_levels),
        "--fps",
        str(args.fps),
        "--snapshot_every",
        str(args.snapshot_every),
        "--out_dir",
        str(out_dir),
        "--run_id",
        case_name,
    ]

    subprocess.run(cmd, check=True, cwd=str(repo_root))
    return out_dir


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_case(run_dir: Path) -> Dict[str, float]:
    debug = _read_json(run_dir / "debug.json")
    seam_stats = debug.get("seam_keyframe_stats", []) or []

    def _mean_from_key(key: str) -> float:
        vals = [float(item.get(key, 0.0)) for item in seam_stats]
        return float(sum(vals) / len(vals)) if vals else 0.0

    metrics = _read_json(run_dir / "metrics_preview.json")
    summary = {
        "overlap_area_px": _mean_from_key("overlap_area_px"),
        "seam_mask_nonzero_ratio_left": _mean_from_key("seam_mask_nonzero_ratio_left"),
        "seam_mask_nonzero_ratio_right": _mean_from_key("seam_mask_nonzero_ratio_right"),
        "overlap_diff_mean_before": _mean_from_key("overlap_diff_mean_before"),
        "overlap_diff_mean_after": _mean_from_key("overlap_diff_mean_after"),
        "runtime_ms_seam_keyframe": _mean_from_key("runtime_ms_seam_keyframe"),
        "avg_runtime_ms": float(metrics.get("avg_runtime_ms", 0.0)),
        "approx_fps": float(metrics.get("approx_fps", 0.0)),
    }
    return summary


def _copy_compare_artifacts(run_dir: Path, case_name: str, compare_dir: Path, frame_ids: List[int], start: int):
    snapshots = run_dir / "snapshots"
    for frame_id in frame_ids:
        source_idx = start + frame_id
        stitched_src = snapshots / f"frame_{source_idx:06d}_stitched.png"
        if stitched_src.exists():
            shutil.copy2(stitched_src, compare_dir / f"{case_name}_stitched_{source_idx:06d}.png")

        seam_src = snapshots / f"seam_overlay_{source_idx:06d}.png"
        if seam_src.exists():
            shutil.copy2(seam_src, compare_dir / f"{case_name}_seam_overlay_{source_idx:06d}.png")

        overlap_src = snapshots / f"overlap_diff_{source_idx:06d}.png"
        if overlap_src.exists():
            shutil.copy2(overlap_src, compare_dir / f"{case_name}_overlap_diff_{source_idx:06d}.png")


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    ablation_dir = repo_root / "outputs" / "ablations" / args.pair / "seam"
    compare_dir = ablation_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("A_no_seam_feather", "none", "feather", "A"),
        ("B_seam_dp_color_hard", "opencv_dp_color", "none", "B"),
        ("C_seam_dp_color_feather", "opencv_dp_color", "feather", "C"),
        ("D_seam_dp_color_multiband", "opencv_dp_color", "multiband", "D"),
    ]

    rows: List[Dict[str, object]] = []
    for case_name, seam, blend, tag in cases:
        row: Dict[str, object] = {
            "case": tag,
            "run_id": case_name,
            "seam": seam,
            "blend": blend,
            "status": "OK",
            "note": "",
        }
        try:
            run_dir = _run_case(repo_root, args, case_name, seam, blend)
            row.update(_summarize_case(run_dir))
            _copy_compare_artifacts(
                run_dir,
                case_name,
                compare_dir,
                frame_ids=[0, 20, 50],
                start=args.start,
            )
        except subprocess.CalledProcessError as exc:
            row["status"] = "FAILED"
            row["note"] = f"subprocess_failed: {exc.returncode}"
        except Exception as exc:
            row["status"] = "FAILED"
            row["note"] = str(exc)
        rows.append(row)

    summary_path = ablation_dir / "summary_seam.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "run_id",
                "seam",
                "blend",
                "status",
                "overlap_area_px",
                "seam_mask_nonzero_ratio_left",
                "seam_mask_nonzero_ratio_right",
                "overlap_diff_mean_before",
                "overlap_diff_mean_after",
                "runtime_ms_seam_keyframe",
                "avg_runtime_ms",
                "approx_fps",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"ablation_dir={ablation_dir}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
