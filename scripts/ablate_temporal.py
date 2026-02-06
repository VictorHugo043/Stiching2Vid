#!/usr/bin/env python3
"""Run temporal ablation on one pair and summarize jitter/runtime metrics."""

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
    parser = argparse.ArgumentParser(description="Temporal smoothing ablation.")
    parser.add_argument("--pair", required=True, help="Pair id from pairs manifest")
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--keyframe_every", type=int, default=5)
    parser.add_argument("--feature", default="orb")
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min_matches", type=int, default=30)
    parser.add_argument("--ransac_thresh", type=float, default=3.0)
    parser.add_argument("--blend", default="feather", choices=["none", "feather"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=None,
        help="Defaults to keyframe_every to capture keyframe overlays densely.",
    )
    parser.add_argument("--smooth_alpha", type=float, default=0.8)
    return parser


def _run_case(
    repo_root: Path,
    pair_id: str,
    args,
    case_name: str,
    smooth_h: str,
    smooth_alpha: float,
) -> Path:
    out_dir = repo_root / "outputs" / "ablations" / pair_id / "runs" / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = repo_root / "scripts" / "run_baseline_video.py"
    snapshot_every = args.snapshot_every or args.keyframe_every
    cmd = [
        sys.executable,
        str(script_path),
        "--pair",
        pair_id,
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
        "--blend",
        args.blend,
        "--fps",
        str(args.fps),
        "--snapshot_every",
        str(snapshot_every),
        "--smooth_h",
        smooth_h,
        "--smooth_alpha",
        str(smooth_alpha),
        "--out_dir",
        str(out_dir),
        "--run_id",
        case_name,
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    return out_dir


def _read_metrics(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "metrics_preview.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics_preview.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mean_jitter_raw": float(payload.get("mean_jitter_raw", 0.0)),
        "mean_jitter_sm": float(payload.get("mean_jitter_sm", 0.0)),
        "p95_jitter_sm": float(payload.get("p95_jitter_sm", 0.0)),
        "avg_runtime_ms": float(payload.get("avg_runtime_ms", 0.0)),
        "approx_fps": float(payload.get("approx_fps", 0.0)),
    }


def _copy_compare_images(run_dir: Path, mode: str, dst_dir: Path) -> List[str]:
    snapshots_dir = run_dir / "snapshots"
    if mode == "baseline":
        candidates = sorted(snapshots_dir.glob("overlay_raw_*.png"))
    else:
        candidates = sorted(snapshots_dir.glob("overlay_sm_*.png"))

    copied = []
    for src in candidates[:3]:
        dst = dst_dir / f"{mode}_{src.name}"
        shutil.copy2(src, dst)
        copied.append(dst.name)
    return copied


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pair_safe = args.pair.replace("/", "_")
    ablation_dir = repo_root / "outputs" / "ablations" / pair_safe
    compare_dir = ablation_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = _run_case(
        repo_root=repo_root,
        pair_id=args.pair,
        args=args,
        case_name="baseline_none",
        smooth_h="none",
        smooth_alpha=args.smooth_alpha,
    )
    temporal_dir = _run_case(
        repo_root=repo_root,
        pair_id=args.pair,
        args=args,
        case_name="temporal_ema",
        smooth_h="ema",
        smooth_alpha=args.smooth_alpha,
    )

    baseline_metrics = _read_metrics(baseline_dir)
    temporal_metrics = _read_metrics(temporal_dir)

    baseline_copied = _copy_compare_images(baseline_dir, "baseline", compare_dir)
    temporal_copied = _copy_compare_images(temporal_dir, "temporal", compare_dir)

    summary_path = ablation_dir / "summary_temporal.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "mean_jitter_raw",
                "mean_jitter_sm",
                "p95_jitter_sm",
                "avg_runtime_ms",
                "approx_fps",
            ],
        )
        writer.writeheader()
        writer.writerow({"run_id": "baseline_none", **baseline_metrics})
        writer.writerow({"run_id": "temporal_ema", **temporal_metrics})

    notes_path = ablation_dir / "compare" / "copied_images.txt"
    notes_path.write_text(
        "baseline:\n"
        + "\n".join(baseline_copied)
        + "\n\ntemporal:\n"
        + "\n".join(temporal_copied)
        + "\n",
        encoding="utf-8",
    )

    print(f"ablation_dir={ablation_dir}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

