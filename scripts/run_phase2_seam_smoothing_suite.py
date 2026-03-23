#!/usr/bin/env python3
"""Run seam temporal smoothing evaluation on the current recommended preset."""

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

SMOOTH_PRESETS: List[Dict[str, object]] = [
    {"preset_id": "smooth_none", "seam_smooth": "none", "seam_smooth_alpha": 0.8, "seam_smooth_window": 5},
    {"preset_id": "smooth_ema_a080", "seam_smooth": "ema", "seam_smooth_alpha": 0.8, "seam_smooth_window": 5},
    {"preset_id": "smooth_window_5", "seam_smooth": "window", "seam_smooth_alpha": 0.8, "seam_smooth_window": 5},
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate seam temporal smoothing on the current recommended "
            "Phase 2 preset by invoking scripts/run_baseline_video.py."
        )
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional pair ids")
    parser.add_argument("--manifest", default="data/manifests/pairs.yaml", help="Path to pairs manifest")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable used to invoke run_baseline_video.py")
    parser.add_argument("--suite_id", default=None, help="Optional suite id; default is timestamped")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue after a failed case")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--max_frames", type=int, default=6000, help="Maximum number of frames; use a large value to traverse full videos")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--snapshot_every", type=int, default=1000, help="Snapshot interval")
    parser.add_argument("--seam_snapshot_on_recompute", type=int, default=0, choices=[0, 1], help="Whether to save seam event snapshots")
    parser.add_argument("--device", default=None, help="Optional Method B device override")
    parser.add_argument("--force_cpu", action="store_true", help="Force Method B backends to use CPU")
    parser.add_argument("--weights_dir", default=None, help="Optional Method B weights dir")
    parser.add_argument("--max_keypoints", type=int, default=None, help="Optional Method B max_keypoints override")
    parser.add_argument("--resize_long_edge", type=int, default=None, help="Optional Method B resize_long_edge override")
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
        "fixed_geometry",
        "--reuse_mode",
        "frame0_all",
        "--feature_backend",
        "superpoint",
        "--matcher_backend",
        "lightglue",
        "--geometry_backend",
        "opencv_usac_magsac",
        "--seam_policy",
        "trigger",
        "--seam_trigger_diff_threshold",
        "18",
        "--foreground_mode",
        "disagreement",
        "--seam_trigger_foreground_ratio",
        "0.08",
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
    return {
        "suite_id": suite_id,
        "pair_id": pair_id,
        "preset_id": str(preset["preset_id"]),
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(repo_root)) if run_dir.exists() else str(run_dir),
        "processed_frames": metrics.get("processed_frames"),
        "success_frames": metrics.get("success_frames"),
        "fallback_frames": metrics.get("fallback_frames"),
        "seam_recompute_count": metrics.get("seam_recompute_count"),
        "mean_overlap_diff_after": metrics.get("mean_overlap_diff_after"),
        "mean_seam_mask_change_ratio": metrics.get("mean_seam_mask_change_ratio"),
        "mean_stitched_delta": metrics.get("mean_stitched_delta"),
        "mean_foreground_ratio": metrics.get("mean_foreground_ratio"),
        "approx_fps": metrics.get("approx_fps"),
        "seam_smooth": metrics.get("seam_smooth"),
        "seam_smooth_alpha": metrics.get("seam_smooth_alpha"),
        "seam_smooth_window": metrics.get("seam_smooth_window"),
        "warnings_count": len(debug.get("warnings", []) if isinstance(debug.get("warnings"), list) else []),
        "errors_count": len(debug.get("errors", []) if isinstance(debug.get("errors"), list) else []),
    }


def _aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["preset_id"]), []).append(row)

    numeric_fields = [
        "processed_frames",
        "success_frames",
        "fallback_frames",
        "seam_recompute_count",
        "mean_overlap_diff_after",
        "mean_seam_mask_change_ratio",
        "mean_stitched_delta",
        "mean_foreground_ratio",
        "approx_fps",
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
            "seam_smooth": base["seam_smooth"],
            "seam_smooth_alpha": base["seam_smooth_alpha"],
            "seam_smooth_window": base["seam_smooth_window"],
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in preset_rows if row.get(field) is not None]
            summary[field] = float(sum(values) / len(values)) if values else 0.0
        out.append(summary)
    return out


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
    pairs = list(args.pairs) if args.pairs else list(DEFAULT_PAIRS)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_id = args.suite_id or f"{timestamp}_phase2_seam_smoothing"
    suite_dir = repo_root / "outputs" / "video_smoothing" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    overall_rc = 0

    for pair_index, pair_id in enumerate(pairs, start=1):
        for preset_index, preset in enumerate(SMOOTH_PRESETS, start=1):
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

    aggregated = _aggregate_rows([row for row in results if row.get("status") == "passed"])
    _write_csv(suite_dir / "summary.csv", results)
    _write_csv(suite_dir / "smooth_summary.csv", aggregated)
    (suite_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (suite_dir / "smooth_summary.json").write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Seam smoothing suite completed: {suite_dir}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
