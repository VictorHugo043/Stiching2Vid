#!/usr/bin/env python3
"""Run a multi-pair single-frame smoke suite through run_baseline_frame.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Sequence, Tuple


DEFAULT_CASES: List[Tuple[str, int]] = [
    ("mine_source_indoor2_left_right", 0),
    ("kitti_raw_data_2011_09_28_drive_0119_image_02_image_03", 0),
    ("kitti_raw_data_2011_09_26_drive_0005_image_02_image_03", 0),
    ("dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right", 0),
]

PAIR_ALIASES: Dict[str, str] = {
    "mysourceindoor2": "mine_source_indoor2_left_right",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-pair single-frame smoke suite by invoking "
            "scripts/run_baseline_frame.py once per configured pair."
        )
    )
    parser.add_argument(
        "--method",
        default="method_a",
        choices=["method_a", "method_b"],
        help="Backend preset to use for the suite",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional pair ids or known aliases; default uses the built-in smoke suite pairs",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=None,
        help="Override frame index for all pairs; default uses each case's built-in index",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable used to invoke scripts/run_baseline_frame.py",
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
        "--device",
        default=None,
        help="Optional Method B device override passed through to run_baseline_frame.py",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force Method B backends to use CPU",
    )
    parser.add_argument(
        "--weights_dir",
        default=None,
        help="Optional Method B weights dir passed through to run_baseline_frame.py",
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


def _resolve_pairs(requested_pairs: Sequence[str] | None) -> List[Tuple[str, int]]:
    if not requested_pairs:
        return list(DEFAULT_CASES)
    resolved: List[Tuple[str, int]] = []
    for raw_name in requested_pairs:
        pair_id = PAIR_ALIASES.get(raw_name, raw_name)
        resolved.append((pair_id, 0))
    return resolved


def _build_case_command(args: argparse.Namespace, pair_id: str, frame_index: int, run_id: str) -> List[str]:
    cmd = [
        str(args.python_bin),
        "scripts/run_baseline_frame.py",
        "--pair",
        pair_id,
        "--frame_index",
        str(frame_index),
        "--manifest",
        str(args.manifest),
        "--run_id",
        run_id,
    ]
    if args.method == "method_b":
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
        if args.force_cpu:
            cmd.append("--force_cpu")
        if args.weights_dir:
            cmd.extend(["--weights_dir", str(args.weights_dir)])
        if args.max_keypoints is not None:
            cmd.extend(["--max_keypoints", str(args.max_keypoints)])
        if args.resize_long_edge is not None:
            cmd.extend(["--resize_long_edge", str(args.resize_long_edge)])
    return cmd


def _load_debug(run_dir: Path) -> Dict:
    debug_path = run_dir / "debug.json"
    if not debug_path.exists():
        return {}
    try:
        return json.loads(debug_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cases = _resolve_pairs(args.pairs)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_id = args.suite_id or f"{timestamp}_{args.method}_frame_smoke"
    suite_dir = repo_root / "outputs" / "frame_smoke" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    overall_rc = 0

    for index, (pair_id, default_frame_index) in enumerate(cases, start=1):
        frame_index = int(args.frame_index) if args.frame_index is not None else int(default_frame_index)
        safe_pair = pair_id.replace("/", "_").replace(" ", "_")
        run_id = f"{suite_id}__{index:02d}__{safe_pair}"
        cmd = _build_case_command(args, pair_id, frame_index, run_id)
        stdout_path = suite_dir / f"{index:02d}_{safe_pair}.stdout.txt"
        stderr_path = suite_dir / f"{index:02d}_{safe_pair}.stderr.txt"

        result: Dict = {
            "pair_id": pair_id,
            "frame_index": frame_index,
            "method": args.method,
            "run_id": run_id,
            "command": cmd,
            "status": "dry_run" if args.dry_run else "planned",
            "returncode": None,
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
        result["stdout_path"] = str(stdout_path.relative_to(repo_root))
        result["stderr_path"] = str(stderr_path.relative_to(repo_root))

        debug = _load_debug(repo_root / "outputs" / "runs" / run_id)
        if debug:
            result["feature_backend_effective"] = debug.get("feature_backend_effective")
            result["matcher_backend_effective"] = debug.get("matcher_backend_effective")
            result["geometry_backend"] = debug.get("geometry_backend")
            result["n_matches_good"] = debug.get("n_matches_good")
            result["n_inliers"] = debug.get("n_inliers")
            result["inlier_ratio"] = debug.get("inlier_ratio")
            result["reprojection_error"] = debug.get("reprojection_error")

        results.append(result)

        if completed.returncode != 0:
            overall_rc = completed.returncode
            if not args.continue_on_error:
                break

    summary_json = suite_dir / "summary.json"
    summary_csv = suite_dir / "summary.csv"
    summary_json.write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "method": args.method,
                "python_bin": str(args.python_bin),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "pair_id",
        "frame_index",
        "method",
        "run_id",
        "status",
        "returncode",
        "feature_backend_effective",
        "matcher_backend_effective",
        "geometry_backend",
        "n_matches_good",
        "n_inliers",
        "inlier_ratio",
        "reprojection_error",
        "stdout_path",
        "stderr_path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"suite_id={suite_id}")
    print(f"summary_json={summary_json.relative_to(repo_root)}")
    print(f"summary_csv={summary_csv.relative_to(repo_root)}")
    for row in results:
        print(
            f"{row['status']}: pair={row['pair_id']} frame={row['frame_index']} "
            f"run_id={row['run_id']} rc={row['returncode']}"
        )

    return int(overall_rc)


if __name__ == "__main__":
    raise SystemExit(main())
