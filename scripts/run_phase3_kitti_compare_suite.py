#!/usr/bin/env python3
"""Run the formal Phase 3 KITTI color stereo experiment suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Sequence


DEFAULT_PAIRS: List[str] = [
    "kitti_raw_data_2011_09_26_drive_0001_image_02_image_03",
    "kitti_raw_data_2011_09_26_drive_0002_image_02_image_03",
    "kitti_raw_data_2011_09_26_drive_0005_image_02_image_03",
    "kitti_raw_data_2011_09_26_drive_0019_image_02_image_03",
    "kitti_raw_data_2011_09_28_drive_0016_image_02_image_03",
    "kitti_raw_data_2011_09_28_drive_0021_image_02_image_03",
]

PAIR_ALIASES: Dict[str, str] = {
    "kitti0001": "kitti_raw_data_2011_09_26_drive_0001_image_02_image_03",
    "kitti0002": "kitti_raw_data_2011_09_26_drive_0002_image_02_image_03",
    "kitti0005": "kitti_raw_data_2011_09_26_drive_0005_image_02_image_03",
    "kitti0019": "kitti_raw_data_2011_09_26_drive_0019_image_02_image_03",
    "kitti0016": "kitti_raw_data_2011_09_28_drive_0016_image_02_image_03",
    "kitti0021": "kitti_raw_data_2011_09_28_drive_0021_image_02_image_03",
}

DEFAULT_DYNAMIC_PRESETS: List[str] = [
    "baseline_fixed",
    "keyframe_seam10",
    "trigger_fused_d18_fg008",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal Phase 3 KITTI color stereo suite by reusing the "
            "Phase 1 method-compare driver and the Phase 2 dynamic-seam compare driver."
        )
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional KITTI pair ids or aliases; default uses the formal Phase 3 KITTI color stereo set",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable used to invoke child suite scripts",
    )
    parser.add_argument(
        "--suite_id",
        default=None,
        help="Optional parent suite id; default is timestamped",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=6000,
        help="Maximum frames per run; default traverses full KITTI clips",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="FPS override passed through to child suites; KITTI raw color stereo defaults to 10",
    )
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=200,
        help="Snapshot interval for method compare runs",
    )
    parser.add_argument(
        "--dynamic_snapshot_every",
        type=int,
        default=100,
        help="Snapshot interval for dynamic compare runs",
    )
    parser.add_argument(
        "--seam_snapshot_on_recompute",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether dynamic compare runs save seam-event snapshots",
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
        help="Optional Method B weights dir",
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
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["method_a_orb", "method_a_sift", "method_b"],
        choices=["method_a_orb", "method_a_sift", "method_b"],
        help="Methods included in the Phase 3 method compare sub-suite",
    )
    parser.add_argument(
        "--dynamic_presets",
        nargs="*",
        default=list(DEFAULT_DYNAMIC_PRESETS),
        choices=[
            "baseline_fixed",
            "keyframe_seam10",
            "trigger_fused_d18_fg008",
            "adaptive_trigger_fused_d18_fg008",
        ],
        help="Dynamic seam presets included in the Phase 3 dynamic sub-suite",
    )
    parser.add_argument(
        "--skip_method_compare",
        action="store_true",
        help="Skip the method compare sub-suite",
    )
    parser.add_argument(
        "--skip_dynamic_compare",
        action="store_true",
        help="Skip the dynamic seam compare sub-suite",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue after child-suite failures when possible",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print child-suite commands without executing them",
    )
    return parser


def _resolve_pairs(requested_pairs: Sequence[str] | None) -> List[str]:
    if not requested_pairs:
        return list(DEFAULT_PAIRS)
    return [PAIR_ALIASES.get(raw_name, raw_name) for raw_name in requested_pairs]


def _build_method_compare_cmd(args: argparse.Namespace, suite_id: str, pairs: Sequence[str]) -> List[str]:
    cmd = [
        str(args.python_bin),
        "scripts/run_video_compare_suite.py",
        "--manifest",
        str(args.manifest),
        "--suite_id",
        suite_id,
        "--video_mode",
        "1",
        "--reuse_mode",
        "frame0_all",
        "--max_frames",
        str(args.max_frames),
        "--start",
        str(args.start),
        "--stride",
        str(args.stride),
        "--snapshot_every",
        str(args.snapshot_every),
        "--fps",
        str(args.fps),
        "--methods",
        *list(args.methods),
        "--pairs",
        *list(pairs),
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
    if args.continue_on_error:
        cmd.append("--continue_on_error")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def _build_dynamic_compare_cmd(args: argparse.Namespace, suite_id: str, pairs: Sequence[str]) -> List[str]:
    cmd = [
        str(args.python_bin),
        "scripts/run_phase2_dynamic_compare_suite.py",
        "--manifest",
        str(args.manifest),
        "--suite_id",
        suite_id,
        "--max_frames",
        str(args.max_frames),
        "--start",
        str(args.start),
        "--stride",
        str(args.stride),
        "--snapshot_every",
        str(args.dynamic_snapshot_every),
        "--seam_snapshot_on_recompute",
        str(args.seam_snapshot_on_recompute),
        "--fps",
        str(args.fps),
        "--presets",
        *list(args.dynamic_presets),
        "--pairs",
        *list(pairs),
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
    if args.continue_on_error:
        cmd.append("--continue_on_error")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def _run_child(repo_root: Path, suite_dir: Path, name: str, cmd: Sequence[str]) -> int:
    stdout_path = suite_dir / f"{name}.stdout.txt"
    stderr_path = suite_dir / f"{name}.stderr.txt"
    completed = subprocess.run(
        list(cmd),
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pairs = _resolve_pairs(args.pairs)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_id = args.suite_id or f"{timestamp}_phase3_kitti"
    phase3_dir = repo_root / "outputs" / "phase3" / suite_id
    phase3_dir.mkdir(parents=True, exist_ok=True)

    method_suite_id = f"{suite_id}__methods"
    dynamic_suite_id = f"{suite_id}__dynamic"

    manifest = {
        "suite_id": suite_id,
        "pairs": pairs,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "methods": list(args.methods),
        "dynamic_presets": list(args.dynamic_presets),
        "method_suite_id": None if args.skip_method_compare else method_suite_id,
        "dynamic_suite_id": None if args.skip_dynamic_compare else dynamic_suite_id,
    }
    (phase3_dir / "suite_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overall_rc = 0

    if not args.skip_method_compare:
        method_cmd = _build_method_compare_cmd(args, method_suite_id, pairs)
        if args.dry_run:
            (phase3_dir / "method_compare.command.txt").write_text(
                " ".join(method_cmd),
                encoding="utf-8",
            )
        else:
            rc = _run_child(repo_root, phase3_dir, "method_compare", method_cmd)
            overall_rc = rc or overall_rc
            if rc != 0 and not args.continue_on_error:
                return rc

    if not args.skip_dynamic_compare:
        dynamic_cmd = _build_dynamic_compare_cmd(args, dynamic_suite_id, pairs)
        if args.dry_run:
            (phase3_dir / "dynamic_compare.command.txt").write_text(
                " ".join(dynamic_cmd),
                encoding="utf-8",
            )
        else:
            rc = _run_child(repo_root, phase3_dir, "dynamic_compare", dynamic_cmd)
            overall_rc = rc or overall_rc
            if rc != 0 and not args.continue_on_error:
                return rc

            visual_cmd = [
                str(args.python_bin),
                "scripts/build_phase2_visual_summary.py",
                "--suite_id",
                dynamic_suite_id,
            ]
            rc = _run_child(repo_root, phase3_dir, "dynamic_visuals", visual_cmd)
            overall_rc = rc or overall_rc
            if rc != 0 and not args.continue_on_error:
                return rc

    build_cmd = [
        str(args.python_bin),
        "scripts/build_phase3_kitti_summary.py",
        "--suite_id",
        suite_id,
        "--pairs",
        *pairs,
        "--fps",
        str(args.fps),
        "--max_frames",
        str(args.max_frames),
    ]
    if not args.skip_method_compare:
        build_cmd.extend(["--method_suite_id", method_suite_id])
    if not args.skip_dynamic_compare:
        build_cmd.extend(["--dynamic_suite_id", dynamic_suite_id])

    if args.dry_run:
        (phase3_dir / "build_summary.command.txt").write_text(" ".join(build_cmd), encoding="utf-8")
        print(f"Phase 3 dry-run manifest written: {phase3_dir}")
        return 0

    rc = _run_child(repo_root, phase3_dir, "build_summary", build_cmd)
    overall_rc = rc or overall_rc
    if rc != 0 and not args.continue_on_error:
        return rc

    print(f"Phase 3 KITTI suite completed: outputs/phase3/{suite_id}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
