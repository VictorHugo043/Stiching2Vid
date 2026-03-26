#!/usr/bin/env python3
"""Run the formal full-length Method A vs Method B compare across all datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Sequence


KITTI_PAIRS: List[str] = [
    "kitti_raw_data_2011_09_26_drive_0001_image_02_image_03",
    "kitti_raw_data_2011_09_26_drive_0002_image_02_image_03",
    "kitti_raw_data_2011_09_26_drive_0005_image_02_image_03",
    "kitti_raw_data_2011_09_26_drive_0019_image_02_image_03",
    "kitti_raw_data_2011_09_28_drive_0016_image_02_image_03",
    "kitti_raw_data_2011_09_28_drive_0021_image_02_image_03",
]

DYNAMICSTEREO_PAIRS: List[str] = [
    "dynamicstereo_real_000_ignacio_waving_test_frames_rect_left_right",
    "dynamicstereo_real_000_nikita_reading_test_frames_rect_left_right",
    "dynamicstereo_real_000_teddy_static_test_frames_rect_left_right",
]

MINESOURCE_PAIRS: List[str] = [
    "mine_source_bow1_left_right",
    "mine_source_bow2_left_right",
    "mine_source_lake_left_right",
    "mine_source_robot_left_right",
    "mine_source_church_left_right",
    "mine_source_park1_left_right",
    "mine_source_pujiang1_left_right",
    "mine_source_pujiang2_left_right",
    "mine_source_pujiang3_left_right",
    "mine_source_indoor_left_right",
    "mine_source_indoor2_left_right",
    "mine_source_mcd1_left_right",
    "mine_source_mcd2_left_right",
    "mine_source_square_left_right",
    "mine_source_traffic1_left_right",
    "mine_source_traffic2_left_right",
    "mine_source_walking_left_right",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the formal full-length Method A vs Method B compare with richer metrics."
    )
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable for child scripts")
    parser.add_argument("--suite_id", default=None, help="Parent suite id under outputs/phase3/")
    parser.add_argument("--manifest", default="data/manifests/pairs.yaml", help="Path to pairs manifest")
    parser.add_argument("--max_frames", type=int, default=6000, help="Max frames; 6000 means effectively full clips")
    parser.add_argument("--snapshot_every", type=int, default=1000, help="Snapshot interval")
    parser.add_argument("--device", default=None, help="Optional Method B device override")
    parser.add_argument("--force_cpu", action="store_true", help="Force Method B to CPU")
    parser.add_argument("--weights_dir", default=None, help="Optional Method B weights dir")
    parser.add_argument(
        "--suite_tag",
        default=None,
        help="Optional suffix tag appended to dataset-level and overall suite ids, e.g. mps_accuracy_v1",
    )
    parser.add_argument("--continue_on_error", action="store_true", help="Continue after child-suite failures")
    parser.add_argument("--skip_figures", action="store_true", help="Skip figure export step")
    return parser


def _run(repo_root: Path, suite_dir: Path, name: str, cmd: Sequence[str]) -> int:
    stdout_path = suite_dir / f"{name}.stdout.txt"
    stderr_path = suite_dir / f"{name}.stderr.txt"
    completed = subprocess.run(list(cmd), cwd=str(repo_root), text=True, capture_output=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def _build_compare_cmd(
    args: argparse.Namespace,
    suite_id: str,
    pairs: Sequence[str],
    fps_value: float,
) -> List[str]:
    cmd = [
        str(args.python_bin),
        "scripts/eval_method_compare_matrix.py",
        "--python_bin",
        str(args.python_bin),
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
        "--snapshot_every",
        str(args.snapshot_every),
        "--fps",
        str(fps_value),
        "--pairs",
        *list(pairs),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.force_cpu:
        cmd.append("--force_cpu")
    if args.weights_dir:
        cmd.extend(["--weights_dir", str(args.weights_dir)])
    if args.continue_on_error:
        cmd.append("--continue_on_error")
    return cmd


def _build_summary_cmd(
    args: argparse.Namespace,
    suite_id: str,
    method_suite_id: str,
    pairs: Sequence[str],
    fps_value: float,
) -> List[str]:
    return [
        str(args.python_bin),
        "scripts/internal/summarize_method_compare_dataset.py",
        "--suite_id",
        suite_id,
        "--method_suite_id",
        method_suite_id,
        "--pairs",
        *list(pairs),
        "--fps",
        str(fps_value),
        "--max_frames",
        str(args.max_frames),
    ]


def _tagged(base: str, tag: str | None) -> str:
    clean = (tag or "").strip()
    if not clean:
        return base
    return f"{base}_{clean}"


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_id = args.suite_id or f"{timestamp}_method_compare_full"
    suite_dir = repo_root / "outputs" / "phase3" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_tag = (args.suite_tag or "").strip()

    dataset_specs = [
        ("kitti", "kitti_method_compare_rich_v3", KITTI_PAIRS, 10.0),
        ("dynamicstereo", "dynamicstereo_method_compare_rich_v3", DYNAMICSTEREO_PAIRS, 10.0),
        ("minesource", "minesource_method_compare_rich_v3", MINESOURCE_PAIRS, 30.0),
    ]

    overall_rc = 0
    source_suites: List[str] = []
    manifest_rows: List[Dict[str, object]] = []
    for dataset_key, child_suite_id, pairs, fps_value in dataset_specs:
        tagged_child_suite_id = _tagged(child_suite_id, suite_tag)
        method_suite_id = f"{tagged_child_suite_id}__methods"
        compare_cmd = _build_compare_cmd(args, method_suite_id, pairs, fps_value)
        rc = _run(repo_root, suite_dir, f"{dataset_key}_method_compare", compare_cmd)
        overall_rc = rc or overall_rc
        if rc != 0 and not args.continue_on_error:
            return rc

        summary_cmd = _build_summary_cmd(args, tagged_child_suite_id, method_suite_id, pairs, fps_value)
        rc = _run(repo_root, suite_dir, f"{dataset_key}_build_summary", summary_cmd)
        overall_rc = rc or overall_rc
        if rc != 0 and not args.continue_on_error:
            return rc

        source_suites.append(tagged_child_suite_id)
        manifest_rows.append(
            {
                "dataset_key": dataset_key,
                "suite_id": tagged_child_suite_id,
                "method_suite_id": method_suite_id,
                "pair_count": len(pairs),
                "fps": fps_value,
            }
        )

    overall_suite_id = _tagged("overall_method_compare_rich_v3", suite_tag)
    overall_cmd = [
        str(args.python_bin),
        "scripts/internal/summarize_method_compare_overall.py",
        "--suite_id",
        overall_suite_id,
        "--source_suites",
        *source_suites,
    ]
    rc = _run(repo_root, suite_dir, "build_overall_summary", overall_cmd)
    overall_rc = rc or overall_rc
    if rc != 0 and not args.continue_on_error:
        return rc

    if not args.skip_figures:
        figure_cmd = [
            str(args.python_bin),
            "scripts/export_report_figures.py",
            "--suite_id",
            overall_suite_id,
        ]
        rc = _run(repo_root, suite_dir, "build_figures", figure_cmd)
        overall_rc = rc or overall_rc
        if rc != 0 and not args.continue_on_error:
            return rc

    (suite_dir / "suite_manifest.json").write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "source_suites": source_suites,
                "overall_suite_id": overall_suite_id,
                "max_frames": args.max_frames,
                "snapshot_every": args.snapshot_every,
                "weights_dir": args.weights_dir,
                "force_cpu": bool(args.force_cpu),
                "datasets": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"parent_suite={suite_id}")
    print(f"source_suites={','.join(source_suites)}")
    print(f"overall_suite={overall_suite_id}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
