#!/usr/bin/env python3
"""Inspect a pair entry and dump sample frames to disk.

用途/Goal: 快速验证 I/O 是否可读，不触碰 stitching 逻辑。
It reads a single frame index from both sources and saves PNG previews.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


# --- CLI helpers ---

def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the inspection script."""
    parser = argparse.ArgumentParser(description="Inspect a pair and save sample frames.")
    parser.add_argument("--pair", required=True, help="Pair id from pairs.yaml")
    parser.add_argument("--frame_index", type=int, required=True, help="Frame index to read")
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest (default: data/manifests/pairs.yaml)",
    )
    return parser


# --- Entry point ---

def main() -> int:
    """Run the inspection pipeline and write preview images.

    Returns:
        Exit code 0 on success; exceptions will bubble up as errors.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # 默认 INFO，确保 open_pair 的日志可见。
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from stitching.io import load_pairs, get_pair, open_pair  # noqa: E402

    pairs = load_pairs(repo_root / args.manifest)
    pair = get_pair(pairs, args.pair)

    left_source, right_source = open_pair(pair)
    try:
        left_frame = left_source.read(args.frame_index)
        right_frame = right_source.read(args.frame_index)
    finally:
        left_source.close()
        right_source.close()

    # 输出目录按 pair 分桶，避免覆盖其他检查结果。
    output_dir = repo_root / "outputs" / "inspect" / pair.id
    output_dir.mkdir(parents=True, exist_ok=True)
    # 固定文件名便于对比与写报告。
    left_path = output_dir / "left.png"
    right_path = output_dir / "right.png"

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required to save images. Install it first."
        ) from exc

    # 保存 BGR 帧为 PNG，用于快速目视检查。
    cv2.imwrite(str(left_path), left_frame)
    cv2.imwrite(str(right_path), right_frame)

    print(f"pair_id={pair.id}")
    print(f"dataset={pair.dataset} input_type={pair.input_type}")
    print(f"left={pair.left} right={pair.right}")
    print(f"length_left={left_source.length()} length_right={right_source.length()}")
    print(f"fps_left={left_source.fps()} fps_right={right_source.fps()}")
    print(
        f"resolution_left={left_source.resolution()} "
        f"resolution_right={right_source.resolution()}"
    )
    print(f"saved={left_path.relative_to(repo_root)} {right_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
