#!/usr/bin/env python3
"""Run baseline single-frame stitching for a specified pair and frame index."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline single-frame stitching.")
    parser.add_argument("--pair", required=True, help="Pair id from pairs.yaml")
    parser.add_argument("--frame_index", type=int, default=0, help="Frame index to read")
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest",
    )
    parser.add_argument("--feature", default="orb", help="Feature type: orb or sift")
    parser.add_argument("--nfeatures", type=int, default=2000, help="ORB nfeatures")
    parser.add_argument("--ratio", type=float, default=0.75, help="Ratio test threshold")
    parser.add_argument(
        "--min_matches",
        type=int,
        default=30,
        help="Minimum good matches required to proceed",
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Optional run id (default: timestamp_pair_id)",
    )
    return parser


def _setup_logging(log_path: Optional[Path]) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=handlers,
    )


def _write_debug(path: Path, debug: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = _build_parser().parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from stitching.io import load_pairs, get_pair, open_pair  # noqa: E402
    from stitching.features import detect_and_describe  # noqa: E402
    from stitching.matching import match_descriptors, draw_matches  # noqa: E402
    from stitching.geometry import (  # noqa: E402
        estimate_homography,
        compute_canvas_and_transform,
        warp_pair,
    )
    from stitching.blending import feather_blend  # noqa: E402
    from stitching.viz import save_image, overlay_images  # noqa: E402

    run_id = args.run_id
    if run_id is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_pair = args.pair.replace("/", "_").replace(" ", "_")
        run_id = f"{ts}_{safe_pair}"

    output_dir = repo_root / "outputs" / "runs" / run_id
    log_path = output_dir / "logs.txt"
    _setup_logging(log_path)

    debug = {
        "pair_id": args.pair,
        "dataset": None,
        "input_type": None,
        "frame_index": args.frame_index,
        "feature": args.feature,
        "nfeatures": args.nfeatures,
        "ratio": args.ratio,
        "min_matches": args.min_matches,
        "ransac_thresh": args.ransac_thresh,
        "n_kp_left": None,
        "n_kp_right": None,
        "n_matches_raw": None,
        "n_matches_good": None,
        "n_inliers": None,
        "inlier_ratio": None,
        "H": None,
        "canvas_size": None,
        "runtime_ms": None,
        "failure_stage": None,
        "message": None,
    }

    start_time = time.time()
    debug_path = output_dir / "debug.json"

    try:
        pairs = load_pairs(repo_root / args.manifest)
        pair = get_pair(pairs, args.pair)
        debug["dataset"] = pair.dataset
        debug["input_type"] = pair.input_type

        left_source, right_source = open_pair(pair)
        try:
            left = left_source.read(args.frame_index)
            right = right_source.read(args.frame_index)
        finally:
            left_source.close()
            right_source.close()
    except Exception as exc:
        debug["failure_stage"] = "read_frame"
        debug["message"] = (
            f"{exc}. If input is readable, try scripts/inspect_pair.py first."
        )
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.error("Failed to read frames: %s", exc)
        return 1

    save_image(output_dir / "left.png", left)
    save_image(output_dir / "right.png", right)

    try:
        kp_left, desc_left = detect_and_describe(
            left, feature=args.feature, nfeatures=args.nfeatures
        )
        kp_right, desc_right = detect_and_describe(
            right, feature=args.feature, nfeatures=args.nfeatures
        )
        debug["n_kp_left"] = len(kp_left)
        debug["n_kp_right"] = len(kp_right)
    except Exception as exc:
        debug["failure_stage"] = "feature"
        debug["message"] = str(exc)
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.error("Feature detection failed: %s", exc)
        return 1

    good_matches, raw_matches = match_descriptors(
        desc_left, desc_right, method=args.feature, ratio=args.ratio
    )
    debug["n_matches_raw"] = raw_matches
    debug["n_matches_good"] = len(good_matches)

    matches_img = draw_matches(left, kp_left, right, kp_right, good_matches)
    save_image(output_dir / "matches.png", matches_img)

    if len(good_matches) < args.min_matches:
        debug["failure_stage"] = "matching"
        debug["message"] = (
            f"Not enough matches: {len(good_matches)} < {args.min_matches}"
        )
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.warning("Not enough matches; aborting.")
        return 1

    H, mask = estimate_homography(
        kp_left, kp_right, good_matches, ransac_thresh=args.ransac_thresh
    )
    if H is None or mask is None:
        debug["failure_stage"] = "homography"
        debug["message"] = "cv2.findHomography failed to estimate H."
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.error("Homography estimation failed.")
        return 1

    inliers = mask.ravel().tolist()
    inlier_count = int(sum(inliers))
    debug["n_inliers"] = inlier_count
    debug["inlier_ratio"] = (
        float(inlier_count) / float(len(inliers)) if inliers else 0.0
    )
    debug["H"] = H.tolist()

    inliers_img = draw_matches(
        left, kp_left, right, kp_right, good_matches, inlier_mask=inliers
    )
    save_image(output_dir / "inliers.png", inliers_img)

    try:
        canvas_size, T = compute_canvas_and_transform(
            (left.shape[0], left.shape[1]),
            (right.shape[0], right.shape[1]),
            H,
        )
        debug["canvas_size"] = [int(canvas_size[0]), int(canvas_size[1])]

        left_warped, right_warped = warp_pair(left, right, H, canvas_size, T)
        overlay = overlay_images(left_warped, right_warped, alpha=0.5)
        save_image(output_dir / "warp_overlay.png", overlay)
    except Exception as exc:
        debug["failure_stage"] = "warp"
        debug["message"] = str(exc)
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.error("Warp failed: %s", exc)
        return 1

    try:
        stitched = feather_blend(left_warped, right_warped)
        save_image(output_dir / "stitched_frame.png", stitched)
    except Exception as exc:
        debug["failure_stage"] = "blend"
        debug["message"] = str(exc)
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.error("Blending failed: %s", exc)
        return 1

    debug["runtime_ms"] = int((time.time() - start_time) * 1000)
    _write_debug(debug_path, debug)
    logging.info("Run completed: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
