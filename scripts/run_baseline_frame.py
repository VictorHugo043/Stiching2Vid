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
    parser = argparse.ArgumentParser(
        description=(
            "Single-frame stitching with a shared Method A / Method B backend skeleton "
            "and a VideoStitcher-based frame quality preview compose path."
        )
    )
    parser.add_argument("--pair", required=True, help="Pair id from pairs.yaml")
    parser.add_argument("--frame_index", type=int, default=0, help="Frame index to read")
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest",
    )
    parser.add_argument("--feature", default="orb", help="Feature type: orb or sift")
    parser.add_argument(
        "--feature_backend",
        default=None,
        help=(
            "Optional feature backend override "
            "(supported: opencv_orb, opencv_sift, superpoint if optional deps are installed)"
        ),
    )
    parser.add_argument(
        "--matcher_backend",
        default=None,
        help=(
            "Optional matcher backend override "
            "(supported: opencv_bf_ratio, lightglue if optional deps are installed)"
        ),
    )
    parser.add_argument(
        "--geometry_backend",
        default=None,
        help="Optional geometry backend override (supported now: opencv_ransac, opencv_usac_magsac)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Method B device override (auto/cpu/mps/cuda/cuda:0)",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force Method B backends to use CPU even if GPU is available",
    )
    parser.add_argument(
        "--weights_dir",
        default=None,
        help="Optional directory or file path for Method B weights",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=2048,
        help="Maximum keypoints for SuperPoint backend (<=0 means no cap)",
    )
    parser.add_argument(
        "--resize_long_edge",
        type=int,
        default=None,
        help=(
            "Optional SuperPoint preprocess resize. "
            "None uses package default (1024), <=0 disables auto-resize."
        ),
    )
    parser.add_argument(
        "--depth_confidence",
        type=float,
        default=None,
        help="Optional LightGlue depth_confidence override",
    )
    parser.add_argument(
        "--width_confidence",
        type=float,
        default=None,
        help="Optional LightGlue width_confidence override",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=None,
        help="Optional LightGlue filter_threshold override",
    )
    parser.add_argument(
        "--feature_fallback_backend",
        default=None,
        help="Optional fallback feature backend when requested backend fails",
    )
    parser.add_argument(
        "--matcher_fallback_backend",
        default=None,
        help="Optional fallback matcher backend when requested backend fails",
    )
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
    parser.add_argument(
        "--blend",
        default="feather",
        choices=["none", "feather", "multiband"],
        help="Blending mode for frame_quality_preview compose",
    )
    parser.add_argument(
        "--mb_levels",
        type=int,
        default=5,
        help="Number of bands when --blend=multiband",
    )
    parser.add_argument(
        "--seam",
        default="opencv_dp_color",
        choices=["none", "opencv_dp_color", "opencv_dp_colorgrad", "opencv_voronoi"],
        help="Seam finder mode for frame_quality_preview compose",
    )
    parser.add_argument(
        "--seam_megapix",
        type=float,
        default=0.1,
        help="Megapixel budget used for seam estimation scale",
    )
    parser.add_argument(
        "--seam_dilate",
        type=int,
        default=1,
        help="Dilate iterations for seam mask before resizing to full compose mask",
    )
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop",
        dest="crop",
        action="store_true",
        default=True,
        help="Enable LIR crop before seam estimation (default: enabled)",
    )
    crop_group.add_argument(
        "--no_crop",
        "--no-crop",
        dest="crop",
        action="store_false",
        help="Disable crop before seam estimation",
    )
    parser.add_argument(
        "--lir_method",
        default="auto",
        choices=["auto", "lir", "fallback"],
        help="LIR backend method for cropper",
    )
    parser.add_argument(
        "--lir_erode",
        type=int,
        default=2,
        help="Erode iterations for fallback LIR method",
    )
    parser.add_argument(
        "--crop_debug",
        type=int,
        default=1,
        choices=[0, 1],
        help="Emit extra crop snapshots in frame_quality_preview output",
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


def _resolve_feature_backend(feature: str, feature_backend: Optional[str]) -> str:
    if feature_backend:
        return feature_backend.strip().lower()
    feature_name = feature.strip().lower()
    if feature_name == "orb":
        return "opencv_orb"
    if feature_name == "sift":
        return "opencv_sift"
    return feature_name


def _resolve_matcher_backend(matcher_backend: Optional[str]) -> str:
    return matcher_backend.strip().lower() if matcher_backend else "opencv_bf_ratio"


def _resolve_geometry_backend(geometry_backend: Optional[str]) -> str:
    return geometry_backend.strip().lower() if geometry_backend else "opencv_ransac"


def _resolve_optional_backend(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().lower()


def main() -> int:
    args = _build_parser().parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from stitching.io import load_pairs, get_pair, open_pair  # noqa: E402
    from stitching.features import detect_and_describe_result  # noqa: E402
    from stitching.matching import match_feature_results, draw_matches  # noqa: E402
    from stitching.geometry import (  # noqa: E402
        estimate_homography_result,
        compute_canvas_and_transform,
        warp_pair,
    )
    from stitching.frame_quality_preview import compose_frame_quality_preview  # noqa: E402
    from stitching.viz import save_image, overlay_images  # noqa: E402

    run_id = args.run_id
    if run_id is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_pair = args.pair.replace("/", "_").replace(" ", "_")
        run_id = f"{ts}_{safe_pair}"

    output_dir = repo_root / "outputs" / "runs" / run_id
    log_path = output_dir / "logs.txt"
    _setup_logging(log_path)

    feature_backend = _resolve_feature_backend(args.feature, args.feature_backend)
    matcher_backend = _resolve_matcher_backend(args.matcher_backend)
    geometry_backend = _resolve_geometry_backend(args.geometry_backend)
    feature_fallback_backend = _resolve_optional_backend(args.feature_fallback_backend)
    matcher_fallback_backend = _resolve_optional_backend(args.matcher_fallback_backend)

    debug = {
        "pair_id": args.pair,
        "dataset": None,
        "input_type": None,
        "frame_index": args.frame_index,
        "pipeline_interface": "result_objects_v1",
        "feature": args.feature,
        "feature_backend": feature_backend,
        "matcher_backend": matcher_backend,
        "geometry_backend": geometry_backend,
        "feature_backend_effective": feature_backend,
        "matcher_backend_effective": matcher_backend,
        "nfeatures": args.nfeatures,
        "ratio": args.ratio,
        "min_matches": args.min_matches,
        "ransac_thresh": args.ransac_thresh,
        "device": args.device,
        "force_cpu": args.force_cpu,
        "weights_dir": args.weights_dir,
        "max_keypoints": args.max_keypoints,
        "resize_long_edge": args.resize_long_edge,
        "depth_confidence": args.depth_confidence,
        "width_confidence": args.width_confidence,
        "filter_threshold": args.filter_threshold,
        "feature_fallback_backend": feature_fallback_backend,
        "matcher_fallback_backend": matcher_fallback_backend,
        "n_kp_left": None,
        "n_kp_right": None,
        "n_matches_raw": None,
        "n_matches_good": None,
        "n_inliers": None,
        "inlier_ratio": None,
        "reprojection_error": None,
        "H": None,
        "canvas_size": None,
        "stage_runtimes_ms": {},
        "feature_stage": {},
        "matching_stage": {},
        "geometry_stage": {},
        "compose_stage": {},
        "compose_backend": None,
        "blend": args.blend,
        "mb_levels": args.mb_levels,
        "seam": args.seam,
        "seam_megapix": args.seam_megapix,
        "seam_dilate": args.seam_dilate,
        "crop": args.crop,
        "lir_method": args.lir_method,
        "lir_erode": args.lir_erode,
        "crop_debug": args.crop_debug,
        "overlap_area": None,
        "crop_applied": None,
        "crop_method": None,
        "crop_rect": None,
        "output_bbox": None,
        "runtime_ms": None,
        "failure_stage": None,
        "message": None,
        "fallback_events": [],
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

    def _extract_pair_with_backend(backend_name: str):
        left_result = detect_and_describe_result(
            left,
            feature=args.feature,
            feature_backend=backend_name,
            nfeatures=args.nfeatures,
            device=args.device,
            force_cpu=args.force_cpu,
            weights_dir=args.weights_dir,
            max_keypoints=args.max_keypoints,
            resize_long_edge=args.resize_long_edge,
        )
        right_result = detect_and_describe_result(
            right,
            feature=args.feature,
            feature_backend=backend_name,
            nfeatures=args.nfeatures,
            device=args.device,
            force_cpu=args.force_cpu,
            weights_dir=args.weights_dir,
            max_keypoints=args.max_keypoints,
            resize_long_edge=args.resize_long_edge,
        )
        return left_result, right_result

    feature_requested_error = None
    feature_requested_diagnostics = None
    try:
        feature_left, feature_right = _extract_pair_with_backend(feature_backend)
    except Exception as exc:
        feature_requested_error = str(exc)
        feature_requested_diagnostics = getattr(exc, "diagnostics", None)
        if feature_fallback_backend and feature_fallback_backend != feature_backend:
            logging.warning(
                "Feature backend %s failed, fallback to %s: %s",
                feature_backend,
                feature_fallback_backend,
                exc,
            )
            debug["fallback_events"].append(
                {
                    "stage": "feature",
                    "requested_backend": feature_backend,
                    "fallback_backend": feature_fallback_backend,
                    "reason": str(exc),
                    "diagnostics": feature_requested_diagnostics,
                }
            )
            try:
                feature_left, feature_right = _extract_pair_with_backend(feature_fallback_backend)
                debug["feature_backend_effective"] = feature_fallback_backend
            except Exception as fallback_exc:
                debug["feature_stage"] = {
                    "requested_backend": feature_backend,
                    "effective_backend": feature_fallback_backend,
                    "requested_error": feature_requested_error,
                    "requested_diagnostics": feature_requested_diagnostics,
                    "fallback_error": str(fallback_exc),
                    "fallback_diagnostics": getattr(fallback_exc, "diagnostics", None),
                }
                debug["failure_stage"] = "feature"
                debug["message"] = str(fallback_exc)
                debug["runtime_ms"] = int((time.time() - start_time) * 1000)
                _write_debug(debug_path, debug)
                logging.error("Feature detection fallback failed: %s", fallback_exc)
                return 1
        else:
            debug["feature_stage"] = {
                "requested_backend": feature_backend,
                "effective_backend": feature_backend,
                "requested_error": feature_requested_error,
                "requested_diagnostics": feature_requested_diagnostics,
            }
            debug["failure_stage"] = "feature"
            debug["message"] = str(exc)
            debug["runtime_ms"] = int((time.time() - start_time) * 1000)
            _write_debug(debug_path, debug)
            logging.error("Feature detection failed: %s", exc)
            return 1

    debug["n_kp_left"] = feature_left.n_keypoints
    debug["n_kp_right"] = feature_right.n_keypoints
    debug["feature_backend_effective"] = feature_left.backend_name
    debug["stage_runtimes_ms"]["feature_left"] = float(feature_left.runtime_ms)
    debug["stage_runtimes_ms"]["feature_right"] = float(feature_right.runtime_ms)
    debug["feature_stage"] = {
        "requested_backend": feature_backend,
        "effective_backend": debug["feature_backend_effective"],
        "requested_error": feature_requested_error,
        "requested_diagnostics": feature_requested_diagnostics,
        "left": {
            "backend_name": feature_left.backend_name,
            "n_keypoints": int(feature_left.n_keypoints),
            "runtime_ms": float(feature_left.runtime_ms),
            "meta": feature_left.meta,
        },
        "right": {
            "backend_name": feature_right.backend_name,
            "n_keypoints": int(feature_right.n_keypoints),
            "runtime_ms": float(feature_right.runtime_ms),
            "meta": feature_right.meta,
        },
    }

    matching_requested_error = None
    matching_requested_diagnostics = None
    try:
        match_result = match_feature_results(
            feature_left,
            feature_right,
            matcher_backend=matcher_backend,
            ratio=args.ratio,
            device=args.device,
            force_cpu=args.force_cpu,
            weights_dir=args.weights_dir,
            depth_confidence=args.depth_confidence,
            width_confidence=args.width_confidence,
            filter_threshold=args.filter_threshold,
        )
    except Exception as exc:
        matching_requested_error = str(exc)
        matching_requested_diagnostics = getattr(exc, "diagnostics", None)
        if matcher_fallback_backend and matcher_fallback_backend != matcher_backend:
            logging.warning(
                "Matcher backend %s failed, fallback to %s: %s",
                matcher_backend,
                matcher_fallback_backend,
                exc,
            )
            debug["fallback_events"].append(
                {
                    "stage": "matching",
                    "requested_backend": matcher_backend,
                    "fallback_backend": matcher_fallback_backend,
                    "reason": str(exc),
                    "diagnostics": matching_requested_diagnostics,
                }
            )
            try:
                match_result = match_feature_results(
                    feature_left,
                    feature_right,
                    matcher_backend=matcher_fallback_backend,
                    ratio=args.ratio,
                    device=args.device,
                    force_cpu=args.force_cpu,
                    weights_dir=args.weights_dir,
                    depth_confidence=args.depth_confidence,
                    width_confidence=args.width_confidence,
                    filter_threshold=args.filter_threshold,
                )
                debug["matcher_backend_effective"] = matcher_fallback_backend
            except Exception as fallback_exc:
                debug["matching_stage"] = {
                    "requested_backend": matcher_backend,
                    "effective_backend": matcher_fallback_backend,
                    "requested_error": matching_requested_error,
                    "requested_diagnostics": matching_requested_diagnostics,
                    "fallback_error": str(fallback_exc),
                    "fallback_diagnostics": getattr(fallback_exc, "diagnostics", None),
                }
                debug["failure_stage"] = "matching"
                debug["message"] = str(fallback_exc)
                debug["runtime_ms"] = int((time.time() - start_time) * 1000)
                _write_debug(debug_path, debug)
                logging.error("Descriptor matching fallback failed: %s", fallback_exc)
                return 1
        else:
            debug["matching_stage"] = {
                "requested_backend": matcher_backend,
                "effective_backend": matcher_backend,
                "requested_error": matching_requested_error,
                "requested_diagnostics": matching_requested_diagnostics,
            }
            debug["failure_stage"] = "matching"
            debug["message"] = str(exc)
            debug["runtime_ms"] = int((time.time() - start_time) * 1000)
            _write_debug(debug_path, debug)
            logging.error("Descriptor matching failed: %s", exc)
            return 1

    debug["n_matches_raw"] = match_result.tentative_count
    debug["n_matches_good"] = match_result.good_count
    debug["matcher_backend_effective"] = match_result.backend_name
    debug["stage_runtimes_ms"]["matching"] = float(match_result.runtime_ms)
    debug["matching_stage"] = {
        "requested_backend": matcher_backend,
        "effective_backend": debug["matcher_backend_effective"],
        "requested_error": matching_requested_error,
        "requested_diagnostics": matching_requested_diagnostics,
        "backend_name": match_result.backend_name,
        "tentative_count": int(match_result.tentative_count),
        "good_count": int(match_result.good_count),
        "runtime_ms": float(match_result.runtime_ms),
        "meta": match_result.meta,
    }

    matches_img = draw_matches(
        left,
        feature_left.cv_keypoints,
        right,
        feature_right.cv_keypoints,
        match_result.cv_matches,
    )
    save_image(output_dir / "matches.png", matches_img)

    if match_result.good_count < args.min_matches:
        debug["failure_stage"] = "matching"
        debug["message"] = (
            f"Not enough matches: {match_result.good_count} < {args.min_matches}"
        )
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.warning("Not enough matches; aborting.")
        return 1

    try:
        geometry_result = estimate_homography_result(
            feature_left,
            feature_right,
            match_result,
            geometry_backend=geometry_backend,
            ransac_thresh=args.ransac_thresh,
        )
    except Exception as exc:
        debug["failure_stage"] = "homography"
        debug["message"] = str(exc)
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.error("Homography backend failed: %s", exc)
        return 1

    debug["stage_runtimes_ms"]["geometry"] = float(geometry_result.runtime_ms)
    debug["geometry_stage"] = {
        "backend_name": geometry_result.backend_name,
        "status": geometry_result.status,
        "runtime_ms": float(geometry_result.runtime_ms),
        "meta": geometry_result.meta,
    }
    if geometry_result.H is None or geometry_result.inlier_mask is None:
        debug["failure_stage"] = "homography"
        debug["message"] = (
            f"Geometry estimation failed with status={geometry_result.status}"
        )
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.error("Homography estimation failed: %s", geometry_result.status)
        return 1

    inliers = geometry_result.inlier_mask
    debug["n_inliers"] = geometry_result.inlier_count
    debug["inlier_ratio"] = geometry_result.inlier_ratio
    debug["reprojection_error"] = geometry_result.reprojection_error
    debug["H"] = geometry_result.H.tolist()

    inliers_img = draw_matches(
        left,
        feature_left.cv_keypoints,
        right,
        feature_right.cv_keypoints,
        match_result.cv_matches,
        inlier_mask=inliers,
    )
    save_image(output_dir / "inliers.png", inliers_img)

    try:
        canvas_size, T = compute_canvas_and_transform(
            (left.shape[0], left.shape[1]),
            (right.shape[0], right.shape[1]),
            geometry_result.H,
        )
        debug["canvas_size"] = [int(canvas_size[0]), int(canvas_size[1])]

        left_warped, right_warped = warp_pair(left, right, geometry_result.H, canvas_size, T)
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
        compose_result = compose_frame_quality_preview(
            left,
            right,
            geometry_result.H,
            T,
            canvas_size,
            output_dir=output_dir,
            frame_idx=args.frame_index,
            blend_mode=args.blend,
            mb_levels=args.mb_levels,
            seam_mode=args.seam,
            seam_megapix=args.seam_megapix,
            seam_dilate=args.seam_dilate,
            crop_enabled=args.crop,
            lir_method=args.lir_method,
            lir_erode=args.lir_erode,
            crop_debug=args.crop_debug,
        )
        stitched = compose_result.stitched
        save_image(output_dir / "stitched_frame.png", stitched)
        debug["stage_runtimes_ms"]["compose"] = float(compose_result.runtime_ms)
        debug["compose_backend"] = compose_result.backend_name
        debug["compose_stage"] = {
            "backend_name": compose_result.backend_name,
            "runtime_ms": float(compose_result.runtime_ms),
            "warnings": compose_result.warnings,
            "meta": compose_result.meta,
        }
        debug["overlap_area"] = int(compose_result.overlap_area)
        debug["crop_applied"] = bool(compose_result.crop_applied)
        debug["crop_method"] = compose_result.crop_method
        debug["crop_rect"] = compose_result.crop_rect
        debug["output_bbox"] = compose_result.output_bbox
    except Exception as exc:
        debug["failure_stage"] = "compose"
        debug["message"] = str(exc)
        debug["runtime_ms"] = int((time.time() - start_time) * 1000)
        _write_debug(debug_path, debug)
        logging.error("Frame quality compose failed: %s", exc)
        return 1

    debug["runtime_ms"] = int((time.time() - start_time) * 1000)
    _write_debug(debug_path, debug)
    logging.info("Run completed: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
