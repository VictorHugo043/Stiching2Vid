#!/usr/bin/env python3
"""Baseline video-level stitching with optional temporal smoothing."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple


# --- CLI parsing ---


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline video-level stitching.")
    parser.add_argument("--pair", required=True, help="Pair id from pairs.yaml")
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument(
        "--keyframe_every",
        type=int,
        default=5,
        help="Geometry re-estimate cadence when --geometry_mode=keyframe_update",
    )
    parser.add_argument(
        "--geometry_mode",
        default=None,
        choices=["fixed_geometry", "keyframe_update", "adaptive_update"],
        help=(
            "User-facing geometry update mode. "
            "fixed_geometry maps to legacy video_mode=1, "
            "keyframe_update maps to legacy video_mode=0, "
            "adaptive_update keeps cached execution but refreshes geometry on seam events."
        ),
    )
    parser.add_argument("--feature", default="orb", help="Feature type: orb or sift")
    parser.add_argument(
        "--feature_backend",
        default=None,
        help=(
            "Optional feature backend override "
            "(supported: opencv_orb, opencv_sift, superpoint)"
        ),
    )
    parser.add_argument(
        "--matcher_backend",
        default=None,
        help=(
            "Optional matcher backend override "
            "(supported: opencv_bf_ratio, lightglue)"
        ),
    )
    parser.add_argument(
        "--geometry_backend",
        default=None,
        help="Optional geometry backend override (supported: opencv_ransac, opencv_usac_magsac)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Method B device override (auto/cpu/cuda/cuda:0)",
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
        help="Minimum good matches required to estimate H",
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold",
    )
    parser.add_argument(
        "--blend",
        default="feather",
        choices=["none", "feather", "multiband"],
        help="Blending mode",
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
        help="Seam finder mode (computed on warped low-res ROI)",
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
    parser.add_argument(
        "--seam_smooth",
        default="none",
        choices=["none", "ema", "window"],
        help="Temporal smoothing policy for final seam masks",
    )
    parser.add_argument(
        "--seam_smooth_alpha",
        type=float,
        default=0.8,
        help="EMA alpha for seam mask smoothing",
    )
    parser.add_argument(
        "--seam_smooth_window",
        type=int,
        default=5,
        help="Window size for seam mask smoothing when --seam_smooth=window",
    )
    parser.add_argument(
        "--seam_policy",
        default="auto",
        choices=["auto", "fixed", "keyframe", "trigger"],
        help="Seam update policy shell; does not change OpenCV seam backend itself",
    )
    parser.add_argument(
        "--seam_keyframe_every",
        type=int,
        default=0,
        help=(
            "Seam refresh cadence for seam_policy=keyframe "
            "(0 -> derive from geometry mode / keyframe_every)"
        ),
    )
    parser.add_argument(
        "--seam_trigger_overlap_ratio",
        type=float,
        default=0.0,
        help="Trigger seam refresh when current overlap ratio drops below this value (0 disables)",
    )
    parser.add_argument(
        "--seam_trigger_diff_threshold",
        type=float,
        default=20.0,
        help="Trigger seam refresh when overlap diff before seam exceeds this threshold",
    )
    parser.add_argument(
        "--seam_trigger_foreground_ratio",
        type=float,
        default=0.0,
        help="Trigger seam refresh when foreground/object-aware protected ratio exceeds this value (0 disables)",
    )
    parser.add_argument(
        "--seam_trigger_cooldown_frames",
        type=int,
        default=0,
        help="Cooldown frames after a trigger seam event before another trigger-driven refresh is allowed",
    )
    parser.add_argument(
        "--seam_trigger_hysteresis_ratio",
        type=float,
        default=1.0,
        help="Re-arm ratio for trigger hysteresis in (0,1]; smaller means stronger hysteresis",
    )
    parser.add_argument(
        "--foreground_mode",
        default="off",
        choices=["off", "disagreement"],
        help="Compatible foreground/object-aware mode; disagreement builds a protected region from cross-view overlap diff",
    )
    parser.add_argument(
        "--foreground_diff_threshold",
        type=float,
        default=24.0,
        help="Pixel diff threshold used by foreground_mode=disagreement",
    )
    parser.add_argument(
        "--foreground_dilate",
        type=int,
        default=5,
        help="Dilate iterations for foreground/object-aware protected region",
    )
    parser.add_argument(
        "--seam_snapshot_on_recompute",
        type=int,
        default=1,
        choices=[0, 1],
        help="Save seam/mask and stitched snapshots when seam is recomputed",
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
        help="Emit extra crop snapshots on seam keyframes",
    )
    parser.add_argument(
        "--video_mode",
        type=int,
        default=None,
        choices=[0, 1],
        help=(
            "Legacy internal geometry-path selector kept for compatibility only: "
            "1=fixed_geometry cached path, 0=keyframe_update path. "
            "Prefer --geometry_mode for new runs."
        ),
    )
    parser.add_argument(
        "--reuse_mode",
        default="frame0_all",
        choices=["frame0_all", "frame0_geom", "frame0_seam", "emaH"],
        help=(
            "Legacy state reuse strategy when --video_mode=1; "
            "current geometry semantics remain fixed_geometry"
        ),
    )
    parser.add_argument(
        "--reinit_every",
        type=int,
        default=0,
        help="Reinitialize every N processed frames in video mode (0 disables)",
    )
    parser.add_argument(
        "--reinit_on_low_overlap_ratio",
        type=float,
        default=0.0,
        help="Reinitialize when overlap_area_current < ratio * overlap_area_init (0 disables)",
    )
    parser.add_argument(
        "--smooth_h",
        default="none",
        choices=["none", "ema", "window"],
        help="Temporal smoothing policy for homography stream",
    )
    parser.add_argument(
        "--smooth_alpha",
        type=float,
        default=0.8,
        help="EMA alpha (higher is smoother)",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Window size for moving-average smoothing",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback fps for frames input when not provided elsewhere",
    )
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=50,
        help="Save snapshot images every K processed frames",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory (default: outputs/runs/<run_id>)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Optional run id (default: YYYYMMDD-HHMM_<pair_id>_<feature>)",
    )
    return parser


# --- Logging and serialization helpers ---


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=handlers,
    )


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v


def _quantile95(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round(0.95 * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


# --- FPS decision policy ---


def _resolve_fps(
    pair_cfg,
    left_source,
    right_source,
    cli_fps: float,
) -> Tuple[float, str, List[str]]:
    """Resolve fps using the required precedence rules."""

    candidates: List[str] = []

    if pair_cfg.input_type == "video":
        src_fps = _safe_float(left_source.fps()) or _safe_float(right_source.fps())
        candidates.append(f"video_source={src_fps}")
        if src_fps is not None:
            return src_fps, "video_source", candidates

    meta_fps = _safe_float((pair_cfg.meta or {}).get("fps"))
    candidates.append(f"meta_fps={meta_fps}")
    if meta_fps is not None:
        return meta_fps, "meta_fps", candidates

    cli_fps_safe = _safe_float(cli_fps)
    candidates.append(f"cli_fps={cli_fps_safe}")
    if cli_fps_safe is not None:
        return cli_fps_safe, "cli_fps", candidates

    candidates.append("fallback=30.0")
    return 30.0, "fallback_30", candidates


# --- Frame processing helpers ---


def _flatten_h(H) -> List[Optional[float]]:
    if H is None:
        return [None] * 9
    return [float(H[r, c]) for r in range(3) for c in range(3)]


def _skip_frames(left_source, right_source, count: int) -> int:
    skipped = 0
    for _ in range(count):
        left = left_source.read_next()
        right = right_source.read_next()
        if left is None or right is None:
            break
        skipped += 1
    return skipped


def _consume_stride(left_source, right_source, stride: int) -> int:
    if stride <= 1:
        return 0
    return _skip_frames(left_source, right_source, stride - 1)


def _save_keyframe_debug(
    output_dir: Path,
    frame_idx: int,
    matches_img,
    inliers_img=None,
) -> None:
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    from stitching.viz import save_image  # noqa: WPS433,E402

    save_image(snapshots_dir / f"matches_{frame_idx:06d}.png", matches_img)
    if inliers_img is not None:
        save_image(snapshots_dir / f"inliers_{frame_idx:06d}.png", inliers_img)


def _save_snapshot(
    output_dir: Path,
    frame_idx: int,
    left_bgr,
    right_bgr,
    stitched_bgr,
    overlay_bgr,
    overlay_raw_bgr,
    overlay_sm_bgr,
) -> None:
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    from stitching.viz import save_image  # noqa: WPS433,E402

    save_image(snapshots_dir / f"frame_{frame_idx:06d}_left.png", left_bgr)
    save_image(snapshots_dir / f"frame_{frame_idx:06d}_right.png", right_bgr)
    save_image(snapshots_dir / f"frame_{frame_idx:06d}_stitched.png", stitched_bgr)
    save_image(snapshots_dir / f"frame_{frame_idx:06d}_overlay.png", overlay_bgr)
    save_image(snapshots_dir / f"overlay_raw_{frame_idx:06d}.png", overlay_raw_bgr)
    save_image(snapshots_dir / f"overlay_sm_{frame_idx:06d}.png", overlay_sm_bgr)


def _seam_cli_to_method(seam_mode: str) -> str:
    mapping = {
        "none": "none",
        "opencv_dp_color": "dp_color",
        "opencv_dp_colorgrad": "dp_colorgrad",
        "opencv_voronoi": "voronoi",
    }
    return mapping.get(seam_mode, "dp_color")


def _resolve_geometry_mode(video_mode: int) -> str:
    return "fixed_geometry" if int(video_mode) == 1 else "keyframe_update"


def _geometry_mode_to_video_mode(geometry_mode: str) -> int:
    normalized = str(geometry_mode).strip().lower()
    if normalized == "fixed_geometry":
        return 1
    if normalized == "keyframe_update":
        return 0
    if normalized == "adaptive_update":
        return 1
    raise ValueError(f"Unsupported geometry mode: {geometry_mode}")


def _resolve_execution_mode(
    requested_geometry_mode: Optional[str],
    requested_video_mode: Optional[int],
) -> Tuple[int, str, str, Optional[str]]:
    geometry_mode_raw = (requested_geometry_mode or "").strip().lower()
    if geometry_mode_raw:
        effective_video_mode = _geometry_mode_to_video_mode(geometry_mode_raw)
        warning = None
        if requested_video_mode is not None:
            requested_from_legacy = _resolve_geometry_mode(int(requested_video_mode))
            if requested_from_legacy != geometry_mode_raw:
                warning = (
                    "Conflicting mode request detected: "
                    f"--geometry_mode={geometry_mode_raw} overrides "
                    f"--video_mode={int(requested_video_mode)} ({requested_from_legacy})"
                )
        return effective_video_mode, geometry_mode_raw, "geometry_mode", warning

    effective_video_mode = 1 if requested_video_mode is None else int(requested_video_mode)
    return (
        effective_video_mode,
        _resolve_geometry_mode(effective_video_mode),
        "legacy_video_mode",
        None,
    )


def _effective_geometry_keyframe_every(geometry_mode: str, keyframe_every: int) -> int:
    return max(1, int(keyframe_every)) if geometry_mode == "keyframe_update" else 0


def _jitter_meaningful_for_geometry_mode(geometry_mode: str) -> int:
    return 0 if geometry_mode == "fixed_geometry" else 1


def _jitter_scope_for_geometry_mode(geometry_mode: str) -> str:
    return "geometry_only" if geometry_mode == "fixed_geometry" else "geometry_stream"


def _primary_temporal_metric_for_geometry_mode(geometry_mode: str) -> str:
    return "mean_jitter_sm" if _jitter_meaningful_for_geometry_mode(geometry_mode) else "mean_overlap_diff_after"


def _describe_mode_semantics(
    geometry_mode: str,
    video_mode: int,
    reuse_mode: str,
    geometry_mode_source: str,
) -> Dict[str, object]:
    return {
        "geometry_mode": geometry_mode,
        "geometry_mode_source": geometry_mode_source,
        "jitter_meaningful": bool(_jitter_meaningful_for_geometry_mode(geometry_mode)),
        "jitter_scope": _jitter_scope_for_geometry_mode(geometry_mode),
        "adaptive_update_available": True,
        "adaptive_update_strategy": (
            "seam_driven_geometry_refresh" if geometry_mode == "adaptive_update" else "disabled"
        ),
        "legacy_video_mode": int(video_mode),
        "legacy_reuse_mode": reuse_mode,
        "geometry_keyframe_control": "keyframe_every (only when geometry_mode=keyframe_update)",
        "seam_keyframe_control": "seam_policy + seam_keyframe_every",
        "adaptive_geometry_control": "seam_policy event -> geometry refresh (only when geometry_mode=adaptive_update)",
    }


def _compose_single_roi_on_canvas(canvas_size, roi_img, roi_mask, corner):
    import numpy as np  # type: ignore

    from stitching.seam_opencv import place_mask_on_canvas, place_roi_on_canvas  # noqa: WPS433,E402

    canvas_w, canvas_h = int(canvas_size[0]), int(canvas_size[1])
    canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    place_roi_on_canvas(canvas_img, roi_img, corner)
    place_mask_on_canvas(canvas_mask, roi_mask, corner)
    return canvas_img, canvas_mask


def _mask_ratio(mask) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(mask)
    if arr.size == 0:
        return 0.0
    return float((arr > 0).sum()) / float(arr.size)


def _h_delta_norm(H_curr, H_prev) -> float:
    """L2 norm of normalized homography delta for stability diagnostics."""

    import numpy as np  # type: ignore

    if H_curr is None or H_prev is None:
        return 0.0

    curr = np.asarray(H_curr, dtype=np.float64)
    prev = np.asarray(H_prev, dtype=np.float64)
    if curr.shape != (3, 3) or prev.shape != (3, 3):
        return 0.0

    curr = curr / max(abs(float(curr[2, 2])), 1e-8)
    prev = prev / max(abs(float(prev[2, 2])), 1e-8)
    return float(np.linalg.norm(curr - prev))


def _warn_and_record(debug: Dict, message: str) -> None:
    debug.setdefault("warnings", []).append(message)
    logging.warning(message)


def _mask_bbox_area(mask) -> int:
    import numpy as np  # type: ignore

    valid = np.asarray(mask) > 0
    if not valid.any():
        return 0
    ys, xs = np.where(valid)
    return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def _mask_union_bbox(mask_a, mask_b) -> Optional[Tuple[int, int, int, int]]:
    import numpy as np  # type: ignore

    union = (np.asarray(mask_a) > 0) | (np.asarray(mask_b) > 0)
    if not union.any():
        return None
    ys, xs = np.where(union)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)


def _crop_frame_to_rect(img, rect: Tuple[int, int, int, int]):
    x, y, w, h = [int(v) for v in rect]
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid crop rect size: {rect}")
    ih, iw = int(img.shape[0]), int(img.shape[1])
    if x < 0 or y < 0 or (x + w) > iw or (y + h) > ih:
        raise ValueError(f"crop rect out of frame bounds: {rect} vs {(iw, ih)}")
    return img[y : y + h, x : x + w]


def _compose_masks_panorama(masks, corners, sizes):
    """Compose masks into one panorama canvas by OR, with corner offsets."""

    import numpy as np  # type: ignore

    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    min_x = min(int(c[0]) for c in corners)
    min_y = min(int(c[1]) for c in corners)
    max_x = max(int(c[0]) + int(s[0]) for c, s in zip(corners, sizes))
    max_y = max(int(c[1]) + int(s[1]) for c, s in zip(corners, sizes))

    pano_w = max(1, int(max_x - min_x))
    pano_h = max(1, int(max_y - min_y))
    panorama = np.zeros((pano_h, pano_w), dtype=np.uint8)

    for mask, corner in zip(masks, corners):
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = np.where(arr > 0, 255, 0).astype(np.uint8)
        x = int(corner[0]) - min_x
        y = int(corner[1]) - min_y
        h, w = arr.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(pano_w, x + w)
        y1 = min(pano_h, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        mx0 = x0 - x
        my0 = y0 - y
        mx1 = mx0 + (x1 - x0)
        my1 = my0 + (y1 - y0)
        panorama[y0:y1, x0:x1] = np.maximum(
            panorama[y0:y1, x0:x1],
            arr[my0:my1, mx0:mx1],
        )
    return panorama


def _compose_as_is_preview(imgs, masks, corners):
    """Build a local panorama preview from ROI tensors and their corners."""

    import numpy as np  # type: ignore

    from stitching.seam_opencv import place_mask_on_canvas, place_roi_on_canvas  # noqa: WPS433,E402

    sizes = [(int(img.shape[1]), int(img.shape[0])) for img in imgs]
    min_x = min(int(c[0]) for c in corners)
    min_y = min(int(c[1]) for c in corners)
    max_x = max(int(c[0]) + int(s[0]) for c, s in zip(corners, sizes))
    max_y = max(int(c[1]) + int(s[1]) for c, s in zip(corners, sizes))
    pano_w = max(1, int(max_x - min_x))
    pano_h = max(1, int(max_y - min_y))

    canvas = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((pano_h, pano_w), dtype=np.uint8)
    for img, mask, corner in zip(imgs, masks, corners):
        shifted = (int(corner[0]) - min_x, int(corner[1]) - min_y)
        place_roi_on_canvas(canvas, img, shifted)
        place_mask_on_canvas(canvas_mask, mask, shifted)
    return canvas, canvas_mask


def _stack_masks_horizontally(masks):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if not masks:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    vis_masks = []
    max_h = max(int(m.shape[0]) for m in masks)
    for mask in masks:
        arr = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        if arr.shape[0] < max_h:
            pad_h = max_h - arr.shape[0]
            arr = np.pad(arr, ((0, pad_h), (0, 0)), mode="constant", constant_values=0)
        vis_masks.append(cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR))
    return cv2.hconcat(vis_masks)


def _validate_rois_or_raise(name: str, imgs, masks, corners, sizes, canvas_size=None) -> None:
    if not (len(imgs) == len(masks) == len(corners) == len(sizes)):
        raise ValueError(f"{name}: imgs/masks/corners/sizes length mismatch")
    canvas_w = int(canvas_size[0]) if canvas_size is not None else None
    canvas_h = int(canvas_size[1]) if canvas_size is not None else None
    for idx, (img, mask, corner, size) in enumerate(zip(imgs, masks, corners, sizes)):
        h, w = int(img.shape[0]), int(img.shape[1])
        mh, mw = int(mask.shape[0]), int(mask.shape[1])
        if h != int(size[1]) or w != int(size[0]):
            raise ValueError(
                f"{name}[{idx}] size mismatch: img={w}x{h}, declared={size[0]}x{size[1]}",
            )
        if h != mh or w != mw:
            raise ValueError(
                f"{name}[{idx}] mask mismatch: img={w}x{h}, mask={mw}x{mh}",
            )
        if int(corner[0]) < 0 or int(corner[1]) < 0:
            raise ValueError(f"{name}[{idx}] corner is negative: {corner}")
        if canvas_w is not None and (int(corner[0]) + w) > canvas_w:
            raise ValueError(f"{name}[{idx}] exceeds canvas width: {corner} + {w} > {canvas_w}")
        if canvas_h is not None and (int(corner[1]) + h) > canvas_h:
            raise ValueError(f"{name}[{idx}] exceeds canvas height: {corner} + {h} > {canvas_h}")


def _resolve_seam_masks(
    left_mask_full,
    right_mask_full,
    seam_left_full,
    seam_right_full,
    limit_mask_full=None,
):
    import numpy as np  # type: ignore

    l_valid = left_mask_full > 0
    r_valid = right_mask_full > 0
    overlap = l_valid & r_valid
    left_only = l_valid & (~r_valid)
    right_only = r_valid & (~l_valid)

    seam_left = (seam_left_full > 0) & overlap
    seam_right = (seam_right_full > 0) & overlap

    unresolved = overlap & (~(seam_left | seam_right))
    seam_left = seam_left | unresolved

    final_left = (left_only | seam_left).astype(np.uint8) * 255
    final_right = (right_only | seam_right).astype(np.uint8) * 255

    if limit_mask_full is not None:
        limit = np.asarray(limit_mask_full) > 0
        final_left = (((final_left > 0) & limit).astype(np.uint8)) * 255
        final_right = (((final_right > 0) & limit).astype(np.uint8)) * 255

    return final_left, final_right


def _mean_overlap_diff(left_img, right_img, left_mask, right_mask) -> float:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    overlap = (left_mask > 0) & (right_mask > 0)
    if not overlap.any():
        return 0.0

    diff = cv2.absdiff(left_img, right_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return float(gray[overlap].mean())


def _blend_frames(
    left_warped,
    right_warped,
    mode: str,
    left_mask=None,
    right_mask=None,
    mb_levels: int = 5,
):
    from stitching.blending import (  # noqa: WPS433,E402
        blend_none,
        feather_blend,
        multiband_blend,
    )

    if mode == "none":
        return blend_none(left_warped, right_warped, left_mask=left_mask, right_mask=right_mask)
    if mode == "multiband":
        return multiband_blend(
            left_warped,
            right_warped,
            left_mask=left_mask,
            right_mask=right_mask,
            levels=mb_levels,
        )
    return feather_blend(left_warped, right_warped, left_mask=left_mask, right_mask=right_mask)


def _save_seam_debug(
    output_dir: Path,
    frame_idx: int,
    left_low,
    right_low,
    left_mask_low,
    right_mask_low,
    seam_left_low,
    seam_right_low,
    seam_overlay_low,
    overlap_diff_low,
) -> None:
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    from stitching.viz import save_image  # noqa: WPS433,E402

    save_image(snapshots_dir / f"warp_left_roi_{frame_idx:06d}.png", left_low)
    save_image(snapshots_dir / f"warp_right_roi_{frame_idx:06d}.png", right_low)
    save_image(snapshots_dir / f"mask_left_roi_{frame_idx:06d}.png", left_mask_low)
    save_image(snapshots_dir / f"mask_right_roi_{frame_idx:06d}.png", right_mask_low)
    save_image(snapshots_dir / f"seam_mask_left_{frame_idx:06d}.png", seam_left_low)
    save_image(snapshots_dir / f"seam_mask_right_{frame_idx:06d}.png", seam_right_low)
    save_image(snapshots_dir / f"seam_overlay_{frame_idx:06d}.png", seam_overlay_low)
    save_image(snapshots_dir / f"overlap_diff_{frame_idx:06d}.png", overlap_diff_low)

    # Stable aliases for quick inspection of latest keyframe.
    save_image(snapshots_dir / "warp_left_roi.png", left_low)
    save_image(snapshots_dir / "warp_right_roi.png", right_low)
    save_image(snapshots_dir / "mask_left_roi.png", left_mask_low)
    save_image(snapshots_dir / "mask_right_roi.png", right_mask_low)
    save_image(snapshots_dir / "seam_mask_left.png", seam_left_low)
    save_image(snapshots_dir / "seam_mask_right.png", seam_right_low)
    save_image(snapshots_dir / "seam_overlay.png", seam_overlay_low)
    save_image(snapshots_dir / "overlap_diff.png", overlap_diff_low)


def _build_transform_columns() -> List[str]:
    columns = [
        "frame_idx",
        "is_keyframe",
        "status",
        "n_kp_left",
        "n_kp_right",
        "n_matches_raw",
        "n_matches_good",
        "n_inliers",
        "inlier_ratio",
        "jitter_raw",
        "jitter_raw_max",
        "jitter_sm",
        "jitter_sm_max",
    ]
    for prefix in ["H", "Hraw", "Hsm"]:
        for i in range(3):
            for j in range(3):
                columns.append(f"{prefix}_{i}{j}")
    columns.extend(
        [
            "video_mode",
            "reuse_mode",
            "geometry_mode",
            "jitter_meaningful",
            "geometry_recomputed",
            "geometry_update_reason",
            "reprojection_error",
            "inlier_spatial_coverage",
            "feature_runtime_ms_left",
            "feature_runtime_ms_right",
            "matching_runtime_ms",
            "geometry_runtime_ms",
            "H_delta_norm",
            "overlap_area_current",
            "seam_policy",
            "seam_recomputed",
            "foreground_ratio",
            "trigger_armed",
            "overlap_diff_before",
            "overlap_diff_after",
            "seam_band_ratio",
            "seam_band_illuminance_diff",
            "seam_band_gradient_disagreement",
            "seam_band_flicker",
            "stitched_delta_mean",
            "seam_mask_change_ratio",
            "crop_applied",
            "crop_method",
            "crop_lir_x",
            "crop_lir_y",
            "crop_lir_w",
            "crop_lir_h",
        ]
    )
    columns.extend(["runtime_ms", "note"])
    return columns


# --- Main pipeline ---


def main() -> int:
    args = _build_parser().parse_args()
    try:
        video_mode_effective, geometry_mode, geometry_mode_source, mode_override_warning = _resolve_execution_mode(
            args.geometry_mode,
            args.video_mode,
        )
    except (NotImplementedError, ValueError) as exc:
        raise SystemExit(str(exc))
    mode_semantics = _describe_mode_semantics(
        geometry_mode,
        video_mode_effective,
        args.reuse_mode,
        geometry_mode_source,
    )
    geometry_mode = str(mode_semantics["geometry_mode"])
    jitter_meaningful = int(mode_semantics["jitter_meaningful"])

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "OpenCV (cv2) and numpy are required for video stitching."
        ) from exc

    from stitching.io import get_pair, load_pairs, open_pair  # noqa: E402
    from stitching.frame_pair_pipeline import (  # noqa: E402
        estimate_frame_pair_geometry,
        resolve_feature_backend,
        resolve_geometry_backend,
        resolve_matcher_backend,
        resolve_optional_backend,
    )
    from stitching.seam_policy import (  # noqa: E402
        decide_seam_update,
        resolve_seam_keyframe_every,
        resolve_seam_policy,
    )
    from stitching.foreground import (  # noqa: E402
        apply_protect_mask_assignment,
        compute_disagreement_mask,
        compute_mask_ratio,
    )
    from stitching.geometry import (  # noqa: E402
        compute_canvas_and_transform,
        warp_pair,
    )
    from stitching.temporal import (  # noqa: E402
        HomographySmoother,
        SeamMaskSmoother,
        compute_frame_absdiff_mean,
        compute_jitter,
        compute_mask_change_ratio,
        transform_corners,
    )
    from stitching.seam_opencv import (  # noqa: E402
        compute_seam_masks_opencv,
        compute_seam_scale,
        overlap_absdiff_preview,
        place_mask_on_canvas,
        place_roi_on_canvas,
        resize_seam_to_compose,
        scale_homography,
        seam_overlay_preview,
        summarize_overlap,
        warp_to_roi,
    )
    from stitching.cropper import Cropper  # noqa: E402
    from stitching.video_stitcher import VideoStitcher, _compute_seam_band_metrics  # noqa: E402
    from stitching.viz import overlay_images  # noqa: E402

    start_time = time.perf_counter()

    keyframe_every = max(1, int(args.keyframe_every))
    geometry_keyframe_every_effective = _effective_geometry_keyframe_every(geometry_mode, keyframe_every)
    stride = max(1, int(args.stride))
    snapshot_every = max(1, int(args.snapshot_every))
    seam_policy_effective = resolve_seam_policy(args.seam_policy, video_mode_effective, args.reuse_mode)
    seam_keyframe_every_effective = resolve_seam_keyframe_every(
        args.seam_policy,
        int(args.seam_keyframe_every),
        keyframe_every,
        int(video_mode_effective),
        args.reuse_mode,
    )
    feature_backend = resolve_feature_backend(args.feature, args.feature_backend)
    matcher_backend = resolve_matcher_backend(args.matcher_backend)
    geometry_backend = resolve_geometry_backend(args.geometry_backend)
    feature_fallback_backend = resolve_optional_backend(args.feature_fallback_backend)
    matcher_fallback_backend = resolve_optional_backend(args.matcher_fallback_backend)

    run_id = args.run_id
    if run_id is None:
        ts = time.strftime("%Y%m%d-%H%M")
        safe_pair = args.pair.replace("/", "_").replace(" ", "_")
        run_id = f"{ts}_{safe_pair}_{args.feature}"

    output_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / "outputs" / "runs" / run_id
    )
    log_path = output_dir / "logs.txt"
    _setup_logging(log_path)

    debug: Dict = {
        "pair_id": args.pair,
        "dataset": None,
        "input_type": None,
        "params": {
            "start": args.start,
            "max_frames": args.max_frames,
            "stride": stride,
            "keyframe_every": keyframe_every,
            "geometry_keyframe_every_effective": int(geometry_keyframe_every_effective),
            "geometry_mode_requested": args.geometry_mode,
            "geometry_mode_source": geometry_mode_source,
            "adaptive_update_strategy": mode_semantics.get("adaptive_update_strategy"),
            "feature": args.feature,
            "feature_backend": feature_backend,
            "matcher_backend": matcher_backend,
            "geometry_backend": geometry_backend,
            "device": args.device,
            "force_cpu": bool(args.force_cpu),
            "weights_dir": args.weights_dir,
            "max_keypoints": int(args.max_keypoints),
            "resize_long_edge": args.resize_long_edge,
            "depth_confidence": args.depth_confidence,
            "width_confidence": args.width_confidence,
            "filter_threshold": args.filter_threshold,
            "feature_fallback_backend": feature_fallback_backend,
            "matcher_fallback_backend": matcher_fallback_backend,
            "nfeatures": args.nfeatures,
            "ratio": args.ratio,
            "min_matches": args.min_matches,
            "ransac_thresh": args.ransac_thresh,
            "blend": args.blend,
            "mb_levels": args.mb_levels,
            "seam": args.seam,
            "seam_megapix": args.seam_megapix,
            "seam_dilate": args.seam_dilate,
            "seam_smooth": args.seam_smooth,
            "seam_smooth_alpha": args.seam_smooth_alpha,
            "seam_smooth_window": int(args.seam_smooth_window),
            "seam_policy": args.seam_policy,
            "seam_policy_effective": seam_policy_effective,
            "seam_keyframe_every": int(args.seam_keyframe_every),
            "seam_keyframe_every_effective": int(seam_keyframe_every_effective),
            "seam_trigger_overlap_ratio": float(args.seam_trigger_overlap_ratio),
            "seam_trigger_diff_threshold": float(args.seam_trigger_diff_threshold),
            "seam_trigger_foreground_ratio": float(args.seam_trigger_foreground_ratio),
            "seam_trigger_cooldown_frames": int(args.seam_trigger_cooldown_frames),
            "seam_trigger_hysteresis_ratio": float(args.seam_trigger_hysteresis_ratio),
            "seam_snapshot_on_recompute": int(args.seam_snapshot_on_recompute),
            "foreground_mode": args.foreground_mode,
            "foreground_diff_threshold": float(args.foreground_diff_threshold),
            "foreground_dilate": int(args.foreground_dilate),
            "crop": bool(args.crop),
            "lir_method": args.lir_method,
            "lir_erode": int(args.lir_erode),
            "crop_debug": int(args.crop_debug),
            "video_mode_requested": args.video_mode,
            "video_mode": int(video_mode_effective),
            "reuse_mode": args.reuse_mode,
            "geometry_mode": geometry_mode,
            "jitter_meaningful": jitter_meaningful,
            "reinit_every": int(args.reinit_every),
            "reinit_on_low_overlap_ratio": float(args.reinit_on_low_overlap_ratio),
            "smooth_h": args.smooth_h,
            "smooth_alpha": args.smooth_alpha,
            "smooth_window": args.smooth_window,
            "fps_cli": args.fps,
            "snapshot_every": snapshot_every,
        },
        "fps": None,
        "fps_source": None,
        "fps_candidates": [],
        "warnings": [],
        "length_left": None,
        "length_right": None,
        "min_length": None,
        "total_frames": None,
        "processed_frames": 0,
        "success_frames": 0,
        "fallback_frames": 0,
        "initial_failure": None,
        "canvas_size": None,
        "smooth_h": args.smooth_h,
        "smooth_alpha": args.smooth_alpha,
        "smooth_window": args.smooth_window,
        "seam": args.seam,
        "seam_megapix": args.seam_megapix,
        "seam_dilate": args.seam_dilate,
        "seam_smooth": args.seam_smooth,
        "seam_smooth_alpha": args.seam_smooth_alpha,
        "seam_smooth_window": int(args.seam_smooth_window),
        "seam_policy": seam_policy_effective,
        "seam_keyframe_every_effective": int(seam_keyframe_every_effective),
        "seam_trigger_overlap_ratio": float(args.seam_trigger_overlap_ratio),
        "seam_trigger_diff_threshold": float(args.seam_trigger_diff_threshold),
        "seam_trigger_foreground_ratio": float(args.seam_trigger_foreground_ratio),
        "seam_trigger_cooldown_frames": int(args.seam_trigger_cooldown_frames),
        "seam_trigger_hysteresis_ratio": float(args.seam_trigger_hysteresis_ratio),
        "seam_snapshot_on_recompute": int(args.seam_snapshot_on_recompute),
        "seam_scale": None,
        "seam_recompute_count": 0,
        "seam_snapshot_count": 0,
        "seam_policy_events": [],
        "foreground_mode": args.foreground_mode,
        "foreground_diff_threshold": float(args.foreground_diff_threshold),
        "foreground_dilate": int(args.foreground_dilate),
        "foreground_ratio_mean": 0.0,
        "foreground_triggered_count": 0,
        "crop_enabled": bool(args.crop),
        "lir_method": args.lir_method,
        "lir_erode": int(args.lir_erode),
        "crop_debug": int(args.crop_debug),
        "crop_fallback_to_no_crop": False,
        "output_crop_rect": None,
        "crop_keyframe_stats": [],
        "video_mode": int(video_mode_effective),
        "video_mode_requested": args.video_mode,
        "reuse_mode": args.reuse_mode,
        "geometry_mode": geometry_mode,
        "geometry_mode_requested": args.geometry_mode,
        "geometry_mode_source": geometry_mode_source,
        "geometry_keyframe_every_effective": int(geometry_keyframe_every_effective),
        "jitter_meaningful": jitter_meaningful,
        "adaptive_update_strategy": mode_semantics.get("adaptive_update_strategy"),
        "geometry_update_count": 0,
        "geometry_update_events": [],
        "mode_semantics": mode_semantics,
        "reinit_every": int(args.reinit_every),
        "reinit_on_low_overlap_ratio": float(args.reinit_on_low_overlap_ratio),
        "init_frame_index": None,
        "overlap_area_init": 0,
        "overlap_area_current": 0,
        "overlap_area_samples": [],
        "reinit_count": 0,
        "time_breakdown_ms": {"init_ms": [], "per_frame_ms": []},
        "seam_keyframe_stats": [],
        "jitter_summary": {},
        "temporal_eval_summary": {},
        "notes": [],
        "errors": [],
        "runtime_ms": None,
        "feature_backend": feature_backend,
        "matcher_backend": matcher_backend,
        "geometry_backend": geometry_backend,
        "feature_backend_effective": feature_backend,
        "matcher_backend_effective": matcher_backend,
        "geometry_backend_effective": geometry_backend,
        "backend_fallback_events": [],
        "last_feature_stage": {},
        "last_matching_stage": {},
        "last_geometry_stage": {},
    }
    if mode_override_warning:
        _warn_and_record(debug, mode_override_warning)

    transforms_path = output_dir / "transforms.csv"
    metrics_path = output_dir / "metrics_preview.json"
    debug_path = output_dir / "debug.json"
    jitter_path = output_dir / "jitter_timeseries.csv"
    video_path = output_dir / "stitched.mp4"

    try:
        pairs = load_pairs(repo_root / args.manifest)
        pair_cfg = get_pair(pairs, args.pair)
        debug["dataset"] = pair_cfg.dataset
        debug["input_type"] = pair_cfg.input_type
    except Exception as exc:
        debug["errors"].append(f"load_pairs: {exc}")
        debug["runtime_ms"] = int((time.perf_counter() - start_time) * 1000)
        _write_json(debug_path, debug)
        logging.error("Failed to load pair: %s", exc)
        return 1

    left_source, right_source = open_pair(pair_cfg)

    try:
        length_left = left_source.length()
        length_right = right_source.length()
        debug["length_left"] = length_left
        debug["length_right"] = length_right
        if length_left is not None and length_right is not None:
            min_length = min(length_left, length_right)
            debug["min_length"] = min_length
        else:
            min_length = None

        skipped = _skip_frames(left_source, right_source, max(0, int(args.start)))
        if skipped < max(0, int(args.start)):
            raise RuntimeError(
                f"Unable to skip to start={args.start}; stream ended at {skipped}."
            )
    except Exception as exc:
        left_source.close()
        right_source.close()
        debug["errors"].append(f"start_skip: {exc}")
        debug["runtime_ms"] = int((time.perf_counter() - start_time) * 1000)
        _write_json(debug_path, debug)
        logging.error("Failed during start skip: %s", exc)
        return 1

    fps_value, fps_source, fps_candidates = _resolve_fps(
        pair_cfg,
        left_source,
        right_source,
        args.fps,
    )
    debug["fps"] = fps_value
    debug["fps_source"] = fps_source
    debug["fps_candidates"] = fps_candidates
    if fps_source == "fallback_30":
        warning = "FPS fallback to 30.0; provide meta.fps or --fps for accuracy."
        debug["warnings"].append(warning)
        logging.warning(warning)

    if min_length is not None:
        available = max(0, min_length - int(args.start))
        debug["total_frames"] = (
            min(available, args.max_frames) if args.max_frames else available
        )

    columns = _build_transform_columns()

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_H_raw = None
    canvas_size: Optional[Tuple[int, int]] = None
    T = None
    writer = None
    output_crop_rect: Optional[Tuple[int, int, int, int]] = None

    last_stats = {
        "n_kp_left": None,
        "n_kp_right": None,
        "n_matches_raw": None,
        "n_matches_good": None,
        "n_inliers": None,
        "inlier_ratio": None,
        "reprojection_error": None,
        "inlier_spatial_coverage": None,
        "feature_runtime_ms_left": None,
        "feature_runtime_ms_right": None,
        "matching_runtime_ms": None,
        "geometry_runtime_ms": None,
    }

    def _update_last_stats_from_pair_result(pair_result) -> None:
        feature_left = getattr(pair_result, "feature_left", None)
        feature_right = getattr(pair_result, "feature_right", None)
        match_result = getattr(pair_result, "match_result", None)
        geometry_result = getattr(pair_result, "geometry_result", None)
        last_stats["n_kp_left"] = (
            int(feature_left.n_keypoints) if feature_left is not None else None
        )
        last_stats["n_kp_right"] = (
            int(feature_right.n_keypoints) if feature_right is not None else None
        )
        last_stats["n_matches_raw"] = (
            int(match_result.tentative_count) if match_result is not None else None
        )
        last_stats["n_matches_good"] = (
            int(match_result.good_count) if match_result is not None else None
        )
        last_stats["n_inliers"] = (
            int(geometry_result.inlier_count)
            if geometry_result is not None and geometry_result.inlier_mask is not None
            else None
        )
        last_stats["inlier_ratio"] = (
            float(geometry_result.inlier_ratio)
            if geometry_result is not None and geometry_result.inlier_mask is not None
            else None
        )
        last_stats["reprojection_error"] = (
            float(geometry_result.reprojection_error)
            if geometry_result is not None and geometry_result.reprojection_error is not None
            else None
        )
        last_stats["inlier_spatial_coverage"] = (
            float(geometry_result.inlier_spatial_coverage)
            if geometry_result is not None and geometry_result.inlier_spatial_coverage is not None
            else None
        )
        feature_stage = getattr(pair_result, "feature_stage", {}) or {}
        matching_stage = getattr(pair_result, "matching_stage", {}) or {}
        geometry_stage = getattr(pair_result, "geometry_stage", {}) or {}
        last_stats["feature_runtime_ms_left"] = (
            float(feature_stage.get("left", {}).get("runtime_ms"))
            if feature_stage.get("left", {}).get("runtime_ms") is not None
            else None
        )
        last_stats["feature_runtime_ms_right"] = (
            float(feature_stage.get("right", {}).get("runtime_ms"))
            if feature_stage.get("right", {}).get("runtime_ms") is not None
            else None
        )
        last_stats["matching_runtime_ms"] = (
            float(matching_stage.get("runtime_ms"))
            if matching_stage.get("runtime_ms") is not None
            else None
        )
        last_stats["geometry_runtime_ms"] = (
            float(geometry_stage.get("runtime_ms"))
            if geometry_stage.get("runtime_ms") is not None
            else None
        )

    def _record_pair_result_debug(pair_result, frame_idx: int) -> None:
        if getattr(pair_result, "feature_backend_effective", None):
            debug["feature_backend_effective"] = pair_result.feature_backend_effective
        if getattr(pair_result, "matcher_backend_effective", None):
            debug["matcher_backend_effective"] = pair_result.matcher_backend_effective
        if getattr(pair_result, "geometry_backend_effective", None):
            debug["geometry_backend_effective"] = pair_result.geometry_backend_effective
        if getattr(pair_result, "feature_stage", None):
            debug["last_feature_stage"] = pair_result.feature_stage
        if getattr(pair_result, "matching_stage", None):
            debug["last_matching_stage"] = pair_result.matching_stage
        if getattr(pair_result, "geometry_stage", None):
            debug["last_geometry_stage"] = pair_result.geometry_stage
        for event in getattr(pair_result, "fallback_events", []) or []:
            enriched = dict(event)
            enriched["frame_idx"] = int(frame_idx)
            debug["backend_fallback_events"].append(enriched)

    def _estimate_geometry_for_current_frame(frame_idx: int, left, right):
        matches_img = None
        pair_result = estimate_frame_pair_geometry(
            left,
            right,
            feature=args.feature,
            feature_backend=feature_backend,
            matcher_backend=matcher_backend,
            geometry_backend=geometry_backend,
            nfeatures=args.nfeatures,
            ratio=args.ratio,
            min_matches=args.min_matches,
            ransac_thresh=args.ransac_thresh,
            device=args.device,
            force_cpu=args.force_cpu,
            weights_dir=args.weights_dir,
            max_keypoints=args.max_keypoints,
            resize_long_edge=args.resize_long_edge,
            depth_confidence=args.depth_confidence,
            width_confidence=args.width_confidence,
            filter_threshold=args.filter_threshold,
            feature_fallback_backend=feature_fallback_backend,
            matcher_fallback_backend=matcher_fallback_backend,
        )
        _update_last_stats_from_pair_result(pair_result)
        _record_pair_result_debug(pair_result, frame_idx)
        matches_img = pair_result.matches_img
        if not pair_result.ok:
            raise RuntimeError(pair_result.message or pair_result.status)
        _save_keyframe_debug(
            output_dir,
            frame_idx,
            pair_result.matches_img,
            pair_result.inliers_img,
        )
        inlier_count = int(pair_result.geometry_result.inlier_count)
        inlier_ratio = float(pair_result.geometry_result.inlier_ratio)
        inliers_ok.append(inlier_count)
        inlier_ratios_ok.append(inlier_ratio)
        _record_geometry_event_metrics(pair_result)
        return pair_result, pair_result.geometry_result.H, matches_img

    smoother = HomographySmoother(
        method=args.smooth_h,
        alpha=args.smooth_alpha,
        window=args.smooth_window,
    )
    smoother.reset()

    inliers_ok: List[int] = []
    inlier_ratios_ok: List[float] = []
    frame_runtimes: List[float] = []
    jitter_raw_values: List[float] = []
    jitter_sm_values: List[float] = []
    seam_keyframe_ms: List[float] = []
    overlap_diff_before_values: List[float] = []
    overlap_diff_after_values: List[float] = []
    seam_mask_change_values: List[float] = []
    stitched_delta_values: List[float] = []
    reprojection_error_values: List[float] = []
    inlier_spatial_coverage_values: List[float] = []
    feature_runtime_left_values: List[float] = []
    feature_runtime_right_values: List[float] = []
    matching_runtime_values: List[float] = []
    geometry_runtime_values: List[float] = []
    seam_band_ratio_values: List[float] = []
    seam_band_illuminance_values: List[float] = []
    seam_band_gradient_values: List[float] = []
    seam_band_flicker_values: List[float] = []
    foreground_ratio_values: List[float] = []
    prev_raw_corners = None
    prev_sm_corners = None
    prev_stitched_written = None
    prev_seam_band_mask = None
    seam_cache: Optional[Dict[str, object]] = None
    video_stitcher = VideoStitcher(
        seam_method=_seam_cli_to_method(args.seam),
        seam_megapix=args.seam_megapix,
        seam_dilate=args.seam_dilate,
        seam_smooth_method=args.seam_smooth,
        seam_smooth_alpha=args.seam_smooth_alpha,
        seam_smooth_window=args.seam_smooth_window,
        blend_mode=args.blend,
        mb_levels=args.mb_levels,
        crop_enabled=bool(args.crop),
        lir_method=args.lir_method,
        lir_erode=args.lir_erode,
        crop_debug=args.crop_debug,
        reuse_mode=args.reuse_mode,
        output_dir=output_dir,
        warning_handler=lambda msg: _warn_and_record(debug, msg),
    )
    seam_mask_smoother = SeamMaskSmoother(
        method=args.seam_smooth,
        alpha=args.seam_smooth_alpha,
        window=args.seam_smooth_window,
    )
    video_prev_H = None
    frames_since_video_init = 0

    def _record_geometry_event_metrics(pair_result) -> None:
        geometry_result = getattr(pair_result, "geometry_result", None)
        feature_stage = getattr(pair_result, "feature_stage", {}) or {}
        matching_stage = getattr(pair_result, "matching_stage", {}) or {}
        geometry_stage = getattr(pair_result, "geometry_stage", {}) or {}
        if geometry_result is not None and geometry_result.reprojection_error is not None:
            reprojection_error_values.append(float(geometry_result.reprojection_error))
        if geometry_result is not None and geometry_result.inlier_spatial_coverage is not None:
            inlier_spatial_coverage_values.append(float(geometry_result.inlier_spatial_coverage))
        if feature_stage.get("left", {}).get("runtime_ms") is not None:
            feature_runtime_left_values.append(float(feature_stage["left"]["runtime_ms"]))
        if feature_stage.get("right", {}).get("runtime_ms") is not None:
            feature_runtime_right_values.append(float(feature_stage["right"]["runtime_ms"]))
        if matching_stage.get("runtime_ms") is not None:
            matching_runtime_values.append(float(matching_stage["runtime_ms"]))
        if geometry_stage.get("runtime_ms") is not None:
            geometry_runtime_values.append(float(geometry_stage["runtime_ms"]))

    processed_idx = 0
    source_idx = int(args.start)

    try:
        with transforms_path.open("w", newline="", encoding="utf-8") as f_csv, jitter_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f_jitter:
            writer_csv = csv.DictWriter(f_csv, fieldnames=columns)
            writer_csv.writeheader()

            jitter_writer = csv.DictWriter(
                f_jitter,
                fieldnames=["frame_idx", "jitter_raw", "jitter_sm", "status"],
            )
            jitter_writer.writeheader()

            while True:
                if args.max_frames is not None and processed_idx >= args.max_frames:
                    break
                if (
                    debug["total_frames"] is not None
                    and processed_idx >= debug["total_frames"]
                ):
                    break

                left = left_source.read_next()
                right = right_source.read_next()
                if left is None or right is None:
                    break

                frame_start = time.perf_counter()
                is_keyframe = (
                    1 if (processed_idx == 0 or processed_idx % keyframe_every == 0) else 0
                )
                status = "OK"
                note_parts: List[str] = []
                geometry_recomputed = False
                geometry_update_reason = "reuse"

                if int(video_mode_effective) == 1:
                    need_init = False
                    reinit_reason = ""
                    overlap_init = int(video_stitcher.state.metadata.get("overlap_area_init", 0) or 0)
                    overlap_curr_prev = int(
                        video_stitcher.state.metadata.get("overlap_area_current", overlap_init) or 0
                    )
                    if not video_stitcher.state.initialized:
                        need_init = True
                        reinit_reason = "frame0"
                    elif int(args.reinit_every) > 0 and frames_since_video_init >= int(args.reinit_every):
                        need_init = True
                        reinit_reason = f"reinit_every={int(args.reinit_every)}"
                    elif (
                        float(args.reinit_on_low_overlap_ratio) > 0.0
                        and overlap_init > 0
                        and overlap_curr_prev
                        < float(args.reinit_on_low_overlap_ratio) * float(overlap_init)
                    ):
                        need_init = True
                        reinit_reason = (
                            "low_overlap="
                            f"{overlap_curr_prev}/{overlap_init}"
                            f"<{float(args.reinit_on_low_overlap_ratio):.3f}"
                        )

                    if need_init:
                        init_t0 = time.perf_counter()
                        is_keyframe = 1
                        matches_img = None
                        H_seed = None
                        try:
                            pair_result, H_seed, matches_img = _estimate_geometry_for_current_frame(
                                source_idx,
                                left,
                                right,
                            )
                        except Exception as init_exc:
                            if matches_img is not None:
                                _save_keyframe_debug(output_dir, source_idx, matches_img, None)
                            _warn_and_record(
                                debug,
                                f"video_init_fallback frame={source_idx}: {init_exc}",
                            )
                            if video_stitcher.state.initialized and video_stitcher.state.H_or_cameras is not None:
                                H_seed = video_stitcher.state.H_or_cameras
                                status = "FALLBACK"
                            elif valid_H_raw is not None:
                                H_seed = valid_H_raw
                                status = "FALLBACK"
                            else:
                                H_seed = np.eye(3, dtype=float)
                                status = "FAIL_INIT" if processed_idx == 0 else "FAIL_EST"
                            if processed_idx == 0:
                                debug["initial_failure"] = str(init_exc)

                        if args.reuse_mode == "emaH":
                            H_use_init = smoother.update(
                                H_seed,
                                image_size=(right.shape[1], right.shape[0]),
                            )
                        else:
                            H_use_init = H_seed
                            smoother.reset()

                        valid_H_raw = H_use_init
                        if canvas_size is None or T is None:
                            canvas_size, T = compute_canvas_and_transform(
                                (left.shape[0], left.shape[1]),
                                (right.shape[0], right.shape[1]),
                                H_use_init,
                            )
                            debug["canvas_size"] = [int(canvas_size[0]), int(canvas_size[1])]

                        if video_stitcher.state.initialized:
                            debug["reinit_count"] = int(debug.get("reinit_count", 0)) + 1
                        init_out = video_stitcher.initialize_from_first_frame(
                            left,
                            right,
                            H_use_init,
                            T,
                            canvas_size,
                            source_idx,
                        )
                        debug["init_frame_index"] = int(source_idx)
                        debug["overlap_area_init"] = int(
                            video_stitcher.state.metadata.get("overlap_area_init", 0)
                        )
                        debug["seam_scale"] = float(
                            video_stitcher.state.metadata.get("seam_scale", 0.0)
                        )
                        debug["time_breakdown_ms"]["init_ms"].append(float(init_out.get("init_ms", 0.0)))
                        debug["notes"].append(f"video_init frame={source_idx} reason={reinit_reason}")
                        debug["seam_recompute_count"] = int(debug.get("seam_recompute_count", 0)) + 1
                        frames_since_video_init = 0
                        geometry_recomputed = True
                        geometry_update_reason = reinit_reason

                    if not video_stitcher.state.initialized:
                        raise RuntimeError("video_mode enabled but stitcher state is not initialized")

                    H_video = video_stitcher.state.H_or_cameras
                    if args.reuse_mode == "emaH":
                        H_video = smoother.update(
                            H_video,
                            image_size=(right.shape[1], right.shape[0]),
                        )
                    seam_keyframe_due = bool(
                        seam_keyframe_every_effective > 0
                        and processed_idx > 0
                        and processed_idx % seam_keyframe_every_effective == 0
                    )
                    stitch_out = video_stitcher.stitch_frame(
                        left,
                        right,
                        H_video,
                        T,
                        canvas_size,
                        source_idx,
                        recompute_seam=False,
                        seam_policy=seam_policy_effective,
                        seam_keyframe=seam_keyframe_due,
                        seam_trigger_overlap_ratio=args.seam_trigger_overlap_ratio,
                        seam_trigger_diff_threshold=args.seam_trigger_diff_threshold,
                        seam_trigger_foreground_ratio=args.seam_trigger_foreground_ratio,
                        seam_trigger_cooldown_frames=args.seam_trigger_cooldown_frames,
                        seam_trigger_hysteresis_ratio=args.seam_trigger_hysteresis_ratio,
                        foreground_mode=args.foreground_mode,
                        foreground_diff_threshold=args.foreground_diff_threshold,
                        foreground_dilate=args.foreground_dilate,
                        save_seam_event_snapshots=bool(int(args.seam_snapshot_on_recompute)),
                    )
                    stitched = stitch_out["stitched"]
                    stitched_written = stitched
                    overlap_current = int(stitch_out.get("overlap_area", 0))
                    overlap_diff_before = float(stitch_out.get("overlap_diff_before", 0.0))
                    overlap_diff_after = float(stitch_out.get("overlap_diff_after", 0.0))
                    seam_recomputed = bool(stitch_out.get("seam_recomputed", False))
                    seam_trigger_reason = str(stitch_out.get("seam_trigger_reason", "none"))
                    seam_trigger_flags = dict(stitch_out.get("seam_trigger_flags", {}) or {})
                    seam_mask_change_ratio = float(stitch_out.get("seam_mask_change_ratio", 0.0))
                    seam_policy_used = str(stitch_out.get("seam_policy", seam_policy_effective))
                    foreground_ratio = float(stitch_out.get("foreground_ratio", 0.0))
                    foreground_protect_ratio = float(stitch_out.get("foreground_protect_ratio", 0.0))
                    seam_band_ratio = float(stitch_out.get("seam_band_ratio", 0.0))
                    seam_band_illuminance_diff = float(
                        stitch_out.get("seam_band_illuminance_diff", 0.0)
                    )
                    seam_band_gradient_disagreement = float(
                        stitch_out.get("seam_band_gradient_disagreement", 0.0)
                    )
                    seam_band_mask = stitch_out.get("seam_band_mask")
                    trigger_armed_next = int(bool(stitch_out.get("trigger_armed_next", True)))
                    trigger_states_next = dict(stitch_out.get("trigger_states_next", {}) or {})
                    proposed_output_bbox = stitch_out.get("output_bbox")

                    if geometry_mode == "adaptive_update" and seam_recomputed:
                        adaptive_reason = seam_trigger_reason
                        previous_final_masks = list(video_stitcher.state.seam_masks_final or [])
                        try:
                            pair_result, H_seed, _ = _estimate_geometry_for_current_frame(
                                source_idx,
                                left,
                                right,
                            )
                            if args.reuse_mode == "emaH":
                                H_use_update = smoother.update(
                                    H_seed,
                                    image_size=(right.shape[1], right.shape[0]),
                                )
                            else:
                                H_use_update = H_seed
                                smoother.reset()

                            valid_H_raw = H_use_update
                            if video_stitcher.state.initialized:
                                debug["reinit_count"] = int(debug.get("reinit_count", 0)) + 1
                            init_out = video_stitcher.initialize_from_first_frame(
                                left,
                                right,
                                H_use_update,
                                T,
                                canvas_size,
                                source_idx,
                            )
                            debug["time_breakdown_ms"]["init_ms"].append(
                                float(init_out.get("init_ms", 0.0))
                            )
                            debug["geometry_update_count"] = int(debug.get("geometry_update_count", 0)) + 1
                            debug["geometry_update_events"].append(
                                {
                                    "frame_idx": int(source_idx),
                                    "reason": adaptive_reason,
                                    "strategy": "seam_driven_geometry_refresh",
                                    "seam_policy": seam_policy_used,
                                    "status": "updated",
                                    "init_ms": float(init_out.get("init_ms", 0.0)),
                                    "inlier_count": int(pair_result.geometry_result.inlier_count),
                                    "inlier_ratio": float(pair_result.geometry_result.inlier_ratio),
                                    "reprojection_error": (
                                        float(pair_result.geometry_result.reprojection_error)
                                        if pair_result.geometry_result.reprojection_error is not None
                                        else None
                                    ),
                                }
                            )
                            debug["notes"].append(
                                f"adaptive_geometry_update frame={source_idx} reason={adaptive_reason}"
                            )
                            debug["init_frame_index"] = int(source_idx)
                            debug["overlap_area_init"] = int(
                                video_stitcher.state.metadata.get("overlap_area_init", 0)
                            )
                            debug["seam_scale"] = float(
                                video_stitcher.state.metadata.get("seam_scale", 0.0)
                            )
                            video_stitcher.state.metadata["seam_trigger_armed"] = bool(trigger_armed_next)
                            video_stitcher.state.metadata["seam_trigger_states"] = dict(trigger_states_next)
                            frames_since_video_init = 0
                            is_keyframe = 1
                            geometry_recomputed = True
                            geometry_update_reason = f"adaptive:{adaptive_reason}"
                            H_video = video_stitcher.state.H_or_cameras
                            stitched = init_out["stitched"]
                            stitched_written = stitched
                            overlap_current = int(init_out.get("overlap_area", overlap_current))
                            overlap_diff_before = float(
                                init_out.get("overlap_diff_before", overlap_diff_before)
                            )
                            overlap_diff_after = float(
                                init_out.get("overlap_diff_after", overlap_diff_after)
                            )
                            seam_band_ratio = float(init_out.get("seam_band_ratio", seam_band_ratio))
                            seam_band_illuminance_diff = float(
                                init_out.get(
                                    "seam_band_illuminance_diff",
                                    seam_band_illuminance_diff,
                                )
                            )
                            seam_band_gradient_disagreement = float(
                                init_out.get(
                                    "seam_band_gradient_disagreement",
                                    seam_band_gradient_disagreement,
                                )
                            )
                            seam_band_mask = init_out.get("seam_band_mask", seam_band_mask)
                            proposed_output_bbox = init_out.get("output_bbox")
                            current_final_masks = list(video_stitcher.state.seam_masks_final or [])
                            if len(previous_final_masks) == 2 and len(current_final_masks) == 2:
                                mask_changes = [
                                    compute_mask_change_ratio(
                                        previous_final_masks[0],
                                        current_final_masks[0],
                                    ),
                                    compute_mask_change_ratio(
                                        previous_final_masks[1],
                                        current_final_masks[1],
                                    ),
                                ]
                                valid_mask_changes = [
                                    float(value) for value in mask_changes if value is not None
                                ]
                                if valid_mask_changes:
                                    seam_mask_change_ratio = float(
                                        sum(valid_mask_changes) / len(valid_mask_changes)
                                    )
                            seam_trigger_reason = f"{adaptive_reason}->adaptive_geometry_refresh"
                        except Exception as adaptive_exc:
                            _warn_and_record(
                                debug,
                                f"adaptive_geometry_update_fallback frame={source_idx}: {adaptive_exc}",
                            )
                            debug["geometry_update_events"].append(
                                {
                                    "frame_idx": int(source_idx),
                                    "reason": adaptive_reason,
                                    "strategy": "seam_driven_geometry_refresh",
                                    "seam_policy": seam_policy_used,
                                    "status": "fallback",
                                    "error": str(adaptive_exc),
                                }
                            )
                            if status == "OK":
                                status = "FALLBACK"
                            geometry_recomputed = False
                            geometry_update_reason = f"adaptive_fallback:{adaptive_reason}"

                    overlap_diff_before_values.append(overlap_diff_before)
                    overlap_diff_after_values.append(overlap_diff_after)
                    seam_mask_change_values.append(seam_mask_change_ratio)
                    seam_band_ratio_values.append(float(seam_band_ratio))
                    seam_band_illuminance_values.append(float(seam_band_illuminance_diff))
                    seam_band_gradient_values.append(float(seam_band_gradient_disagreement))
                    foreground_ratio_values.append(float(foreground_ratio))
                    debug["overlap_area_current"] = int(overlap_current)
                    debug["overlap_area_samples"].append(
                        {"frame_idx": int(source_idx), "overlap_area_current": overlap_current}
                    )
                    if bool(seam_trigger_flags.get("foreground_triggered")):
                        debug["foreground_triggered_count"] = int(debug.get("foreground_triggered_count", 0)) + 1
                    if seam_recomputed:
                        debug["seam_recompute_count"] = int(debug.get("seam_recompute_count", 0)) + 1
                        debug["seam_policy_events"].append(
                            {
                                "frame_idx": int(source_idx),
                                "policy": seam_policy_used,
                                "reason": seam_trigger_reason,
                                "overlap_area_current": int(overlap_current),
                                "overlap_diff_before": overlap_diff_before,
                                "overlap_diff_after": overlap_diff_after,
                                "foreground_ratio": float(foreground_ratio),
                                "foreground_protect_ratio": float(foreground_protect_ratio),
                                "trigger_flags": seam_trigger_flags,
                                "trigger_states_next": trigger_states_next,
                                "geometry_recomputed": bool(geometry_recomputed),
                                "geometry_update_reason": geometry_update_reason,
                            }
                        )
                        seam_keyframe_ms.append(float(stitch_out.get("seam_compute_ms", 0.0)))
                        debug["seam_keyframe_stats"].append(
                            {
                                "frame_idx": int(source_idx),
                                "method": _seam_cli_to_method(args.seam),
                                "policy": seam_policy_used,
                                "trigger_reason": seam_trigger_reason,
                                "overlap_area_px": int(overlap_current),
                                "overlap_diff_mean_before": overlap_diff_before,
                                "overlap_diff_mean_after": overlap_diff_after,
                                "runtime_ms_seam_keyframe": float(stitch_out.get("seam_compute_ms", 0.0)),
                                "crop_applied": bool(stitch_out.get("crop_applied")),
                                "geometry_recomputed": bool(geometry_recomputed),
                                "geometry_update_reason": geometry_update_reason,
                            }
                        )
                    if (
                        bool(args.crop)
                        and output_crop_rect is None
                        and proposed_output_bbox is not None
                    ):
                        output_crop_rect = tuple(int(v) for v in proposed_output_bbox)
                        debug["output_crop_rect"] = {
                            "x": int(output_crop_rect[0]),
                            "y": int(output_crop_rect[1]),
                            "w": int(output_crop_rect[2]),
                            "h": int(output_crop_rect[3]),
                        }
                    if bool(args.crop) and output_crop_rect is not None:
                        stitched_written = _crop_frame_to_rect(stitched, output_crop_rect)
                        if seam_band_mask is not None:
                            seam_band_mask = _crop_frame_to_rect(seam_band_mask, output_crop_rect)
                    stitched_delta_mean = compute_frame_absdiff_mean(prev_stitched_written, stitched_written)
                    seam_band_flicker = None
                    flicker_mask = seam_band_mask
                    if prev_seam_band_mask is not None and seam_band_mask is not None:
                        prev_mask_arr = np.asarray(prev_seam_band_mask) > 0
                        curr_mask_arr = np.asarray(seam_band_mask) > 0
                        if prev_mask_arr.shape == curr_mask_arr.shape:
                            flicker_mask = ((prev_mask_arr | curr_mask_arr).astype(np.uint8)) * 255
                    seam_band_flicker = compute_frame_absdiff_mean(
                        prev_stitched_written,
                        stitched_written,
                        mask=flicker_mask,
                    )
                    prev_stitched_written = stitched_written
                    prev_seam_band_mask = seam_band_mask
                    if stitched_delta_mean is not None:
                        stitched_delta_values.append(float(stitched_delta_mean))
                    if seam_band_flicker is not None:
                        seam_band_flicker_values.append(float(seam_band_flicker))

                    if writer is None:
                        out_h, out_w = int(stitched_written.shape[0]), int(stitched_written.shape[1])
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(video_path), fourcc, fps_value, (out_w, out_h))
                        if not writer.isOpened():
                            raise RuntimeError("Failed to open VideoWriter for stitched.mp4")
                    writer.write(stitched_written)

                    H_raw = H_video
                    H_sm = H_video
                    H_active = H_video
                    h_delta = _h_delta_norm(H_active, video_prev_H)
                    video_prev_H = H_active

                    raw_corners = transform_corners(
                        H_raw,
                        image_size=(right.shape[1], right.shape[0]),
                        pre_transform=T,
                    )
                    sm_corners = transform_corners(
                        H_sm,
                        image_size=(right.shape[1], right.shape[0]),
                        pre_transform=T,
                    )
                    jitter_raw_stats = compute_jitter(prev_raw_corners, raw_corners)
                    jitter_sm_stats = compute_jitter(prev_sm_corners, sm_corners)
                    prev_raw_corners = raw_corners
                    prev_sm_corners = sm_corners
                    if jitter_raw_stats.mean is not None:
                        jitter_raw_values.append(jitter_raw_stats.mean)
                    if jitter_sm_stats.mean is not None:
                        jitter_sm_values.append(jitter_sm_stats.mean)
                    jitter_writer.writerow(
                        {
                            "frame_idx": source_idx,
                            "jitter_raw": jitter_raw_stats.mean,
                            "jitter_sm": jitter_sm_stats.mean,
                            "status": status,
                        }
                    )

                    should_save_frame_snapshot = (
                        processed_idx % snapshot_every == 0
                        or (bool(int(args.seam_snapshot_on_recompute)) and bool(seam_recomputed))
                    )
                    if should_save_frame_snapshot:
                        # Skip expensive debug overlays on non-snapshot frames.
                        left_warped_v, right_warped_v = warp_pair(left, right, H_video, canvas_size, T)
                        overlay_raw = overlay_images(left_warped_v, right_warped_v, alpha=0.5)
                        overlay_sm = overlay_raw
                        overlay_active = overlay_raw
                        _save_snapshot(
                            output_dir,
                            source_idx,
                            left,
                            right,
                            stitched_written,
                            overlay_active,
                            overlay_raw,
                            overlay_sm,
                        )
                        if bool(seam_recomputed) and bool(int(args.seam_snapshot_on_recompute)):
                            debug["seam_snapshot_count"] = int(debug.get("seam_snapshot_count", 0)) + 1

                    runtime_ms = (time.perf_counter() - frame_start) * 1000.0
                    frame_runtimes.append(runtime_ms)
                    debug["time_breakdown_ms"]["per_frame_ms"].append(float(runtime_ms))
                    if status == "OK":
                        debug["success_frames"] += 1
                    if status == "FALLBACK":
                        debug["fallback_frames"] += 1

                    note_parts.append(f"video_mode={int(video_mode_effective)}")
                    note_parts.append(f"reuse_mode={args.reuse_mode}")
                    note_parts.append(f"geometry_mode={geometry_mode}")
                    note_parts.append(f"geometry_keyframe_every={geometry_keyframe_every_effective}")
                    note_parts.append(f"jitter_meaningful={jitter_meaningful}")
                    note_parts.append(f"geometry_recomputed={int(geometry_recomputed)}")
                    note_parts.append(f"geometry_update_reason={geometry_update_reason}")
                    note_parts.append(f"H_delta_norm={h_delta:.6f}")
                    note_parts.append(f"overlap_area={overlap_current}")
                    note_parts.append(f"seam_policy={seam_policy_used}")
                    note_parts.append(f"seam_recomputed={int(seam_recomputed)}")
                    note_parts.append(f"seam_reason={seam_trigger_reason}")
                    note_parts.append(f"foreground_ratio={foreground_ratio:.4f}")
                    note_parts.append(f"trigger_armed={trigger_armed_next}")
                    note_parts.append(f"overlap_diff={overlap_diff_before:.2f}->{overlap_diff_after:.2f}")

                    row = {
                        "frame_idx": source_idx,
                        "is_keyframe": int(is_keyframe),
                        "status": status,
                        "n_kp_left": last_stats["n_kp_left"],
                        "n_kp_right": last_stats["n_kp_right"],
                        "n_matches_raw": last_stats["n_matches_raw"],
                        "n_matches_good": last_stats["n_matches_good"],
                        "n_inliers": last_stats["n_inliers"],
                        "inlier_ratio": last_stats["inlier_ratio"],
                        "jitter_raw": jitter_raw_stats.mean,
                        "jitter_raw_max": jitter_raw_stats.max,
                        "jitter_sm": jitter_sm_stats.mean,
                        "jitter_sm_max": jitter_sm_stats.max,
                        "video_mode": int(video_mode_effective),
                        "reuse_mode": args.reuse_mode,
                        "geometry_mode": geometry_mode,
                        "jitter_meaningful": jitter_meaningful,
                        "geometry_recomputed": int(geometry_recomputed),
                        "geometry_update_reason": geometry_update_reason,
                        "reprojection_error": last_stats["reprojection_error"],
                        "inlier_spatial_coverage": last_stats["inlier_spatial_coverage"],
                        "feature_runtime_ms_left": last_stats["feature_runtime_ms_left"],
                        "feature_runtime_ms_right": last_stats["feature_runtime_ms_right"],
                        "matching_runtime_ms": last_stats["matching_runtime_ms"],
                        "geometry_runtime_ms": last_stats["geometry_runtime_ms"],
                        "H_delta_norm": float(h_delta),
                        "overlap_area_current": int(overlap_current),
                        "seam_policy": seam_policy_used,
                        "seam_recomputed": int(seam_recomputed),
                        "foreground_ratio": float(foreground_ratio),
                        "trigger_armed": int(trigger_armed_next),
                        "overlap_diff_before": float(overlap_diff_before),
                        "overlap_diff_after": float(overlap_diff_after),
                        "seam_band_ratio": float(seam_band_ratio),
                        "seam_band_illuminance_diff": float(seam_band_illuminance_diff),
                        "seam_band_gradient_disagreement": float(
                            seam_band_gradient_disagreement
                        ),
                        "seam_band_flicker": seam_band_flicker,
                        "stitched_delta_mean": stitched_delta_mean,
                        "seam_mask_change_ratio": float(seam_mask_change_ratio),
                        "crop_applied": int(1 if stitch_out.get("crop_applied") else 0),
                        "crop_method": stitch_out.get("crop_method", "none"),
                        "crop_lir_x": (
                            int(stitch_out["crop_rect"]["x"])
                            if stitch_out.get("crop_rect") is not None
                            else None
                        ),
                        "crop_lir_y": (
                            int(stitch_out["crop_rect"]["y"])
                            if stitch_out.get("crop_rect") is not None
                            else None
                        ),
                        "crop_lir_w": (
                            int(stitch_out["crop_rect"]["w"])
                            if stitch_out.get("crop_rect") is not None
                            else None
                        ),
                        "crop_lir_h": (
                            int(stitch_out["crop_rect"]["h"])
                            if stitch_out.get("crop_rect") is not None
                            else None
                        ),
                        "runtime_ms": int(runtime_ms),
                        "note": ";".join(note_parts),
                    }
                    for idx, value in enumerate(_flatten_h(H_active)):
                        row[f"H_{idx // 3}{idx % 3}"] = value
                    for idx, value in enumerate(_flatten_h(H_raw)):
                        row[f"Hraw_{idx // 3}{idx % 3}"] = value
                    for idx, value in enumerate(_flatten_h(H_sm)):
                        row[f"Hsm_{idx // 3}{idx % 3}"] = value
                    writer_csv.writerow(row)

                    processed_idx += 1
                    source_idx += 1
                    frames_since_video_init += 1
                    consumed = _consume_stride(left_source, right_source, stride)
                    source_idx += consumed
                    if consumed < stride - 1:
                        break
                    continue

                H_raw = valid_H_raw
                keyframe_failed = False
                geometry_recomputed = bool(is_keyframe)
                geometry_update_reason = "keyframe" if is_keyframe else "reuse"

                if not is_keyframe:
                    note_parts.append("reuse_last_H")

                if is_keyframe:
                    matches_img = None
                    try:
                        pair_result = estimate_frame_pair_geometry(
                            left,
                            right,
                            feature=args.feature,
                            feature_backend=feature_backend,
                            matcher_backend=matcher_backend,
                            geometry_backend=geometry_backend,
                            nfeatures=args.nfeatures,
                            ratio=args.ratio,
                            min_matches=args.min_matches,
                            ransac_thresh=args.ransac_thresh,
                            device=args.device,
                            force_cpu=args.force_cpu,
                            weights_dir=args.weights_dir,
                            max_keypoints=args.max_keypoints,
                            resize_long_edge=args.resize_long_edge,
                            depth_confidence=args.depth_confidence,
                            width_confidence=args.width_confidence,
                            filter_threshold=args.filter_threshold,
                            feature_fallback_backend=feature_fallback_backend,
                            matcher_fallback_backend=matcher_fallback_backend,
                        )
                        _update_last_stats_from_pair_result(pair_result)
                        _record_pair_result_debug(pair_result, source_idx)
                        matches_img = pair_result.matches_img

                        if not pair_result.ok:
                            raise RuntimeError(pair_result.message or pair_result.status)

                        inlier_count = int(pair_result.geometry_result.inlier_count)
                        inlier_ratio = float(pair_result.geometry_result.inlier_ratio)

                        _save_keyframe_debug(
                            output_dir,
                            source_idx,
                            pair_result.matches_img,
                            pair_result.inliers_img,
                        )

                        valid_H_raw = pair_result.geometry_result.H
                        H_raw = valid_H_raw
                        inliers_ok.append(inlier_count)
                        inlier_ratios_ok.append(inlier_ratio)
                        _record_geometry_event_metrics(pair_result)
                        note_parts.append("estimation=OK")
                    except Exception as exc:
                        keyframe_failed = True
                        note_parts.append(str(exc))
                        if matches_img is not None:
                            _save_keyframe_debug(output_dir, source_idx, matches_img, None)
                        if valid_H_raw is None:
                            valid_H_raw = np.eye(3, dtype=float)
                            H_raw = valid_H_raw
                            status = "FAIL_INIT" if processed_idx == 0 else "FAIL_EST"
                            note_parts.append("fallback=identity")
                        else:
                            H_raw = valid_H_raw
                            status = "FAIL_INIT" if processed_idx == 0 else "FALLBACK"

                if keyframe_failed and processed_idx == 0:
                    debug["initial_failure"] = note_parts[-1] if note_parts else "init_failure"

                if H_raw is None:
                    H_raw = np.eye(3, dtype=float)
                    valid_H_raw = H_raw
                    status = "FAIL_EST"
                    note_parts.append("fallback=identity")

                H_sm = smoother.update(
                    H_raw,
                    image_size=(right.shape[1], right.shape[0]),
                )
                H_active = H_sm if args.smooth_h != "none" else H_raw

                if canvas_size is None or T is None:
                    canvas_size, T = compute_canvas_and_transform(
                        (left.shape[0], left.shape[1]),
                        (right.shape[0], right.shape[1]),
                        H_active,
                    )
                    debug["canvas_size"] = [int(canvas_size[0]), int(canvas_size[1])]

                left_raw_warped, right_raw_warped = warp_pair(left, right, H_raw, canvas_size, T)
                left_sm_warped, right_sm_warped = warp_pair(left, right, H_sm, canvas_size, T)

                if args.smooth_h == "none":
                    left_active = left_raw_warped
                    right_active = right_raw_warped
                else:
                    left_active = left_sm_warped
                    right_active = right_sm_warped

                overlay_raw = overlay_images(left_raw_warped, right_raw_warped, alpha=0.5)
                overlay_sm = overlay_images(left_sm_warped, right_sm_warped, alpha=0.5)
                overlay_active = overlay_sm if args.smooth_h != "none" else overlay_raw

                seam_method = _seam_cli_to_method(args.seam)
                seam_compute_ms = 0.0
                overlap_diff_before = 0.0
                overlap_diff_after = 0.0
                overlap_area_current = 0
                seam_recomputed = False
                seam_trigger_reason = "none"
                seam_trigger_flags: Dict[str, object] = {}
                seam_mask_change_ratio = 0.0
                seam_band_ratio = 0.0
                seam_band_illuminance_diff = 0.0
                seam_band_gradient_disagreement = 0.0
                seam_band_mask = None
                crop_applied = 0
                crop_method_used = "none"
                crop_lir_rect = None
                output_bbox_current = None
                foreground_ratio = 0.0
                trigger_armed_next = 1
                trigger_states_next: Dict[str, object] = {}
                foreground_mask_full = None

                stitched = _blend_frames(
                    left_active,
                    right_active,
                    args.blend,
                    mb_levels=args.mb_levels,
                )

                if seam_method != "none":
                    try:
                        # Build full-resolution warped ROIs with corner metadata.
                        M_left_full = T
                        M_right_full = T @ H_active
                        left_roi_full, left_roi_mask_full, left_corner_full = warp_to_roi(
                            left,
                            M_left_full,
                        )
                        right_roi_full, right_roi_mask_full, right_corner_full = warp_to_roi(
                            right,
                            M_right_full,
                        )

                        left_canvas, left_mask_full = _compose_single_roi_on_canvas(
                            canvas_size,
                            left_roi_full,
                            left_roi_mask_full,
                            left_corner_full,
                        )
                        right_canvas, right_mask_full = _compose_single_roi_on_canvas(
                            canvas_size,
                            right_roi_full,
                            right_roi_mask_full,
                            right_corner_full,
                        )

                        overlap_diff_before = _mean_overlap_diff(
                            left_canvas,
                            right_canvas,
                            left_mask_full,
                            right_mask_full,
                        )
                        overlap_area_current = int(
                            summarize_overlap(left_mask_full, right_mask_full).get("overlap_area", 0)
                        )
                        overlap_mask_full = (
                            (left_mask_full > 0) & (right_mask_full > 0)
                        ).astype(np.uint8) * 255
                        if str(args.foreground_mode).strip().lower() == "disagreement":
                            foreground_mask_full = compute_disagreement_mask(
                                left_canvas,
                                right_canvas,
                                left_mask_full,
                                right_mask_full,
                                diff_threshold=float(args.foreground_diff_threshold),
                                dilate_iter=int(args.foreground_dilate),
                            )
                            foreground_ratio = compute_mask_ratio(
                                foreground_mask_full,
                                overlap_mask_full,
                            )

                        seam_keyframe_due = bool(
                            seam_keyframe_every_effective > 0
                            and processed_idx > 0
                            and processed_idx % seam_keyframe_every_effective == 0
                        )
                        seam_decision = decide_seam_update(
                            policy=seam_policy_effective,
                            seam_cache_available=seam_cache is not None,
                            is_seam_keyframe=seam_keyframe_due,
                            seam_age_frames=(
                                max(0, int(source_idx) - int(seam_cache.get("last_seam_frame_index", source_idx)))
                                if seam_cache is not None
                                else 0
                            ),
                            overlap_area_current=int(overlap_area_current),
                            overlap_area_reference=(
                                int(debug.get("overlap_area_init", 0) or 0)
                                if int(debug.get("overlap_area_init", 0) or 0) > 0
                                else int(overlap_area_current)
                            ),
                            overlap_diff_before=float(overlap_diff_before),
                            trigger_overlap_ratio=float(args.seam_trigger_overlap_ratio),
                            trigger_diff_threshold=float(args.seam_trigger_diff_threshold),
                            foreground_ratio=float(foreground_ratio),
                            trigger_foreground_ratio=float(args.seam_trigger_foreground_ratio),
                            cooldown_frames=int(args.seam_trigger_cooldown_frames),
                            hysteresis_ratio=float(args.seam_trigger_hysteresis_ratio),
                            trigger_armed=bool(seam_cache.get("trigger_armed", True))
                            if seam_cache is not None
                            else True,
                            trigger_states=(
                                seam_cache.get("trigger_states")
                                if seam_cache is not None
                                else None
                            ),
                            force_recompute=False,
                        )
                        seam_recomputed = bool(seam_decision.recompute)
                        seam_trigger_reason = str(seam_decision.reason)
                        seam_trigger_flags = dict(seam_decision.trigger_flags)
                        trigger_armed_next = int(bool(seam_decision.trigger_armed_next))
                        trigger_states_next = dict(seam_decision.trigger_states_next)

                        if seam_decision.recompute:
                            seam_t0 = time.perf_counter()
                            seam_scale = compute_seam_scale(
                                args.seam_megapix,
                                image_shape=(left.shape[0], left.shape[1]),
                            )

                            left_w = max(1, int(round(left.shape[1] * seam_scale)))
                            left_h = max(1, int(round(left.shape[0] * seam_scale)))
                            right_w = max(1, int(round(right.shape[1] * seam_scale)))
                            right_h = max(1, int(round(right.shape[0] * seam_scale)))

                            left_small = cv2.resize(
                                left,
                                (left_w, left_h),
                                interpolation=cv2.INTER_AREA,
                            )
                            right_small = cv2.resize(
                                right,
                                (right_w, right_h),
                                interpolation=cv2.INTER_AREA,
                            )

                            T_small = scale_homography(T, seam_scale)
                            H_small = scale_homography(H_active, seam_scale)

                            left_low, left_low_mask, left_low_corner = warp_to_roi(left_small, T_small)
                            right_low, right_low_mask, right_low_corner = warp_to_roi(
                                right_small,
                                T_small @ H_small,
                            )

                            low_imgs_raw = [left_low, right_low]
                            low_masks_raw = [left_low_mask, right_low_mask]
                            low_corners_raw = [left_low_corner, right_low_corner]
                            low_sizes_raw = [
                                (int(left_low.shape[1]), int(left_low.shape[0])),
                                (int(right_low.shape[1]), int(right_low.shape[0])),
                            ]
                            final_imgs_raw = [left_roi_full, right_roi_full]
                            final_masks_raw = [left_roi_mask_full, right_roi_mask_full]
                            final_corners_raw = [left_corner_full, right_corner_full]
                            final_sizes_raw = [
                                (int(left_roi_full.shape[1]), int(left_roi_full.shape[0])),
                                (int(right_roi_full.shape[1]), int(right_roi_full.shape[0])),
                            ]

                            seam_input_low_imgs = list(low_imgs_raw)
                            seam_input_low_masks = list(low_masks_raw)
                            seam_input_low_corners = list(low_corners_raw)
                            seam_input_low_sizes = list(low_sizes_raw)
                            seam_input_final_masks = list(final_masks_raw)
                            seam_input_final_corners_abs = list(final_corners_raw)

                            cropper = None
                            crop_aspect = 1.0
                            crop_keyframe_stat = {
                                "frame_idx": int(source_idx),
                                "crop_enabled": bool(args.crop),
                                "crop_applied": False,
                                "lir_method_requested": args.lir_method,
                                "lir_method_used": "disabled",
                                "lir_rect": None,
                                "crop_time_ms": 0.0,
                                "crop_fallback_to_no_crop": False,
                                "mask_area_before": 0,
                                "mask_area_after": 0,
                                "mask_bbox_area_before": 0,
                                "mask_bbox_area_after": 0,
                                "black_border_ratio_low": 0.0,
                            }

                            if args.crop:
                                crop_t0 = time.perf_counter()

                                def _crop_warn(message: str) -> None:
                                    _warn_and_record(
                                        debug,
                                        f"crop_warning frame={source_idx}: {message}",
                                    )

                                cropper = Cropper(
                                    crop=True,
                                    lir_method=args.lir_method,
                                    lir_erode=args.lir_erode,
                                    warning_handler=_crop_warn,
                                )
                                cropper.prepare(
                                    seam_input_low_imgs,
                                    seam_input_low_masks,
                                    seam_input_low_corners,
                                    seam_input_low_sizes,
                                )

                                crop_keyframe_stat["crop_time_ms"] = float(
                                    (time.perf_counter() - crop_t0) * 1000.0
                                )

                                if cropper.panorama_mask is None or cropper.lir_rectangle is None:
                                    raise RuntimeError("cropper did not produce panorama mask / lir rect")

                                panorama_mask_low = cropper.panorama_mask
                                lir_rect = cropper.lir_rectangle
                                crop_keyframe_stat["lir_method_used"] = cropper.lir_method_used
                                crop_keyframe_stat["lir_rect"] = {
                                    "x": int(lir_rect.x),
                                    "y": int(lir_rect.y),
                                    "w": int(lir_rect.w),
                                    "h": int(lir_rect.h),
                                }
                                crop_keyframe_stat["mask_area_before"] = int(
                                    (panorama_mask_low > 0).sum()
                                )
                                crop_keyframe_stat["mask_bbox_area_before"] = int(
                                    _mask_bbox_area(panorama_mask_low)
                                )

                                lir_ratio = float(lir_rect.area) / float(
                                    max(1, crop_keyframe_stat["mask_bbox_area_before"])
                                )
                                if lir_ratio < 0.30:
                                    crop_keyframe_stat["crop_fallback_to_no_crop"] = True
                                    debug["crop_fallback_to_no_crop"] = True
                                    crop_method_used = f"{cropper.lir_method_used}_rejected_small_lir"
                                    crop_lir_rect = lir_rect
                                    _warn_and_record(
                                        debug,
                                        (
                                            f"crop_fallback_to_no_crop frame={source_idx}: "
                                            f"lir_ratio={lir_ratio:.3f} < 0.30"
                                        ),
                                    )
                                    if int(args.crop_debug) == 1:
                                        from stitching.viz import save_image  # noqa: WPS433,E402

                                        snapshots_dir = output_dir / "snapshots"
                                        snapshots_dir.mkdir(parents=True, exist_ok=True)

                                        lir_vis = lir_rect.draw_on(
                                            panorama_mask_low.copy(),
                                            color=(0, 0, 255),
                                            size=2,
                                        )
                                        low_masks_vis = _stack_masks_horizontally(low_masks_raw)
                                        final_as_is, _ = _compose_as_is_preview(
                                            final_imgs_raw,
                                            final_masks_raw,
                                            final_corners_raw,
                                        )
                                        save_image(snapshots_dir / "panorama_mask_low.png", panorama_mask_low)
                                        save_image(snapshots_dir / "lir_on_mask_low.png", lir_vis)
                                        save_image(snapshots_dir / "cropped_low_masks.png", low_masks_vis)
                                        save_image(
                                            snapshots_dir
                                            / f"cropped_final_frame_as_is_{source_idx:06d}.png",
                                            final_as_is,
                                        )
                                    cropper = None
                                else:
                                    seam_input_low_imgs = list(cropper.crop_images(low_imgs_raw))
                                    seam_input_low_masks = list(cropper.crop_images(low_masks_raw))
                                    seam_input_low_corners, seam_input_low_sizes = cropper.crop_rois(
                                        low_corners_raw,
                                        low_sizes_raw,
                                    )

                                    # Use conservative low->final ratio from actual ROI tensor sizes.
                                    # This avoids +1 overflow caused by mixed rounding during seam-scale resize.
                                    aspect_candidates = []
                                    for (f_w, f_h), (l_w, l_h) in zip(final_sizes_raw, low_sizes_raw):
                                        aspect_candidates.append(float(f_w) / float(max(1, l_w)))
                                        aspect_candidates.append(float(f_h) / float(max(1, l_h)))
                                    crop_aspect = min(aspect_candidates) - 1e-6
                                    crop_aspect = max(crop_aspect, 1e-6)
                                    seam_input_final_imgs = list(
                                        cropper.crop_images(final_imgs_raw, aspect=crop_aspect)
                                    )
                                    seam_input_final_masks = list(
                                        cropper.crop_images(final_masks_raw, aspect=crop_aspect)
                                    )
                                    abs_overlaps = cropper.get_overlaps_absolute(aspect=crop_aspect)
                                    seam_input_final_corners_abs = [rect.corner for rect in abs_overlaps]
                                    seam_input_final_sizes = [
                                        (int(img.shape[1]), int(img.shape[0]))
                                        for img in seam_input_final_imgs
                                    ]

                                    _validate_rois_or_raise(
                                        "cropped_low",
                                        seam_input_low_imgs,
                                        seam_input_low_masks,
                                        seam_input_low_corners,
                                        seam_input_low_sizes,
                                    )
                                    _validate_rois_or_raise(
                                        "cropped_final_abs",
                                        seam_input_final_imgs,
                                        seam_input_final_masks,
                                        seam_input_final_corners_abs,
                                        seam_input_final_sizes,
                                        canvas_size=canvas_size,
                                    )

                                    cropped_panorama_mask_low = _compose_masks_panorama(
                                        seam_input_low_masks,
                                        seam_input_low_corners,
                                        seam_input_low_sizes,
                                    )
                                    crop_keyframe_stat["mask_area_after"] = int(
                                        (cropped_panorama_mask_low > 0).sum()
                                    )
                                    crop_keyframe_stat["mask_bbox_area_after"] = int(
                                        _mask_bbox_area(cropped_panorama_mask_low)
                                    )
                                    crop_keyframe_stat["black_border_ratio_low"] = float(
                                        1.0
                                        - (
                                            float(crop_keyframe_stat["mask_area_after"])
                                            / float(max(1, crop_keyframe_stat["mask_bbox_area_after"]))
                                        )
                                    )
                                    crop_keyframe_stat["crop_applied"] = True
                                    crop_applied = 1
                                    crop_method_used = cropper.lir_method_used
                                    crop_lir_rect = cropper.lir_rectangle

                                    logging.info(
                                        "crop frame=%d low_shapes_before=%s low_shapes_after=%s "
                                        "corners_before=%s corners_after=%s",
                                        source_idx,
                                        [img.shape[:2] for img in low_imgs_raw],
                                        [img.shape[:2] for img in seam_input_low_imgs],
                                        low_corners_raw,
                                        seam_input_low_corners,
                                    )

                                    if int(args.crop_debug) == 1:
                                        from stitching.viz import save_image  # noqa: WPS433,E402

                                        snapshots_dir = output_dir / "snapshots"
                                        snapshots_dir.mkdir(parents=True, exist_ok=True)

                                        lir_vis = cropper.lir_rectangle.draw_on(
                                            panorama_mask_low.copy(),
                                            color=(0, 0, 255),
                                            size=2,
                                        )
                                        cropped_low_masks_vis = _stack_masks_horizontally(
                                            seam_input_low_masks
                                        )
                                        cropped_final_as_is, _ = _compose_as_is_preview(
                                            seam_input_final_imgs,
                                            seam_input_final_masks,
                                            cropper.crop_rois(
                                                final_corners_raw,
                                                final_sizes_raw,
                                                aspect=crop_aspect,
                                            )[0],
                                        )

                                        save_image(
                                            snapshots_dir / "panorama_mask_low.png",
                                            panorama_mask_low,
                                        )
                                        save_image(snapshots_dir / "lir_on_mask_low.png", lir_vis)
                                        save_image(
                                            snapshots_dir / "cropped_low_masks.png",
                                            cropped_low_masks_vis,
                                        )
                                        save_image(
                                            snapshots_dir
                                            / f"cropped_final_frame_as_is_{source_idx:06d}.png",
                                            cropped_final_as_is,
                                        )
                                if crop_keyframe_stat["mask_area_after"] == 0:
                                    crop_keyframe_stat["mask_area_after"] = int(
                                        crop_keyframe_stat["mask_area_before"]
                                    )
                                if crop_keyframe_stat["mask_bbox_area_after"] == 0:
                                    crop_keyframe_stat["mask_bbox_area_after"] = int(
                                        crop_keyframe_stat["mask_bbox_area_before"]
                                    )
                                if not crop_keyframe_stat["crop_applied"]:
                                    crop_keyframe_stat["black_border_ratio_low"] = float(
                                        1.0
                                        - (
                                            float(crop_keyframe_stat["mask_area_after"])
                                            / float(max(1, crop_keyframe_stat["mask_bbox_area_after"]))
                                        )
                                    )
                            else:
                                panorama_mask_low = _compose_masks_panorama(
                                    seam_input_low_masks,
                                    seam_input_low_corners,
                                    seam_input_low_sizes,
                                )
                                crop_keyframe_stat["mask_area_before"] = int(
                                    (panorama_mask_low > 0).sum()
                                )
                                crop_keyframe_stat["mask_area_after"] = int(
                                    crop_keyframe_stat["mask_area_before"]
                                )
                                crop_keyframe_stat["mask_bbox_area_before"] = int(
                                    _mask_bbox_area(panorama_mask_low)
                                )
                                crop_keyframe_stat["mask_bbox_area_after"] = int(
                                    crop_keyframe_stat["mask_bbox_area_before"]
                                )
                                crop_keyframe_stat["black_border_ratio_low"] = float(
                                    1.0
                                    - (
                                        float(crop_keyframe_stat["mask_area_before"])
                                        / float(max(1, crop_keyframe_stat["mask_bbox_area_before"]))
                                    )
                                )

                            debug["crop_keyframe_stats"].append(crop_keyframe_stat)

                            seam_masks_low = compute_seam_masks_opencv(
                                seam_input_low_imgs,
                                seam_input_low_corners,
                                seam_input_low_masks,
                                method=seam_method,
                            )
                            if len(seam_masks_low) != 2:
                                raise RuntimeError("seam finder did not return 2 seam masks")

                            seam_cache = {
                                "seam_scale": seam_scale,
                                "left_mask_low": seam_masks_low[0],
                                "right_mask_low": seam_masks_low[1],
                                "left_mask_final": None,
                                "right_mask_final": None,
                                "crop_applied": bool(crop_applied),
                                "cropper": cropper if crop_applied else None,
                                "crop_aspect": float(crop_aspect),
                                "crop_method": crop_method_used,
                                "last_seam_frame_index": int(source_idx),
                                "trigger_armed": bool(seam_decision.trigger_armed_next),
                                "trigger_states": dict(seam_decision.trigger_states_next),
                                "crop_lir_rect": (
                                    (
                                        int(crop_lir_rect.x),
                                        int(crop_lir_rect.y),
                                        int(crop_lir_rect.w),
                                        int(crop_lir_rect.h),
                                    )
                                    if crop_lir_rect is not None
                                    else None
                                ),
                            }

                            seam_compute_ms = (time.perf_counter() - seam_t0) * 1000.0
                            seam_keyframe_ms.append(seam_compute_ms)
                            debug["seam_recompute_count"] = int(debug.get("seam_recompute_count", 0)) + 1
                            debug["seam_policy_events"].append(
                                {
                                    "frame_idx": int(source_idx),
                                    "policy": seam_policy_effective,
                                    "reason": seam_trigger_reason,
                                    "overlap_area_current": int(overlap_area_current),
                                    "overlap_diff_before": float(overlap_diff_before),
                                    "overlap_diff_after": float(overlap_diff_after),
                                    "foreground_ratio": float(foreground_ratio),
                                    "foreground_protect_ratio": float(foreground_protect_ratio),
                                    "trigger_flags": seam_trigger_flags,
                                    "trigger_states_next": trigger_states_next,
                                    "geometry_recomputed": bool(geometry_recomputed),
                                    "geometry_update_reason": geometry_update_reason,
                                }
                            )

                            # Debug: project low-res seam/masks to one low-res canvas.
                            canvas_low_size = (
                                max(
                                    1,
                                    max(
                                        int(c[0]) + int(s[0])
                                        for c, s in zip(seam_input_low_corners, seam_input_low_sizes)
                                    ),
                                ),
                                max(
                                    1,
                                    max(
                                        int(c[1]) + int(s[1])
                                        for c, s in zip(seam_input_low_corners, seam_input_low_sizes)
                                    ),
                                ),
                            )
                            left_low_canvas, left_low_full_mask = _compose_single_roi_on_canvas(
                                canvas_low_size,
                                seam_input_low_imgs[0],
                                seam_input_low_masks[0],
                                seam_input_low_corners[0],
                            )
                            right_low_canvas, right_low_full_mask = _compose_single_roi_on_canvas(
                                canvas_low_size,
                                seam_input_low_imgs[1],
                                seam_input_low_masks[1],
                                seam_input_low_corners[1],
                            )

                            seam_left_low_full = np.zeros_like(left_low_full_mask, dtype=np.uint8)
                            seam_right_low_full = np.zeros_like(right_low_full_mask, dtype=np.uint8)
                            place_mask_on_canvas(
                                seam_left_low_full,
                                seam_masks_low[0],
                                seam_input_low_corners[0],
                            )
                            place_mask_on_canvas(
                                seam_right_low_full,
                                seam_masks_low[1],
                                seam_input_low_corners[1],
                            )

                            seam_overlay_low = seam_overlay_preview(
                                left_low_canvas,
                                right_low_canvas,
                                left_low_full_mask,
                                right_low_full_mask,
                                seam_left_low_full,
                                seam_right_low_full,
                            )
                            overlap_diff_low = overlap_absdiff_preview(
                                left_low_canvas,
                                right_low_canvas,
                                left_low_full_mask,
                                right_low_full_mask,
                            )

                            if bool(int(args.seam_snapshot_on_recompute)):
                                _save_seam_debug(
                                    output_dir=output_dir,
                                    frame_idx=source_idx,
                                    left_low=left_low_canvas,
                                    right_low=right_low_canvas,
                                    left_mask_low=left_low_full_mask,
                                    right_mask_low=right_low_full_mask,
                                    seam_left_low=seam_left_low_full,
                                    seam_right_low=seam_right_low_full,
                                    seam_overlay_low=seam_overlay_low,
                                    overlap_diff_low=overlap_diff_low,
                                )
                                if foreground_mask_full is not None:
                                    from stitching.viz import save_image  # noqa: WPS433,E402

                                    snapshots_dir = output_dir / "snapshots"
                                    snapshots_dir.mkdir(parents=True, exist_ok=True)
                                    save_image(
                                        snapshots_dir / f"seam_event_{source_idx:06d}_foreground_mask.png",
                                        foreground_mask_full,
                                    )

                        if seam_cache is None:
                            raise RuntimeError("seam_cache is unavailable while seam mode is enabled")
                        seam_cache["trigger_armed"] = bool(seam_decision.trigger_armed_next)
                        seam_cache["trigger_states"] = dict(seam_decision.trigger_states_next)

                        seam_target_masks = [left_roi_mask_full, right_roi_mask_full]
                        seam_target_corners_abs = [left_corner_full, right_corner_full]
                        if bool(seam_cache.get("crop_applied", False)):
                            cached_cropper = seam_cache.get("cropper")
                            if cached_cropper is None:
                                raise RuntimeError("crop_applied but cropper is missing in seam_cache")
                            cached_aspect = float(seam_cache.get("crop_aspect", 1.0))
                            seam_target_imgs = list(
                                cached_cropper.crop_images([left_roi_full, right_roi_full], cached_aspect)
                            )
                            seam_target_masks = list(
                                cached_cropper.crop_images(
                                    [left_roi_mask_full, right_roi_mask_full],
                                    cached_aspect,
                                )
                            )
                            seam_target_sizes = [
                                (int(img.shape[1]), int(img.shape[0])) for img in seam_target_imgs
                            ]
                            seam_target_corners_abs = [
                                rect.corner
                                for rect in cached_cropper.get_overlaps_absolute(aspect=cached_aspect)
                            ]
                            _validate_rois_or_raise(
                                "cached_cropped_final_abs",
                                seam_target_imgs,
                                seam_target_masks,
                                seam_target_corners_abs,
                                seam_target_sizes,
                                canvas_size=canvas_size,
                            )
                            crop_applied = 1
                            crop_method_used = str(seam_cache.get("crop_method", "fallback"))
                            if getattr(cached_cropper, "lir_rectangle", None) is not None:
                                crop_lir_rect = cached_cropper.lir_rectangle

                        seam_left_roi = resize_seam_to_compose(
                            seam_cache["left_mask_low"],
                            seam_target_masks[0],
                            dilate_iter=args.seam_dilate,
                        )
                        seam_right_roi = resize_seam_to_compose(
                            seam_cache["right_mask_low"],
                            seam_target_masks[1],
                            dilate_iter=args.seam_dilate,
                        )

                        seam_left_full = np.zeros_like(left_mask_full, dtype=np.uint8)
                        seam_right_full = np.zeros_like(right_mask_full, dtype=np.uint8)
                        place_mask_on_canvas(seam_left_full, seam_left_roi, seam_target_corners_abs[0])
                        place_mask_on_canvas(seam_right_full, seam_right_roi, seam_target_corners_abs[1])

                        crop_limit_full = None
                        if crop_applied:
                            crop_limit_full = np.zeros_like(left_mask_full, dtype=np.uint8)
                            place_mask_on_canvas(
                                crop_limit_full,
                                seam_target_masks[0],
                                seam_target_corners_abs[0],
                            )
                            place_mask_on_canvas(
                                crop_limit_full,
                                seam_target_masks[1],
                                seam_target_corners_abs[1],
                            )

                        final_left_mask, final_right_mask = _resolve_seam_masks(
                            left_mask_full,
                            right_mask_full,
                            seam_left_full,
                            seam_right_full,
                            limit_mask_full=crop_limit_full,
                        )
                        prev_final_left = seam_cache.get("left_mask_final")
                        prev_final_right = seam_cache.get("right_mask_final")
                        foreground_protect_ratio = 0.0
                        if foreground_mask_full is not None:
                            final_left_mask, final_right_mask, foreground_protect_ratio = apply_protect_mask_assignment(
                                final_left_mask,
                                final_right_mask,
                                left_mask_full,
                                right_mask_full,
                                foreground_mask_full,
                                prefer_left_mask=prev_final_left,
                                prefer_right_mask=prev_final_right,
                                limit_mask_full=crop_limit_full,
                            )
                        final_left_mask, final_right_mask = seam_mask_smoother.update(
                            final_left_mask,
                            final_right_mask,
                            left_mask_full,
                            right_mask_full,
                            limit_mask=crop_limit_full,
                        )
                        mask_changes = [
                            compute_mask_change_ratio(prev_final_left, final_left_mask),
                            compute_mask_change_ratio(prev_final_right, final_right_mask),
                        ]
                        valid_mask_changes = [value for value in mask_changes if value is not None]
                        if valid_mask_changes:
                            seam_mask_change_ratio = float(
                                sum(valid_mask_changes) / len(valid_mask_changes)
                            )
                            seam_mask_change_values.append(seam_mask_change_ratio)
                        seam_cache["left_mask_final"] = final_left_mask
                        seam_cache["right_mask_final"] = final_right_mask
                        if seam_recomputed:
                            seam_cache["last_seam_frame_index"] = int(source_idx)
                        output_bbox_current = _mask_union_bbox(final_left_mask, final_right_mask)

                        overlap_diff_after = _mean_overlap_diff(
                            left_canvas,
                            right_canvas,
                            final_left_mask,
                            final_right_mask,
                        )
                        seam_band_metrics = _compute_seam_band_metrics(
                            left_canvas,
                            right_canvas,
                            left_mask_full,
                            right_mask_full,
                            final_left_mask,
                            final_right_mask,
                        )
                        seam_band_ratio = float(seam_band_metrics["seam_band_ratio"])
                        seam_band_illuminance_diff = float(
                            seam_band_metrics["seam_band_illuminance_diff"]
                        )
                        seam_band_gradient_disagreement = float(
                            seam_band_metrics["seam_band_gradient_disagreement"]
                        )
                        seam_band_mask = seam_band_metrics["seam_band_mask"]
                        overlap_diff_before_values.append(float(overlap_diff_before))
                        overlap_diff_after_values.append(float(overlap_diff_after))
                        seam_band_ratio_values.append(float(seam_band_ratio))
                        seam_band_illuminance_values.append(float(seam_band_illuminance_diff))
                        seam_band_gradient_values.append(float(seam_band_gradient_disagreement))

                        if seam_recomputed:
                            overlap_stats = summarize_overlap(left_mask_full, right_mask_full)
                            debug["seam_keyframe_stats"].append(
                                {
                                    "frame_idx": int(source_idx),
                                    "method": seam_method,
                                    "policy": seam_policy_effective,
                                    "trigger_reason": seam_trigger_reason,
                                    "seam_scale": float(seam_cache["seam_scale"]),
                                    "overlap_area_px": int(overlap_stats["overlap_area"]),
                                    "seam_mask_nonzero_ratio_left": _mask_ratio(final_left_mask),
                                    "seam_mask_nonzero_ratio_right": _mask_ratio(final_right_mask),
                                    "overlap_diff_mean_before": float(overlap_diff_before),
                                    "overlap_diff_mean_after": float(overlap_diff_after),
                                    "runtime_ms_seam_keyframe": float(seam_compute_ms),
                                    "crop_applied": bool(crop_applied),
                                }
                            )

                        stitched = _blend_frames(
                            left_canvas,
                            right_canvas,
                            args.blend,
                            left_mask=final_left_mask,
                            right_mask=final_right_mask,
                            mb_levels=args.mb_levels,
                        )
                    except Exception as seam_exc:
                        _warn_and_record(debug, f"seam_fallback frame={source_idx}: {seam_exc}")
                        note_parts.append("seam_fallback")
                foreground_ratio_values.append(float(foreground_ratio))
                if bool(seam_trigger_flags.get("foreground_triggered")):
                    debug["foreground_triggered_count"] = int(debug.get("foreground_triggered_count", 0)) + 1

                stitched_written = stitched
                if bool(args.crop) and output_crop_rect is None and output_bbox_current is not None:
                    output_crop_rect = tuple(int(v) for v in output_bbox_current)
                    debug["output_crop_rect"] = {
                        "x": int(output_crop_rect[0]),
                        "y": int(output_crop_rect[1]),
                        "w": int(output_crop_rect[2]),
                        "h": int(output_crop_rect[3]),
                    }
                if bool(args.crop) and output_crop_rect is not None:
                    stitched_written = _crop_frame_to_rect(stitched, output_crop_rect)
                    if seam_band_mask is not None:
                        seam_band_mask = _crop_frame_to_rect(seam_band_mask, output_crop_rect)
                stitched_delta_mean = compute_frame_absdiff_mean(prev_stitched_written, stitched_written)
                seam_band_flicker = None
                flicker_mask = seam_band_mask
                if prev_seam_band_mask is not None and seam_band_mask is not None:
                    prev_mask_arr = np.asarray(prev_seam_band_mask) > 0
                    curr_mask_arr = np.asarray(seam_band_mask) > 0
                    if prev_mask_arr.shape == curr_mask_arr.shape:
                        flicker_mask = ((prev_mask_arr | curr_mask_arr).astype(np.uint8)) * 255
                seam_band_flicker = compute_frame_absdiff_mean(
                    prev_stitched_written,
                    stitched_written,
                    mask=flicker_mask,
                )
                prev_stitched_written = stitched_written
                prev_seam_band_mask = seam_band_mask
                if stitched_delta_mean is not None:
                    stitched_delta_values.append(float(stitched_delta_mean))
                if seam_band_flicker is not None:
                    seam_band_flicker_values.append(float(seam_band_flicker))

                if writer is None:
                    out_h, out_w = int(stitched_written.shape[0]), int(stitched_written.shape[1])
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, fps_value, (out_w, out_h))
                    if not writer.isOpened():
                        raise RuntimeError("Failed to open VideoWriter for stitched.mp4")
                writer.write(stitched_written)

                raw_corners = transform_corners(
                    H_raw,
                    image_size=(right.shape[1], right.shape[0]),
                    pre_transform=T,
                )
                sm_corners = transform_corners(
                    H_sm,
                    image_size=(right.shape[1], right.shape[0]),
                    pre_transform=T,
                )
                jitter_raw_stats = compute_jitter(prev_raw_corners, raw_corners)
                jitter_sm_stats = compute_jitter(prev_sm_corners, sm_corners)
                prev_raw_corners = raw_corners
                prev_sm_corners = sm_corners

                if jitter_raw_stats.mean is not None:
                    jitter_raw_values.append(jitter_raw_stats.mean)
                if jitter_sm_stats.mean is not None:
                    jitter_sm_values.append(jitter_sm_stats.mean)

                jitter_writer.writerow(
                    {
                        "frame_idx": source_idx,
                        "jitter_raw": jitter_raw_stats.mean,
                        "jitter_sm": jitter_sm_stats.mean,
                        "status": status,
                    }
                )

                should_save_frame_snapshot = (
                    processed_idx % snapshot_every == 0
                    or (bool(int(args.seam_snapshot_on_recompute)) and bool(seam_recomputed))
                )
                if should_save_frame_snapshot:
                    _save_snapshot(
                        output_dir,
                        source_idx,
                        left,
                        right,
                        stitched_written,
                        overlay_active,
                        overlay_raw,
                        overlay_sm,
                    )
                    if bool(seam_recomputed) and bool(int(args.seam_snapshot_on_recompute)):
                        debug["seam_snapshot_count"] = int(debug.get("seam_snapshot_count", 0)) + 1

                runtime_ms = (time.perf_counter() - frame_start) * 1000.0
                frame_runtimes.append(runtime_ms)

                if status == "OK":
                    debug["success_frames"] += 1
                if status == "FALLBACK":
                    debug["fallback_frames"] += 1
                h_delta = _h_delta_norm(H_active, video_prev_H)
                video_prev_H = H_active
                if seam_method != "none":
                    note_parts.append(f"video_mode={int(video_mode_effective)}")
                    note_parts.append(f"geometry_mode={geometry_mode}")
                    note_parts.append(f"geometry_keyframe_every={geometry_keyframe_every_effective}")
                    note_parts.append(f"geometry_recomputed={int(geometry_recomputed)}")
                    note_parts.append(f"geometry_update_reason={geometry_update_reason}")
                    note_parts.append(f"seam_policy={seam_policy_effective}")
                    note_parts.append(f"seam_recomputed={int(seam_recomputed)}")
                    note_parts.append(f"seam_reason={seam_trigger_reason}")
                    note_parts.append(f"seam={seam_method}")
                    note_parts.append(f"foreground_ratio={foreground_ratio:.4f}")
                    note_parts.append(f"trigger_armed={trigger_armed_next}")
                    if seam_compute_ms > 0:
                        note_parts.append(f"seam_ms={seam_compute_ms:.2f}")
                    note_parts.append(
                        f"overlap_diff={overlap_diff_before:.2f}->{overlap_diff_after:.2f}"
                    )
                    note_parts.append(f"overlap_area={overlap_area_current}")
                    note_parts.append(f"crop_applied={crop_applied}")
                    if crop_method_used != "none":
                        note_parts.append(f"crop_method={crop_method_used}")

                row = {
                    "frame_idx": source_idx,
                    "is_keyframe": is_keyframe,
                    "status": status,
                    "n_kp_left": last_stats["n_kp_left"],
                    "n_kp_right": last_stats["n_kp_right"],
                    "n_matches_raw": last_stats["n_matches_raw"],
                    "n_matches_good": last_stats["n_matches_good"],
                    "n_inliers": last_stats["n_inliers"],
                    "inlier_ratio": last_stats["inlier_ratio"],
                    "jitter_raw": jitter_raw_stats.mean,
                    "jitter_raw_max": jitter_raw_stats.max,
                    "jitter_sm": jitter_sm_stats.mean,
                    "jitter_sm_max": jitter_sm_stats.max,
                    "video_mode": int(video_mode_effective),
                    "reuse_mode": "baseline",
                    "geometry_mode": geometry_mode,
                    "jitter_meaningful": jitter_meaningful,
                    "geometry_recomputed": int(geometry_recomputed),
                    "geometry_update_reason": geometry_update_reason,
                    "reprojection_error": last_stats["reprojection_error"],
                    "inlier_spatial_coverage": last_stats["inlier_spatial_coverage"],
                    "feature_runtime_ms_left": last_stats["feature_runtime_ms_left"],
                    "feature_runtime_ms_right": last_stats["feature_runtime_ms_right"],
                    "matching_runtime_ms": last_stats["matching_runtime_ms"],
                    "geometry_runtime_ms": last_stats["geometry_runtime_ms"],
                    "H_delta_norm": float(h_delta),
                    "overlap_area_current": int(overlap_area_current),
                    "seam_policy": seam_policy_effective,
                    "seam_recomputed": int(seam_recomputed),
                    "foreground_ratio": float(foreground_ratio),
                    "trigger_armed": int(trigger_armed_next),
                    "overlap_diff_before": float(overlap_diff_before),
                    "overlap_diff_after": float(overlap_diff_after),
                    "seam_band_ratio": float(seam_band_ratio),
                    "seam_band_illuminance_diff": float(seam_band_illuminance_diff),
                    "seam_band_gradient_disagreement": float(
                        seam_band_gradient_disagreement
                    ),
                    "seam_band_flicker": seam_band_flicker,
                    "stitched_delta_mean": stitched_delta_mean,
                    "seam_mask_change_ratio": float(seam_mask_change_ratio),
                    "crop_applied": int(crop_applied),
                    "crop_method": crop_method_used,
                    "crop_lir_x": int(crop_lir_rect.x) if crop_lir_rect is not None else None,
                    "crop_lir_y": int(crop_lir_rect.y) if crop_lir_rect is not None else None,
                    "crop_lir_w": int(crop_lir_rect.w) if crop_lir_rect is not None else None,
                    "crop_lir_h": int(crop_lir_rect.h) if crop_lir_rect is not None else None,
                    "runtime_ms": int(runtime_ms),
                    "note": ";".join(note_parts) if note_parts else "",
                }

                for idx, value in enumerate(_flatten_h(H_active)):
                    row[f"H_{idx // 3}{idx % 3}"] = value
                for idx, value in enumerate(_flatten_h(H_raw)):
                    row[f"Hraw_{idx // 3}{idx % 3}"] = value
                for idx, value in enumerate(_flatten_h(H_sm)):
                    row[f"Hsm_{idx // 3}{idx % 3}"] = value

                writer_csv.writerow(row)

                processed_idx += 1
                source_idx += 1

                consumed = _consume_stride(left_source, right_source, stride)
                source_idx += consumed
                if consumed < stride - 1:
                    break
    except Exception as exc:
        debug["errors"].append(str(exc))
        logging.error("Pipeline aborted: %s", exc)
        if writer is not None:
            writer.release()
        left_source.close()
        right_source.close()
        debug["processed_frames"] = processed_idx
        debug["runtime_ms"] = int((time.perf_counter() - start_time) * 1000)
        _write_json(debug_path, debug)
        return 1
    finally:
        if writer is not None:
            writer.release()
        left_source.close()
        right_source.close()

    debug["processed_frames"] = processed_idx
    debug["runtime_ms"] = int((time.perf_counter() - start_time) * 1000)

    processed_frames = processed_idx
    success_frames = debug["success_frames"]
    fallback_frames = debug["fallback_frames"]

    mean_inliers = float(sum(inliers_ok) / len(inliers_ok)) if inliers_ok else 0.0
    mean_inlier_ratio = (
        float(sum(inlier_ratios_ok) / len(inlier_ratios_ok)) if inlier_ratios_ok else 0.0
    )
    avg_runtime_ms = (
        float(sum(frame_runtimes) / len(frame_runtimes)) if frame_runtimes else 0.0
    )
    approx_fps = 1000.0 / avg_runtime_ms if avg_runtime_ms > 0 else 0.0

    jitter_raw_mean = (
        float(sum(jitter_raw_values) / len(jitter_raw_values)) if jitter_raw_values else 0.0
    )
    jitter_sm_mean = (
        float(sum(jitter_sm_values) / len(jitter_sm_values)) if jitter_sm_values else 0.0
    )
    jitter_raw_p95 = _quantile95(jitter_raw_values)
    jitter_sm_p95 = _quantile95(jitter_sm_values)
    overlap_diff_before_mean = (
        float(sum(overlap_diff_before_values) / len(overlap_diff_before_values))
        if overlap_diff_before_values
        else 0.0
    )
    overlap_diff_after_mean = (
        float(sum(overlap_diff_after_values) / len(overlap_diff_after_values))
        if overlap_diff_after_values
        else 0.0
    )
    seam_mask_change_mean = (
        float(sum(seam_mask_change_values) / len(seam_mask_change_values))
        if seam_mask_change_values
        else 0.0
    )
    stitched_delta_mean = (
        float(sum(stitched_delta_values) / len(stitched_delta_values))
        if stitched_delta_values
        else 0.0
    )
    reprojection_error_mean = (
        float(sum(reprojection_error_values) / len(reprojection_error_values))
        if reprojection_error_values
        else 0.0
    )
    inlier_spatial_coverage_mean = (
        float(sum(inlier_spatial_coverage_values) / len(inlier_spatial_coverage_values))
        if inlier_spatial_coverage_values
        else 0.0
    )
    feature_runtime_left_mean = (
        float(sum(feature_runtime_left_values) / len(feature_runtime_left_values))
        if feature_runtime_left_values
        else 0.0
    )
    feature_runtime_right_mean = (
        float(sum(feature_runtime_right_values) / len(feature_runtime_right_values))
        if feature_runtime_right_values
        else 0.0
    )
    matching_runtime_mean = (
        float(sum(matching_runtime_values) / len(matching_runtime_values))
        if matching_runtime_values
        else 0.0
    )
    geometry_runtime_mean = (
        float(sum(geometry_runtime_values) / len(geometry_runtime_values))
        if geometry_runtime_values
        else 0.0
    )
    seam_band_ratio_mean = (
        float(sum(seam_band_ratio_values) / len(seam_band_ratio_values))
        if seam_band_ratio_values
        else 0.0
    )
    seam_band_illuminance_mean = (
        float(sum(seam_band_illuminance_values) / len(seam_band_illuminance_values))
        if seam_band_illuminance_values
        else 0.0
    )
    seam_band_gradient_mean = (
        float(sum(seam_band_gradient_values) / len(seam_band_gradient_values))
        if seam_band_gradient_values
        else 0.0
    )
    seam_band_flicker_mean = (
        float(sum(seam_band_flicker_values) / len(seam_band_flicker_values))
        if seam_band_flicker_values
        else 0.0
    )
    foreground_ratio_mean = (
        float(sum(foreground_ratio_values) / len(foreground_ratio_values))
        if foreground_ratio_values
        else 0.0
    )
    temporal_primary_metric = _primary_temporal_metric_for_geometry_mode(geometry_mode)
    temporal_primary_value = (
        jitter_sm_mean if temporal_primary_metric == "mean_jitter_sm" else overlap_diff_after_mean
    )
    debug["foreground_ratio_mean"] = float(foreground_ratio_mean)

    debug["jitter_summary"] = {
        "mean_raw": jitter_raw_mean,
        "p95_raw": jitter_raw_p95,
        "mean_sm": jitter_sm_mean,
        "p95_sm": jitter_sm_p95,
        "meaningful_under_current_geometry_mode": bool(jitter_meaningful),
        "scope": _jitter_scope_for_geometry_mode(geometry_mode),
    }
    debug["temporal_eval_summary"] = {
        "primary_metric": temporal_primary_metric,
        "primary_value": float(temporal_primary_value),
        "mean_overlap_diff_before": float(overlap_diff_before_mean),
        "mean_overlap_diff_after": float(overlap_diff_after_mean),
        "mean_seam_mask_change_ratio": float(seam_mask_change_mean),
        "mean_reprojection_error": float(reprojection_error_mean),
        "mean_inlier_spatial_coverage": float(inlier_spatial_coverage_mean),
        "avg_feature_runtime_ms_left": float(feature_runtime_left_mean),
        "avg_feature_runtime_ms_right": float(feature_runtime_right_mean),
        "avg_matching_runtime_ms": float(matching_runtime_mean),
        "avg_geometry_runtime_ms": float(geometry_runtime_mean),
        "mean_seam_band_ratio": float(seam_band_ratio_mean),
        "mean_seam_band_illuminance_diff": float(seam_band_illuminance_mean),
        "mean_seam_band_gradient_disagreement": float(seam_band_gradient_mean),
        "mean_seam_band_flicker": float(seam_band_flicker_mean),
        "mean_stitched_delta": float(stitched_delta_mean),
        "mean_foreground_ratio": float(foreground_ratio_mean),
        "foreground_triggered_count": int(debug.get("foreground_triggered_count", 0)),
        "jitter_scope": _jitter_scope_for_geometry_mode(geometry_mode),
        "meaningful_under_current_geometry_mode": {
            "jitter": bool(jitter_meaningful),
            "seam_visibility": True,
            "visual_delta": True,
        },
    }

    seam_stats = debug.get("seam_keyframe_stats", [])
    if seam_stats:
        overlap_areas = [float(s.get("overlap_area_px", 0)) for s in seam_stats]
        overlap_before_vals = [
            float(s.get("overlap_diff_mean_before", 0.0)) for s in seam_stats
        ]
        overlap_after_vals = [
            float(s.get("overlap_diff_mean_after", 0.0)) for s in seam_stats
        ]
        debug["seam_summary"] = {
            "keyframe_count": len(seam_stats),
            "runtime_ms_mean": float(sum(seam_keyframe_ms) / len(seam_keyframe_ms))
            if seam_keyframe_ms
            else 0.0,
            "overlap_area_mean": float(sum(overlap_areas) / len(overlap_areas)),
            "overlap_diff_mean_before": float(
                sum(overlap_before_vals) / len(overlap_before_vals)
            ),
            "overlap_diff_mean_after": float(
                sum(overlap_after_vals) / len(overlap_after_vals)
            ),
        }

    crop_stats = debug.get("crop_keyframe_stats", []) or []
    if crop_stats:
        mask_before_vals = [float(s.get("mask_area_before", 0.0)) for s in crop_stats]
        mask_after_vals = [float(s.get("mask_area_after", 0.0)) for s in crop_stats]
        bbox_before_vals = [float(s.get("mask_bbox_area_before", 0.0)) for s in crop_stats]
        bbox_after_vals = [float(s.get("mask_bbox_area_after", 0.0)) for s in crop_stats]
        border_vals = [float(s.get("black_border_ratio_low", 0.0)) for s in crop_stats]
        debug["crop_summary"] = {
            "keyframe_count": len(crop_stats),
            "crop_applied_count": int(sum(1 for s in crop_stats if s.get("crop_applied"))),
            "mask_area_before_mean": float(sum(mask_before_vals) / len(mask_before_vals)),
            "mask_area_after_mean": float(sum(mask_after_vals) / len(mask_after_vals)),
            "mask_bbox_before_mean": float(sum(bbox_before_vals) / len(bbox_before_vals)),
            "mask_bbox_after_mean": float(sum(bbox_after_vals) / len(bbox_after_vals)),
            "black_border_ratio_low_mean": float(sum(border_vals) / len(border_vals)),
        }

    if int(video_mode_effective) == 1:
        init_vals = [float(v) for v in debug.get("time_breakdown_ms", {}).get("init_ms", [])]
        frame_vals = [float(v) for v in debug.get("time_breakdown_ms", {}).get("per_frame_ms", [])]
        debug["time_breakdown_summary"] = {
            "init_ms_mean": float(sum(init_vals) / len(init_vals)) if init_vals else 0.0,
            "per_frame_ms_mean": float(sum(frame_vals) / len(frame_vals)) if frame_vals else 0.0,
            "init_count": int(len(init_vals)),
        }

    metrics_preview = {
        "total_frames": (
            debug["total_frames"] if debug["total_frames"] is not None else processed_frames
        ),
        "processed_frames": processed_frames,
        "success_frames": success_frames,
        "fallback_frames": fallback_frames,
        "mean_inliers": mean_inliers,
        "mean_inlier_ratio": mean_inlier_ratio,
        "avg_runtime_ms": avg_runtime_ms,
        "approx_fps": approx_fps,
        "mean_jitter_raw": jitter_raw_mean,
        "mean_jitter_sm": jitter_sm_mean,
        "p95_jitter_raw": jitter_raw_p95,
        "p95_jitter_sm": jitter_sm_p95,
        "mean_overlap_diff_before": overlap_diff_before_mean,
        "mean_overlap_diff_after": overlap_diff_after_mean,
        "mean_seam_mask_change_ratio": seam_mask_change_mean,
        "mean_reprojection_error": reprojection_error_mean,
        "mean_inlier_spatial_coverage": inlier_spatial_coverage_mean,
        "avg_feature_runtime_ms_left": feature_runtime_left_mean,
        "avg_feature_runtime_ms_right": feature_runtime_right_mean,
        "avg_matching_runtime_ms": matching_runtime_mean,
        "avg_geometry_runtime_ms": geometry_runtime_mean,
        "mean_seam_band_ratio": seam_band_ratio_mean,
        "mean_seam_band_illuminance_diff": seam_band_illuminance_mean,
        "mean_seam_band_gradient_disagreement": seam_band_gradient_mean,
        "mean_seam_band_flicker": seam_band_flicker_mean,
        "mean_stitched_delta": stitched_delta_mean,
        "mean_foreground_ratio": foreground_ratio_mean,
        "temporal_primary_metric": temporal_primary_metric,
        "temporal_primary_value": float(temporal_primary_value),
        "jitter_scope": _jitter_scope_for_geometry_mode(geometry_mode),
        "seam_keyframe_count": len(debug.get("seam_keyframe_stats", [])),
        "seam_runtime_ms_mean": float(sum(seam_keyframe_ms) / len(seam_keyframe_ms))
        if seam_keyframe_ms
        else 0.0,
        "seam_smooth": args.seam_smooth,
        "seam_smooth_alpha": float(args.seam_smooth_alpha),
        "seam_smooth_window": int(args.seam_smooth_window),
        "seam_policy": seam_policy_effective,
        "seam_keyframe_every_effective": int(seam_keyframe_every_effective),
        "seam_recompute_count": int(debug.get("seam_recompute_count", 0)),
        "seam_snapshot_count": int(debug.get("seam_snapshot_count", 0)),
        "seam_snapshot_on_recompute": int(args.seam_snapshot_on_recompute),
        "foreground_triggered_count": int(debug.get("foreground_triggered_count", 0)),
        "geometry_keyframe_every_effective": int(geometry_keyframe_every_effective),
        "geometry_update_count": int(debug.get("geometry_update_count", 0)),
        "adaptive_update_strategy": mode_semantics.get("adaptive_update_strategy"),
        "crop_keyframe_count": len(debug.get("crop_keyframe_stats", [])),
        "crop_black_border_ratio_low_mean": float(
            debug.get("crop_summary", {}).get("black_border_ratio_low_mean", 0.0)
        ),
        "video_mode": int(video_mode_effective),
        "video_mode_requested": args.video_mode,
        "reuse_mode": args.reuse_mode if int(video_mode_effective) == 1 else "baseline",
        "geometry_mode": geometry_mode,
        "geometry_mode_requested": args.geometry_mode,
        "geometry_mode_source": geometry_mode_source,
        "jitter_meaningful": jitter_meaningful,
        "jitter_scope": _jitter_scope_for_geometry_mode(geometry_mode),
        "reinit_count": int(debug.get("reinit_count", 0)),
        "init_ms_mean": float(debug.get("time_breakdown_summary", {}).get("init_ms_mean", 0.0)),
        "per_frame_ms_mean": float(
            debug.get("time_breakdown_summary", {}).get("per_frame_ms_mean", 0.0)
        ),
        "reuse_per_frame_ms_mean": float(
            debug.get("time_breakdown_summary", {}).get("per_frame_ms_mean", 0.0)
        ),
        "feature_backend_effective": debug.get("feature_backend_effective"),
        "matcher_backend_effective": debug.get("matcher_backend_effective"),
        "geometry_backend_effective": debug.get("geometry_backend_effective"),
    }

    _write_json(metrics_path, metrics_preview)
    _write_json(debug_path, debug)

    try:
        stitched_path_log = video_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        stitched_path_log = video_path

    logging.info("Run completed: %s", output_dir)
    logging.info("stitched.mp4: %s", stitched_path_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
