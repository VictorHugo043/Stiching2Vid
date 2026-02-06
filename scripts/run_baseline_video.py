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
        help="Re-estimate homography every N frames",
    )
    parser.add_argument("--feature", default="orb", help="Feature type: orb or sift")
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
        choices=["none", "feather", "hard"],
        help="Blending mode",
    )
    parser.add_argument(
        "--seam",
        default="none",
        choices=["none", "hard"],
        help="Seam mode for overlap selection",
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
        "--debug_masks",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable mask debug snapshots and stats logging",
    )
    parser.add_argument(
        "--snapshot_stride",
        type=int,
        default=50,
        help="Stride for mask debug snapshots",
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


def _blend_frames(
    left_warped,
    right_warped,
    mode: str,
    mask_left=None,
    mask_right=None,
    seam_side_mask=None,
):
    from stitching.blending import (  # noqa: WPS433,E402
        blend_none,
        composite_hard_seam,
        feather_blend,
    )

    if mode == "none":
        return blend_none(left_warped, right_warped)
    if mode == "hard":
        if mask_left is None or mask_right is None:
            raise ValueError("mask_left/mask_right are required for hard blending.")
        return composite_hard_seam(
            left_warped,
            right_warped,
            mask_left=mask_left,
            mask_right=mask_right,
            seam_side_mask=seam_side_mask,
        )
    return feather_blend(left_warped, right_warped)


def _compute_alpha_for_blend(left_bgr, right_bgr, mode: str, seam_side_mask=None):
    """Compute left alpha used by the current blend mode, constrained by validity."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    mask_left = ((left_bgr.sum(axis=2) > 0).astype(np.uint8) * 255)
    mask_right = ((right_bgr.sum(axis=2) > 0).astype(np.uint8) * 255)
    left_bin = (mask_left > 0).astype(np.uint8)
    right_bin = (mask_right > 0).astype(np.uint8)
    overlap_bin = (left_bin & right_bin).astype(np.uint8)

    if mode == "none":
        alpha = left_bin.astype(np.float32)
    elif mode == "hard":
        if seam_side_mask is None:
            seam_bool = left_bin.astype(bool)
        else:
            seam_arr = np.asarray(seam_side_mask)
            if seam_arr.ndim == 3:
                seam_arr = seam_arr[..., 0]
            seam_bool = seam_arr > 0
        alpha = np.zeros_like(left_bin, dtype=np.float32)
        alpha[(left_bin == 1) & (right_bin == 0)] = 1.0
        alpha[(left_bin == 0) & (right_bin == 1)] = 0.0
        alpha[(left_bin == 1) & (right_bin == 1)] = seam_bool[(left_bin == 1) & (right_bin == 1)]
    else:
        dist_left = cv2.distanceTransform(left_bin, cv2.DIST_L2, 3)
        dist_right = cv2.distanceTransform(right_bin, cv2.DIST_L2, 3)
        denom = dist_left + dist_right
        denom[denom == 0] = 1.0
        alpha = (dist_left / denom).astype(np.float32)
        alpha[(left_bin == 1) & (right_bin == 0)] = 1.0
        alpha[(left_bin == 0) & (right_bin == 1)] = 0.0
        alpha[(left_bin == 0) & (right_bin == 0)] = 0.0

    return alpha, mask_left, mask_right, overlap_bin


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
    columns.extend(["runtime_ms", "note"])
    return columns


# --- Main pipeline ---


def main() -> int:
    args = _build_parser().parse_args()

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
    from stitching.features import detect_and_describe  # noqa: E402
    from stitching.matching import draw_matches, match_descriptors  # noqa: E402
    from stitching.mask_utils import (  # noqa: E402
        ensure_float_mask,
        ensure_uint8_mask,
        save_mask_png,
        summarize_mask,
    )
    from stitching.geometry import (  # noqa: E402
        compute_canvas_and_transform,
        estimate_homography,
        warp_pair,
    )
    from stitching.temporal import (  # noqa: E402
        HomographySmoother,
        compute_jitter,
        transform_corners,
    )
    from stitching.viz import overlay_images  # noqa: E402

    start_time = time.perf_counter()

    keyframe_every = max(1, int(args.keyframe_every))
    stride = max(1, int(args.stride))
    snapshot_every = max(1, int(args.snapshot_every))
    snapshot_stride = max(1, int(args.snapshot_stride))
    debug_masks = bool(args.debug_masks)

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
            "feature": args.feature,
            "nfeatures": args.nfeatures,
            "ratio": args.ratio,
            "min_matches": args.min_matches,
            "ransac_thresh": args.ransac_thresh,
            "blend": args.blend,
            "seam": args.seam,
            "smooth_h": args.smooth_h,
            "smooth_alpha": args.smooth_alpha,
            "smooth_window": args.smooth_window,
            "fps_cli": args.fps,
            "snapshot_every": snapshot_every,
            "debug_masks": int(debug_masks),
            "snapshot_stride": snapshot_stride,
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
        "blend": args.blend,
        "seam": args.seam,
        "smooth_h": args.smooth_h,
        "smooth_alpha": args.smooth_alpha,
        "smooth_window": args.smooth_window,
        "jitter_summary": {},
        "notes": [],
        "errors": [],
        "runtime_ms": None,
    }

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

    last_stats = {
        "n_kp_left": None,
        "n_kp_right": None,
        "n_matches_raw": None,
        "n_matches_good": None,
        "n_inliers": None,
        "inlier_ratio": None,
    }

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
    prev_raw_corners = None
    prev_sm_corners = None

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

                H_raw = valid_H_raw
                keyframe_failed = False

                if not is_keyframe:
                    note_parts.append("reuse_last_H")

                if is_keyframe:
                    matches_img = None
                    try:
                        kp_left, desc_left = detect_and_describe(
                            left,
                            feature=args.feature,
                            nfeatures=args.nfeatures,
                        )
                        kp_right, desc_right = detect_and_describe(
                            right,
                            feature=args.feature,
                            nfeatures=args.nfeatures,
                        )

                        good_matches, raw_matches = match_descriptors(
                            desc_left,
                            desc_right,
                            method=args.feature,
                            ratio=args.ratio,
                        )

                        last_stats["n_kp_left"] = len(kp_left)
                        last_stats["n_kp_right"] = len(kp_right)
                        last_stats["n_matches_raw"] = raw_matches
                        last_stats["n_matches_good"] = len(good_matches)

                        matches_img = draw_matches(left, kp_left, right, kp_right, good_matches)

                        if len(good_matches) < args.min_matches:
                            raise RuntimeError(
                                f"not_enough_matches: {len(good_matches)} < {args.min_matches}"
                            )

                        H_est, mask = estimate_homography(
                            kp_left,
                            kp_right,
                            good_matches,
                            ransac_thresh=args.ransac_thresh,
                        )
                        if H_est is None or mask is None:
                            raise RuntimeError("findHomography_failed")

                        inliers_mask = mask.ravel().tolist()
                        inlier_count = int(sum(inliers_mask))
                        inlier_ratio = (
                            float(inlier_count) / float(len(inliers_mask))
                            if inliers_mask
                            else 0.0
                        )

                        last_stats["n_inliers"] = inlier_count
                        last_stats["inlier_ratio"] = inlier_ratio

                        inliers_img = draw_matches(
                            left,
                            kp_left,
                            right,
                            kp_right,
                            good_matches,
                            inlier_mask=inliers_mask,
                        )

                        _save_keyframe_debug(output_dir, source_idx, matches_img, inliers_img)

                        valid_H_raw = H_est
                        H_raw = valid_H_raw
                        inliers_ok.append(inlier_count)
                        inlier_ratios_ok.append(inlier_ratio)
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

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, fps_value, canvas_size)
                    if not writer.isOpened():
                        raise RuntimeError("Failed to open VideoWriter for stitched.mp4")

                left_raw_warped, right_raw_warped = warp_pair(
                    left,
                    right,
                    H_raw,
                    canvas_size,
                    T,
                )
                left_sm_warped, right_sm_warped = warp_pair(
                    left,
                    right,
                    H_sm,
                    canvas_size,
                    T,
                )

                if args.smooth_h == "none":
                    left_active = left_raw_warped
                    right_active = right_raw_warped
                else:
                    left_active = left_sm_warped
                    right_active = right_sm_warped

                overlay_raw = overlay_images(left_raw_warped, right_raw_warped, alpha=0.5)
                overlay_sm = overlay_images(left_sm_warped, right_sm_warped, alpha=0.5)
                overlay_active = overlay_sm if args.smooth_h != "none" else overlay_raw
                seam_side_mask = None
                if args.blend == "hard" or args.seam == "hard":
                    from stitching.blending import resolve_hard_seam_side  # noqa: E402

                    _, base_mask_left, base_mask_right, _ = _compute_alpha_for_blend(
                        left_active,
                        right_active,
                        mode="none",
                    )
                    seam_side_mask = resolve_hard_seam_side(
                        base_mask_left,
                        base_mask_right,
                        seam_side_mask=None,
                    )
                    stitched = _blend_frames(
                        left_active,
                        right_active,
                        args.blend,
                        mask_left=base_mask_left,
                        mask_right=base_mask_right,
                        seam_side_mask=seam_side_mask,
                    )
                else:
                    stitched = _blend_frames(left_active, right_active, args.blend)

                alpha_left, mask_left, mask_right, overlap_bin = _compute_alpha_for_blend(
                    left_active,
                    right_active,
                    args.blend,
                    seam_side_mask=seam_side_mask,
                )
                alpha_float = ensure_float_mask(alpha_left, shape=mask_left.shape)
                alpha_u8 = (alpha_float * 255.0).clip(0, 255).astype("uint8")
                overlap_u8 = ensure_uint8_mask(overlap_bin, shape=mask_left.shape)

                if debug_masks and (processed_idx % snapshot_stride == 0):
                    snapshots_dir = output_dir / "snapshots"
                    save_mask_png(snapshots_dir / f"mask_left_{source_idx:06d}.png", mask_left)
                    save_mask_png(snapshots_dir / f"mask_right_{source_idx:06d}.png", mask_right)
                    save_mask_png(
                        snapshots_dir / f"mask_overlap_{source_idx:06d}.png",
                        overlap_u8,
                    )
                    save_mask_png(snapshots_dir / f"alpha_{source_idx:06d}.png", alpha_u8)
                    if args.blend == "hard":
                        seam_hard_u8 = ensure_uint8_mask(
                            seam_side_mask.astype("uint8"),
                            shape=mask_left.shape,
                        )
                        save_mask_png(
                            snapshots_dir / f"seam_hard_{source_idx:06d}.png",
                            seam_hard_u8,
                        )
                        from stitching.viz import save_image  # noqa: E402

                        save_image(
                            snapshots_dir / f"stitched_hard_{source_idx:06d}.png",
                            stitched,
                        )

                    for name, m in [
                        ("mask_left", mask_left),
                        ("mask_right", mask_right),
                        ("mask_overlap", overlap_u8),
                        ("alpha", alpha_u8),
                    ]:
                        stats = summarize_mask(name, m)
                        logging.info(
                            "mask_stats frame=%s name=%s min=%.3f max=%.3f mean=%.3f unique=%.0f overlap_ratio=%.4f",
                            source_idx,
                            name,
                            stats["min"],
                            stats["max"],
                            stats["mean"],
                            stats["unique_count"],
                            stats["overlap_ratio"],
                        )

                writer.write(stitched)

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

                if processed_idx % snapshot_every == 0:
                    _save_snapshot(
                        output_dir,
                        source_idx,
                        left,
                        right,
                        stitched,
                        overlay_active,
                        overlay_raw,
                        overlay_sm,
                    )

                runtime_ms = (time.perf_counter() - frame_start) * 1000.0
                frame_runtimes.append(runtime_ms)

                if status == "OK":
                    debug["success_frames"] += 1
                if status == "FALLBACK":
                    debug["fallback_frames"] += 1

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

    debug["jitter_summary"] = {
        "mean_raw": jitter_raw_mean,
        "p95_raw": jitter_raw_p95,
        "mean_sm": jitter_sm_mean,
        "p95_sm": jitter_sm_p95,
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
