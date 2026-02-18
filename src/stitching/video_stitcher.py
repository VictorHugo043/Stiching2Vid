"""Frame0-reuse video stitching pipeline.

This module keeps the project-specific two-view data flow:
warp -> crop -> seam -> blend.
It does not estimate homography itself; geometry is provided by caller.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from stitching.video_state import VideoStitchState


def _as_u8(mask):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    arr = mask
    if hasattr(arr, "get"):
        arr = arr.get()
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
    return (arr > 0).astype(np.uint8) * 255


def _resolve_seam_masks(left_mask_full, right_mask_full, seam_left_full, seam_right_full):
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
    return final_left, final_right


def _compose_single_roi_on_canvas(canvas_size, roi_img, roi_mask, corner):
    import numpy as np  # type: ignore

    from stitching.seam_opencv import place_mask_on_canvas, place_roi_on_canvas

    canvas_w, canvas_h = int(canvas_size[0]), int(canvas_size[1])
    canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    place_roi_on_canvas(canvas_img, roi_img, corner)
    place_mask_on_canvas(canvas_mask, roi_mask, corner)
    return canvas_img, canvas_mask


def _mask_bbox_area(mask) -> int:
    import numpy as np  # type: ignore

    valid = np.asarray(mask) > 0
    if not valid.any():
        return 0
    ys, xs = np.where(valid)
    return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def _compose_masks_panorama(masks, corners, sizes):
    import numpy as np  # type: ignore

    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    min_x = min(int(c[0]) for c in corners)
    min_y = min(int(c[1]) for c in corners)
    max_x = max(int(c[0]) + int(s[0]) for c, s in zip(corners, sizes))
    max_y = max(int(c[1]) + int(s[1]) for c, s in zip(corners, sizes))
    pano = np.zeros((max(1, max_y - min_y), max(1, max_x - min_x)), dtype=np.uint8)
    for mask, corner in zip(masks, corners):
        arr = _as_u8(mask)
        x = int(corner[0]) - min_x
        y = int(corner[1]) - min_y
        h, w = arr.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(pano.shape[1], x + w)
        y1 = min(pano.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        mx0 = x0 - x
        my0 = y0 - y
        mx1 = mx0 + (x1 - x0)
        my1 = my0 + (y1 - y0)
        pano[y0:y1, x0:x1] = np.maximum(pano[y0:y1, x0:x1], arr[my0:my1, mx0:mx1])
    return pano


class VideoStitcher:
    """Stitcher that supports frame0 initialization and frame reuse."""

    def __init__(
        self,
        *,
        seam_method: str,
        seam_megapix: float,
        seam_dilate: int,
        blend_mode: str,
        mb_levels: int,
        crop_enabled: bool,
        lir_method: str,
        lir_erode: int,
        crop_debug: int,
        reuse_mode: str,
        output_dir: Path,
        warning_handler: Optional[Callable[[str], None]] = None,
    ):
        self.seam_method = seam_method
        self.seam_megapix = float(seam_megapix)
        self.seam_dilate = int(seam_dilate)
        self.blend_mode = blend_mode
        self.mb_levels = int(mb_levels)
        self.crop_enabled = bool(crop_enabled)
        self.lir_method = lir_method
        self.lir_erode = int(lir_erode)
        self.crop_debug = int(crop_debug)
        self.reuse_mode = reuse_mode
        self.output_dir = output_dir
        self.warning_handler = warning_handler
        self.state = VideoStitchState()

    def _warn(self, message: str) -> None:
        logging.warning(message)
        if self.warning_handler is not None:
            self.warning_handler(message)

    def _blend(self, left_canvas, right_canvas, left_mask, right_mask):
        from stitching.blending import blend_none, feather_blend, multiband_blend

        if self.blend_mode == "none":
            return blend_none(left_canvas, right_canvas, left_mask=left_mask, right_mask=right_mask)
        if self.blend_mode == "multiband":
            return multiband_blend(
                left_canvas,
                right_canvas,
                left_mask=left_mask,
                right_mask=right_mask,
                levels=self.mb_levels,
            )
        return feather_blend(left_canvas, right_canvas, left_mask=left_mask, right_mask=right_mask)

    def _compute_low(self, left_frame, right_frame, H, T):
        import cv2  # type: ignore

        from stitching.seam_opencv import compute_seam_scale, scale_homography, warp_to_roi

        seam_scale = compute_seam_scale(self.seam_megapix, (left_frame.shape[0], left_frame.shape[1]))
        left_w = max(1, int(round(left_frame.shape[1] * seam_scale)))
        left_h = max(1, int(round(left_frame.shape[0] * seam_scale)))
        right_w = max(1, int(round(right_frame.shape[1] * seam_scale)))
        right_h = max(1, int(round(right_frame.shape[0] * seam_scale)))
        left_small = cv2.resize(left_frame, (left_w, left_h), interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_frame, (right_w, right_h), interpolation=cv2.INTER_AREA)
        T_small = scale_homography(T, seam_scale)
        H_small = scale_homography(H, seam_scale)
        left_low, left_low_mask, left_low_corner = warp_to_roi(left_small, T_small)
        right_low, right_low_mask, right_low_corner = warp_to_roi(right_small, T_small @ H_small)
        return {
            "seam_scale": seam_scale,
            "imgs": [left_low, right_low],
            "masks": [left_low_mask, right_low_mask],
            "corners": [left_low_corner, right_low_corner],
            "sizes": [
                (int(left_low.shape[1]), int(left_low.shape[0])),
                (int(right_low.shape[1]), int(right_low.shape[0])),
            ],
        }

    def _compute_final_rois(self, left_frame, right_frame, H, T, canvas_size):
        from stitching.seam_opencv import summarize_overlap, warp_to_roi

        left_roi, left_mask, left_corner = warp_to_roi(left_frame, T)
        right_roi, right_mask, right_corner = warp_to_roi(right_frame, T @ H)
        left_canvas, left_mask_full = _compose_single_roi_on_canvas(canvas_size, left_roi, left_mask, left_corner)
        right_canvas, right_mask_full = _compose_single_roi_on_canvas(
            canvas_size,
            right_roi,
            right_mask,
            right_corner,
        )
        overlap_stats = summarize_overlap(left_mask_full, right_mask_full)
        return {
            "imgs": [left_roi, right_roi],
            "masks": [left_mask, right_mask],
            "corners": [left_corner, right_corner],
            "sizes": [
                (int(left_roi.shape[1]), int(left_roi.shape[0])),
                (int(right_roi.shape[1]), int(right_roi.shape[0])),
            ],
            "canvas_imgs": [left_canvas, right_canvas],
            "canvas_masks": [left_mask_full, right_mask_full],
            "overlap_area": int(overlap_stats.get("overlap_area", 0)),
        }

    def _maybe_crop(self, low_data: Dict[str, object], final_data: Dict[str, object], frame_idx: int):
        crop_output = {
            "crop_applied": False,
            "crop_method": "none",
            "cropper": None,
            "lir_rect": None,
            "mask_area_before": 0,
            "mask_area_after": 0,
            "mask_bbox_before": 0,
            "mask_bbox_after": 0,
        }
        if not self.crop_enabled:
            return low_data, final_data, crop_output

        from stitching.cropper import Cropper
        from stitching.viz import save_image

        def _crop_warn(msg: str) -> None:
            self._warn(f"crop_warning frame={frame_idx}: {msg}")

        cropper = Cropper(
            crop=True,
            lir_method=self.lir_method,
            lir_erode=self.lir_erode,
            warning_handler=_crop_warn,
        )
        cropper.prepare(
            low_data["imgs"],
            low_data["masks"],
            low_data["corners"],
            low_data["sizes"],
        )
        if cropper.panorama_mask is None or cropper.lir_rectangle is None:
            return low_data, final_data, crop_output

        mask_before = cropper.panorama_mask
        bbox_before = _mask_bbox_area(mask_before)
        lir_ratio = float(cropper.lir_rectangle.area) / float(max(1, bbox_before))
        if lir_ratio < 0.30:
            self._warn(
                f"crop_fallback_to_no_crop frame={frame_idx}: lir_ratio={lir_ratio:.3f} < 0.30",
            )
            if self.crop_debug == 1:
                snapshots_dir = self.output_dir / "snapshots"
                snapshots_dir.mkdir(parents=True, exist_ok=True)
                save_image(snapshots_dir / "frame0_panorama_mask.png", mask_before)
                save_image(
                    snapshots_dir / "frame0_lir.png",
                    cropper.lir_rectangle.draw_on(mask_before.copy(), color=(0, 0, 255), size=2),
                )
            return low_data, final_data, crop_output

        try:
            low_imgs_c = list(cropper.crop_images(low_data["imgs"]))
            low_masks_c = list(cropper.crop_images(low_data["masks"]))
            low_corners_c, low_sizes_c = cropper.crop_rois(low_data["corners"], low_data["sizes"])

            aspect_candidates: List[float] = []
            for (fw, fh), (lw, lh) in zip(final_data["sizes"], low_data["sizes"]):
                aspect_candidates.append(float(fw) / float(max(1, lw)))
                aspect_candidates.append(float(fh) / float(max(1, lh)))
            crop_aspect = max(min(aspect_candidates) - 1e-6, 1e-6)

            final_imgs_c = list(cropper.crop_images(final_data["imgs"], aspect=crop_aspect))
            final_masks_c = list(cropper.crop_images(final_data["masks"], aspect=crop_aspect))
            abs_overlaps = cropper.get_overlaps_absolute(aspect=crop_aspect)
            final_corners_abs = [rect.corner for rect in abs_overlaps]
            final_sizes_abs = [(int(img.shape[1]), int(img.shape[0])) for img in final_imgs_c]
        except Exception as crop_exc:
            self._warn(f"crop_fallback_to_no_crop frame={frame_idx}: {crop_exc}")
            return low_data, final_data, crop_output

        low_c = dict(low_data)
        low_c["imgs"] = low_imgs_c
        low_c["masks"] = low_masks_c
        low_c["corners"] = low_corners_c
        low_c["sizes"] = low_sizes_c

        final_c = dict(final_data)
        final_c["imgs"] = final_imgs_c
        final_c["masks"] = final_masks_c
        final_c["corners"] = final_corners_abs
        final_c["sizes"] = final_sizes_abs

        mask_after = _compose_masks_panorama(low_masks_c, low_corners_c, low_sizes_c)
        crop_output.update(
            {
                "crop_applied": True,
                "crop_method": cropper.lir_method_used,
                "cropper": cropper,
                "crop_aspect": crop_aspect,
                "lir_rect": cropper.lir_rectangle,
                "mask_area_before": int((mask_before > 0).sum()),
                "mask_area_after": int((mask_after > 0).sum()),
                "mask_bbox_before": int(bbox_before),
                "mask_bbox_after": int(_mask_bbox_area(mask_after)),
                "panorama_mask_low": mask_before,
            }
        )
        if self.crop_debug == 1:
            snapshots_dir = self.output_dir / "snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            save_image(snapshots_dir / "frame0_panorama_mask.png", mask_before)
            save_image(
                snapshots_dir / "frame0_lir.png",
                cropper.lir_rectangle.draw_on(mask_before.copy(), color=(0, 0, 255), size=2),
            )
        return low_c, final_c, crop_output

    def _compute_seam_masks_low(self, imgs, corners, masks):
        from stitching.seam_opencv import compute_seam_masks_opencv

        return compute_seam_masks_opencv(imgs, corners, masks, method=self.seam_method)

    def _compose_final_masks(self, seam_masks_low, final_data):
        import numpy as np  # type: ignore

        from stitching.seam_opencv import place_mask_on_canvas, resize_seam_to_compose

        left_roi_mask, right_roi_mask = final_data["masks"]
        left_corner, right_corner = final_data["corners"]
        left_canvas_mask, right_canvas_mask = final_data["canvas_masks"]

        seam_left_roi = resize_seam_to_compose(
            seam_masks_low[0],
            left_roi_mask,
            dilate_iter=self.seam_dilate,
        )
        seam_right_roi = resize_seam_to_compose(
            seam_masks_low[1],
            right_roi_mask,
            dilate_iter=self.seam_dilate,
        )
        seam_left_full = np.zeros_like(left_canvas_mask, dtype=np.uint8)
        seam_right_full = np.zeros_like(right_canvas_mask, dtype=np.uint8)
        place_mask_on_canvas(seam_left_full, seam_left_roi, left_corner)
        place_mask_on_canvas(seam_right_full, seam_right_roi, right_corner)

        final_left_mask, final_right_mask = _resolve_seam_masks(
            left_canvas_mask,
            right_canvas_mask,
            seam_left_full,
            seam_right_full,
        )
        return final_left_mask, final_right_mask

    def initialize_from_first_frame(
        self,
        left_frame,
        right_frame,
        H,
        T,
        canvas_size,
        frame_idx: int,
    ) -> Dict[str, object]:
        """Initialize reusable state from first (or re-init) frame."""

        from stitching.seam_opencv import seam_overlay_preview
        from stitching.viz import save_image

        init_t0 = time.perf_counter()
        low_data = self._compute_low(left_frame, right_frame, H, T)
        final_data = self._compute_final_rois(left_frame, right_frame, H, T, canvas_size)
        low_data, final_data, crop_out = self._maybe_crop(low_data, final_data, frame_idx)

        seam_low = self._compute_seam_masks_low(low_data["imgs"], low_data["corners"], low_data["masks"])
        final_left_mask, final_right_mask = self._compose_final_masks(seam_low, final_data)
        stitched = self._blend(
            final_data["canvas_imgs"][0],
            final_data["canvas_imgs"][1],
            final_left_mask,
            final_right_mask,
        )

        # Frame0 snapshots used to prove initialization correctness.
        snapshots_dir = self.output_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np  # type: ignore

        from stitching.seam_opencv import place_mask_on_canvas

        canvas_low_size = (
            max(1, max(int(c[0]) + int(s[0]) for c, s in zip(low_data["corners"], low_data["sizes"]))),
            max(1, max(int(c[1]) + int(s[1]) for c, s in zip(low_data["corners"], low_data["sizes"]))),
        )
        left_low_canvas, left_low_full_mask = _compose_single_roi_on_canvas(
            canvas_low_size,
            low_data["imgs"][0],
            low_data["masks"][0],
            low_data["corners"][0],
        )
        right_low_canvas, right_low_full_mask = _compose_single_roi_on_canvas(
            canvas_low_size,
            low_data["imgs"][1],
            low_data["masks"][1],
            low_data["corners"][1],
        )
        seam_left_low_full = np.zeros_like(left_low_full_mask, dtype=np.uint8)
        seam_right_low_full = np.zeros_like(right_low_full_mask, dtype=np.uint8)
        place_mask_on_canvas(seam_left_low_full, seam_low[0], low_data["corners"][0])
        place_mask_on_canvas(seam_right_low_full, seam_low[1], low_data["corners"][1])
        seam_overlay = seam_overlay_preview(
            left_low_canvas,
            right_low_canvas,
            left_low_full_mask,
            right_low_full_mask,
            seam_left_low_full,
            seam_right_low_full,
        )
        save_image(snapshots_dir / "frame0_warp_low.png", left_low_canvas)
        save_image(snapshots_dir / "frame0_mask_low.png", left_low_full_mask)
        if crop_out.get("panorama_mask_low") is not None:
            save_image(snapshots_dir / "frame0_panorama_mask.png", crop_out["panorama_mask_low"])
        if crop_out.get("lir_rect") is not None and crop_out.get("panorama_mask_low") is not None:
            save_image(
                snapshots_dir / "frame0_lir.png",
                crop_out["lir_rect"].draw_on(crop_out["panorama_mask_low"].copy(), color=(0, 0, 255), size=2),
            )
        save_image(snapshots_dir / "frame0_seam_mask_left.png", _as_u8(seam_low[0]))
        save_image(snapshots_dir / "frame0_seam_mask_right.png", _as_u8(seam_low[1]))
        save_image(snapshots_dir / "frame0_seam_overlay.png", seam_overlay)

        self.state.initialized = True
        self.state.H_or_cameras = H
        self.state.cropper_state = crop_out.get("cropper")
        self.state.seam_masks_low = [seam_low[0], seam_low[1]]
        self.state.corners_low = list(low_data["corners"])
        self.state.sizes_low = list(low_data["sizes"])
        self.state.corners_final = list(final_data["corners"])
        self.state.sizes_final = list(final_data["sizes"])
        self.state.frame0_index = int(frame_idx)
        self.state.init_timestamp = time.time()
        self.state.init_count += 1
        self.state.metadata = {
            "seam_method": self.seam_method,
            "seam_scale": float(low_data["seam_scale"]),
            "crop_applied": bool(crop_out.get("crop_applied", False)),
            "crop_method": crop_out.get("crop_method", "none"),
            "crop_rect": (
                {
                    "x": int(crop_out["lir_rect"].x),
                    "y": int(crop_out["lir_rect"].y),
                    "w": int(crop_out["lir_rect"].w),
                    "h": int(crop_out["lir_rect"].h),
                }
                if crop_out.get("lir_rect") is not None
                else None
            ),
            "overlap_area_init": int(final_data["overlap_area"]),
            "overlap_area_current": int(final_data["overlap_area"]),
            "mask_area_before": int(crop_out.get("mask_area_before", 0)),
            "mask_area_after": int(crop_out.get("mask_area_after", 0)),
            "mask_bbox_before": int(crop_out.get("mask_bbox_before", 0)),
            "mask_bbox_after": int(crop_out.get("mask_bbox_after", 0)),
            "init_ms": float((time.perf_counter() - init_t0) * 1000.0),
        }
        return {
            "stitched": stitched,
            "overlap_area": int(final_data["overlap_area"]),
            "init_ms": float(self.state.metadata["init_ms"]),
            "crop_applied": bool(crop_out.get("crop_applied", False)),
            "crop_method": str(crop_out.get("crop_method", "none")),
            "crop_rect": self.state.metadata.get("crop_rect"),
            "seam_compute_ms": 0.0,
        }

    def stitch_frame(
        self,
        left_frame,
        right_frame,
        H,
        T,
        canvas_size,
        frame_idx: int,
        recompute_seam: bool = False,
    ) -> Dict[str, object]:
        """Stitch one frame using cached state with optional seam refresh."""

        if not self.state.initialized:
            raise RuntimeError("VideoStitcher state is not initialized")

        final_data = self._compute_final_rois(left_frame, right_frame, H, T, canvas_size)
        low_data = self._compute_low(left_frame, right_frame, H, T)
        crop_applied = False
        crop_method = "none"
        crop_rect = self.state.metadata.get("crop_rect")

        if self.state.cropper_state is not None:
            cropper = self.state.cropper_state
            try:
                crop_applied = True
                crop_method = str(self.state.metadata.get("crop_method", "fallback"))
                low_data["imgs"] = list(cropper.crop_images(low_data["imgs"]))
                low_data["masks"] = list(cropper.crop_images(low_data["masks"]))
                low_data["corners"], low_data["sizes"] = cropper.crop_rois(
                    low_data["corners"],
                    low_data["sizes"],
                )

                aspect_candidates: List[float] = []
                for (fw, fh), (lw, lh) in zip(final_data["sizes"], low_data["sizes"]):
                    aspect_candidates.append(float(fw) / float(max(1, lw)))
                    aspect_candidates.append(float(fh) / float(max(1, lh)))
                crop_aspect = max(min(aspect_candidates) - 1e-6, 1e-6)
                final_data["imgs"] = list(cropper.crop_images(final_data["imgs"], aspect=crop_aspect))
                final_data["masks"] = list(cropper.crop_images(final_data["masks"], aspect=crop_aspect))
                final_data["corners"] = [
                    rect.corner for rect in cropper.get_overlaps_absolute(aspect=crop_aspect)
                ]
                final_data["sizes"] = [(int(img.shape[1]), int(img.shape[0])) for img in final_data["imgs"]]
            except Exception as crop_exc:
                self._warn(f"crop_fallback_to_no_crop frame={frame_idx}: {crop_exc}")
                crop_applied = False
                crop_method = "fallback_no_crop"

        seam_t0 = time.perf_counter()
        if recompute_seam or self.state.seam_masks_low is None:
            seam_low = self._compute_seam_masks_low(low_data["imgs"], low_data["corners"], low_data["masks"])
            seam_compute_ms = (time.perf_counter() - seam_t0) * 1000.0
            if self.reuse_mode in {"frame0_all", "frame0_seam"}:
                self.state.seam_masks_low = [seam_low[0], seam_low[1]]
        else:
            seam_low = self.state.seam_masks_low
            seam_compute_ms = 0.0

        final_left_mask, final_right_mask = self._compose_final_masks(seam_low, final_data)
        stitched = self._blend(
            final_data["canvas_imgs"][0],
            final_data["canvas_imgs"][1],
            final_left_mask,
            final_right_mask,
        )
        self.state.metadata["overlap_area_current"] = int(final_data["overlap_area"])
        return {
            "stitched": stitched,
            "overlap_area": int(final_data["overlap_area"]),
            "seam_compute_ms": float(seam_compute_ms),
            "crop_applied": bool(crop_applied),
            "crop_method": crop_method,
            "crop_rect": crop_rect,
        }
