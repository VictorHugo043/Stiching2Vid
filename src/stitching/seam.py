"""Graph-cut style seam estimation constrained to overlap ROI."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple


def _as_bool(mask, shape):
    import numpy as np  # type: ignore

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape:
        raise ValueError(f"mask shape mismatch: expected {shape}, got {arr.shape}")
    if arr.dtype == np.bool_:
        return arr
    return arr > 0


def _bbox_from_mask(mask_bool):
    ys, xs = mask_bool.nonzero()
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _resize_bool(mask_bool, new_w, new_h):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    resized = cv2.resize(
        mask_bool.astype("uint8"),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.bool_)


def _try_opencv_graphcut(left_roi, right_roi, overlap_roi):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if hasattr(cv2, "detail_GraphCutSeamFinder"):
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    elif hasattr(cv2, "detail") and hasattr(cv2.detail, "GraphCutSeamFinder"):
        finder = cv2.detail.GraphCutSeamFinder("COST_COLOR")
    else:
        raise RuntimeError("OpenCV GraphCutSeamFinder not available")

    masks = [
        (overlap_roi.astype("uint8") * 255),
        (overlap_roi.astype("uint8") * 255),
    ]
    images = [left_roi.astype("int16"), right_roi.astype("int16")]
    corners = [(0, 0), (0, 0)]
    finder.find(images, corners, masks)
    seam_left = (masks[0] > 0) & overlap_roi
    return seam_left.astype(np.bool_)


def _dp_vertical_seam(left_roi, right_roi, overlap_roi):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    h, w = overlap_roi.shape
    if h == 0 or w == 0:
        return None

    left_gray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY).astype("float32")
    right_gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY).astype("float32")
    gx_l = cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_l = cv2.Sobel(left_gray, cv2.CV_32F, 0, 1, ksize=3)
    gx_r = cv2.Sobel(right_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(right_gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_diff = ((gx_l - gx_r) ** 2 + (gy_l - gy_r) ** 2) ** 0.5
    color_diff = abs(left_gray - right_gray)
    cost = color_diff + 0.5 * grad_diff
    cost[~overlap_roi] = 1e6

    dp = cost.copy()
    parent = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h):
        for x in range(w):
            candidates = [x]
            if x > 0:
                candidates.append(x - 1)
            if x + 1 < w:
                candidates.append(x + 1)
            best_prev = min(candidates, key=lambda c: dp[y - 1, c])
            dp[y, x] += dp[y - 1, best_prev]
            parent[y, x] = best_prev

    end_x = int(dp[h - 1].argmin())
    seam_cols = [end_x]
    for y in range(h - 1, 0, -1):
        seam_cols.append(int(parent[y, seam_cols[-1]]))
    seam_cols.reverse()

    seam_left = np.zeros((h, w), dtype=np.bool_)
    for y in range(h):
        x_cut = seam_cols[y]
        seam_left[y, : x_cut + 1] = True
    seam_left &= overlap_roi
    return seam_left


def compute_seam_mask_graphcut(
    left,
    right_warp,
    mask_left,
    mask_right,
    scale: float = 0.5,
    min_overlap_area: int = 400,
):
    """Compute full-size seam side mask with ROI-only seam optimization.

    Returns:
        seam_side_mask: bool mask where True selects left.
        meta: dict with overlap/bbox/method/compute time.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    t0 = time.perf_counter()
    h, w = left.shape[:2]
    left_valid = _as_bool(mask_left, (h, w))
    right_valid = _as_bool(mask_right, (h, w))
    overlap = left_valid & right_valid
    overlap_area = int(overlap.sum())

    x0y0x1y1 = _bbox_from_mask(overlap)
    meta = {
        "overlap_area": overlap_area,
        "overlap_bbox": None,
        "seam_method": "graphcut",
        "seam_scale": float(scale),
        "seam_compute_ms": 0.0,
        "fallback": False,
        "note": "",
    }

    # Deterministic selection outside overlap.
    seam_side_full = left_valid.copy()
    seam_side_full[right_valid & (~left_valid)] = False

    if x0y0x1y1 is None or overlap_area < min_overlap_area:
        meta["fallback"] = True
        meta["note"] = "overlap_too_small"
        meta["seam_compute_ms"] = (time.perf_counter() - t0) * 1000.0
        return None, meta

    x0, y0, x1, y1 = x0y0x1y1
    meta["overlap_bbox"] = [x0, y0, x1, y1]
    roi_left = left[y0 : y1 + 1, x0 : x1 + 1]
    roi_right = right_warp[y0 : y1 + 1, x0 : x1 + 1]
    roi_overlap = overlap[y0 : y1 + 1, x0 : x1 + 1]
    roi_h, roi_w = roi_overlap.shape

    scale = min(1.0, max(0.2, float(scale)))
    new_w = max(8, int(round(roi_w * scale)))
    new_h = max(8, int(round(roi_h * scale)))

    left_s = cv2.resize(roi_left, (new_w, new_h), interpolation=cv2.INTER_AREA)
    right_s = cv2.resize(roi_right, (new_w, new_h), interpolation=cv2.INTER_AREA)
    overlap_s = _resize_bool(roi_overlap, new_w, new_h)

    try:
        seam_left_s = _try_opencv_graphcut(left_s, right_s, overlap_s)
    except Exception:
        seam_left_s = _dp_vertical_seam(left_s, right_s, overlap_s)
        meta["fallback"] = True
        meta["seam_method"] = "dp_fallback"

    if seam_left_s is None:
        meta["fallback"] = True
        meta["note"] = "seam_failed"
        meta["seam_compute_ms"] = (time.perf_counter() - t0) * 1000.0
        return None, meta

    seam_left_roi = cv2.resize(
        seam_left_s.astype("uint8"),
        (roi_w, roi_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.bool_)
    seam_left_roi &= roi_overlap
    seam_side_full[y0 : y1 + 1, x0 : x1 + 1][roi_overlap] = seam_left_roi[roi_overlap]

    meta["seam_compute_ms"] = (time.perf_counter() - t0) * 1000.0
    return seam_side_full.astype(np.bool_), meta


def overlap_diff_roi_image(left, right_warp, overlap_bbox):
    """Return ROI absolute difference image for debug visualization."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    h, w = left.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if not overlap_bbox:
        return out
    x0, y0, x1, y1 = overlap_bbox
    x0 = max(0, min(w - 1, int(x0)))
    x1 = max(0, min(w - 1, int(x1)))
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(0, min(h - 1, int(y1)))
    if x1 < x0 or y1 < y0:
        return out
    diff = cv2.absdiff(left[y0 : y1 + 1, x0 : x1 + 1], right_warp[y0 : y1 + 1, x0 : x1 + 1])
    out[y0 : y1 + 1, x0 : x1 + 1] = diff
    return out

