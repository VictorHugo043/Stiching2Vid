"""OpenCV-style seam utilities for two-view stitching.

This module follows the same principle used by OpenCV stitching_detailed:
1) compute seam on warped low-resolution images,
2) resize seam mask to full-resolution compose mask,
3) bitwise_and with final valid warp mask.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def _as_uint8_mask(mask) -> "np.ndarray":
    """Normalize mask to uint8 {0,255}."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if mask is None:
        raise ValueError("mask cannot be None")
    if hasattr(mask, "get"):
        mask = mask.get()
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = np.where(arr > 0, 255, 0).astype(np.uint8)
    else:
        arr = np.where(arr > 0, 255, 0).astype(np.uint8)
    return arr


def scale_homography(H, scale: float):
    """Scale homography between full-res and seam-res coordinate systems.

    Formula follows OpenCV-style convention: H_s = S * H * S^-1,
    where S = diag(scale, scale, 1).
    """

    import numpy as np  # type: ignore

    if H is None:
        return None
    s = float(scale)
    if s <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    S = np.array(
        [
            [s, 0.0, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    S_inv = np.array(
        [
            [1.0 / s, 0.0, 0.0],
            [0.0, 1.0 / s, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return S @ H @ S_inv


def compute_seam_scale(megapix: float, image_shape: Tuple[int, int]) -> float:
    """Compute seam scale from megapix budget.

    Args:
        megapix: target megapixels for seam estimation, e.g. 0.1.
        image_shape: (height, width) of source image.

    Returns:
        Scale in (0, 1].
    """

    import numpy as np  # type: ignore

    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        return 1.0
    if megapix <= 0:
        return 1.0

    scale = np.sqrt((float(megapix) * 1_000_000.0) / float(w * h))
    return float(min(1.0, max(scale, 1e-6)))


def warp_to_roi(img, M_3x3) -> Tuple["np.ndarray", "np.ndarray", Tuple[int, int]]:
    """Warp one image to a tight ROI in panorama coordinates.

    Args:
        img: Source BGR image.
        M_3x3: 3x3 transform mapping source coordinates -> panorama coordinates.

    Returns:
        roi_img: warped ROI image.
        roi_mask_u8: warped valid mask in uint8 (0/255).
        corner_xy: top-left ROI corner in panorama coordinates.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if img is None:
        raise ValueError("img cannot be None")
    if M_3x3 is None:
        raise ValueError("M_3x3 cannot be None")

    h, w = img.shape[:2]
    src_corners = np.array(
        [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    dst_corners = cv2.perspectiveTransform(src_corners, M_3x3).reshape(-1, 2)

    min_x = int(np.floor(dst_corners[:, 0].min()))
    min_y = int(np.floor(dst_corners[:, 1].min()))
    max_x = int(np.ceil(dst_corners[:, 0].max()))
    max_y = int(np.ceil(dst_corners[:, 1].max()))

    roi_w = max(1, max_x - min_x + 1)
    roi_h = max(1, max_y - min_y + 1)

    shift = np.array(
        [
            [1.0, 0.0, -float(min_x)],
            [0.0, 1.0, -float(min_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    M_roi = shift @ M_3x3

    roi_img = cv2.warpPerspective(
        img,
        M_roi,
        (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    src_mask = np.full((h, w), 255, dtype=np.uint8)
    roi_mask = cv2.warpPerspective(
        src_mask,
        M_roi,
        (roi_w, roi_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    roi_mask = _as_uint8_mask(roi_mask)
    return roi_img, roi_mask, (min_x, min_y)


def compute_seam_masks_opencv(
    warped_roi_imgs: Sequence,
    corners: Sequence[Tuple[int, int]],
    warped_roi_masks: Sequence,
    method: str = "dp_color",
) -> List["np.ndarray"]:
    """Compute seam masks using OpenCV seam finders.

    Args:
        warped_roi_imgs: list of warped ROI images in the same pano coordinate system.
        corners: list of ROI top-left corners in pano coordinates.
        warped_roi_masks: list of valid ROI masks (uint8 0/255).
        method: one of `dp_color`, `dp_colorgrad`, `voronoi`, `none`.

    Returns:
        List of seam masks in uint8 {0,255}.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if len(warped_roi_imgs) != len(corners) or len(corners) != len(warped_roi_masks):
        raise ValueError("warped_roi_imgs/corners/warped_roi_masks length mismatch")

    method = (method or "dp_color").lower()
    masks = [_as_uint8_mask(m).copy() for m in warped_roi_masks]
    if method in {"none", "no"}:
        return masks

    imgs_f = [img.astype(np.float32) for img in warped_roi_imgs]
    corners_i = [(int(c[0]), int(c[1])) for c in corners]

    if method == "dp_color":
        seam_finder = cv2.detail_DpSeamFinder("COLOR")
    elif method == "dp_colorgrad":
        seam_finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
    elif method == "voronoi":
        seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
    else:
        raise ValueError(f"Unsupported seam method: {method}")

    out = seam_finder.find(imgs_f, corners_i, masks)

    if out is None:
        seam_masks = masks
    else:
        seam_masks = list(out)

    normalized = [_as_uint8_mask(m) for m in seam_masks]
    return normalized


def resize_seam_to_compose(
    seam_mask_low,
    target_mask_full,
    dilate_iter: int = 1,
) -> "np.ndarray":
    """Resize seam mask from seam-scale ROI to full compose ROI.

    Following stitching_detailed style:
    - optional dilate,
    - resize to target mask size,
    - bitwise_and with final valid warp mask.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    seam_mask = _as_uint8_mask(seam_mask_low)
    target_mask = _as_uint8_mask(target_mask_full)

    if dilate_iter > 0:
        seam_mask = cv2.dilate(seam_mask, None, iterations=int(dilate_iter))

    interp_exact = getattr(cv2, "INTER_LINEAR_EXACT", cv2.INTER_LINEAR)
    seam_resized = cv2.resize(
        seam_mask,
        (target_mask.shape[1], target_mask.shape[0]),
        interpolation=interp_exact,
    )
    seam_resized = np.where(seam_resized > 0, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(seam_resized, target_mask)


def place_roi_on_canvas(canvas, roi, corner: Tuple[int, int]) -> None:
    """Paste ROI into canvas with clipping."""

    x, y = int(corner[0]), int(corner[1])
    h, w = roi.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas_w, x + w)
    y1 = min(canvas_h, y + h)

    if x1 <= x0 or y1 <= y0:
        return

    rx0 = x0 - x
    ry0 = y0 - y
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)

    canvas[y0:y1, x0:x1] = roi[ry0:ry1, rx0:rx1]


def place_mask_on_canvas(canvas_mask, roi_mask, corner: Tuple[int, int]) -> None:
    """Paste ROI mask into canvas mask with max composition."""

    import numpy as np  # type: ignore

    x, y = int(corner[0]), int(corner[1])
    h, w = roi_mask.shape[:2]
    canvas_h, canvas_w = canvas_mask.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas_w, x + w)
    y1 = min(canvas_h, y + h)

    if x1 <= x0 or y1 <= y0:
        return

    rx0 = x0 - x
    ry0 = y0 - y
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)

    canvas_mask[y0:y1, x0:x1] = np.maximum(
        canvas_mask[y0:y1, x0:x1],
        roi_mask[ry0:ry1, rx0:rx1],
    )


def summarize_overlap(mask_left, mask_right) -> Dict[str, int]:
    """Return overlap area and bbox for debugging."""

    import numpy as np  # type: ignore

    left = _as_uint8_mask(mask_left) > 0
    right = _as_uint8_mask(mask_right) > 0
    overlap = left & right
    area = int(overlap.sum())
    if area == 0:
        return {"overlap_area": 0, "x": -1, "y": -1, "w": 0, "h": 0}

    ys, xs = np.where(overlap)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "overlap_area": area,
        "x": x0,
        "y": y0,
        "w": int(x1 - x0 + 1),
        "h": int(y1 - y0 + 1),
    }


def seam_overlay_preview(
    left_img,
    right_img,
    left_mask,
    right_mask,
    seam_left,
    seam_right,
) -> "np.ndarray":
    """Create seam debug overlay in one canvas (expects same size inputs)."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    base = cv2.addWeighted(left_img, 0.5, right_img, 0.5, 0.0)

    l_mask = _as_uint8_mask(left_mask) > 0
    r_mask = _as_uint8_mask(right_mask) > 0
    overlap = l_mask & r_mask

    seam_l = (_as_uint8_mask(seam_left) > 0) & overlap
    seam_r = (_as_uint8_mask(seam_right) > 0) & overlap

    border_l = cv2.Canny((seam_l.astype(np.uint8) * 255), 80, 160)
    border_r = cv2.Canny((seam_r.astype(np.uint8) * 255), 80, 160)

    out = base.copy()
    out[border_l > 0] = (0, 255, 0)
    out[border_r > 0] = (0, 0, 255)
    return out


def overlap_absdiff_preview(left_img, right_img, left_mask, right_mask) -> "np.ndarray":
    """Visualize abs diff on overlap region only."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    l_mask = _as_uint8_mask(left_mask) > 0
    r_mask = _as_uint8_mask(right_mask) > 0
    overlap = l_mask & r_mask

    diff = cv2.absdiff(left_img, right_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    vis = np.zeros_like(left_img)
    if overlap.any():
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        vis[overlap] = heat[overlap]
    return vis
