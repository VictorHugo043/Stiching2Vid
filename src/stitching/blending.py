"""Blending helpers for stitched frames."""

from __future__ import annotations

from typing import Optional


def _resolve_mask(img, mask=None):
    """Resolve mask to uint8 {0,255} from explicit mask or non-black pixels."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if mask is None:
        resolved = (img.sum(axis=2) > 0).astype(np.uint8) * 255
        return resolved

    arr = mask
    if hasattr(arr, "get"):
        arr = arr.get()
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
    arr = np.where(arr > 0, 255, 0).astype(np.uint8)
    return arr


def blend_none(left_bgr, right_bgr, left_mask: Optional[object] = None, right_mask: Optional[object] = None):
    """Hard compose by mask.

    Non-overlap keeps its source unchanged; overlap defaults to left priority.
    This is useful for seam diagnostics because it avoids averaging artifacts.
    """

    import numpy as np  # type: ignore

    l_mask = _resolve_mask(left_bgr, left_mask) > 0
    r_mask = _resolve_mask(right_bgr, right_mask) > 0

    out = np.zeros_like(left_bgr)
    out[r_mask] = right_bgr[r_mask]
    out[l_mask] = left_bgr[l_mask]
    return out


def feather_blend(left_bgr, right_bgr, left_mask: Optional[object] = None, right_mask: Optional[object] = None):
    """Distance-transform feather blend with explicit validity masks."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    l_mask = _resolve_mask(left_bgr, left_mask)
    r_mask = _resolve_mask(right_bgr, right_mask)

    l_valid = l_mask > 0
    r_valid = r_mask > 0
    if not l_valid.any() and not r_valid.any():
        return np.zeros_like(left_bgr)
    if not r_valid.any():
        return left_bgr.copy()
    if not l_valid.any():
        return right_bgr.copy()

    dist_left = cv2.distanceTransform((l_valid.astype(np.uint8)), cv2.DIST_L2, 3)
    dist_right = cv2.distanceTransform((r_valid.astype(np.uint8)), cv2.DIST_L2, 3)

    w_left = dist_left.astype(np.float32)
    w_right = dist_right.astype(np.float32)
    w_left[~l_valid] = 0.0
    w_right[~r_valid] = 0.0

    weight_sum = w_left + w_right
    nonzero = weight_sum > 1e-6
    w_left[nonzero] = w_left[nonzero] / weight_sum[nonzero]
    w_right[nonzero] = w_right[nonzero] / weight_sum[nonzero]

    blended = (
        left_bgr.astype(np.float32) * w_left[..., None]
        + right_bgr.astype(np.float32) * w_right[..., None]
    )
    blended[~(l_valid | r_valid)] = 0
    return blended.astype(np.uint8)


def multiband_blend(
    left_bgr,
    right_bgr,
    left_mask: Optional[object] = None,
    right_mask: Optional[object] = None,
    levels: int = 5,
):
    """Multi-band blend via OpenCV detail blender with explicit masks."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    l_mask = _resolve_mask(left_bgr, left_mask)
    r_mask = _resolve_mask(right_bgr, right_mask)

    if not (l_mask > 0).any() and not (r_mask > 0).any():
        return np.zeros_like(left_bgr)
    if not (r_mask > 0).any():
        return left_bgr.copy()
    if not (l_mask > 0).any():
        return right_bgr.copy()

    # OpenCV detail blender expects int16 images and uint8 masks.
    h, w = left_bgr.shape[:2]
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(max(1, int(levels)))
    blender.prepare((0, 0, w, h))
    blender.feed(cv2.UMat(left_bgr.astype(np.int16)), l_mask, (0, 0))
    blender.feed(cv2.UMat(right_bgr.astype(np.int16)), r_mask, (0, 0))

    result, _ = blender.blend(None, None)
    return cv2.convertScaleAbs(result)
