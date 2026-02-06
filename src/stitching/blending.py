"""Blending helpers for stitched frames."""

from __future__ import annotations


def _as_bool_mask(mask, shape):
    import numpy as np  # type: ignore

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape:
        raise ValueError(f"Mask shape mismatch: expected {shape}, got {arr.shape}")
    if arr.dtype == np.bool_:
        return arr
    return arr > 0


def resolve_hard_seam_side(mask_left, mask_right, seam_side_mask=None):
    """Resolve seam side selection (True=left, False=right) for hard compositing."""
    import numpy as np  # type: ignore

    left_valid = _as_bool_mask(mask_left, mask_left.shape)
    right_valid = _as_bool_mask(mask_right, mask_left.shape)
    overlap = left_valid & right_valid
    left_only = left_valid & (~right_valid)
    right_only = right_valid & (~left_valid)

    side = np.zeros(mask_left.shape, dtype=bool)
    side[left_only] = True
    side[right_only] = False

    if overlap.any():
        if seam_side_mask is not None:
            seam_bool = _as_bool_mask(seam_side_mask, mask_left.shape)
            side[overlap] = seam_bool[overlap]
        else:
            ys, xs = np.where(overlap)
            x_mid = int((int(xs.min()) + int(xs.max())) // 2)
            overlap_left = overlap.copy()
            overlap_left[:, :] = False
            overlap_left[:, : x_mid + 1] = True
            side[overlap] = overlap_left[overlap]
    return side


def composite_hard_seam(
    left,
    right_warp,
    mask_left,
    mask_right,
    seam_side_mask=None,
):
    """Compose with hard seam: no averaging anywhere."""
    import numpy as np  # type: ignore

    left_valid = _as_bool_mask(mask_left, mask_left.shape)
    right_valid = _as_bool_mask(mask_right, mask_left.shape)
    overlap = left_valid & right_valid
    left_only = left_valid & (~right_valid)
    right_only = right_valid & (~left_valid)
    side = resolve_hard_seam_side(mask_left, mask_right, seam_side_mask=seam_side_mask)

    out = np.zeros_like(left)
    out[left_only] = left[left_only]
    out[right_only] = right_warp[right_only]
    out[overlap & side] = left[overlap & side]
    out[overlap & (~side)] = right_warp[overlap & (~side)]
    return out


def blend_none(left_bgr, right_bgr):
    """Blend by selecting left pixels when available, otherwise right.

    This is a diagnostic mode: it avoids averaging across overlap so that
    geometric misalignment and boundary placement issues are easier to see.
    """

    import numpy as np  # type: ignore

    left_mask = (left_bgr.sum(axis=2) > 0)[..., None]
    return np.where(left_mask, left_bgr, right_bgr)


def feather_blend(left_bgr, right_bgr):
    """Feather blend two warped images.

    Args:
        left_bgr: Left image on canvas.
        right_bgr: Right image on canvas.

    Returns:
        Blended BGR image.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    left_mask = (left_bgr.sum(axis=2) > 0).astype("uint8")
    right_mask = (right_bgr.sum(axis=2) > 0).astype("uint8")

    dist_left = cv2.distanceTransform(left_mask, cv2.DIST_L2, 3)
    dist_right = cv2.distanceTransform(right_mask, cv2.DIST_L2, 3)

    weight_sum = dist_left + dist_right
    weight_sum[weight_sum == 0] = 1.0

    w_left = dist_left / weight_sum
    w_right = dist_right / weight_sum

    blended = (
        left_bgr.astype("float32") * w_left[..., None]
        + right_bgr.astype("float32") * w_right[..., None]
    )
    return blended.astype("uint8")
