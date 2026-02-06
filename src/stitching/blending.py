"""Blending helpers for stitched frames."""

from __future__ import annotations


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
