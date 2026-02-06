"""Geometry helpers: homography estimation and warping utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple


def estimate_homography(kp_left, kp_right, matches, ransac_thresh: float = 3.0):
    """Estimate homography using RANSAC.

    Args:
        kp_left: Left keypoints list.
        kp_right: Right keypoints list.
        matches: List of cv2.DMatch.
        ransac_thresh: Reprojection threshold for RANSAC.

    Returns:
        (H, mask) where H is 3x3 homography or None, mask is inlier mask.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if len(matches) < 4:
        return None, None

    pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Map right -> left so the left frame remains the reference plane.
    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, ransac_thresh)
    return H, mask


def compute_canvas_and_transform(
    left_shape: Tuple[int, int],
    right_shape: Tuple[int, int],
    H_right_to_left,
) -> Tuple[Tuple[int, int], "np.ndarray"]:
    """Compute output canvas size and translation matrix.

    Args:
        left_shape: (height, width) of left image.
        right_shape: (height, width) of right image.
        H_right_to_left: Homography mapping right -> left.

    Returns:
        (canvas_size, T) where canvas_size=(width, height) and T is translation.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    h_left, w_left = left_shape
    h_right, w_right = right_shape

    corners_left = np.array(
        [[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]], dtype=np.float32
    ).reshape(-1, 1, 2)
    corners_right = np.array(
        [[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]], dtype=np.float32
    ).reshape(-1, 1, 2)

    warped_right = cv2.perspectiveTransform(corners_right, H_right_to_left)
    all_corners = np.concatenate([corners_left, warped_right], axis=0)

    min_xy = all_corners.min(axis=0).ravel()
    max_xy = all_corners.max(axis=0).ravel()

    min_x, min_y = min_xy
    max_x, max_y = max_xy

    # Translate so all coordinates are positive on the canvas.
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0

    canvas_width = int(round(max_x + tx))
    canvas_height = int(round(max_y + ty))

    T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    return (canvas_width, canvas_height), T


def warp_pair(
    left_bgr,
    right_bgr,
    H_right_to_left,
    canvas_size: Tuple[int, int],
    T,
):
    """Warp both frames onto a shared canvas.

    Args:
        left_bgr: Left image.
        right_bgr: Right image.
        H_right_to_left: Homography mapping right -> left.
        canvas_size: Output canvas size (width, height).
        T: Translation matrix to shift into positive canvas space.

    Returns:
        (left_warped, right_warped)
    """

    import cv2  # type: ignore

    left_warped = cv2.warpPerspective(left_bgr, T, canvas_size)
    right_warped = cv2.warpPerspective(right_bgr, T @ H_right_to_left, canvas_size)
    return left_warped, right_warped
