"""Geometry helpers: homography estimation and warping utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple


@dataclass
class GeometryResult:
    H: Optional[object]
    inlier_mask: Optional[List[int]]
    inlier_count: int
    inlier_ratio: float
    reprojection_error: Optional[float]
    inlier_spatial_coverage: Optional[float]
    backend_name: str
    runtime_ms: float
    status: str
    meta: Dict[str, object] = field(default_factory=dict)


def _find_homography_method(geometry_backend: str):
    import cv2  # type: ignore

    backend_name = geometry_backend.strip().lower()
    if backend_name == "opencv_ransac":
        return backend_name, cv2.RANSAC
    if backend_name in {"opencv_usac_magsac", "opencv_magsac"}:
        if not hasattr(cv2, "USAC_MAGSAC"):
            raise ValueError("OpenCV USAC_MAGSAC is unavailable in this environment.")
        return "opencv_usac_magsac", cv2.USAC_MAGSAC
    if backend_name in {"magsac++", "magsac"}:
        if not hasattr(cv2, "USAC_MAGSAC"):
            raise ValueError("OpenCV USAC_MAGSAC is unavailable in this environment.")
        return "opencv_usac_magsac", cv2.USAC_MAGSAC
    raise ValueError(f"Unsupported geometry backend: {geometry_backend}")


def _compute_reprojection_error(H, pts_right, pts_left, inlier_mask) -> Optional[float]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if H is None or pts_right.size == 0 or pts_left.size == 0:
        return None

    projected = cv2.perspectiveTransform(pts_right, H).reshape(-1, 2)
    target = pts_left.reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    if inlier_mask:
        keep = np.asarray(inlier_mask, dtype=bool)
        if keep.any():
            errors = errors[keep]
    if errors.size == 0:
        return None
    return float(errors.mean())


def _compute_inlier_spatial_coverage(feature_left, match_result, inlier_mask) -> Optional[float]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if not inlier_mask or not getattr(match_result, "matches_lr", None):
        return None
    image_w, image_h = feature_left.image_size
    image_area = float(max(1, int(image_w) * int(image_h)))
    inlier_points = [
        feature_left.keypoints_xy[q_idx]
        for (q_idx, _), keep in zip(match_result.matches_lr, inlier_mask)
        if int(keep) > 0
    ]
    if not inlier_points:
        return None

    pts = np.asarray(inlier_points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        area = float(cv2.contourArea(hull))
    elif pts.shape[0] == 2:
        xs = pts[:, 0]
        ys = pts[:, 1]
        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
    else:
        area = 0.0
    return float(min(1.0, max(0.0, area / image_area)))


def estimate_homography_result(
    feature_left,
    feature_right,
    match_result,
    geometry_backend: str = "opencv_ransac",
    ransac_thresh: float = 3.0,
    confidence: float = 0.995,
    max_iters: int = 2000,
) -> GeometryResult:
    """Estimate homography from normalized feature and match results."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    start_time = time.perf_counter()
    backend_name, method = _find_homography_method(geometry_backend)

    if len(match_result.matches_lr) < 4:
        return GeometryResult(
            H=None,
            inlier_mask=None,
            inlier_count=0,
            inlier_ratio=0.0,
            reprojection_error=None,
            inlier_spatial_coverage=None,
            backend_name=backend_name,
            runtime_ms=float((time.perf_counter() - start_time) * 1000.0),
            status="not_enough_matches",
            meta={"match_count": int(len(match_result.matches_lr))},
        )

    pts_left = np.float32(
        [feature_left.keypoints_xy[q_idx] for q_idx, _ in match_result.matches_lr]
    ).reshape(-1, 1, 2)
    pts_right = np.float32(
        [feature_right.keypoints_xy[t_idx] for _, t_idx in match_result.matches_lr]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts_right,
        pts_left,
        method,
        ransac_thresh,
        None,
        int(max_iters),
        float(confidence),
    )
    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    if H is None or mask is None:
        return GeometryResult(
            H=None,
            inlier_mask=None,
            inlier_count=0,
            inlier_ratio=0.0,
            reprojection_error=None,
            inlier_spatial_coverage=None,
            backend_name=backend_name,
            runtime_ms=float(runtime_ms),
            status="homography_failed",
            meta={
                "match_count": int(len(match_result.matches_lr)),
                "ransac_thresh": float(ransac_thresh),
            },
        )

    inlier_mask = [int(v) for v in mask.ravel().tolist()]
    inlier_count = int(sum(inlier_mask))
    inlier_ratio = (
        float(inlier_count) / float(len(inlier_mask))
        if inlier_mask
        else 0.0
    )
    reprojection_error = _compute_reprojection_error(H, pts_right, pts_left, inlier_mask)
    inlier_spatial_coverage = _compute_inlier_spatial_coverage(
        feature_left,
        match_result,
        inlier_mask,
    )
    return GeometryResult(
        H=H,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        reprojection_error=reprojection_error,
        inlier_spatial_coverage=inlier_spatial_coverage,
        backend_name=backend_name,
        runtime_ms=float(runtime_ms),
        status="ok",
        meta={
            "match_count": int(len(match_result.matches_lr)),
            "ransac_thresh": float(ransac_thresh),
            "confidence": float(confidence),
            "max_iters": int(max_iters),
        },
    )


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
