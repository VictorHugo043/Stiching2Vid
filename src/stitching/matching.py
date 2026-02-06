"""Descriptor matching helpers (KNN + ratio test) and visualization."""

from __future__ import annotations

from typing import List, Optional, Tuple


def match_descriptors(
    desc_left,
    desc_right,
    method: str = "orb",
    ratio: float = 0.75,
) -> Tuple[List, int]:
    """Match descriptors with KNN + ratio test.

    Args:
        desc_left: Left descriptors (numpy array).
        desc_right: Right descriptors (numpy array).
        method: Feature type ("orb" or "sift") to choose distance metric.
        ratio: Lowe's ratio threshold.

    Returns:
        (good_matches, raw_match_count).
    """

    import cv2  # type: ignore

    if desc_left is None or desc_right is None:
        return [], 0

    if method.lower() == "orb":
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn_matches = matcher.knnMatch(desc_left, desc_right, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches, len(knn_matches)


def draw_matches(
    image_left,
    kp_left,
    image_right,
    kp_right,
    matches,
    inlier_mask: Optional[List[int]] = None,
):
    """Draw matches between two images.

    Args:
        image_left: Left BGR image.
        kp_left: Left keypoints list.
        image_right: Right BGR image.
        kp_right: Right keypoints list.
        matches: List of cv2.DMatch.
        inlier_mask: Optional list of 0/1 indicating inliers.

    Returns:
        Composite image with matches drawn.
    """

    import cv2  # type: ignore

    if inlier_mask is not None:
        inliers = [m for m, keep in zip(matches, inlier_mask) if keep]
        return cv2.drawMatches(
            image_left,
            kp_left,
            image_right,
            kp_right,
            inliers,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    return cv2.drawMatches(
        image_left,
        kp_left,
        image_right,
        kp_right,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
