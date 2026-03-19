"""Descriptor matching helpers (KNN + ratio test) and visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple


@dataclass
class MatchResult:
    matches_lr: List[Tuple[int, int]]
    match_scores: List[float]
    tentative_count: int
    good_count: int
    backend_name: str
    runtime_ms: float
    meta: Dict[str, object] = field(default_factory=dict)
    cv_matches: List[object] = field(default_factory=list)


def _norm_for_feature_backend_name(feature_backend_name: str):
    import cv2  # type: ignore

    if feature_backend_name == "opencv_orb":
        return cv2.NORM_HAMMING, "hamming"
    return cv2.NORM_L2, "l2"


def match_feature_results(
    feature_left,
    feature_right,
    matcher_backend: str = "opencv_bf_ratio",
    ratio: float = 0.75,
) -> MatchResult:
    """Match two normalized feature results using a named matcher backend."""

    import cv2  # type: ignore

    backend_name = matcher_backend.strip().lower()
    start_time = time.perf_counter()

    if feature_left.descriptors is None or feature_right.descriptors is None:
        return MatchResult(
            matches_lr=[],
            match_scores=[],
            tentative_count=0,
            good_count=0,
            backend_name=backend_name,
            runtime_ms=float((time.perf_counter() - start_time) * 1000.0),
            meta={"ratio": float(ratio), "message": "descriptors_missing"},
            cv_matches=[],
        )

    if backend_name == "lightglue":
        raise NotImplementedError(
            "Matcher backend 'lightglue' is planned for Method B but not implemented in this subtask."
        )
    if backend_name != "opencv_bf_ratio":
        raise ValueError(f"Unsupported matcher backend: {backend_name}")

    norm, norm_name = _norm_for_feature_backend_name(feature_left.backend_name)
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn_matches = matcher.knnMatch(feature_left.descriptors, feature_right.descriptors, k=2)

    good_matches = []
    matches_lr: List[Tuple[int, int]] = []
    match_scores: List[float] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)
            matches_lr.append((int(m.queryIdx), int(m.trainIdx)))
            match_scores.append(float(m.distance))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return MatchResult(
        matches_lr=matches_lr,
        match_scores=match_scores,
        tentative_count=len(knn_matches),
        good_count=len(good_matches),
        backend_name=backend_name,
        runtime_ms=float(runtime_ms),
        meta={
            "ratio": float(ratio),
            "distance_norm": norm_name,
            "feature_backend_left": feature_left.backend_name,
            "feature_backend_right": feature_right.backend_name,
        },
        cv_matches=good_matches,
    )


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
