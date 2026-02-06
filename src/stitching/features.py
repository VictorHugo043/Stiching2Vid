"""Feature detection/description helpers for baseline stitching."""

from __future__ import annotations

from typing import List, Tuple, Optional


def detect_and_describe(
    image_bgr,
    feature: str = "orb",
    nfeatures: int = 2000,
) -> Tuple[List, Optional[object]]:
    """Detect keypoints and compute descriptors.

    Args:
        image_bgr: Input BGR image (numpy ndarray).
        feature: Feature type ("orb" or "sift").
        nfeatures: Max features for ORB (ignored by SIFT).

    Returns:
        (keypoints, descriptors). Descriptors may be None if detection fails.

    Raises:
        ValueError: If feature type is unsupported or SIFT unavailable.
    """

    import cv2  # type: ignore

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if feature.lower() == "orb":
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif feature.lower() == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise ValueError("SIFT not available in this OpenCV build.")
        detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unsupported feature type: {feature}")

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints or [], descriptors
