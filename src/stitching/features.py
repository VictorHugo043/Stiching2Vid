"""Feature detection/description helpers for baseline stitching."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple


@dataclass
class FeatureResult:
    keypoints_xy: List[Tuple[float, float]]
    descriptors: Optional[object]
    scores: Optional[List[float]]
    image_size: Tuple[int, int]
    backend_name: str
    runtime_ms: float
    meta: Dict[str, object] = field(default_factory=dict)
    cv_keypoints: List[object] = field(default_factory=list)

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoints_xy)


def _normalize_feature_backend(feature: str = "orb", feature_backend: Optional[str] = None) -> str:
    if feature_backend:
        return feature_backend.strip().lower()
    feature_name = feature.strip().lower()
    if feature_name == "orb":
        return "opencv_orb"
    if feature_name == "sift":
        return "opencv_sift"
    return feature_name


def detect_and_describe_result(
    image_bgr,
    feature: str = "orb",
    feature_backend: Optional[str] = None,
    nfeatures: int = 2000,
) -> FeatureResult:
    """Detect keypoints and compute descriptors using a named feature backend."""

    import cv2  # type: ignore

    backend_name = _normalize_feature_backend(feature=feature, feature_backend=feature_backend)
    start_time = time.perf_counter()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if backend_name == "opencv_orb":
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif backend_name == "opencv_sift":
        if not hasattr(cv2, "SIFT_create"):
            raise ValueError("SIFT not available in this OpenCV build.")
        detector = cv2.SIFT_create()
    elif backend_name == "superpoint":
        raise NotImplementedError(
            "Feature backend 'superpoint' is planned for Method B but not implemented in this subtask."
        )
    else:
        raise ValueError(f"Unsupported feature backend: {backend_name}")

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    keypoints = keypoints or []
    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return FeatureResult(
        keypoints_xy=[(float(kp.pt[0]), float(kp.pt[1])) for kp in keypoints],
        descriptors=descriptors,
        scores=[float(getattr(kp, "response", 0.0)) for kp in keypoints],
        image_size=(int(image_bgr.shape[1]), int(image_bgr.shape[0])),
        backend_name=backend_name,
        runtime_ms=float(runtime_ms),
        meta={
            "legacy_feature": feature,
            "nfeatures": int(nfeatures),
        },
        # Keep the OpenCV keypoints alongside the normalized arrays so current
        # debug visualization can still draw matches while Method B evolves.
        cv_keypoints=list(keypoints),
    )


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
    result = detect_and_describe_result(
        image_bgr,
        feature=feature,
        feature_backend=None,
        nfeatures=nfeatures,
    )
    return result.cv_keypoints, result.descriptors
