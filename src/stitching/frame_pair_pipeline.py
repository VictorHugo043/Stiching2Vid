"""Shared frame-pair feature/matching/geometry pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FramePairPipelineResult:
    status: str
    failure_stage: Optional[str]
    message: Optional[str]
    feature_backend_requested: str
    feature_backend_effective: Optional[str]
    matcher_backend_requested: str
    matcher_backend_effective: Optional[str]
    geometry_backend_requested: str
    geometry_backend_effective: Optional[str]
    feature_left: Optional[object] = None
    feature_right: Optional[object] = None
    match_result: Optional[object] = None
    geometry_result: Optional[object] = None
    matches_img: Optional[object] = None
    inliers_img: Optional[object] = None
    feature_stage: Dict[str, object] = field(default_factory=dict)
    matching_stage: Dict[str, object] = field(default_factory=dict)
    geometry_stage: Dict[str, object] = field(default_factory=dict)
    fallback_events: List[Dict[str, object]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def resolve_feature_backend(feature: str, feature_backend: Optional[str]) -> str:
    if feature_backend:
        return feature_backend.strip().lower()
    feature_name = feature.strip().lower()
    if feature_name == "orb":
        return "opencv_orb"
    if feature_name == "sift":
        return "opencv_sift"
    return feature_name


def resolve_matcher_backend(matcher_backend: Optional[str]) -> str:
    return matcher_backend.strip().lower() if matcher_backend else "opencv_bf_ratio"


def resolve_geometry_backend(geometry_backend: Optional[str]) -> str:
    return geometry_backend.strip().lower() if geometry_backend else "opencv_ransac"


def resolve_optional_backend(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().lower()


def estimate_frame_pair_geometry(
    left,
    right,
    *,
    feature: str = "orb",
    feature_backend: Optional[str] = None,
    matcher_backend: Optional[str] = None,
    geometry_backend: Optional[str] = None,
    nfeatures: int = 2000,
    ratio: float = 0.75,
    min_matches: int = 30,
    ransac_thresh: float = 3.0,
    device: Optional[str] = None,
    force_cpu: bool = False,
    weights_dir: Optional[str] = None,
    max_keypoints: int = 2048,
    resize_long_edge: Optional[int] = None,
    depth_confidence: Optional[float] = None,
    width_confidence: Optional[float] = None,
    filter_threshold: Optional[float] = None,
    feature_fallback_backend: Optional[str] = None,
    matcher_fallback_backend: Optional[str] = None,
) -> FramePairPipelineResult:
    from stitching.features import detect_and_describe_result
    from stitching.geometry import estimate_homography_result
    from stitching.matching import draw_matches, match_feature_results

    resolved_feature_backend = resolve_feature_backend(feature, feature_backend)
    resolved_matcher_backend = resolve_matcher_backend(matcher_backend)
    resolved_geometry_backend = resolve_geometry_backend(geometry_backend)
    resolved_feature_fallback = resolve_optional_backend(feature_fallback_backend)
    resolved_matcher_fallback = resolve_optional_backend(matcher_fallback_backend)

    def _extract_pair_with_backend(backend_name: str):
        left_result = detect_and_describe_result(
            left,
            feature=feature,
            feature_backend=backend_name,
            nfeatures=nfeatures,
            device=device,
            force_cpu=force_cpu,
            weights_dir=weights_dir,
            max_keypoints=max_keypoints,
            resize_long_edge=resize_long_edge,
        )
        right_result = detect_and_describe_result(
            right,
            feature=feature,
            feature_backend=backend_name,
            nfeatures=nfeatures,
            device=device,
            force_cpu=force_cpu,
            weights_dir=weights_dir,
            max_keypoints=max_keypoints,
            resize_long_edge=resize_long_edge,
        )
        return left_result, right_result

    fallback_events: List[Dict[str, object]] = []
    feature_requested_error = None
    feature_requested_diagnostics = None
    try:
        feature_left, feature_right = _extract_pair_with_backend(resolved_feature_backend)
    except Exception as exc:
        feature_requested_error = str(exc)
        feature_requested_diagnostics = getattr(exc, "diagnostics", None)
        if resolved_feature_fallback and resolved_feature_fallback != resolved_feature_backend:
            fallback_events.append(
                {
                    "stage": "feature",
                    "requested_backend": resolved_feature_backend,
                    "fallback_backend": resolved_feature_fallback,
                    "reason": str(exc),
                    "diagnostics": feature_requested_diagnostics,
                }
            )
            try:
                feature_left, feature_right = _extract_pair_with_backend(resolved_feature_fallback)
            except Exception as fallback_exc:
                return FramePairPipelineResult(
                    status="feature_failed",
                    failure_stage="feature",
                    message=str(fallback_exc),
                    feature_backend_requested=resolved_feature_backend,
                    feature_backend_effective=resolved_feature_fallback,
                    matcher_backend_requested=resolved_matcher_backend,
                    matcher_backend_effective=None,
                    geometry_backend_requested=resolved_geometry_backend,
                    geometry_backend_effective=None,
                    feature_stage={
                        "requested_backend": resolved_feature_backend,
                        "effective_backend": resolved_feature_fallback,
                        "requested_error": feature_requested_error,
                        "requested_diagnostics": feature_requested_diagnostics,
                        "fallback_error": str(fallback_exc),
                        "fallback_diagnostics": getattr(fallback_exc, "diagnostics", None),
                    },
                    fallback_events=fallback_events,
                )
        else:
            return FramePairPipelineResult(
                status="feature_failed",
                failure_stage="feature",
                message=str(exc),
                feature_backend_requested=resolved_feature_backend,
                feature_backend_effective=resolved_feature_backend,
                matcher_backend_requested=resolved_matcher_backend,
                matcher_backend_effective=None,
                geometry_backend_requested=resolved_geometry_backend,
                geometry_backend_effective=None,
                feature_stage={
                    "requested_backend": resolved_feature_backend,
                    "effective_backend": resolved_feature_backend,
                    "requested_error": feature_requested_error,
                    "requested_diagnostics": feature_requested_diagnostics,
                },
                fallback_events=fallback_events,
            )

    feature_stage = {
        "requested_backend": resolved_feature_backend,
        "effective_backend": feature_left.backend_name,
        "requested_error": feature_requested_error,
        "requested_diagnostics": feature_requested_diagnostics,
        "left": {
            "backend_name": feature_left.backend_name,
            "n_keypoints": int(feature_left.n_keypoints),
            "runtime_ms": float(feature_left.runtime_ms),
            "meta": feature_left.meta,
        },
        "right": {
            "backend_name": feature_right.backend_name,
            "n_keypoints": int(feature_right.n_keypoints),
            "runtime_ms": float(feature_right.runtime_ms),
            "meta": feature_right.meta,
        },
    }

    matching_requested_error = None
    matching_requested_diagnostics = None
    try:
        match_result = match_feature_results(
            feature_left,
            feature_right,
            matcher_backend=resolved_matcher_backend,
            ratio=ratio,
            device=device,
            force_cpu=force_cpu,
            weights_dir=weights_dir,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
            filter_threshold=filter_threshold,
        )
    except Exception as exc:
        matching_requested_error = str(exc)
        matching_requested_diagnostics = getattr(exc, "diagnostics", None)
        if resolved_matcher_fallback and resolved_matcher_fallback != resolved_matcher_backend:
            fallback_events.append(
                {
                    "stage": "matching",
                    "requested_backend": resolved_matcher_backend,
                    "fallback_backend": resolved_matcher_fallback,
                    "reason": str(exc),
                    "diagnostics": matching_requested_diagnostics,
                }
            )
            try:
                match_result = match_feature_results(
                    feature_left,
                    feature_right,
                    matcher_backend=resolved_matcher_fallback,
                    ratio=ratio,
                    device=device,
                    force_cpu=force_cpu,
                    weights_dir=weights_dir,
                    depth_confidence=depth_confidence,
                    width_confidence=width_confidence,
                    filter_threshold=filter_threshold,
                )
            except Exception as fallback_exc:
                return FramePairPipelineResult(
                    status="matching_failed",
                    failure_stage="matching",
                    message=str(fallback_exc),
                    feature_backend_requested=resolved_feature_backend,
                    feature_backend_effective=feature_left.backend_name,
                    matcher_backend_requested=resolved_matcher_backend,
                    matcher_backend_effective=resolved_matcher_fallback,
                    geometry_backend_requested=resolved_geometry_backend,
                    geometry_backend_effective=None,
                    feature_left=feature_left,
                    feature_right=feature_right,
                    feature_stage=feature_stage,
                    matching_stage={
                        "requested_backend": resolved_matcher_backend,
                        "effective_backend": resolved_matcher_fallback,
                        "requested_error": matching_requested_error,
                        "requested_diagnostics": matching_requested_diagnostics,
                        "fallback_error": str(fallback_exc),
                        "fallback_diagnostics": getattr(fallback_exc, "diagnostics", None),
                    },
                    fallback_events=fallback_events,
                )
        else:
            return FramePairPipelineResult(
                status="matching_failed",
                failure_stage="matching",
                message=str(exc),
                feature_backend_requested=resolved_feature_backend,
                feature_backend_effective=feature_left.backend_name,
                matcher_backend_requested=resolved_matcher_backend,
                matcher_backend_effective=resolved_matcher_backend,
                geometry_backend_requested=resolved_geometry_backend,
                geometry_backend_effective=None,
                feature_left=feature_left,
                feature_right=feature_right,
                feature_stage=feature_stage,
                matching_stage={
                    "requested_backend": resolved_matcher_backend,
                    "effective_backend": resolved_matcher_backend,
                    "requested_error": matching_requested_error,
                    "requested_diagnostics": matching_requested_diagnostics,
                },
                fallback_events=fallback_events,
            )

    matching_stage = {
        "requested_backend": resolved_matcher_backend,
        "effective_backend": match_result.backend_name,
        "requested_error": matching_requested_error,
        "requested_diagnostics": matching_requested_diagnostics,
        "backend_name": match_result.backend_name,
        "tentative_count": int(match_result.tentative_count),
        "good_count": int(match_result.good_count),
        "runtime_ms": float(match_result.runtime_ms),
        "meta": match_result.meta,
    }
    matches_img = draw_matches(
        left,
        feature_left.cv_keypoints,
        right,
        feature_right.cv_keypoints,
        match_result.cv_matches,
    )

    if match_result.good_count < int(min_matches):
        return FramePairPipelineResult(
            status="not_enough_matches",
            failure_stage="matching",
            message=f"Not enough matches: {match_result.good_count} < {int(min_matches)}",
            feature_backend_requested=resolved_feature_backend,
            feature_backend_effective=feature_left.backend_name,
            matcher_backend_requested=resolved_matcher_backend,
            matcher_backend_effective=match_result.backend_name,
            geometry_backend_requested=resolved_geometry_backend,
            geometry_backend_effective=None,
            feature_left=feature_left,
            feature_right=feature_right,
            match_result=match_result,
            matches_img=matches_img,
            feature_stage=feature_stage,
            matching_stage=matching_stage,
            fallback_events=fallback_events,
        )

    try:
        geometry_result = estimate_homography_result(
            feature_left,
            feature_right,
            match_result,
            geometry_backend=resolved_geometry_backend,
            ransac_thresh=ransac_thresh,
        )
    except Exception as exc:
        return FramePairPipelineResult(
            status="homography_failed",
            failure_stage="homography",
            message=str(exc),
            feature_backend_requested=resolved_feature_backend,
            feature_backend_effective=feature_left.backend_name,
            matcher_backend_requested=resolved_matcher_backend,
            matcher_backend_effective=match_result.backend_name,
            geometry_backend_requested=resolved_geometry_backend,
            geometry_backend_effective=resolved_geometry_backend,
            feature_left=feature_left,
            feature_right=feature_right,
            match_result=match_result,
            matches_img=matches_img,
            feature_stage=feature_stage,
            matching_stage=matching_stage,
            fallback_events=fallback_events,
        )

    geometry_stage = {
        "backend_name": geometry_result.backend_name,
        "status": geometry_result.status,
        "runtime_ms": float(geometry_result.runtime_ms),
        "meta": geometry_result.meta,
    }
    if geometry_result.H is None or geometry_result.inlier_mask is None:
        return FramePairPipelineResult(
            status="homography_failed",
            failure_stage="homography",
            message=f"Geometry estimation failed with status={geometry_result.status}",
            feature_backend_requested=resolved_feature_backend,
            feature_backend_effective=feature_left.backend_name,
            matcher_backend_requested=resolved_matcher_backend,
            matcher_backend_effective=match_result.backend_name,
            geometry_backend_requested=resolved_geometry_backend,
            geometry_backend_effective=geometry_result.backend_name,
            feature_left=feature_left,
            feature_right=feature_right,
            match_result=match_result,
            geometry_result=geometry_result,
            matches_img=matches_img,
            feature_stage=feature_stage,
            matching_stage=matching_stage,
            geometry_stage=geometry_stage,
            fallback_events=fallback_events,
        )

    inliers_img = draw_matches(
        left,
        feature_left.cv_keypoints,
        right,
        feature_right.cv_keypoints,
        match_result.cv_matches,
        inlier_mask=geometry_result.inlier_mask,
    )
    return FramePairPipelineResult(
        status="ok",
        failure_stage=None,
        message=None,
        feature_backend_requested=resolved_feature_backend,
        feature_backend_effective=feature_left.backend_name,
        matcher_backend_requested=resolved_matcher_backend,
        matcher_backend_effective=match_result.backend_name,
        geometry_backend_requested=resolved_geometry_backend,
        geometry_backend_effective=geometry_result.backend_name,
        feature_left=feature_left,
        feature_right=feature_right,
        match_result=match_result,
        geometry_result=geometry_result,
        matches_img=matches_img,
        inliers_img=inliers_img,
        feature_stage=feature_stage,
        matching_stage=matching_stage,
        geometry_stage=geometry_stage,
        fallback_events=fallback_events,
    )
