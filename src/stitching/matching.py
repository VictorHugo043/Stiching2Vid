"""Descriptor matching helpers and visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple

from stitching.method_b_runtime import (
    MethodBBackendError,
    ensure_method_b_dependencies,
    maybe_load_state_dict,
    resolve_method_b_device,
    resolve_method_b_weights,
)


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


_LIGHTGLUE_CACHE: Dict[Tuple[object, ...], object] = {}


def _norm_for_feature_backend_name(feature_backend_name: str):
    import cv2  # type: ignore

    if feature_backend_name == "opencv_orb":
        return cv2.NORM_HAMMING, "hamming"
    return cv2.NORM_L2, "l2"


def _resolve_lightglue_weights(weights_dir: Optional[str]) -> Dict[str, object]:
    return resolve_method_b_weights(
        backend_name="lightglue",
        weights_dir=weights_dir,
        candidate_names=(
            "lightglue_superpoint.pth",
            "lightglue_superpoint.pt",
            "lightglue.pth",
            "lightglue.pt",
        ),
    )


def _instantiate_lightglue(
    *,
    depth_confidence: Optional[float],
    width_confidence: Optional[float],
    filter_threshold: Optional[float],
):
    from lightglue import LightGlue  # type: ignore

    kwargs = {"features": "superpoint"}
    if depth_confidence is not None:
        kwargs["depth_confidence"] = float(depth_confidence)
    if width_confidence is not None:
        kwargs["width_confidence"] = float(width_confidence)
    if filter_threshold is not None:
        kwargs["filter_threshold"] = float(filter_threshold)

    try:
        return LightGlue(**kwargs), kwargs
    except TypeError:
        return LightGlue(features="superpoint"), {"features": "superpoint"}


def _load_lightglue_matcher(
    *,
    device: Optional[str],
    force_cpu: bool,
    weights_dir: Optional[str],
    depth_confidence: Optional[float],
    width_confidence: Optional[float],
    filter_threshold: Optional[float],
):
    dependency_status = ensure_method_b_dependencies("lightglue", ("torch", "lightglue"))
    device_info = resolve_method_b_device(requested_device=device, force_cpu=force_cpu)
    if not device_info.get("resolved_device"):
        raise MethodBBackendError(
            "Unable to resolve a runtime device for LightGlue.",
            diagnostics={
                "backend_name": "lightglue",
                "dependency_status": dependency_status,
                "device_info": device_info,
            },
        )

    weights_info = _resolve_lightglue_weights(weights_dir)
    cache_key = (
        str(device_info["resolved_device"]),
        weights_info.get("weights_path"),
        depth_confidence,
        width_confidence,
        filter_threshold,
    )
    if cache_key in _LIGHTGLUE_CACHE:
        return _LIGHTGLUE_CACHE[cache_key], device_info, weights_info, dependency_status

    matcher, init_kwargs = _instantiate_lightglue(
        depth_confidence=depth_confidence,
        width_confidence=width_confidence,
        filter_threshold=filter_threshold,
    )
    load_info = maybe_load_state_dict(matcher, weights_info.get("weights_path"))
    matcher = matcher.eval().to(str(device_info["resolved_device"]))
    _LIGHTGLUE_CACHE[cache_key] = matcher
    weights_info = dict(weights_info)
    weights_info["load_info"] = load_info
    weights_info["init_kwargs"] = init_kwargs
    return matcher, device_info, weights_info, dependency_status


def _tensor_to_numpy(tensor_like):
    import numpy as np  # type: ignore

    if isinstance(tensor_like, (list, tuple)):
        if len(tensor_like) == 0:
            return np.asarray([])
        if len(tensor_like) == 1:
            return _tensor_to_numpy(tensor_like[0])
        return np.asarray([_tensor_to_numpy(item) for item in tensor_like], dtype=object)
    if hasattr(tensor_like, "detach"):
        arr = tensor_like.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_like)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _extract_lightglue_pairs(prediction):
    import numpy as np  # type: ignore

    pairs = prediction.get("matches")
    scores = prediction.get("scores")
    if pairs is not None:
        if isinstance(pairs, (list, tuple)):
            pairs = pairs[0] if len(pairs) > 0 else []
        if isinstance(scores, (list, tuple)):
            scores = scores[0] if len(scores) > 0 else []
        pairs_np = _tensor_to_numpy(pairs)
        if pairs_np.size == 0:
            return [], []
        pairs_np = np.asarray(pairs_np, dtype=np.int32).reshape(-1, 2)
        if scores is not None:
            scores_np = _tensor_to_numpy(scores).astype(np.float32).reshape(-1)
        else:
            scores_np = np.ones((pairs_np.shape[0],), dtype=np.float32)
        return (
            [(int(q), int(t)) for q, t in pairs_np.tolist()],
            [float(v) for v in scores_np.tolist()],
        )

    matches0 = prediction.get("matches0")
    if matches0 is None:
        raise MethodBBackendError(
            "LightGlue output does not contain supported match tensors.",
            diagnostics={"available_keys": sorted(prediction.keys())},
        )
    if isinstance(matches0, (list, tuple)):
        matches0 = matches0[0] if len(matches0) > 0 else []
    matches0_np = _tensor_to_numpy(matches0).astype(np.int32).reshape(-1)
    scores0 = prediction.get("matching_scores0")
    if scores0 is None:
        scores0 = prediction.get("scores0")
    if isinstance(scores0, (list, tuple)):
        scores0 = scores0[0] if len(scores0) > 0 else []
    if scores0 is not None:
        scores0_np = _tensor_to_numpy(scores0).astype(np.float32).reshape(-1)
    else:
        scores0_np = None

    valid_indices = np.where(matches0_np >= 0)[0].tolist()
    pairs_out: List[Tuple[int, int]] = []
    scores_out: List[float] = []
    for idx in valid_indices:
        pairs_out.append((int(idx), int(matches0_np[idx])))
        if scores0_np is None:
            scores_out.append(1.0)
        else:
            scores_out.append(float(scores0_np[idx]))
    return pairs_out, scores_out


def _extract_lightglue_meta(prediction) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    stop = prediction.get("stop")
    if stop is not None:
        stop_np = _tensor_to_numpy(stop)
        try:
            meta["stop_layer"] = int(stop_np.reshape(-1)[0])
        except Exception:
            meta["stop_layer"] = stop

    for key in ("prune0", "prune1"):
        value = prediction.get(key)
        if value is None:
            continue
        value_np = _tensor_to_numpy(value)
        try:
            flat = value_np.astype("float32").reshape(-1)
            if flat.size:
                meta[f"{key}_mean"] = float(flat.mean())
                meta[f"{key}_max"] = float(flat.max())
        except Exception:
            meta[f"{key}_available"] = True
    return meta


def _build_cv_matches(matches_lr: List[Tuple[int, int]], match_scores: List[float]):
    import cv2  # type: ignore

    matches = []
    for idx, (query_idx, train_idx) in enumerate(matches_lr):
        score = float(match_scores[idx]) if idx < len(match_scores) else 1.0
        distance = float(max(0.0, 1.0 - score)) if 0.0 <= score <= 1.0 else float(score)
        matches.append(cv2.DMatch(int(query_idx), int(train_idx), distance))
    return matches


def match_feature_results(
    feature_left,
    feature_right,
    matcher_backend: str = "opencv_bf_ratio",
    ratio: float = 0.75,
    device: Optional[str] = None,
    force_cpu: bool = False,
    weights_dir: Optional[str] = None,
    depth_confidence: Optional[float] = None,
    width_confidence: Optional[float] = None,
    filter_threshold: Optional[float] = None,
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
        if feature_left.backend_name != "superpoint" or feature_right.backend_name != "superpoint":
            raise MethodBBackendError(
                "LightGlue matcher requires SuperPoint feature inputs on both sides.",
                diagnostics={
                    "backend_name": "lightglue",
                    "feature_backend_left": feature_left.backend_name,
                    "feature_backend_right": feature_right.backend_name,
                },
            )
        if feature_left.backend_payload is None or feature_right.backend_payload is None:
            raise MethodBBackendError(
                "LightGlue matcher requires backend payload from SuperPoint extraction.",
                diagnostics={
                    "backend_name": "lightglue",
                    "left_payload_available": feature_left.backend_payload is not None,
                    "right_payload_available": feature_right.backend_payload is not None,
                },
            )

        try:
            matcher, device_info, weights_info, dependency_status = _load_lightglue_matcher(
                device=device,
                force_cpu=force_cpu,
                weights_dir=weights_dir,
                depth_confidence=depth_confidence,
                width_confidence=width_confidence,
                filter_threshold=filter_threshold,
            )
            import torch  # type: ignore

            with torch.inference_mode():
                prediction = matcher(
                    {"image0": feature_left.backend_payload, "image1": feature_right.backend_payload}
                )
            matches_lr, match_scores = _extract_lightglue_pairs(prediction)
            prediction_meta = _extract_lightglue_meta(prediction)
        except MethodBBackendError:
            raise
        except Exception as exc:
            raise MethodBBackendError(
                f"LightGlue backend failed: {exc}",
                diagnostics={
                    "backend_name": "lightglue",
                    "error_type": type(exc).__name__,
                    "device": device,
                    "force_cpu": bool(force_cpu),
                    "weights_dir": weights_dir,
                    "depth_confidence": depth_confidence,
                    "width_confidence": width_confidence,
                    "filter_threshold": filter_threshold,
                },
            ) from exc

        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        return MatchResult(
            matches_lr=matches_lr,
            match_scores=match_scores,
            tentative_count=len(matches_lr),
            good_count=len(matches_lr),
            backend_name=backend_name,
            runtime_ms=float(runtime_ms),
            meta={
                "ratio": float(ratio),
                "distance_norm": "lightglue_confidence",
                "feature_backend_left": feature_left.backend_name,
                "feature_backend_right": feature_right.backend_name,
                "dependency_status": dependency_status,
                "device_info": device_info,
                "weights_info": weights_info,
                "depth_confidence": depth_confidence,
                "width_confidence": width_confidence,
                "filter_threshold": filter_threshold,
                "tentative_count_semantics": "filtered_lightglue_matches",
                "prediction_meta": prediction_meta,
            },
            cv_matches=_build_cv_matches(matches_lr, match_scores),
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
    """Match descriptors with KNN + ratio test."""

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
    """Draw matches between two images."""

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
