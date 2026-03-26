"""Feature detection/description helpers for baseline stitching."""

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
class FeatureResult:
    keypoints_xy: List[Tuple[float, float]]
    descriptors: Optional[object]
    scores: Optional[List[float]]
    image_size: Tuple[int, int]
    backend_name: str
    runtime_ms: float
    meta: Dict[str, object] = field(default_factory=dict)
    cv_keypoints: List[object] = field(default_factory=list)
    backend_payload: Optional[object] = None

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoints_xy)


_SUPERPOINT_CACHE: Dict[Tuple[object, ...], object] = {}
_USE_SUPERPOINT_DEFAULT_RESIZE = object()


def _normalize_feature_backend(feature: str = "orb", feature_backend: Optional[str] = None) -> str:
    if feature_backend:
        return feature_backend.strip().lower()
    feature_name = feature.strip().lower()
    if feature_name == "orb":
        return "opencv_orb"
    if feature_name == "sift":
        return "opencv_sift"
    return feature_name


def _to_numpy_2d(tensor_like):
    arr = tensor_like.detach().cpu().numpy()
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _build_cv_keypoints(keypoints_xy: List[Tuple[float, float]], scores: List[float]):
    import cv2  # type: ignore

    keypoints = []
    for idx, (x, y) in enumerate(keypoints_xy):
        response = float(scores[idx]) if idx < len(scores) else 0.0
        keypoints.append(cv2.KeyPoint(float(x), float(y), 1.0, -1.0, response))
    return keypoints


def _resolve_superpoint_weights(weights_dir: Optional[str]) -> Dict[str, object]:
    return resolve_method_b_weights(
        backend_name="superpoint",
        weights_dir=weights_dir,
        candidate_names=(
            "superpoint.pth",
            "superpoint.pt",
            "superpoint_v1.pth",
            "superpoint_v1.pt",
        ),
    )


def _normalize_superpoint_max_keypoints(max_keypoints: Optional[int]) -> Optional[int]:
    if max_keypoints is None:
        return None
    value = int(max_keypoints)
    return None if value <= 0 else value


def _resolve_superpoint_resize(resize_long_edge: Optional[int]):
    if resize_long_edge is None:
        return _USE_SUPERPOINT_DEFAULT_RESIZE
    value = int(resize_long_edge)
    return None if value <= 0 else value


def _instantiate_superpoint(max_keypoints: Optional[int]):
    from lightglue import SuperPoint  # type: ignore

    normalized_max_keypoints = _normalize_superpoint_max_keypoints(max_keypoints)
    init_candidates = (
        {"max_num_keypoints": normalized_max_keypoints},
        {"max_keypoints": normalized_max_keypoints},
        {},
    )
    last_error = None
    for kwargs in init_candidates:
        try:
            return SuperPoint(**kwargs), kwargs
        except TypeError as exc:
            last_error = exc
    raise MethodBBackendError(
        "Unable to instantiate SuperPoint with the supported constructor variants.",
        diagnostics={"last_error": str(last_error), "max_keypoints": normalized_max_keypoints},
    )


def _load_superpoint_extractor(
    *,
    device: Optional[str],
    force_cpu: bool,
    weights_dir: Optional[str],
    max_keypoints: int,
):
    dependency_status = ensure_method_b_dependencies("superpoint", ("torch", "lightglue"))
    device_info = resolve_method_b_device(requested_device=device, force_cpu=force_cpu)
    if not device_info.get("resolved_device"):
        raise MethodBBackendError(
            "Unable to resolve a runtime device for SuperPoint.",
            diagnostics={
                "backend_name": "superpoint",
                "dependency_status": dependency_status,
                "device_info": device_info,
            },
        )

    weights_info = _resolve_superpoint_weights(weights_dir)
    normalized_max_keypoints = _normalize_superpoint_max_keypoints(max_keypoints)
    cache_key = (
        str(device_info["resolved_device"]),
        normalized_max_keypoints,
        weights_info.get("weights_path"),
    )
    if cache_key in _SUPERPOINT_CACHE:
        return _SUPERPOINT_CACHE[cache_key], device_info, weights_info, dependency_status

    extractor, init_kwargs = _instantiate_superpoint(normalized_max_keypoints)
    load_info = maybe_load_state_dict(extractor, weights_info.get("weights_path"))

    extractor = extractor.eval().to(str(device_info["resolved_device"]))
    _SUPERPOINT_CACHE[cache_key] = extractor
    weights_info = dict(weights_info)
    weights_info["load_info"] = load_info
    weights_info["init_kwargs"] = init_kwargs
    return extractor, device_info, weights_info, dependency_status


def _extract_superpoint_features(
    image_bgr,
    *,
    extractor,
    resolved_device: str,
    resize_long_edge: Optional[int],
):
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore

    # SuperPoint converts RGB inputs to grayscale internally. Precomputing grayscale
    # here preserves the effective input while avoiding a 3-channel host->device copy.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(np.ascontiguousarray(gray)).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device=str(resolved_device), dtype=torch.float32)
    tensor.div_(255.0)
    preprocess_resize = _resolve_superpoint_resize(resize_long_edge)
    extract_kwargs = {}
    if preprocess_resize is not _USE_SUPERPOINT_DEFAULT_RESIZE:
        extract_kwargs["resize"] = preprocess_resize

    with torch.inference_mode():
        if hasattr(extractor, "extract"):
            pred = extractor.extract(tensor, **extract_kwargs)
        else:
            pred = extractor({"image": tensor})

    keypoints = _to_numpy_2d(pred["keypoints"]).astype(np.float32)
    descriptors = _to_numpy_2d(pred["descriptors"]).astype(np.float32)
    if descriptors.ndim == 2 and descriptors.shape[0] != keypoints.shape[0]:
        if descriptors.shape[1] == keypoints.shape[0]:
            descriptors = descriptors.T
    score_tensor = pred.get("keypoint_scores")
    if score_tensor is None:
        score_tensor = pred.get("scores")
    if score_tensor is None:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)
    else:
        scores = _to_numpy_2d(score_tensor).astype(np.float32).reshape(-1)

    feature_payload = dict(pred)
    return {
        "keypoints_xy": [(float(x), float(y)) for x, y in keypoints.tolist()],
        "descriptors": descriptors,
        "scores": [float(v) for v in scores.tolist()],
        "preprocess_resize": (
            "package_default_1024"
            if preprocess_resize is _USE_SUPERPOINT_DEFAULT_RESIZE
            else preprocess_resize
        ),
        "preprocess_input_channels": 1,
        "payload_image_size": tuple(int(v) for v in pred["image_size"].reshape(-1).tolist())
        if "image_size" in pred
        else None,
        "payload": feature_payload,
    }


def detect_and_describe_result(
    image_bgr,
    feature: str = "orb",
    feature_backend: Optional[str] = None,
    nfeatures: int = 2000,
    device: Optional[str] = None,
    force_cpu: bool = False,
    weights_dir: Optional[str] = None,
    max_keypoints: int = 2048,
    resize_long_edge: Optional[int] = None,
) -> FeatureResult:
    """Detect keypoints and compute descriptors using a named feature backend."""

    import cv2  # type: ignore

    backend_name = _normalize_feature_backend(feature=feature, feature_backend=feature_backend)
    start_time = time.perf_counter()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if backend_name == "opencv_orb":
        detector = cv2.ORB_create(nfeatures=nfeatures)
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
            cv_keypoints=list(keypoints),
        )

    if backend_name == "opencv_sift":
        if not hasattr(cv2, "SIFT_create"):
            raise ValueError("SIFT not available in this OpenCV build.")
        detector = cv2.SIFT_create()
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
            cv_keypoints=list(keypoints),
        )

    if backend_name != "superpoint":
        raise ValueError(f"Unsupported feature backend: {backend_name}")

    try:
        extractor, device_info, weights_info, dependency_status = _load_superpoint_extractor(
            device=device,
            force_cpu=force_cpu,
            weights_dir=weights_dir,
            max_keypoints=max_keypoints,
        )
        extracted = _extract_superpoint_features(
            image_bgr,
            extractor=extractor,
            resolved_device=str(device_info["resolved_device"]),
            resize_long_edge=resize_long_edge,
        )
    except MethodBBackendError:
        raise
    except Exception as exc:
        raise MethodBBackendError(
            f"SuperPoint backend failed: {exc}",
            diagnostics={
                "backend_name": "superpoint",
                "error_type": type(exc).__name__,
                "device": device,
                "force_cpu": bool(force_cpu),
                "weights_dir": weights_dir,
                "max_keypoints": _normalize_superpoint_max_keypoints(max_keypoints),
                "resize_long_edge": resize_long_edge,
            },
        ) from exc

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    cv_keypoints = _build_cv_keypoints(extracted["keypoints_xy"], extracted["scores"])
    return FeatureResult(
        keypoints_xy=extracted["keypoints_xy"],
        descriptors=extracted["descriptors"],
        scores=extracted["scores"],
        image_size=(int(image_bgr.shape[1]), int(image_bgr.shape[0])),
        backend_name=backend_name,
        runtime_ms=float(runtime_ms),
        meta={
            "legacy_feature": feature,
            "dependency_status": dependency_status,
            "device_info": device_info,
            "weights_info": weights_info,
            "max_keypoints": _normalize_superpoint_max_keypoints(max_keypoints),
            "resize_long_edge": resize_long_edge,
            "superpoint_preprocess_resize": extracted["preprocess_resize"],
            "payload_image_size": extracted["payload_image_size"],
            "payload_format": "lightglue_superpoint_v1",
        },
        cv_keypoints=cv_keypoints,
        backend_payload=extracted["payload"],
    )


def detect_and_describe(
    image_bgr,
    feature: str = "orb",
    nfeatures: int = 2000,
) -> Tuple[List, Optional[object]]:
    """Detect keypoints and compute descriptors."""

    result = detect_and_describe_result(
        image_bgr,
        feature=feature,
        feature_backend=None,
        nfeatures=nfeatures,
    )
    return result.cv_keypoints, result.descriptors
