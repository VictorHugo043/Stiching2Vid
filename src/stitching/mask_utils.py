"""Mask utilities for safe stitching composition and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


def ensure_uint8_mask(mask, shape: Tuple[int, int]):
    """Convert mask to uint8 in {0,255} with target shape (h, w)."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    h, w = int(shape[0]), int(shape[1])
    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != (h, w):
        arr = cv2.resize(arr.astype("float32"), (w, h), interpolation=cv2.INTER_NEAREST)

    if arr.dtype == np.bool_:
        bin_mask = arr.astype(np.uint8)
    else:
        arr_f = arr.astype("float32")
        if arr_f.max() <= 1.0:
            bin_mask = (arr_f > 0.5).astype(np.uint8)
        else:
            bin_mask = (arr_f > 127.0).astype(np.uint8)
    return (bin_mask * 255).astype(np.uint8)


def ensure_float_mask(mask, shape: Tuple[int, int]):
    """Convert mask to float32 in [0,1] with target shape (h, w)."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    h, w = int(shape[0]), int(shape[1])
    if mask is None:
        return np.zeros((h, w), dtype=np.float32)

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != (h, w):
        arr = cv2.resize(arr.astype("float32"), (w, h), interpolation=cv2.INTER_LINEAR)

    arr_f = arr.astype("float32")
    if arr_f.max() > 1.0:
        arr_f = arr_f / 255.0
    return arr_f.clip(0.0, 1.0).astype(np.float32)


def summarize_mask(name: str, mask) -> Dict[str, float]:
    """Return concise stats for debug logging."""
    import numpy as np  # type: ignore

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr_f = arr.astype("float32")
    if arr_f.max() > 1.0:
        arr_f = arr_f / 255.0

    total = float(arr_f.size) if arr_f.size else 1.0
    overlap_ratio = float((arr_f > 0).sum()) / total
    return {
        "name": name,
        "min": float(arr_f.min()) if arr_f.size else 0.0,
        "max": float(arr_f.max()) if arr_f.size else 0.0,
        "mean": float(arr_f.mean()) if arr_f.size else 0.0,
        "unique_count": float(len(np.unique(arr_f))) if arr_f.size else 0.0,
        "overlap_ratio": overlap_ratio,
    }


def save_mask_png(path: Path, mask) -> None:
    """Save mask visualization to png as uint8 in [0,255]."""
    import cv2  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)

