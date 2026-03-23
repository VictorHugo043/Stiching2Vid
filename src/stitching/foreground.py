"""Compatible foreground/object-aware helpers for seam MVP.

This module does not introduce a new seam backend. It provides:
1) a lightweight disagreement-based foreground proxy in overlap ROI,
2) a conservative mask reassignment step that keeps seam from cutting
   through protected regions when previous seam assignment is available.
"""

from __future__ import annotations

from typing import Optional, Tuple


def compute_disagreement_mask(
    left_img,
    right_img,
    left_mask,
    right_mask,
    *,
    diff_threshold: float,
    dilate_iter: int = 0,
):
    """Compute a protected-region mask from cross-view disagreement."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    left_arr = np.asarray(left_img)
    right_arr = np.asarray(right_img)
    left_valid = np.asarray(left_mask) > 0
    right_valid = np.asarray(right_mask) > 0
    overlap = left_valid & right_valid

    protect = np.zeros(left_valid.shape, dtype=np.uint8)
    if not overlap.any():
        return protect

    diff = cv2.absdiff(left_arr, right_arr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
    protect[overlap & (gray >= float(diff_threshold))] = 255

    if int(dilate_iter) > 0 and protect.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        protect = cv2.dilate(protect, kernel, iterations=int(dilate_iter))
        protect = np.where(overlap, protect, 0).astype(np.uint8)
    return protect


def compute_mask_ratio(mask, reference_mask=None) -> float:
    """Return non-zero mask ratio, optionally normalized by reference region."""

    import numpy as np  # type: ignore

    arr = np.asarray(mask) > 0
    if reference_mask is None:
        denom = int(arr.size)
        return float(arr.sum()) / float(max(1, denom))

    ref = np.asarray(reference_mask) > 0
    denom = int(ref.sum())
    if denom <= 0:
        return 0.0
    return float((arr & ref).sum()) / float(denom)


def apply_protect_mask_assignment(
    final_left_mask,
    final_right_mask,
    left_canvas_mask,
    right_canvas_mask,
    protect_mask_full,
    *,
    prefer_left_mask=None,
    prefer_right_mask=None,
    limit_mask_full=None,
) -> Tuple["np.ndarray", "np.ndarray", float]:
    """Reassign protected overlap region using previous seam preference."""

    import numpy as np  # type: ignore

    final_left = (np.asarray(final_left_mask) > 0).copy()
    final_right = (np.asarray(final_right_mask) > 0).copy()
    left_valid = np.asarray(left_canvas_mask) > 0
    right_valid = np.asarray(right_canvas_mask) > 0
    overlap = left_valid & right_valid
    protect = overlap & (np.asarray(protect_mask_full) > 0)

    if limit_mask_full is not None:
        protect &= np.asarray(limit_mask_full) > 0

    protect_ratio = compute_mask_ratio(protect.astype(np.uint8) * 255, overlap.astype(np.uint8) * 255)
    if not protect.any():
        return final_left.astype(np.uint8) * 255, final_right.astype(np.uint8) * 255, float(protect_ratio)

    if prefer_left_mask is not None and prefer_right_mask is not None:
        prefer_left = np.asarray(prefer_left_mask) > 0
        prefer_right = np.asarray(prefer_right_mask) > 0
    else:
        prefer_left = final_left.copy()
        prefer_right = final_right.copy()

    assign_left = protect & prefer_left & left_valid
    assign_right = protect & (~assign_left) & prefer_right & right_valid

    unresolved = protect & (~assign_left) & (~assign_right)
    assign_left |= unresolved & final_left & left_valid
    assign_right |= unresolved & (~assign_left) & final_right & right_valid

    unresolved = protect & (~assign_left) & (~assign_right)
    assign_left |= unresolved & left_valid
    assign_right |= protect & (~assign_left) & right_valid

    final_left &= ~protect
    final_right &= ~protect
    final_left |= assign_left
    final_right |= assign_right
    final_right &= ~assign_left

    if limit_mask_full is not None:
        limit = np.asarray(limit_mask_full) > 0
        final_left &= limit
        final_right &= limit

    return final_left.astype(np.uint8) * 255, final_right.astype(np.uint8) * 255, float(protect_ratio)
