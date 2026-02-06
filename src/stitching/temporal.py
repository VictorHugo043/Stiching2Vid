"""Temporal utilities for homography stabilization and jitter diagnostics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class JitterStats:
    """Per-frame jitter statistics between consecutive corner projections."""

    mean: Optional[float]
    max: Optional[float]


def _normalize_h(H):
    """Normalize homography so H[2,2] is close to 1 when possible."""
    if H is None:
        return None
    denom = float(H[2, 2]) if H.shape == (3, 3) else 0.0
    if abs(denom) < 1e-8:
        return H
    return H / denom


def _source_corners(image_size: Tuple[int, int]):
    """Return canonical image corners as float32 [4, 2]."""
    import numpy as np  # type: ignore

    width, height = int(image_size[0]), int(image_size[1])
    return np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype="float32",
    )


def transform_corners(H, image_size: Tuple[int, int], pre_transform=None):
    """Project image corners with homography (optionally prepended by pre_transform)."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if H is None:
        return None
    corners = _source_corners(image_size).reshape(-1, 1, 2)
    H_use = H
    if pre_transform is not None:
        H_use = pre_transform @ H
    projected = cv2.perspectiveTransform(corners, H_use)
    return projected.reshape(-1, 2).astype(np.float32)


def compute_jitter(prev_corners, curr_corners) -> JitterStats:
    """Compute jitter as inter-frame corner displacement.

    Formula:
    - d_i = ||c_i(t) - c_i(t-1)||_2 for 4 transformed corners
    - jitter_mean = mean(d_i), jitter_max = max(d_i)
    """
    import numpy as np  # type: ignore

    if prev_corners is None or curr_corners is None:
        return JitterStats(mean=None, max=None)
    diff = curr_corners - prev_corners
    dists = np.linalg.norm(diff, axis=1)
    return JitterStats(mean=float(np.mean(dists)), max=float(np.max(dists)))


class HomographySmoother:
    """Homography smoother with pluggable policies.

    Supported methods:
    - none: passthrough (fallback to last valid or identity on missing)
    - ema: exponential moving average on transformed corner trajectories
    - window: moving average over a fixed window on corner trajectories
    """

    def __init__(self, method: str = "none", alpha: float = 0.8, window: int = 5) -> None:
        self.method = method
        self.alpha = float(alpha)
        self.window = max(1, int(window))
        self._last_smoothed_H = None
        self._smoothed_corners = None
        self._history: Deque = deque(maxlen=self.window)

    def reset(self) -> None:
        """Reset internal states for a new stream/pair."""
        self._last_smoothed_H = None
        self._smoothed_corners = None
        self._history.clear()

    def update(self, raw_H, image_size: Tuple[int, int]):
        """Update smoother with current raw homography and return smoothed homography.

        For missing raw_H (e.g., FALLBACK/FAIL_EST), smoother keeps previous smoothed H
        to maintain temporal continuity.
        """
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if self.method == "none":
            if raw_H is not None:
                self._last_smoothed_H = _normalize_h(raw_H)
                return self._last_smoothed_H
            if self._last_smoothed_H is not None:
                return self._last_smoothed_H
            self._last_smoothed_H = np.eye(3, dtype="float64")
            return self._last_smoothed_H

        if raw_H is None:
            if self._last_smoothed_H is not None:
                return self._last_smoothed_H
            self._last_smoothed_H = np.eye(3, dtype="float64")
            return self._last_smoothed_H

        raw_H = _normalize_h(raw_H)
        raw_corners = transform_corners(raw_H, image_size=image_size)
        if raw_corners is None:
            if self._last_smoothed_H is not None:
                return self._last_smoothed_H
            self._last_smoothed_H = np.eye(3, dtype="float64")
            return self._last_smoothed_H

        if self.method == "window":
            self._history.append(raw_corners)

        if self._smoothed_corners is None:
            if self.method == "window":
                smoothed_corners = np.mean(np.stack(list(self._history), axis=0), axis=0)
            else:
                smoothed_corners = raw_corners
        elif self.method == "ema":
            # alpha 越大越平滑，越信任历史轨迹。
            smoothed_corners = (
                self.alpha * self._smoothed_corners + (1.0 - self.alpha) * raw_corners
            )
        elif self.method == "window":
            smoothed_corners = np.mean(np.stack(list(self._history), axis=0), axis=0)
        else:
            raise ValueError(f"Unsupported smooth method: {self.method}")

        if self.method != "window":
            self._history.append(raw_corners)
        self._smoothed_corners = smoothed_corners.astype("float32")

        src = _source_corners(image_size).astype("float32")
        H_sm = cv2.getPerspectiveTransform(src, self._smoothed_corners.astype("float32"))
        H_sm = _normalize_h(H_sm)
        if H_sm is None:
            if self._last_smoothed_H is not None:
                return self._last_smoothed_H
            self._last_smoothed_H = np.eye(3, dtype="float64")
            return self._last_smoothed_H

        self._last_smoothed_H = H_sm
        return self._last_smoothed_H
