"""Visualization helpers for baseline stitching outputs."""

from __future__ import annotations

from pathlib import Path


def save_image(path: Path, image) -> None:
    """Save a BGR image to disk, creating parent directories if needed."""

    import cv2  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def overlay_images(left_bgr, right_bgr, alpha: float = 0.5):
    """Overlay two images for quick visual comparison."""

    import numpy as np  # type: ignore

    left = left_bgr.astype("float32")
    right = right_bgr.astype("float32")
    overlay = (left * alpha + right * (1.0 - alpha)).clip(0, 255)
    return overlay.astype("uint8")
