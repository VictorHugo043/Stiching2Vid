"""Crop helpers aligned with OpenStitching semantics.

This module is intentionally compatible with the basic interfaces from
`stitching/cropper.py` in OpenStitching:
- `Rectangle`
- `Cropper.prepare(...)`
- `Cropper.crop_images(...)`
- `Cropper.crop_rois(...)`

Coordinate-system notes:
- Input `corners` are panorama-space top-left coordinates for each warped ROI.
- `estimate_panorama_mask()` builds one panorama mask in a local canvas whose
  origin is `(min_corner_x, min_corner_y)` across all inputs.
- `get_zero_center_corners()` applies the same origin shift so rectangle
  intersections are computed in the same coordinate system as the panorama mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


def _as_uint8_mask(mask) -> "np.ndarray":
    """Normalize any mask-like input to uint8 {0,255}."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if mask is None:
        raise ValueError("mask cannot be None")
    if hasattr(mask, "get"):
        mask = mask.get()
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangle in integer pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return int(max(0, self.w) * max(0, self.h))

    @property
    def corner(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    @property
    def size(self) -> Tuple[int, int]:
        return int(self.w), int(self.h)

    @property
    def width(self) -> int:
        return int(self.w)

    @property
    def height(self) -> int:
        return int(self.h)

    @property
    def x2(self) -> int:
        return int(self.x + self.w)

    @property
    def y2(self) -> int:
        return int(self.y + self.h)

    def times(self, scale: float) -> "Rectangle":
        sx = float(scale)
        return Rectangle(
            int(round(self.x * sx)),
            int(round(self.y * sx)),
            int(round(self.w * sx)),
            int(round(self.h * sx)),
        )

    def draw_on(self, img, color=(0, 0, 255), size: int = 1):
        """Draw the rectangle on BGR/gray image."""

        import cv2  # type: ignore

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        start = (int(self.x), int(self.y))
        end = (int(self.x2 - 1), int(self.y2 - 1))
        cv2.rectangle(img, start, end, color, int(size))
        return img


class Cropper:
    """Largest-interior-rectangle cropper.

    Args:
        crop: enable crop pipeline.
        lir_method: `auto|lir|fallback`.
        lir_erode: erosion iterations used by fallback path.
        warning_handler: optional callback for explicit warning reporting.
    """

    DEFAULT_CROP = True

    def __init__(
        self,
        crop: bool = DEFAULT_CROP,
        lir_method: str = "auto",
        lir_erode: int = 2,
        warning_handler: Optional[Callable[[str], None]] = None,
    ):
        method = str(lir_method).lower().strip()
        if method not in {"auto", "lir", "fallback"}:
            raise ValueError(f"Unsupported lir_method: {lir_method}")

        self.do_crop = bool(crop)
        self.lir_method = method
        self.lir_erode = max(0, int(lir_erode))
        self.warning_handler = warning_handler

        self.overlapping_rectangles: List[Rectangle] = []
        self.intersection_rectangles: List[Rectangle] = []
        self.panorama_mask = None
        self.lir_rectangle: Optional[Rectangle] = None
        self.lir_method_used = "disabled"

        # Origin of the local panorama mask coordinate system.
        self.panorama_origin: Tuple[int, int] = (0, 0)
        # Min corner used by `get_zero_center_corners` for input corners.
        self.zero_center_origin: Tuple[int, int] = (0, 0)

    def _warn(self, message: str) -> None:
        logging.warning(message)
        if self.warning_handler is not None:
            self.warning_handler(message)

    def prepare(self, imgs, masks, corners, sizes) -> None:
        """Prepare per-image crop rectangles from warped panorama inputs."""

        self.overlapping_rectangles = []
        self.intersection_rectangles = []
        self.panorama_mask = None
        self.lir_rectangle = None
        self.lir_method_used = "disabled"
        self.panorama_origin = self._result_roi_origin(corners)
        self.zero_center_origin = self.panorama_origin

        if not self.do_crop:
            return

        mask = self.estimate_panorama_mask(imgs, masks, corners, sizes)
        lir = self.estimate_largest_interior_rectangle(mask)
        zero_corners = self.get_zero_center_corners(corners)
        rectangles = self.get_rectangles(zero_corners, sizes)
        overlaps = self.get_overlaps(rectangles, lir)
        intersections = self.get_intersections(rectangles, overlaps)

        self.panorama_mask = mask
        self.lir_rectangle = lir
        self.overlapping_rectangles = overlaps
        self.intersection_rectangles = intersections

    def crop_images(self, imgs: Iterable, aspect: float = 1):
        for idx, img in enumerate(imgs):
            yield self.crop_img(img, idx, aspect)

    def crop_img(self, img, idx: int, aspect: float = 1):
        if not self.do_crop:
            return img
        if idx < 0 or idx >= len(self.intersection_rectangles):
            raise IndexError(f"crop index out of range: {idx}")
        intersection = self.intersection_rectangles[idx]
        scaled = intersection.times(aspect)
        return self.crop_rectangle(img, scaled)

    def crop_rois(self, corners, sizes, aspect: float = 1):
        """Return cropped ROI `corners/sizes` in zero-centered crop canvas."""

        if not self.do_crop:
            return corners, sizes

        scaled_overlaps = [rect.times(aspect) for rect in self.overlapping_rectangles]
        cropped_corners = [rect.corner for rect in scaled_overlaps]
        cropped_corners = self.get_zero_center_corners(cropped_corners)
        cropped_sizes = [rect.size for rect in scaled_overlaps]
        return cropped_corners, cropped_sizes

    def get_overlaps_absolute(self, aspect: float = 1) -> List[Rectangle]:
        """Return overlaps back in panorama-space absolute coordinates.

        This is useful when seam/blend logic needs to paste cropped masks back
        onto the original full panorama canvas.
        """

        if not self.do_crop:
            return []
        ox = int(round(self.panorama_origin[0] * float(aspect)))
        oy = int(round(self.panorama_origin[1] * float(aspect)))
        overlaps_abs: List[Rectangle] = []
        for rect in self.overlapping_rectangles:
            scaled = rect.times(aspect)
            overlaps_abs.append(Rectangle(ox + scaled.x, oy + scaled.y, scaled.w, scaled.h))
        return overlaps_abs

    @staticmethod
    def estimate_panorama_mask(imgs, masks, corners, sizes):
        """Compose all warped masks into one panorama mask via OR blending."""

        import numpy as np  # type: ignore

        if not corners or not sizes:
            raise ValueError("corners/sizes cannot be empty")
        if len(corners) != len(sizes):
            raise ValueError("corners/sizes length mismatch")
        if masks is None or len(masks) != len(corners):
            raise ValueError("masks length mismatch")

        min_x = min(int(c[0]) for c in corners)
        min_y = min(int(c[1]) for c in corners)
        max_x = max(int(c[0]) + int(s[0]) for c, s in zip(corners, sizes))
        max_y = max(int(c[1]) + int(s[1]) for c, s in zip(corners, sizes))
        pano_w = max(1, int(max_x - min_x))
        pano_h = max(1, int(max_y - min_y))

        panorama_mask = np.zeros((pano_h, pano_w), dtype=np.uint8)
        for idx, corner in enumerate(corners):
            x = int(corner[0]) - min_x
            y = int(corner[1]) - min_y
            mask = _as_uint8_mask(masks[idx])
            h, w = mask.shape[:2]

            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(pano_w, x + w)
            y1 = min(pano_h, y + h)
            if x1 <= x0 or y1 <= y0:
                continue

            mx0 = x0 - x
            my0 = y0 - y
            mx1 = mx0 + (x1 - x0)
            my1 = my0 + (y1 - y0)
            panorama_mask[y0:y1, x0:x1] = np.maximum(
                panorama_mask[y0:y1, x0:x1],
                mask[my0:my1, mx0:mx1],
            )
        return panorama_mask

    def estimate_largest_interior_rectangle(self, mask) -> Rectangle:
        """Estimate largest interior rectangle from panorama mask."""

        method = self.lir_method
        if method in {"auto", "lir"}:
            try:
                rect = self._estimate_lir_with_package(mask)
                self.lir_method_used = "lir"
                return rect
            except Exception as exc:
                self._warn(f"crop_lir_package_unavailable: {exc}; fallback to conservative LIR.")
                if method == "lir":
                    self._warn("crop_lir_method=lir requested, but package path failed; using fallback.")

        rect = self._estimate_lir_fallback(mask)
        self.lir_method_used = "fallback"
        return rect

    def _estimate_lir_with_package(self, mask) -> Rectangle:
        import cv2  # type: ignore
        import largestinteriorrectangle  # type: ignore
        import numpy as np  # type: ignore

        mask_u8 = _as_uint8_mask(mask)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("empty contours for LIR")
        contour = max(contours, key=cv2.contourArea)[:, 0, :]
        lir = largestinteriorrectangle.lir(mask_u8 > 0, contour)
        if lir is None or len(lir) != 4:
            raise ValueError(f"invalid LIR result: {lir}")
        rect = Rectangle(int(lir[0]), int(lir[1]), int(lir[2]), int(lir[3]))
        self._validate_rect_inside(mask_u8, rect)
        return rect

    def _estimate_lir_fallback(self, mask) -> Rectangle:
        """Conservative fallback that guarantees rectangle lies inside mask."""

        import cv2  # type: ignore
        import numpy as np  # type: ignore

        mask_u8 = _as_uint8_mask(mask)
        if not (mask_u8 > 0).any():
            raise ValueError("panorama mask is empty; cannot estimate crop rectangle")

        work = (mask_u8 > 0).astype(np.uint8)
        if self.lir_erode > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            eroded = cv2.erode(work * 255, kernel, iterations=int(self.lir_erode))
            eroded = (eroded > 0).astype(np.uint8)
            if eroded.any():
                work = eroded
            else:
                self._warn("crop_fallback_erode_empty: keep non-eroded mask for safety.")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            work = (labels == largest_label).astype(np.uint8)

        rect = self._largest_rectangle_in_binary(work > 0)
        self._validate_rect_inside(mask_u8, rect)
        return rect

    @staticmethod
    def _largest_rectangle_in_binary(binary_mask) -> Rectangle:
        """Largest axis-aligned rectangle of ones using histogram stack DP."""

        import numpy as np  # type: ignore

        binary = np.asarray(binary_mask).astype(bool)
        h, w = binary.shape[:2]
        heights = np.zeros((w,), dtype=np.int32)
        best_area = 0
        best = Rectangle(0, 0, 0, 0)

        for y in range(h):
            heights = np.where(binary[y], heights + 1, 0)
            stack: List[Tuple[int, int]] = []
            for x in range(w + 1):
                curr = int(heights[x]) if x < w else 0
                start = x
                while stack and stack[-1][1] > curr:
                    left_idx, bar_h = stack.pop()
                    bar_w = x - left_idx
                    area = bar_h * bar_w
                    if area > best_area and bar_h > 0 and bar_w > 0:
                        best_area = area
                        best = Rectangle(
                            int(left_idx),
                            int(y - bar_h + 1),
                            int(bar_w),
                            int(bar_h),
                        )
                    start = left_idx
                if not stack or stack[-1][1] < curr:
                    stack.append((start, curr))

        if best.area <= 0:
            ys, xs = np.where(binary)
            if len(xs) == 0:
                raise ValueError("cannot build fallback rectangle from empty mask")
            return Rectangle(int(xs[0]), int(ys[0]), 1, 1)
        return best

    @staticmethod
    def _validate_rect_inside(mask_u8, rect: Rectangle) -> None:
        if rect.w <= 0 or rect.h <= 0:
            raise ValueError(f"invalid rectangle size: {rect}")
        if rect.x < 0 or rect.y < 0:
            raise ValueError(f"negative rectangle corner: {rect}")
        if rect.x2 > mask_u8.shape[1] or rect.y2 > mask_u8.shape[0]:
            raise ValueError(f"rectangle outside mask bounds: {rect} vs {mask_u8.shape}")

        roi = mask_u8[rect.y : rect.y2, rect.x : rect.x2]
        if roi.size == 0 or not (roi > 0).all():
            raise ValueError(f"rectangle is not fully inside valid mask: {rect}")

    @staticmethod
    def _result_roi_origin(corners) -> Tuple[int, int]:
        min_x = min(int(corner[0]) for corner in corners)
        min_y = min(int(corner[1]) for corner in corners)
        return min_x, min_y

    @staticmethod
    def get_zero_center_corners(corners):
        min_corner_x = min(int(corner[0]) for corner in corners)
        min_corner_y = min(int(corner[1]) for corner in corners)
        return [(int(x) - min_corner_x, int(y) - min_corner_y) for x, y in corners]

    @staticmethod
    def get_rectangles(corners, sizes):
        rectangles = []
        for corner, size in zip(corners, sizes):
            rect = Rectangle(int(corner[0]), int(corner[1]), int(size[0]), int(size[1]))
            rectangles.append(rect)
        return rectangles

    @staticmethod
    def get_overlaps(rectangles, lir: Rectangle):
        return [Cropper.get_overlap(rect, lir) for rect in rectangles]

    @staticmethod
    def get_overlap(rectangle1: Rectangle, rectangle2: Rectangle):
        x1 = max(rectangle1.x, rectangle2.x)
        y1 = max(rectangle1.y, rectangle2.y)
        x2 = min(rectangle1.x2, rectangle2.x2)
        y2 = min(rectangle1.y2, rectangle2.y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Rectangles do not overlap.")
        return Rectangle(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    @staticmethod
    def get_intersections(rectangles, overlapping_rectangles):
        return [
            Cropper.get_intersection(rect, overlap_rect)
            for rect, overlap_rect in zip(rectangles, overlapping_rectangles)
        ]

    @staticmethod
    def get_intersection(rectangle: Rectangle, overlapping_rectangle: Rectangle):
        x = abs(int(overlapping_rectangle.x) - int(rectangle.x))
        y = abs(int(overlapping_rectangle.y) - int(rectangle.y))
        return Rectangle(int(x), int(y), int(overlapping_rectangle.w), int(overlapping_rectangle.h))

    @staticmethod
    def crop_rectangle(img, rectangle: Rectangle):
        if img is None:
            raise ValueError("img cannot be None")
        h, w = img.shape[:2]
        if rectangle.x < 0 or rectangle.y < 0:
            raise ValueError(f"crop rectangle has negative corner: {rectangle}")
        if rectangle.w <= 0 or rectangle.h <= 0:
            raise ValueError(f"crop rectangle has non-positive size: {rectangle}")
        if rectangle.x2 > w or rectangle.y2 > h:
            raise ValueError(
                f"crop rectangle out of image bounds: {rectangle} vs image (w={w}, h={h})",
            )
        return img[rectangle.y : rectangle.y2, rectangle.x : rectangle.x2]
