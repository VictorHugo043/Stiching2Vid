"""Single-frame quality preview helpers.

This module reuses ``VideoStitcher.initialize_from_first_frame`` so the
single-frame entry can share the same crop/seam/blend quality chain as the
video pipeline without copying the large inline logic from the video script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional


def seam_cli_to_method(seam_mode: str) -> str:
    """Map CLI seam mode names to ``VideoStitcher`` seam method names."""

    normalized = str(seam_mode or "opencv_dp_color").strip().lower()
    mapping = {
        "none": "none",
        "opencv_dp_color": "dp_color",
        "opencv_dp_colorgrad": "dp_colorgrad",
        "opencv_voronoi": "voronoi",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported seam mode: {seam_mode}")
    return mapping[normalized]


@dataclass
class FrameComposeResult:
    stitched: object
    backend_name: str
    runtime_ms: float
    overlap_area: int
    crop_applied: bool
    crop_method: str
    crop_rect: Optional[Dict[str, int]]
    output_bbox: Optional[Dict[str, int]]
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


def compose_frame_quality_preview(
    left_frame,
    right_frame,
    H,
    T,
    canvas_size,
    *,
    output_dir: Path,
    frame_idx: int,
    blend_mode: str,
    mb_levels: int,
    seam_mode: str,
    seam_megapix: float,
    seam_dilate: int,
    crop_enabled: bool,
    lir_method: str,
    lir_erode: int,
    crop_debug: int,
    warning_handler: Optional[Callable[[str], None]] = None,
) -> FrameComposeResult:
    """Compose one stitched frame using the shared video-quality chain."""

    from stitching.video_stitcher import VideoStitcher

    warnings: List[str] = []

    def _warn(message: str) -> None:
        warnings.append(str(message))
        if warning_handler is not None:
            warning_handler(message)

    seam_method = seam_cli_to_method(seam_mode)
    stitcher = VideoStitcher(
        seam_method=seam_method,
        seam_megapix=seam_megapix,
        seam_dilate=seam_dilate,
        blend_mode=blend_mode,
        mb_levels=mb_levels,
        crop_enabled=crop_enabled,
        lir_method=lir_method,
        lir_erode=lir_erode,
        crop_debug=crop_debug,
        reuse_mode="frame0_all",
        output_dir=output_dir,
        warning_handler=_warn,
    )
    result = stitcher.initialize_from_first_frame(
        left_frame,
        right_frame,
        H,
        T,
        canvas_size,
        frame_idx=frame_idx,
    )

    output_bbox = result.get("output_bbox")
    output_bbox_dict = None
    if output_bbox is not None:
        output_bbox_dict = {
            "x": int(output_bbox[0]),
            "y": int(output_bbox[1]),
            "w": int(output_bbox[2]),
            "h": int(output_bbox[3]),
        }

    return FrameComposeResult(
        stitched=result["stitched"],
        backend_name="video_stitcher_initialize_v1",
        runtime_ms=float(result["init_ms"]),
        overlap_area=int(result["overlap_area"]),
        crop_applied=bool(result.get("crop_applied", False)),
        crop_method=str(result.get("crop_method", "none")),
        crop_rect=result.get("crop_rect"),
        output_bbox=output_bbox_dict,
        warnings=warnings,
        meta={
            "seam_cli_mode": str(seam_mode),
            "seam_method": seam_method,
            "blend_mode": str(blend_mode),
            "mb_levels": int(mb_levels),
            "crop_enabled": bool(crop_enabled),
            "lir_method": str(lir_method),
            "lir_erode": int(lir_erode),
            "crop_debug": int(crop_debug),
            "reuse_mode": "frame0_all",
            "seam_compute_ms_split_available": False,
            "state_metadata": dict(stitcher.state.metadata),
        },
    )
