"""State container for frame-reuse video stitching mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VideoStitchState:
    """Persistent state for `VideoStitcher`.

    In `frame0_*` reuse modes, the first successful initialization stores
    geometry/crop/seam artifacts here and subsequent frames reuse them.
    """

    initialized: bool = False
    H_or_cameras: Optional[object] = None
    cropper_state: Optional[object] = None
    seam_masks_low: Optional[List[object]] = None
    seam_masks_final: Optional[List[object]] = None
    corners_low: Optional[List[Tuple[int, int]]] = None
    sizes_low: Optional[List[Tuple[int, int]]] = None
    corners_final: Optional[List[Tuple[int, int]]] = None
    sizes_final: Optional[List[Tuple[int, int]]] = None
    exposure_comp_state: Optional[object] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    frame0_index: Optional[int] = None
    init_timestamp: Optional[float] = None
    init_count: int = 0

