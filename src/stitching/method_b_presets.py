"""Named Method B preset candidates used for controlled optimization sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MethodBPreset:
    name: str
    description: str
    max_keypoints: int
    resize_long_edge: int | None
    depth_confidence: float | None
    width_confidence: float | None
    filter_threshold: float | None


METHOD_B_PRESETS: Dict[str, MethodBPreset] = {
    "accuracy_v1": MethodBPreset(
        name="accuracy_v1",
        description="Current frozen formal baseline: high recall, no LightGlue adaptivity.",
        max_keypoints=4096,
        resize_long_edge=1536,
        depth_confidence=-1.0,
        width_confidence=-1.0,
        filter_threshold=0.1,
    ),
    "kp3072_v1": MethodBPreset(
        name="kp3072_v1",
        description="Reduce SuperPoint keypoints while keeping the rest of accuracy preset fixed.",
        max_keypoints=3072,
        resize_long_edge=1536,
        depth_confidence=-1.0,
        width_confidence=-1.0,
        filter_threshold=0.1,
    ),
}


def get_method_b_preset(name: str) -> MethodBPreset:
    key = str(name).strip().lower()
    if key not in METHOD_B_PRESETS:
        raise KeyError(f"Unknown Method B preset: {name}")
    return METHOD_B_PRESETS[key]


def list_method_b_presets() -> List[MethodBPreset]:
    return list(METHOD_B_PRESETS.values())
