"""Seam update policy helpers for dynamic seam MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SeamPolicyDecision:
    """Decision for whether the current frame should recompute seam."""

    policy: str
    recompute: bool
    reason: str
    trigger_flags: Dict[str, object] = field(default_factory=dict)


def resolve_seam_policy(seam_policy: Optional[str], video_mode: int, reuse_mode: str) -> str:
    """Resolve seam policy while preserving legacy defaults."""

    raw = (seam_policy or "auto").strip().lower()
    if raw != "auto":
        return raw
    if int(video_mode) == 1:
        return "keyframe" if reuse_mode == "frame0_geom" else "fixed"
    return "keyframe"


def resolve_seam_keyframe_every(
    seam_policy: Optional[str],
    seam_keyframe_every: int,
    keyframe_every: int,
    video_mode: int,
    reuse_mode: str,
) -> int:
    """Resolve seam cadence for keyframe/trigger policies."""

    raw = int(seam_keyframe_every)
    if raw > 0:
        return raw

    resolved_policy = resolve_seam_policy(seam_policy, video_mode, reuse_mode)
    if resolved_policy != "keyframe":
        return 0
    if int(video_mode) == 1 and (seam_policy or "auto").strip().lower() == "auto" and reuse_mode == "frame0_geom":
        return 1
    return max(1, int(keyframe_every))


def decide_seam_update(
    *,
    policy: str,
    seam_cache_available: bool,
    is_seam_keyframe: bool,
    seam_age_frames: int,
    overlap_area_current: int,
    overlap_area_reference: int,
    overlap_diff_before: float,
    trigger_overlap_ratio: float,
    trigger_diff_threshold: float,
    force_recompute: bool = False,
) -> SeamPolicyDecision:
    """Decide whether seam should be recomputed on the current frame."""

    normalized_policy = str(policy).strip().lower()
    flags: Dict[str, object] = {
        "seam_cache_available": bool(seam_cache_available),
        "is_seam_keyframe": bool(is_seam_keyframe),
        "seam_age_frames": int(seam_age_frames),
        "overlap_area_current": int(overlap_area_current),
        "overlap_area_reference": int(overlap_area_reference),
        "overlap_diff_before": float(overlap_diff_before),
    }

    if force_recompute:
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=True,
            reason="forced",
            trigger_flags=flags,
        )

    if not seam_cache_available:
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=True,
            reason="no_cache",
            trigger_flags=flags,
        )

    if normalized_policy == "fixed":
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=False,
            reason="reuse_fixed",
            trigger_flags=flags,
        )

    if normalized_policy == "keyframe":
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=bool(is_seam_keyframe),
            reason="keyframe" if is_seam_keyframe else "reuse_keyframe",
            trigger_flags=flags,
        )

    if normalized_policy != "trigger":
        raise ValueError(f"Unsupported seam policy: {policy}")

    overlap_ratio = 1.0
    if int(overlap_area_reference) > 0:
        overlap_ratio = float(overlap_area_current) / float(max(1, int(overlap_area_reference)))
    overlap_triggered = (
        float(trigger_overlap_ratio) > 0.0 and overlap_ratio < float(trigger_overlap_ratio)
    )
    diff_triggered = (
        float(trigger_diff_threshold) > 0.0 and float(overlap_diff_before) >= float(trigger_diff_threshold)
    )
    cadence_triggered = bool(is_seam_keyframe)
    flags.update(
        {
            "overlap_ratio": float(overlap_ratio),
            "trigger_overlap_ratio": float(trigger_overlap_ratio),
            "trigger_diff_threshold": float(trigger_diff_threshold),
            "overlap_triggered": bool(overlap_triggered),
            "diff_triggered": bool(diff_triggered),
            "cadence_triggered": bool(cadence_triggered),
        }
    )

    if overlap_triggered:
        reason = f"trigger_overlap<{float(trigger_overlap_ratio):.3f}"
        return SeamPolicyDecision(policy=normalized_policy, recompute=True, reason=reason, trigger_flags=flags)
    if diff_triggered:
        reason = f"trigger_diff>={float(trigger_diff_threshold):.3f}"
        return SeamPolicyDecision(policy=normalized_policy, recompute=True, reason=reason, trigger_flags=flags)
    if cadence_triggered:
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=True,
            reason="trigger_cadence",
            trigger_flags=flags,
        )
    return SeamPolicyDecision(
        policy=normalized_policy,
        recompute=False,
        reason="reuse_trigger",
        trigger_flags=flags,
    )
