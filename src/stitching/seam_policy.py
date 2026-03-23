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
    trigger_armed_next: bool = True


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
    foreground_ratio: float = 0.0,
    trigger_foreground_ratio: float = 0.0,
    cooldown_frames: int = 0,
    hysteresis_ratio: float = 1.0,
    trigger_armed: bool = True,
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
        "foreground_ratio": float(foreground_ratio),
        "trigger_armed_prev": bool(trigger_armed),
        "cooldown_frames": int(max(0, int(cooldown_frames))),
        "hysteresis_ratio": float(hysteresis_ratio),
    }

    if force_recompute:
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=True,
            reason="forced",
            trigger_flags=flags,
            trigger_armed_next=False,
        )

    if not seam_cache_available:
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=True,
            reason="no_cache",
            trigger_flags=flags,
            trigger_armed_next=False,
        )

    if normalized_policy == "fixed":
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=False,
            reason="reuse_fixed",
            trigger_flags=flags,
            trigger_armed_next=bool(trigger_armed),
        )

    if normalized_policy == "keyframe":
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=bool(is_seam_keyframe),
            reason="keyframe" if is_seam_keyframe else "reuse_keyframe",
            trigger_flags=flags,
            trigger_armed_next=bool(trigger_armed),
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
    foreground_triggered = (
        float(trigger_foreground_ratio) > 0.0 and float(foreground_ratio) >= float(trigger_foreground_ratio)
    )
    cadence_triggered = bool(is_seam_keyframe)
    hyst = min(1.0, max(0.0, float(hysteresis_ratio)))
    overlap_rearm_ready = True
    if float(trigger_overlap_ratio) > 0.0:
        overlap_rearm_threshold = float(trigger_overlap_ratio) + (1.0 - float(trigger_overlap_ratio)) * (1.0 - hyst)
        overlap_rearm_ready = overlap_ratio >= overlap_rearm_threshold
    diff_rearm_ready = True
    if float(trigger_diff_threshold) > 0.0:
        diff_rearm_ready = float(overlap_diff_before) < float(trigger_diff_threshold) * hyst
    foreground_rearm_ready = True
    if float(trigger_foreground_ratio) > 0.0:
        foreground_rearm_ready = float(foreground_ratio) < float(trigger_foreground_ratio) * hyst
    rearm_ready = bool(overlap_rearm_ready and diff_rearm_ready and foreground_rearm_ready)
    cooldown_active = int(seam_age_frames) < max(0, int(cooldown_frames))
    flags.update(
        {
            "overlap_ratio": float(overlap_ratio),
            "trigger_overlap_ratio": float(trigger_overlap_ratio),
            "trigger_diff_threshold": float(trigger_diff_threshold),
            "trigger_foreground_ratio": float(trigger_foreground_ratio),
            "overlap_triggered": bool(overlap_triggered),
            "diff_triggered": bool(diff_triggered),
            "foreground_triggered": bool(foreground_triggered),
            "cadence_triggered": bool(cadence_triggered),
            "cooldown_active": bool(cooldown_active),
            "rearm_ready": bool(rearm_ready),
        }
    )

    if not bool(trigger_armed) and not cadence_triggered:
        if rearm_ready:
            return SeamPolicyDecision(
                policy=normalized_policy,
                recompute=False,
                reason="trigger_rearmed",
                trigger_flags=flags,
                trigger_armed_next=True,
            )
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=False,
            reason="hysteresis_hold",
            trigger_flags=flags,
            trigger_armed_next=False,
        )

    fired_reasons = []
    if overlap_triggered:
        fired_reasons.append(f"trigger_overlap<{float(trigger_overlap_ratio):.3f}")
    if diff_triggered:
        fired_reasons.append(f"trigger_diff>={float(trigger_diff_threshold):.3f}")
    if foreground_triggered:
        fired_reasons.append(f"trigger_fg>={float(trigger_foreground_ratio):.3f}")
    if cadence_triggered:
        fired_reasons.append("trigger_cadence")

    if fired_reasons and cooldown_active and not cadence_triggered:
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=False,
            reason="cooldown:" + "+".join(fired_reasons),
            trigger_flags=flags,
            trigger_armed_next=False,
        )
    if fired_reasons:
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=True,
            reason="+".join(fired_reasons),
            trigger_flags=flags,
            trigger_armed_next=False,
        )
    return SeamPolicyDecision(
        policy=normalized_policy,
        recompute=False,
        reason="reuse_trigger",
        trigger_flags=flags,
        trigger_armed_next=bool(trigger_armed),
    )
