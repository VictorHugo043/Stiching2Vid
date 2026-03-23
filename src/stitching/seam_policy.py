"""Seam update policy helpers for dynamic seam MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


TRIGGER_KEYS = ("overlap", "diff", "foreground")


@dataclass
class SeamPolicyDecision:
    """Decision for whether the current frame should recompute seam."""

    policy: str
    recompute: bool
    reason: str
    trigger_flags: Dict[str, object] = field(default_factory=dict)
    trigger_armed_next: bool = True
    trigger_states_next: Dict[str, bool] = field(default_factory=dict)


def _normalize_trigger_states(
    trigger_states: Optional[Dict[str, object]],
    trigger_armed: bool,
) -> Dict[str, bool]:
    states = {key: bool(trigger_armed) for key in TRIGGER_KEYS}
    if not trigger_states:
        return states
    for key in TRIGGER_KEYS:
        if key in trigger_states:
            states[key] = bool(trigger_states[key])
    return states


def _compute_trigger_armed_any(
    states: Dict[str, bool],
    *,
    trigger_overlap_ratio: float,
    trigger_diff_threshold: float,
    trigger_foreground_ratio: float,
) -> bool:
    active_keys = []
    if float(trigger_overlap_ratio) > 0.0:
        active_keys.append("overlap")
    if float(trigger_diff_threshold) > 0.0:
        active_keys.append("diff")
    if float(trigger_foreground_ratio) > 0.0:
        active_keys.append("foreground")
    if not active_keys:
        return True
    return any(bool(states.get(key, True)) for key in active_keys)


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
    trigger_states: Optional[Dict[str, object]] = None,
    force_recompute: bool = False,
) -> SeamPolicyDecision:
    """Decide whether seam should be recomputed on the current frame."""

    normalized_policy = str(policy).strip().lower()
    states_prev = _normalize_trigger_states(trigger_states, trigger_armed)
    flags: Dict[str, object] = {
        "seam_cache_available": bool(seam_cache_available),
        "is_seam_keyframe": bool(is_seam_keyframe),
        "seam_age_frames": int(seam_age_frames),
        "overlap_area_current": int(overlap_area_current),
        "overlap_area_reference": int(overlap_area_reference),
        "overlap_diff_before": float(overlap_diff_before),
        "foreground_ratio": float(foreground_ratio),
        "trigger_armed_prev": bool(trigger_armed),
        "trigger_states_prev": dict(states_prev),
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
            trigger_states_next={key: False for key in TRIGGER_KEYS},
        )

    if not seam_cache_available:
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=True,
            reason="no_cache",
            trigger_flags=flags,
            trigger_armed_next=False,
            trigger_states_next={key: False for key in TRIGGER_KEYS},
        )

    if normalized_policy == "fixed":
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=False,
            reason="reuse_fixed",
            trigger_flags=flags,
            trigger_armed_next=bool(trigger_armed),
            trigger_states_next=dict(states_prev),
        )

    if normalized_policy == "keyframe":
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=bool(is_seam_keyframe),
            reason="keyframe" if is_seam_keyframe else "reuse_keyframe",
            trigger_flags=flags,
            trigger_armed_next=bool(trigger_armed),
            trigger_states_next=dict(states_prev),
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
    states_after_rearm = dict(states_prev)
    if overlap_rearm_ready:
        states_after_rearm["overlap"] = True
    if diff_rearm_ready:
        states_after_rearm["diff"] = True
    if foreground_rearm_ready:
        states_after_rearm["foreground"] = True
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
            "rearm_ready_overlap": bool(overlap_rearm_ready),
            "rearm_ready_diff": bool(diff_rearm_ready),
            "rearm_ready_foreground": bool(foreground_rearm_ready),
            "trigger_states_after_rearm": dict(states_after_rearm),
        }
    )

    fired_channels = []
    fired_reasons = []
    if overlap_triggered and states_after_rearm["overlap"]:
        fired_channels.append("overlap")
        fired_reasons.append(f"trigger_overlap<{float(trigger_overlap_ratio):.3f}")
    if diff_triggered and states_after_rearm["diff"]:
        fired_channels.append("diff")
        fired_reasons.append(f"trigger_diff>={float(trigger_diff_threshold):.3f}")
    if foreground_triggered and states_after_rearm["foreground"]:
        fired_channels.append("foreground")
        fired_reasons.append(f"trigger_fg>={float(trigger_foreground_ratio):.3f}")
    if cadence_triggered:
        fired_reasons.append("trigger_cadence")
    flags["fired_channels"] = list(fired_channels)

    if fired_channels and cooldown_active and not cadence_triggered:
        states_blocked = dict(states_after_rearm)
        for key in fired_channels:
            states_blocked[key] = False
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=False,
            reason="cooldown:" + "+".join(fired_reasons),
            trigger_flags=flags,
            trigger_armed_next=_compute_trigger_armed_any(
                states_blocked,
                trigger_overlap_ratio=float(trigger_overlap_ratio),
                trigger_diff_threshold=float(trigger_diff_threshold),
                trigger_foreground_ratio=float(trigger_foreground_ratio),
            ),
            trigger_states_next=states_blocked,
        )
    if fired_reasons:
        states_next = dict(states_after_rearm)
        for key in fired_channels:
            states_next[key] = False
        return SeamPolicyDecision(
            policy=normalized_policy,
            recompute=True,
            reason="+".join(fired_reasons),
            trigger_flags=flags,
            trigger_armed_next=_compute_trigger_armed_any(
                states_next,
                trigger_overlap_ratio=float(trigger_overlap_ratio),
                trigger_diff_threshold=float(trigger_diff_threshold),
                trigger_foreground_ratio=float(trigger_foreground_ratio),
            ),
            trigger_states_next=states_next,
        )
    return SeamPolicyDecision(
        policy=normalized_policy,
        recompute=False,
        reason="reuse_trigger",
        trigger_flags=flags,
        trigger_armed_next=_compute_trigger_armed_any(
            states_after_rearm,
            trigger_overlap_ratio=float(trigger_overlap_ratio),
            trigger_diff_threshold=float(trigger_diff_threshold),
            trigger_foreground_ratio=float(trigger_foreground_ratio),
        ),
        trigger_states_next=states_after_rearm,
    )
