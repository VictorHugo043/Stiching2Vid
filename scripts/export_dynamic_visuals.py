#!/usr/bin/env python3
"""Build representative visualization manifest for a dynamic compare suite."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect representative stitched/seam snapshots from a dynamic compare suite "
            "and write visual_manifest.csv plus visual_summary.md."
        )
    )
    parser.add_argument(
        "--suite_id",
        required=True,
        help="Suite id under outputs/video_compare/",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Optional explicit summary.csv path",
    )
    return parser


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _first_file(paths: Sequence[Path]) -> Optional[Path]:
    return paths[0] if paths else None


def _rel(base: Path, target: Optional[Path]) -> str:
    if target is None:
        return ""
    return os.path.relpath(str(target), start=str(base))


def _extract_frame_idx(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    name = path.name
    parts = name.split("_")
    if len(parts) < 3:
        return None
    return parts[2]


def _select_visuals(repo_root: Path, summary_dir: Path, run_dir_rel: str) -> Dict[str, str]:
    run_dir = repo_root / run_dir_rel
    snapshots_dir = run_dir / "snapshots"
    frame_stitched = sorted(snapshots_dir.glob("frame_*_stitched.png"))
    frame_overlay = sorted(snapshots_dir.glob("frame_*_overlay.png"))
    event_overlay = sorted(snapshots_dir.glob("seam_event_*_seam_overlay.png"))
    event_diff = sorted(snapshots_dir.glob("seam_event_*_overlap_diff.png"))
    overlay_sm = sorted(snapshots_dir.glob("overlay_sm_*.png"))
    overlay_raw = sorted(snapshots_dir.glob("overlay_raw_*.png"))

    preview_stitched = _first_file(frame_stitched)
    preview_overlay = _first_file(frame_overlay)
    preview_overlay_sm = _first_file(overlay_sm)
    preview_overlay_raw = _first_file(overlay_raw)
    first_event_overlay = _first_file(event_overlay)
    first_event_diff = _first_file(event_diff)
    first_event_idx = _extract_frame_idx(first_event_overlay)
    first_event_stitched = (
        snapshots_dir / f"frame_{first_event_idx}_stitched.png" if first_event_idx else None
    )
    if first_event_stitched is not None and not first_event_stitched.exists():
        first_event_stitched = None

    return {
        "preview_stitched": _rel(summary_dir, preview_stitched),
        "preview_overlay": _rel(summary_dir, preview_overlay),
        "preview_overlay_raw": _rel(summary_dir, preview_overlay_raw),
        "preview_overlay_sm": _rel(summary_dir, preview_overlay_sm),
        "first_event_idx": first_event_idx or "",
        "first_event_stitched": _rel(summary_dir, first_event_stitched),
        "first_event_overlay": _rel(summary_dir, first_event_overlay),
        "first_event_overlap_diff": _rel(summary_dir, first_event_diff),
    }


def _build_markdown(
    suite_id: str,
    rows: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Dynamic Visual Summary: {suite_id}")
    lines.append("")
    lines.append("本文件汇总正式 dynamic seam 对比矩阵的代表性图像。")
    lines.append("")

    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    preset_order = [
        "baseline_fixed",
        "keyframe_seam10",
        "trigger_fused_d18_fg008",
        "adaptive_trigger_fused_d18_fg008",
    ]
    for pair_id in pair_ids:
        lines.append(f"## {pair_id}")
        lines.append("")
        pair_rows = {str(row["preset_id"]): row for row in rows if str(row["pair_id"]) == pair_id}
        for preset_id in preset_order:
            row = pair_rows.get(preset_id)
            if not row:
                continue
            lines.append(f"### {preset_id}")
            lines.append("")
            lines.append(
                f"- `run_id={row['run_id']}` "
                f"`overlap_diff_after={row['mean_overlap_diff_after']}` "
                f"`stitched_delta={row['mean_stitched_delta']}` "
                f"`seam_recompute={row['seam_recompute_count']}` "
                f"`geometry_update={row['geometry_update_count']}` "
                f"`fps={row['approx_fps']}`"
            )
            if row.get("preview_stitched"):
                lines.append(f"- Preview stitched: ![]({row['preview_stitched']})")
            if row.get("preview_overlay_sm"):
                lines.append(f"- Preview overlay_sm: ![]({row['preview_overlay_sm']})")
            if row.get("first_event_overlay"):
                lines.append(f"- First seam event overlay: ![]({row['first_event_overlay']})")
            if row.get("first_event_stitched"):
                lines.append(f"- First seam event stitched: ![]({row['first_event_stitched']})")
            if row.get("first_event_overlap_diff"):
                lines.append(f"- First seam event overlap diff: ![]({row['first_event_overlap_diff']})")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    suite_dir = repo_root / "outputs" / "video_compare" / args.suite_id
    summary_csv = Path(args.summary_csv) if args.summary_csv else suite_dir / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"summary csv not found: {summary_csv}")

    raw_rows = _read_csv(summary_csv)
    rows: List[Dict[str, object]] = []
    for row in raw_rows:
        if row.get("status") != "passed":
            continue
        visual_paths = _select_visuals(repo_root, suite_dir, str(row["run_dir"]))
        merged = dict(row)
        merged.update(visual_paths)
        rows.append(merged)

    manifest_path = suite_dir / "visual_manifest.csv"
    markdown_path = suite_dir / "visual_summary.md"
    _write_csv(manifest_path, rows)
    markdown_path.write_text(
        _build_markdown(args.suite_id, rows),
        encoding="utf-8",
    )
    print(f"visual_manifest={manifest_path.relative_to(repo_root)}")
    print(f"visual_summary={markdown_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
