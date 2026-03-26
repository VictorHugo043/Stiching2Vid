#!/usr/bin/env python3
"""Build dataset-level summaries from method and dynamic compare suites."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


METHOD_NUMERIC_FIELDS: List[str] = [
    "total_frames",
    "processed_frames",
    "success_frames",
    "fallback_frames",
    "mean_inliers",
    "mean_inlier_ratio",
    "avg_runtime_ms",
    "approx_fps",
    "mean_reprojection_error",
    "mean_inlier_spatial_coverage",
    "mean_overlap_diff_after",
    "mean_seam_band_illuminance_diff",
    "mean_seam_band_gradient_disagreement",
    "mean_seam_band_flicker",
    "mean_stitched_delta",
    "mean_jitter_raw",
    "mean_jitter_sm",
    "p95_jitter_raw",
    "p95_jitter_sm",
    "init_ms_mean",
    "per_frame_ms_mean",
    "avg_feature_runtime_ms_left",
    "avg_feature_runtime_ms_right",
    "avg_matching_runtime_ms",
    "avg_geometry_runtime_ms",
    "avg_geometry_event_total_ms",
    "warnings_count",
    "errors_count",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate dataset-level method and dynamic compare suite outputs into "
            "formal summary tables and an overview markdown."
        )
    )
    parser.add_argument("--suite_id", required=True, help="Parent suite id under outputs/phase3/")
    parser.add_argument("--method_suite_id", default=None, help="Method compare suite id under outputs/video_compare/")
    parser.add_argument("--dynamic_suite_id", default=None, help="Dynamic compare suite id under outputs/video_compare/")
    parser.add_argument("--pairs", nargs="*", default=None, help="Pairs included in the dataset suite")
    parser.add_argument("--fps", type=float, default=10.0, help="Nominal fps used for the suite")
    parser.add_argument("--max_frames", type=int, default=6000, help="Nominal max_frames used for the suite")
    return parser


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
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


def _rel(base: Path, target: Path | None) -> str:
    if target is None:
        return ""
    return os.path.relpath(str(target), start=str(base))


def _safe_float(value) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _aggregate_method_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "passed":
            continue
        grouped.setdefault(str(row["method"]), []).append(row)

    out: List[Dict[str, object]] = []
    for method, method_rows in sorted(grouped.items()):
        base = method_rows[0]
        summary: Dict[str, object] = {
            "method": method,
            "pair_count": len(method_rows),
            "geometry_mode": base.get("geometry_mode"),
            "reuse_mode": base.get("reuse_mode"),
            "feature_backend_effective": base.get("feature_backend_effective"),
            "matcher_backend_effective": base.get("matcher_backend_effective"),
            "geometry_backend_effective": base.get("geometry_backend_effective"),
            "method_b_requested_device": base.get("method_b_requested_device"),
            "method_b_resolved_device": base.get("method_b_resolved_device"),
            "method_b_device_resolution_reason": base.get("method_b_device_resolution_reason"),
        }
        for field in METHOD_NUMERIC_FIELDS:
            summary[field] = _mean(_safe_float(row.get(field)) for row in method_rows)
        out.append(summary)
    return out


def _build_pair_coverage(
    pairs: Sequence[str],
    method_rows: Sequence[Dict[str, str]],
    dynamic_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    by_pair_method: Dict[str, List[Dict[str, str]]] = {}
    by_pair_dynamic: Dict[str, List[Dict[str, str]]] = {}
    for row in method_rows:
        if row.get("status") == "passed":
            by_pair_method.setdefault(str(row["pair_id"]), []).append(row)
    for row in dynamic_rows:
        if row.get("status") == "passed":
            by_pair_dynamic.setdefault(str(row["pair_id"]), []).append(row)

    rows: List[Dict[str, object]] = []
    for pair_id in pairs:
        method_subset = by_pair_method.get(pair_id, [])
        dynamic_subset = by_pair_dynamic.get(pair_id, [])
        any_row = method_subset[0] if method_subset else (dynamic_subset[0] if dynamic_subset else {})
        rows.append(
            {
                "pair_id": pair_id,
                "dataset": any_row.get("dataset"),
                "input_type": any_row.get("input_type"),
                "fps": any_row.get("fps"),
                "fps_source": any_row.get("fps_source"),
                "method_run_count": len(method_subset),
                "dynamic_run_count": len(dynamic_subset),
                "method_processed_frames_mean": _mean(_safe_float(row.get("processed_frames")) for row in method_subset),
                "dynamic_processed_frames_mean": _mean(_safe_float(row.get("processed_frames")) for row in dynamic_subset),
                "method_success_frames_mean": _mean(_safe_float(row.get("success_frames")) for row in method_subset),
                "dynamic_success_frames_mean": _mean(_safe_float(row.get("success_frames")) for row in dynamic_subset),
            }
        )
    return rows


def _build_markdown(
    suite_id: str,
    pairs: Sequence[str],
    fps: float,
    max_frames: int,
    method_suite_id: str | None,
    dynamic_suite_id: str | None,
    method_summary_rows: Sequence[Dict[str, object]],
    dynamic_preset_rows: Sequence[Dict[str, str]],
    dynamic_visual_summary: Path | None,
    out_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append(f"# Dataset Summary: {suite_id}")
    lines.append("")
    lines.append("## 配置")
    lines.append("")
    lines.append(f"- Pairs: `{len(pairs)}`")
    lines.append(f"- FPS override: `{fps}`")
    lines.append(f"- Max frames: `{max_frames}`")
    if method_suite_id:
        lines.append(f"- Method suite: `{method_suite_id}`")
    if dynamic_suite_id:
        lines.append(f"- Dynamic suite: `{dynamic_suite_id}`")
    lines.append("")

    if method_summary_rows:
        lines.append("## Method Summary")
        lines.append("")
        for row in method_summary_rows:
            lines.append(
                f"- `{row['method']}` "
                f"`inliers={row.get('mean_inliers')}` "
                f"`inlier_ratio={row.get('mean_inlier_ratio')}` "
                f"`fps={row.get('approx_fps')}` "
                f"`reproj={row.get('mean_reprojection_error')}` "
                f"`seam_flicker={row.get('mean_seam_band_flicker')}`"
            )
        lines.append("")

    if dynamic_preset_rows:
        lines.append("## Dynamic Summary")
        lines.append("")
        for row in dynamic_preset_rows:
            lines.append(
                f"- `{row['preset_id']}` "
                f"`overlap_diff_after={row.get('mean_overlap_diff_after')}` "
                f"`stitched_delta={row.get('mean_stitched_delta')}` "
                f"`fps={row.get('approx_fps')}` "
                f"`geometry_updates={row.get('geometry_update_count')}`"
            )
        lines.append("")

    lines.append("## 代表性 artefacts")
    lines.append("")
    lines.append(f"- Pair coverage: `{_rel(out_dir, out_dir / 'pair_coverage.csv')}`")
    if method_suite_id:
        lines.append(f"- Method summary: `{_rel(out_dir, out_dir / 'method_summary.csv')}`")
        lines.append(f"- Method pair compare: `{_rel(out_dir, out_dir / 'method_pair_compare.csv')}`")
    if dynamic_suite_id:
        lines.append(f"- Dynamic preset summary: `{_rel(out_dir, out_dir / 'dynamic_preset_summary.csv')}`")
        lines.append(f"- Dynamic pair compare: `{_rel(out_dir, out_dir / 'dynamic_pair_compare.csv')}`")
    if dynamic_visual_summary is not None and dynamic_visual_summary.exists():
        lines.append(f"- Dynamic visual summary: `{_rel(out_dir, dynamic_visual_summary)}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs" / "phase3" / args.suite_id
    out_dir.mkdir(parents=True, exist_ok=True)

    method_suite_dir = (
        repo_root / "outputs" / "video_compare" / args.method_suite_id
        if args.method_suite_id
        else None
    )
    dynamic_suite_dir = (
        repo_root / "outputs" / "video_compare" / args.dynamic_suite_id
        if args.dynamic_suite_id
        else None
    )

    method_rows = _read_csv(method_suite_dir / "summary.csv") if method_suite_dir else []
    method_pair_compare_rows = _read_csv(method_suite_dir / "pair_compare.csv") if method_suite_dir else []
    dynamic_rows = _read_csv(dynamic_suite_dir / "summary.csv") if dynamic_suite_dir else []
    dynamic_preset_rows = _read_csv(dynamic_suite_dir / "preset_summary.csv") if dynamic_suite_dir else []
    dynamic_pair_compare_rows = _read_csv(dynamic_suite_dir / "pair_compare.csv") if dynamic_suite_dir else []

    pairs = list(args.pairs or [])
    if not pairs:
        discovered_pairs = sorted(
            {
                str(row["pair_id"])
                for row in method_rows + dynamic_rows
                if row.get("pair_id")
            }
        )
        pairs = discovered_pairs

    method_summary_rows = _aggregate_method_rows(method_rows)
    pair_coverage_rows = _build_pair_coverage(pairs, method_rows, dynamic_rows)

    _write_csv(out_dir / "method_summary.csv", method_summary_rows)
    _write_csv(out_dir / "method_pair_compare.csv", method_pair_compare_rows)
    _write_csv(out_dir / "dynamic_preset_summary.csv", dynamic_preset_rows)
    _write_csv(out_dir / "dynamic_pair_compare.csv", dynamic_pair_compare_rows)
    _write_csv(out_dir / "pair_coverage.csv", pair_coverage_rows)

    manifest = {
        "suite_id": args.suite_id,
        "pairs": pairs,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "method_suite_id": args.method_suite_id,
        "dynamic_suite_id": args.dynamic_suite_id,
        "method_summary_rows": len(method_summary_rows),
        "dynamic_preset_rows": len(dynamic_preset_rows),
    }
    (out_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dynamic_visual_summary = dynamic_suite_dir / "visual_summary.md" if dynamic_suite_dir else None
    overview_md = _build_markdown(
        suite_id=args.suite_id,
        pairs=pairs,
        fps=args.fps,
        max_frames=args.max_frames,
        method_suite_id=args.method_suite_id,
        dynamic_suite_id=args.dynamic_suite_id,
        method_summary_rows=method_summary_rows,
        dynamic_preset_rows=dynamic_preset_rows,
        dynamic_visual_summary=dynamic_visual_summary,
        out_dir=out_dir,
    )
    (out_dir / "dataset_summary.md").write_text(overview_md, encoding="utf-8")
    print(f"dataset_summary={_rel(repo_root, out_dir / 'dataset_summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
