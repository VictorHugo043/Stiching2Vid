#!/usr/bin/env python3
"""Build a unified Phase 3 summary across multiple dataset-specific suites."""

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
    "mean_jitter_raw",
    "mean_jitter_sm",
    "p95_jitter_raw",
    "p95_jitter_sm",
    "warnings_count",
    "errors_count",
]

DYNAMIC_NUMERIC_FIELDS: List[str] = [
    "processed_frames",
    "success_frames",
    "fallback_frames",
    "seam_recompute_count",
    "geometry_update_count",
    "mean_inliers",
    "mean_inlier_ratio",
    "mean_overlap_diff_after",
    "mean_seam_mask_change_ratio",
    "mean_stitched_delta",
    "mean_foreground_ratio",
    "mean_jitter_sm",
    "temporal_primary_value",
    "approx_fps",
    "seam_snapshot_count",
    "warnings_count",
    "errors_count",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple outputs/phase3/<suite_id> directories into one unified "
            "Phase 3 summary for final report usage."
        )
    )
    parser.add_argument(
        "--suite_id",
        required=True,
        help="Output suite id created under outputs/phase3/",
    )
    parser.add_argument(
        "--source_suites",
        nargs="+",
        required=True,
        help="Source Phase 3 suite ids under outputs/phase3/",
    )
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


def _rel(base: Path, target: Path | None) -> str:
    if target is None:
        return ""
    return os.path.relpath(str(target), start=str(base))


def _infer_dataset_name(pair_rows: Sequence[Dict[str, str]]) -> str:
    datasets = sorted({str(row.get("dataset")) for row in pair_rows if row.get("dataset")})
    if len(datasets) == 1:
        return datasets[0]
    if not datasets:
        return "unknown"
    return "+".join(datasets)


def _aggregate_methods(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
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
        }
        for field in METHOD_NUMERIC_FIELDS:
            summary[field] = _mean(_safe_float(row.get(field)) for row in method_rows)
        out.append(summary)
    return out


def _aggregate_dynamic(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("status") != "passed":
            continue
        grouped.setdefault(str(row["preset_id"]), []).append(row)

    out: List[Dict[str, object]] = []
    for preset_id, preset_rows in sorted(grouped.items()):
        base = preset_rows[0]
        summary: Dict[str, object] = {
            "preset_id": preset_id,
            "pair_count": len(preset_rows),
            "geometry_mode": base.get("geometry_mode"),
            "seam_policy": base.get("seam_policy"),
            "foreground_mode": base.get("foreground_mode"),
            "seam_smooth": base.get("seam_smooth"),
            "temporal_primary_metric": base.get("temporal_primary_metric"),
        }
        for field in DYNAMIC_NUMERIC_FIELDS:
            summary[field] = _mean(_safe_float(row.get(field)) for row in preset_rows)
        out.append(summary)
    return out


def _build_markdown(
    out_dir: Path,
    source_rows: Sequence[Dict[str, object]],
    overall_method_rows: Sequence[Dict[str, object]],
    overall_dynamic_rows: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Phase 3 Overall Summary: {out_dir.name}")
    lines.append("")
    lines.append("## Source Suites")
    lines.append("")
    for row in source_rows:
        lines.append(
            f"- `{row['dataset_name']}` "
            f"`suite={row['source_suite_id']}` "
            f"`pairs={row['pair_count']}` "
            f"`fps={row.get('fps')}` "
            f"`method_suite={row.get('method_suite_id')}` "
            f"`dynamic_suite={row.get('dynamic_suite_id')}`"
        )
    lines.append("")

    if overall_method_rows:
        lines.append("## Overall Method Summary")
        lines.append("")
        for row in overall_method_rows:
            lines.append(
                f"- `{row['method']}` "
                f"`pair_count={row['pair_count']}` "
                f"`inliers={row.get('mean_inliers')}` "
                f"`inlier_ratio={row.get('mean_inlier_ratio')}` "
                f"`fps={row.get('approx_fps')}`"
            )
        lines.append("")

    if overall_dynamic_rows:
        lines.append("## Overall Dynamic Seam Summary")
        lines.append("")
        for row in overall_dynamic_rows:
            lines.append(
                f"- `{row['preset_id']}` "
                f"`pair_count={row['pair_count']}` "
                f"`overlap_diff_after={row.get('mean_overlap_diff_after')}` "
                f"`stitched_delta={row.get('mean_stitched_delta')}` "
                f"`fps={row.get('approx_fps')}`"
            )
        lines.append("")

    lines.append("## Artefacts")
    lines.append("")
    lines.append(f"- Source suites: `{_rel(out_dir, out_dir / 'suite_manifest.csv')}`")
    lines.append(f"- Overall method summary: `{_rel(out_dir, out_dir / 'overall_method_summary.csv')}`")
    lines.append(f"- Overall dynamic summary: `{_rel(out_dir, out_dir / 'overall_dynamic_preset_summary.csv')}`")
    lines.append(f"- By-dataset method summary: `{_rel(out_dir, out_dir / 'overall_method_by_dataset.csv')}`")
    lines.append(f"- By-dataset dynamic summary: `{_rel(out_dir, out_dir / 'overall_dynamic_by_dataset.csv')}`")
    lines.append(f"- Pair coverage: `{_rel(out_dir, out_dir / 'overall_pair_coverage.csv')}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "phase3" / args.suite_id
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_rows: List[Dict[str, object]] = []
    all_method_rows: List[Dict[str, object]] = []
    all_dynamic_rows: List[Dict[str, object]] = []
    all_pair_coverage_rows: List[Dict[str, object]] = []
    method_by_dataset_rows: List[Dict[str, object]] = []
    dynamic_by_dataset_rows: List[Dict[str, object]] = []

    for source_suite_id in args.source_suites:
        suite_dir = repo_root / "outputs" / "phase3" / source_suite_id
        summary_manifest_path = suite_dir / "summary_manifest.json"
        if not summary_manifest_path.exists():
            raise SystemExit(f"missing summary manifest: {summary_manifest_path}")
        summary_manifest = json.loads(summary_manifest_path.read_text(encoding="utf-8"))
        pair_coverage_rows = _read_csv(suite_dir / "pair_coverage.csv")
        dataset_name = _infer_dataset_name(pair_coverage_rows)
        method_suite_id = summary_manifest.get("method_suite_id")
        dynamic_suite_id = summary_manifest.get("dynamic_suite_id")

        method_summary_rows = _read_csv(suite_dir / "method_summary.csv")
        dynamic_preset_rows = _read_csv(suite_dir / "dynamic_preset_summary.csv")
        for row in method_summary_rows:
            method_by_dataset_rows.append(
                {
                    "source_suite_id": source_suite_id,
                    "dataset_name": dataset_name,
                    **row,
                }
            )
        for row in dynamic_preset_rows:
            dynamic_by_dataset_rows.append(
                {
                    "source_suite_id": source_suite_id,
                    "dataset_name": dataset_name,
                    **row,
                }
            )
        for row in pair_coverage_rows:
            all_pair_coverage_rows.append(
                {
                    "source_suite_id": source_suite_id,
                    "dataset_name": dataset_name,
                    **row,
                }
            )

        if method_suite_id:
            method_raw_rows = _read_csv(repo_root / "outputs" / "video_compare" / method_suite_id / "summary.csv")
            for row in method_raw_rows:
                merged = dict(row)
                merged["source_suite_id"] = source_suite_id
                merged["dataset_name"] = dataset_name
                all_method_rows.append(merged)

        if dynamic_suite_id:
            dynamic_raw_rows = _read_csv(repo_root / "outputs" / "video_compare" / dynamic_suite_id / "summary.csv")
            for row in dynamic_raw_rows:
                merged = dict(row)
                merged["source_suite_id"] = source_suite_id
                merged["dataset_name"] = dataset_name
                all_dynamic_rows.append(merged)

        source_manifest_rows.append(
            {
                "source_suite_id": source_suite_id,
                "dataset_name": dataset_name,
                "pair_count": len(pair_coverage_rows),
                "fps": summary_manifest.get("fps"),
                "max_frames": summary_manifest.get("max_frames"),
                "method_suite_id": method_suite_id,
                "dynamic_suite_id": dynamic_suite_id,
            }
        )

    overall_method_rows = _aggregate_methods(all_method_rows)
    overall_dynamic_rows = _aggregate_dynamic(all_dynamic_rows)

    _write_csv(out_dir / "suite_manifest.csv", source_manifest_rows)
    _write_csv(out_dir / "overall_method_summary.csv", overall_method_rows)
    _write_csv(out_dir / "overall_dynamic_preset_summary.csv", overall_dynamic_rows)
    _write_csv(out_dir / "overall_method_by_dataset.csv", method_by_dataset_rows)
    _write_csv(out_dir / "overall_dynamic_by_dataset.csv", dynamic_by_dataset_rows)
    _write_csv(out_dir / "overall_pair_coverage.csv", all_pair_coverage_rows)

    (out_dir / "overall_summary_manifest.json").write_text(
        json.dumps(
            {
                "suite_id": args.suite_id,
                "source_suites": list(args.source_suites),
                "overall_method_rows": len(overall_method_rows),
                "overall_dynamic_rows": len(overall_dynamic_rows),
                "overall_pair_rows": len(all_pair_coverage_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "phase3_overall_summary.md").write_text(
        _build_markdown(out_dir, source_manifest_rows, overall_method_rows, overall_dynamic_rows),
        encoding="utf-8",
    )
    print(f"phase3_overall_summary={_rel(repo_root, out_dir / 'phase3_overall_summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
