#!/usr/bin/env python3
"""Compare two full-length Method B overall suites and export compact tables/figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a two-variant Method B comparison by combining an existing baseline overall "
            "suite with a newly rerun overall suite."
        )
    )
    parser.add_argument("--suite_id", required=True, help="Output suite id under outputs/phase3/")
    parser.add_argument("--baseline_suite_id", required=True, help="Existing baseline overall suite id under outputs/phase3/")
    parser.add_argument("--variant_suite_id", required=True, help="New variant overall suite id under outputs/phase3/")
    parser.add_argument(
        "--baseline_label",
        default="method_b_accuracy_v1_mps",
        help="Output label for the baseline Method B row",
    )
    parser.add_argument(
        "--variant_label",
        default="method_b_native_res_mps",
        help="Output label for the variant Method B row",
    )
    return parser


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: object) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_matplotlib(mpl_config_dir: Path):
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _find_method_row(rows: Sequence[Dict[str, str]], method: str = "method_b") -> Dict[str, str]:
    for row in rows:
        if str(row.get("method")) == method:
            return row
    raise SystemExit(f"missing method row '{method}'")


def _index_dataset_rows(rows: Sequence[Dict[str, str]], method: str = "method_b") -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if str(row.get("method")) != method:
            continue
        out[str(row.get("dataset_name"))] = row
    return out


def _decorate_row(row: Dict[str, str], method_name: str, source_suite_id: str, variant: str) -> Dict[str, object]:
    out = dict(row)
    out["method"] = method_name
    out["source_suite_id"] = source_suite_id
    out["variant"] = variant
    return out


def _build_overall_rows(
    baseline_overall: Sequence[Dict[str, str]],
    variant_overall: Sequence[Dict[str, str]],
    baseline_suite_id: str,
    variant_suite_id: str,
    baseline_label: str,
    variant_label: str,
) -> List[Dict[str, object]]:
    return [
        _decorate_row(_find_method_row(baseline_overall), baseline_label, baseline_suite_id, "baseline"),
        _decorate_row(_find_method_row(variant_overall), variant_label, variant_suite_id, "variant"),
    ]


def _build_by_dataset_rows(
    baseline_by_dataset: Sequence[Dict[str, str]],
    variant_by_dataset: Sequence[Dict[str, str]],
    baseline_suite_id: str,
    variant_suite_id: str,
    baseline_label: str,
    variant_label: str,
) -> List[Dict[str, object]]:
    baseline_rows = _index_dataset_rows(baseline_by_dataset)
    variant_rows = _index_dataset_rows(variant_by_dataset)
    datasets = sorted(set(baseline_rows) | set(variant_rows))
    rows: List[Dict[str, object]] = []
    for dataset_name in datasets:
        if dataset_name in baseline_rows:
            rows.append(_decorate_row(baseline_rows[dataset_name], baseline_label, baseline_suite_id, "baseline"))
        if dataset_name in variant_rows:
            rows.append(_decorate_row(variant_rows[dataset_name], variant_label, variant_suite_id, "variant"))
    return rows


def _build_delta_rows(
    baseline_overall: Sequence[Dict[str, str]],
    variant_overall: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    baseline_row = _find_method_row(baseline_overall)
    variant_row = _find_method_row(variant_overall)
    fields = [
        "mean_inliers",
        "mean_inlier_ratio",
        "approx_fps",
        "mean_reprojection_error",
        "mean_inlier_spatial_coverage",
        "mean_overlap_diff_after",
        "mean_seam_band_illuminance_diff",
        "mean_seam_band_gradient_disagreement",
        "mean_seam_band_flicker",
        "mean_stitched_delta",
        "init_ms_mean",
        "per_frame_ms_mean",
    ]
    rows: List[Dict[str, object]] = []
    for field in fields:
        baseline_value = _safe_float(baseline_row.get(field))
        variant_value = _safe_float(variant_row.get(field))
        delta = None
        if baseline_value is not None and variant_value is not None:
            delta = variant_value - baseline_value
        rows.append(
            {
                "metric": field,
                "baseline_value": baseline_value,
                "variant_value": variant_value,
                "delta_variant_minus_baseline": delta,
            }
        )
    return rows


def _bar_colors(methods: Sequence[str]) -> List[str]:
    palette = {
        "method_b_accuracy_v1_mps": "#4c78a8",
        "method_b_native_res_mps": "#e45756",
    }
    return [palette.get(method, "#4c78a8") for method in methods]


def _series(rows: Sequence[Dict[str, object]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        parsed = _safe_float(row.get(field))
        values.append(parsed if parsed is not None else 0.0)
    return values


def _save_core_figure(plt, rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    methods = [str(row.get("method")) for row in rows]
    colors = _bar_colors(methods)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("mean_inliers", "Mean Inliers"),
        ("mean_inlier_ratio", "Mean Inlier Ratio"),
        ("approx_fps", "Approx FPS"),
        ("mean_reprojection_error", "Mean Reprojection Error"),
    ]
    for ax, (field, title) in zip(axes.ravel(), specs):
        ax.bar(methods, _series(rows, field), color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Method B Variant Compare")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_quality_figure(plt, rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    methods = [str(row.get("method")) for row in rows]
    colors = _bar_colors(methods)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("mean_inlier_spatial_coverage", "Inlier Spatial Coverage"),
        ("mean_overlap_diff_after", "Overlap Diff After"),
        ("mean_seam_band_illuminance_diff", "Seam-band Illuminance Diff"),
        ("mean_seam_band_flicker", "Seam-band Flicker"),
    ]
    for ax, (field, title) in zip(axes.ravel(), specs):
        ax.bar(methods, _series(rows, field), color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Method B Variant Quality Compare")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_markdown(
    out_dir: Path,
    baseline_suite_id: str,
    variant_suite_id: str,
    baseline_label: str,
    variant_label: str,
    overall_rows: Sequence[Dict[str, object]],
) -> str:
    baseline_row = next(row for row in overall_rows if row["method"] == baseline_label)
    variant_row = next(row for row in overall_rows if row["method"] == variant_label)
    lines: List[str] = []
    lines.append("# Method B Variant Compare")
    lines.append("")
    lines.append("## Source Suites")
    lines.append("")
    lines.append(f"- Baseline suite: `{baseline_suite_id}`")
    lines.append(f"- Variant suite: `{variant_suite_id}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    for row in overall_rows:
        lines.append(
            f"- `{row['method']}` "
            f"`inliers={row.get('mean_inliers')}` "
            f"`inlier_ratio={row.get('mean_inlier_ratio')}` "
            f"`fps={row.get('approx_fps')}` "
            f"`reproj={row.get('mean_reprojection_error')}`"
        )
    lines.append("")
    lines.append("## Delta")
    lines.append("")
    lines.append(f"- `mean_inliers`: `{baseline_row.get('mean_inliers')} -> {variant_row.get('mean_inliers')}`")
    lines.append(
        f"- `mean_inlier_ratio`: `{baseline_row.get('mean_inlier_ratio')} -> {variant_row.get('mean_inlier_ratio')}`"
    )
    lines.append(f"- `approx_fps`: `{baseline_row.get('approx_fps')} -> {variant_row.get('approx_fps')}`")
    lines.append(
        f"- `mean_reprojection_error`: `{baseline_row.get('mean_reprojection_error')} -> {variant_row.get('mean_reprojection_error')}`"
    )
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append(f"- Overall compare: `{os.path.relpath(out_dir / 'overall_method_compare.csv', out_dir)}`")
    lines.append(f"- By-dataset compare: `{os.path.relpath(out_dir / 'by_dataset_method_compare.csv', out_dir)}`")
    lines.append(f"- Variant delta: `{os.path.relpath(out_dir / 'method_b_variant_delta.csv', out_dir)}`")
    lines.append(f"- Figure core: `{os.path.relpath(out_dir / 'figures' / 'variant_compare_core_metrics.png', out_dir)}`")
    lines.append(f"- Figure quality: `{os.path.relpath(out_dir / 'figures' / 'variant_compare_quality_metrics.png', out_dir)}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs" / "phase3" / args.suite_id
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = repo_root / "outputs" / "phase3" / args.baseline_suite_id
    variant_dir = repo_root / "outputs" / "phase3" / args.variant_suite_id
    baseline_overall = _read_csv(baseline_dir / "overall_method_summary.csv")
    baseline_by_dataset = _read_csv(baseline_dir / "overall_method_by_dataset.csv")
    variant_overall = _read_csv(variant_dir / "overall_method_summary.csv")
    variant_by_dataset = _read_csv(variant_dir / "overall_method_by_dataset.csv")

    overall_rows = _build_overall_rows(
        baseline_overall,
        variant_overall,
        baseline_suite_id=args.baseline_suite_id,
        variant_suite_id=args.variant_suite_id,
        baseline_label=args.baseline_label,
        variant_label=args.variant_label,
    )
    by_dataset_rows = _build_by_dataset_rows(
        baseline_by_dataset,
        variant_by_dataset,
        baseline_suite_id=args.baseline_suite_id,
        variant_suite_id=args.variant_suite_id,
        baseline_label=args.baseline_label,
        variant_label=args.variant_label,
    )
    delta_rows = _build_delta_rows(baseline_overall, variant_overall)

    _write_csv(out_dir / "overall_method_compare.csv", overall_rows)
    _write_csv(out_dir / "by_dataset_method_compare.csv", by_dataset_rows)
    _write_csv(out_dir / "method_b_variant_delta.csv", delta_rows)
    (out_dir / "summary.md").write_text(
        _build_markdown(
            out_dir,
            baseline_suite_id=args.baseline_suite_id,
            variant_suite_id=args.variant_suite_id,
            baseline_label=args.baseline_label,
            variant_label=args.variant_label,
            overall_rows=overall_rows,
        ),
        encoding="utf-8",
    )
    (out_dir / "compare_manifest.json").write_text(
        json.dumps(
            {
                "suite_id": args.suite_id,
                "baseline_suite_id": args.baseline_suite_id,
                "variant_suite_id": args.variant_suite_id,
                "baseline_label": args.baseline_label,
                "variant_label": args.variant_label,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plt = _require_matplotlib(out_dir / "figures" / ".mplconfig")
    _save_core_figure(plt, overall_rows, out_dir / "figures" / "variant_compare_core_metrics.png")
    _save_quality_figure(plt, overall_rows, out_dir / "figures" / "variant_compare_quality_metrics.png")
    print(f"method_b_variant_compare_summary={out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
