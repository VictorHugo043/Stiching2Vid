#!/usr/bin/env python3
"""Build final-report figures from the formal overall method summary tables."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export richer method-compare figures for the final report."
    )
    parser.add_argument(
        "--suite_id",
        required=True,
        help="Suite id under outputs/phase3/ that contains overall_method_summary.csv",
    )
    return parser


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _series(rows: Sequence[Dict[str, str]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        parsed = _safe_float(row.get(field))
        values.append(parsed if parsed is not None else 0.0)
    return values


def _methods(rows: Sequence[Dict[str, str]]) -> List[str]:
    return [str(row.get("method", "")) for row in rows]


def _datasets(rows: Sequence[Dict[str, str]]) -> List[str]:
    return sorted({str(row.get("dataset_name")) for row in rows if row.get("dataset_name")})


def _grouped_by_dataset(
    rows: Sequence[Dict[str, str]],
    methods: Sequence[str],
    datasets: Sequence[str],
    field: str,
) -> List[List[float]]:
    index: Dict[tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        dataset = str(row.get("dataset_name"))
        method = str(row.get("method"))
        index[(dataset, method)] = row
    out: List[List[float]] = []
    for method in methods:
        method_values: List[float] = []
        for dataset in datasets:
            row = index.get((dataset, method), {})
            method_values.append(_safe_float(row.get(field)) or 0.0)
        out.append(method_values)
    return out


def _bar_colors(methods: Sequence[str]) -> List[str]:
    palette = {
        "method_a_orb": "#1b9e77",
        "method_a_sift": "#d95f02",
        "method_b": "#7570b3",
    }
    return [palette.get(method, "#4c78a8") for method in methods]


def _write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_core_metrics(
    plt,
    overall_rows: Sequence[Dict[str, str]],
    out_path: Path,
) -> None:
    methods = _methods(overall_rows)
    colors = _bar_colors(methods)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        ("mean_inliers", "Mean Inliers"),
        ("mean_inlier_ratio", "Mean Inlier Ratio"),
        ("approx_fps", "Approx FPS"),
        ("mean_reprojection_error", "Mean Reprojection Error"),
    ]
    for ax, (field, title) in zip(axes.ravel(), specs):
        ax.bar(methods, _series(overall_rows, field), color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Method Compare: Core Metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_runtime_metrics(
    plt,
    overall_rows: Sequence[Dict[str, str]],
    out_path: Path,
) -> None:
    methods = _methods(overall_rows)
    colors = _bar_colors(methods)
    feature_totals = [
        (_safe_float(row.get("avg_feature_runtime_ms_left")) or 0.0)
        + (_safe_float(row.get("avg_feature_runtime_ms_right")) or 0.0)
        for row in overall_rows
    ]
    matching = _series(overall_rows, "avg_matching_runtime_ms")
    geometry = _series(overall_rows, "avg_geometry_runtime_ms")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(methods, _series(overall_rows, "init_ms_mean"), color=colors, label="init")
    axes[0].bar(
        methods,
        _series(overall_rows, "per_frame_ms_mean"),
        color=colors,
        alpha=0.55,
        label="per_frame",
    )
    axes[0].set_title("Init / Per-frame Runtime")
    axes[0].set_ylabel("ms")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(methods, feature_totals, label="feature total", color="#4c78a8")
    axes[1].bar(methods, matching, bottom=feature_totals, label="matching", color="#f58518")
    bottom = [feature_totals[idx] + matching[idx] for idx in range(len(methods))]
    axes[1].bar(methods, geometry, bottom=bottom, label="geometry", color="#54a24b")
    axes[1].set_title("Geometry-update Event Runtime Breakdown")
    axes[1].set_ylabel("ms")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("Method Compare: Runtime")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_quality_metrics(
    plt,
    overall_rows: Sequence[Dict[str, str]],
    out_path: Path,
) -> None:
    methods = _methods(overall_rows)
    colors = _bar_colors(methods)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        ("mean_inlier_spatial_coverage", "Inlier Spatial Coverage"),
        ("mean_overlap_diff_after", "Overlap Diff After"),
        ("mean_seam_band_illuminance_diff", "Seam-band Illuminance Diff"),
        ("mean_seam_band_gradient_disagreement", "Seam-band Gradient Disagreement"),
    ]
    for ax, (field, title) in zip(axes.ravel(), specs):
        ax.bar(methods, _series(overall_rows, field), color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Method Compare: Geometry / Seam Quality")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_temporal_metrics(
    plt,
    overall_rows: Sequence[Dict[str, str]],
    out_path: Path,
) -> None:
    methods = _methods(overall_rows)
    colors = _bar_colors(methods)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    specs = [
        ("mean_seam_band_flicker", "Seam-band Flicker"),
        ("mean_stitched_delta", "Mean Stitched Delta"),
    ]
    for ax, (field, title) in zip(axes.ravel(), specs):
        ax.bar(methods, _series(overall_rows, field), color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Method Compare: Temporal Artefacts")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_dataset_breakdown(
    plt,
    by_dataset_rows: Sequence[Dict[str, str]],
    out_path: Path,
) -> None:
    methods = sorted({str(row.get("method")) for row in by_dataset_rows if row.get("method")})
    datasets = _datasets(by_dataset_rows)
    colors = _bar_colors(methods)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("mean_inliers", "Mean Inliers"),
        ("approx_fps", "Approx FPS"),
        ("mean_reprojection_error", "Mean Reprojection Error"),
        ("mean_seam_band_flicker", "Seam-band Flicker"),
    ]
    x = list(range(len(datasets)))
    width = 0.22 if methods else 0.25

    for ax, (field, title) in zip(axes.ravel(), specs):
        grouped = _grouped_by_dataset(by_dataset_rows, methods, datasets, field)
        for idx, method in enumerate(methods):
            offsets = [value + (idx - (len(methods) - 1) / 2.0) * width for value in x]
            ax.bar(offsets, grouped[idx], width=width, label=method, color=colors[idx])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Method Compare: By-dataset Breakdown")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_markdown(path: Path, figure_rows: Sequence[Dict[str, object]]) -> None:
    lines = ["# Report Figures", ""]
    lines.append("## Figure List")
    lines.append("")
    for row in figure_rows:
        lines.append(
            f"- `{row['figure_id']}` `{row['path']}` {row['description']}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    suite_dir = repo_root / "outputs" / "phase3" / args.suite_id
    figures_dir = suite_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = _read_csv(suite_dir / "overall_method_summary.csv")
    by_dataset_rows = _read_csv(suite_dir / "overall_method_by_dataset.csv")
    if not overall_rows or not by_dataset_rows:
        raise SystemExit(
            f"Missing required overall CSVs in {suite_dir}; scripts/internal/summarize_method_compare_overall.py must run first."
        )

    plt = _require_matplotlib(suite_dir / "figures" / ".mplconfig")

    figure_rows: List[Dict[str, object]] = []
    figures = [
        (
            "method_core_metrics",
            "method_core_metrics.png",
            "Overall Method A / Method B core metric bars.",
            _save_core_metrics,
        ),
        (
            "method_runtime_metrics",
            "method_runtime_metrics.png",
            "Overall runtime cost and stage breakdown.",
            _save_runtime_metrics,
        ),
        (
            "method_quality_metrics",
            "method_quality_metrics.png",
            "Overall geometry and seam-quality bars.",
            _save_quality_metrics,
        ),
        (
            "method_temporal_metrics",
            "method_temporal_metrics.png",
            "Overall temporal artefact bars for fixed-geometry compare.",
            _save_temporal_metrics,
        ),
        (
            "method_by_dataset",
            "method_by_dataset.png",
            "By-dataset grouped comparison for core metrics.",
            _save_dataset_breakdown,
        ),
    ]

    for figure_id, filename, description, builder in figures:
        out_path = figures_dir / filename
        if figure_id == "method_by_dataset":
            builder(plt, by_dataset_rows, out_path)
        else:
            builder(plt, overall_rows, out_path)
        figure_rows.append(
            {
                "figure_id": figure_id,
                "path": str(out_path.relative_to(suite_dir)),
                "description": description,
                "source_csvs": "overall_method_summary.csv;overall_method_by_dataset.csv",
            }
        )

    _write_manifest(figures_dir / "figure_manifest.csv", figure_rows)
    _write_markdown(figures_dir / "figures.md", figure_rows)
    print(f"figure_manifest={figures_dir / 'figure_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
