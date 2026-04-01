#!/usr/bin/env python3
"""Compare preserved CPU Method B results against a new device-variant method suite."""

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
            "Build CPU-vs-device comparison tables and figures by combining a preserved "
            "overall method suite with a newly rerun overall method suite."
        )
    )
    parser.add_argument("--suite_id", required=True, help="Output suite id under outputs/phase3/")
    parser.add_argument("--cpu_suite_id", required=True, help="Existing CPU overall suite id under outputs/phase3/")
    parser.add_argument("--device_suite_id", required=True, help="New overall suite id under outputs/phase3/")
    parser.add_argument(
        "--device_label",
        default="mps",
        help="Label used for the rerun Method B variant, e.g. mps",
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


def _find_method_row(rows: Sequence[Dict[str, str]], method: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("method")) == method:
            return row
    raise SystemExit(f"missing method row '{method}'")


def _find_method_row_optional(rows: Sequence[Dict[str, str]], method: str) -> Dict[str, str] | None:
    for row in rows:
        if str(row.get("method")) == method:
            return row
    return None


def _index_dataset_rows(rows: Sequence[Dict[str, str]], method: str) -> Dict[str, Dict[str, str]]:
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
    cpu_overall: Sequence[Dict[str, str]],
    device_overall: Sequence[Dict[str, str]],
    cpu_suite_id: str,
    device_suite_id: str,
    device_label: str,
) -> List[Dict[str, object]]:
    orb_row_device = _find_method_row_optional(device_overall, "method_a_orb")
    orb_row = orb_row_device or _find_method_row(cpu_overall, "method_a_orb")
    orb_source_suite = device_suite_id if orb_row_device is not None else cpu_suite_id
    orb_variant = "current_ref" if orb_row_device is not None else "cpu_ref"
    sift_row_device = _find_method_row_optional(device_overall, "method_a_sift")
    sift_row = sift_row_device or _find_method_row(cpu_overall, "method_a_sift")
    sift_source_suite = device_suite_id if sift_row_device is not None else cpu_suite_id
    sift_variant = "current_ref" if sift_row_device is not None else "cpu_ref"
    return [
        _decorate_row(orb_row, "method_a_orb", orb_source_suite, orb_variant),
        _decorate_row(sift_row, "method_a_sift", sift_source_suite, sift_variant),
        _decorate_row(_find_method_row(cpu_overall, "method_b"), "method_b_accuracy_v1_cpu", cpu_suite_id, "cpu_ref"),
        _decorate_row(_find_method_row(device_overall, "method_b"), f"method_b_accuracy_v1_{device_label}", device_suite_id, device_label),
    ]


def _build_by_dataset_rows(
    cpu_by_dataset: Sequence[Dict[str, str]],
    device_by_dataset: Sequence[Dict[str, str]],
    cpu_suite_id: str,
    device_suite_id: str,
    device_label: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    device_orb = _index_dataset_rows(device_by_dataset, "method_a_orb")
    device_sift = _index_dataset_rows(device_by_dataset, "method_a_sift")
    cpu_orb = _index_dataset_rows(cpu_by_dataset, "method_a_orb")
    cpu_sift = _index_dataset_rows(cpu_by_dataset, "method_a_sift")
    cpu_method_b = _index_dataset_rows(cpu_by_dataset, "method_b")
    device_method_b = _index_dataset_rows(device_by_dataset, "method_b")
    datasets = sorted(
        set(device_orb)
        | set(device_sift)
        | set(cpu_orb)
        | set(cpu_sift)
        | set(cpu_method_b)
        | set(device_method_b)
    )
    for dataset_name in datasets:
        if dataset_name in device_orb:
            rows.append(_decorate_row(device_orb[dataset_name], "method_a_orb", device_suite_id, "current_ref"))
        elif dataset_name in cpu_orb:
            rows.append(_decorate_row(cpu_orb[dataset_name], "method_a_orb", cpu_suite_id, "cpu_ref"))
        if dataset_name in device_sift:
            rows.append(_decorate_row(device_sift[dataset_name], "method_a_sift", device_suite_id, "current_ref"))
        elif dataset_name in cpu_sift:
            rows.append(_decorate_row(cpu_sift[dataset_name], "method_a_sift", cpu_suite_id, "cpu_ref"))
        if dataset_name in cpu_method_b:
            rows.append(_decorate_row(cpu_method_b[dataset_name], "method_b_accuracy_v1_cpu", cpu_suite_id, "cpu_ref"))
        if dataset_name in device_method_b:
            rows.append(_decorate_row(device_method_b[dataset_name], f"method_b_accuracy_v1_{device_label}", device_suite_id, device_label))
    return rows


def _build_delta_rows(
    cpu_overall: Sequence[Dict[str, str]],
    device_overall: Sequence[Dict[str, str]],
    device_label: str,
) -> List[Dict[str, object]]:
    cpu_row = _find_method_row(cpu_overall, "method_b")
    device_row = _find_method_row(device_overall, "method_b")
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
        "steady_frame_ms_mean",
        "steady_approx_fps",
    ]
    rows: List[Dict[str, object]] = []
    for field in fields:
        cpu_value = _safe_float(cpu_row.get(field))
        device_value = _safe_float(device_row.get(field))
        delta = None
        if cpu_value is not None and device_value is not None:
            delta = device_value - cpu_value
        rows.append(
            {
                "metric": field,
                "cpu_value": cpu_value,
                f"{device_label}_value": device_value,
                f"delta_{device_label}_minus_cpu": delta,
            }
        )
    return rows


def _bar_colors(methods: Sequence[str]) -> List[str]:
    palette = {
        "method_a_orb": "#1b9e77",
        "method_a_sift": "#d95f02",
        "method_b_accuracy_v1_cpu": "#4c78a8",
        "method_b_accuracy_v1_mps": "#7b61ff",
        "method_b_accuracy_v1_cuda": "#7b61ff",
    }
    return [palette.get(method, "#4c78a8") for method in methods]


def _series(rows: Sequence[Dict[str, object]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        parsed = _safe_float(row.get(field))
        values.append(parsed if parsed is not None else 0.0)
    return values


def _save_overall_figure(plt, rows: Sequence[Dict[str, object]], out_path: Path) -> None:
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
    fig.suptitle("Method Compare: CPU vs Device Variant")
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
    fig.suptitle("Method Compare: CPU vs Device Quality")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_markdown(
    out_dir: Path,
    cpu_suite_id: str,
    device_suite_id: str,
    device_label: str,
    overall_rows: Sequence[Dict[str, object]],
    delta_rows: Sequence[Dict[str, object]],
) -> str:
    cpu_method_b = next(row for row in overall_rows if row["method"] == "method_b_accuracy_v1_cpu")
    device_method_b = next(row for row in overall_rows if row["method"] == f"method_b_accuracy_v1_{device_label}")
    lines: List[str] = []
    lines.append(f"# CPU vs {device_label.upper()} Method B Compare")
    lines.append("")
    lines.append("## Source Suites")
    lines.append("")
    lines.append(f"- CPU reference suite: `{cpu_suite_id}`")
    lines.append(f"- Device rerun suite: `{device_suite_id}`")
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
    lines.append("## Method B Delta")
    lines.append("")
    lines.append(
        f"- `mean_inliers`: `{cpu_method_b.get('mean_inliers')} -> {device_method_b.get('mean_inliers')}`"
    )
    lines.append(
        f"- `mean_inlier_ratio`: `{cpu_method_b.get('mean_inlier_ratio')} -> {device_method_b.get('mean_inlier_ratio')}`"
    )
    lines.append(
        f"- `approx_fps`: `{cpu_method_b.get('approx_fps')} -> {device_method_b.get('approx_fps')}`"
    )
    lines.append(
        f"- `mean_reprojection_error`: `{cpu_method_b.get('mean_reprojection_error')} -> {device_method_b.get('mean_reprojection_error')}`"
    )
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append(f"- Overall compare: `{os.path.relpath(out_dir / 'overall_method_compare.csv', out_dir)}`")
    lines.append(f"- By-dataset compare: `{os.path.relpath(out_dir / 'by_dataset_method_compare.csv', out_dir)}`")
    lines.append(f"- Device delta: `{os.path.relpath(out_dir / 'method_b_device_delta.csv', out_dir)}`")
    lines.append(f"- Figure core: `{os.path.relpath(out_dir / 'figures' / 'device_compare_core_metrics.png', out_dir)}`")
    lines.append(f"- Figure quality: `{os.path.relpath(out_dir / 'figures' / 'device_compare_quality_metrics.png', out_dir)}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs" / "phase3" / args.suite_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_dir = repo_root / "outputs" / "phase3" / args.cpu_suite_id
    device_dir = repo_root / "outputs" / "phase3" / args.device_suite_id
    cpu_overall = _read_csv(cpu_dir / "overall_method_summary.csv")
    cpu_by_dataset = _read_csv(cpu_dir / "overall_method_by_dataset.csv")
    device_overall = _read_csv(device_dir / "overall_method_summary.csv")
    device_by_dataset = _read_csv(device_dir / "overall_method_by_dataset.csv")

    overall_rows = _build_overall_rows(
        cpu_overall,
        device_overall,
        cpu_suite_id=args.cpu_suite_id,
        device_suite_id=args.device_suite_id,
        device_label=args.device_label,
    )
    by_dataset_rows = _build_by_dataset_rows(
        cpu_by_dataset,
        device_by_dataset,
        cpu_suite_id=args.cpu_suite_id,
        device_suite_id=args.device_suite_id,
        device_label=args.device_label,
    )
    delta_rows = _build_delta_rows(cpu_overall, device_overall, args.device_label)

    _write_csv(out_dir / "overall_method_compare.csv", overall_rows)
    _write_csv(out_dir / "by_dataset_method_compare.csv", by_dataset_rows)
    _write_csv(out_dir / "method_b_device_delta.csv", delta_rows)
    (out_dir / "summary.md").write_text(
        _build_markdown(out_dir, args.cpu_suite_id, args.device_suite_id, args.device_label, overall_rows, delta_rows),
        encoding="utf-8",
    )
    (out_dir / "compare_manifest.json").write_text(
        json.dumps(
            {
                "suite_id": args.suite_id,
                "cpu_suite_id": args.cpu_suite_id,
                "device_suite_id": args.device_suite_id,
                "device_label": args.device_label,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plt = _require_matplotlib(out_dir / "figures" / ".mplconfig")
    _save_overall_figure(plt, overall_rows, out_dir / "figures" / "device_compare_core_metrics.png")
    _save_quality_figure(plt, overall_rows, out_dir / "figures" / "device_compare_quality_metrics.png")
    print(f"device_compare_summary={out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
