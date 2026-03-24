#!/usr/bin/env python3
"""Launch the desktop GUI thin wrapper for the stitching pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the desktop stitching GUI thin wrapper.")
    parser.add_argument(
        "--manifest",
        default="data/manifests/pairs.yaml",
        help="Path to pairs manifest",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    from stitching.gui_thin_wrapper import launch_gui  # noqa: E402

    manifest_path = (repo_root / args.manifest).resolve()
    launch_gui(repo_root=repo_root, manifest_path=manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
