"""Unified data I/O layer for stitching pipelines.

统一 I/O 入口，兼容两类输入：
- video files (avi/mp4/mpeg) via OpenCV VideoCapture
- frame sequences (jpeg/png) via sorted file lists or index CSV
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Logging 默认 INFO (caller 可覆盖). WARNING 用于可恢复问题，
# ERROR 通过异常抛出，保持行为显式。
logger = logging.getLogger(__name__)


@dataclass
class PairConfig:
    """Container for a single pair entry from pairs.yaml."""
    id: str
    dataset: str
    input_type: str
    left: str
    right: str
    calib: Optional[str]
    optional: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    root_dir: Path = field(default_factory=Path, repr=False)


# --- Frame source abstraction ---

class FrameSource:
    """Abstract frame source interface.

    Contract / 契约:
    - read(i) returns the i-th frame (BGR ndarray) or raises on failure.
    - read_next() returns the next frame, or None at end-of-stream.
    - length() returns total frames if known, else None.
    - fps()/resolution() return metadata when available.
    - close() releases resources.
    """

    def read(self, i: int) -> Any:
        """Random access read; should raise if index is invalid."""
        raise NotImplementedError

    def read_next(self) -> Any:
        """Sequential read; returns None when stream ends."""
        raise NotImplementedError

    def length(self) -> Optional[int]:
        """Total number of frames if known, otherwise None."""
        raise NotImplementedError

    def fps(self) -> Optional[float]:
        """Frames per second if known, otherwise None."""
        raise NotImplementedError

    def resolution(self) -> Optional[Tuple[int, int]]:
        """(width, height) if known, otherwise None."""
        raise NotImplementedError

    def close(self) -> None:
        """Release underlying resources."""
        raise NotImplementedError


# --- Frame source implementations ---

class FramesSource(FrameSource):
    """Frame source backed by an explicit list of image file paths.

    Ordering 由传入的列表决定，因此上游排序必须稳定且按时间顺序。
    """

    def __init__(self, frame_paths: List[Path], fps: Optional[float] = None) -> None:
        self._require_cv2()
        if not frame_paths:
            raise FileNotFoundError("No frames found for frames source.")
        self._paths = frame_paths
        self._idx = 0
        self._fps = fps
        self._resolution = None
        self._length_limit = None

        import cv2  # type: ignore

        # Probe first frame 以推断分辨率并尽早失败 (fail fast).
        first = cv2.imread(str(self._paths[0]))
        if first is None:
            raise ValueError(f"Failed to read first frame: {self._paths[0]}")
        self._resolution = (int(first.shape[1]), int(first.shape[0]))

    @staticmethod
    def _require_cv2() -> None:
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenCV (cv2) is required to read frames. Install it first."
            ) from exc

    def set_length_limit(self, limit: int) -> None:
        """Clamp the visible length for alignment across paired sources."""
        self._length_limit = max(0, limit)

    def read(self, i: int) -> Any:
        import cv2  # type: ignore

        limit = self.length()
        if limit is not None and (i < 0 or i >= limit):
            raise IndexError(f"Frame index {i} out of range (limit={limit}).")
        # Direct file read; 顺序来自预排序列表。
        frame = cv2.imread(str(self._paths[i]))
        if frame is None:
            raise ValueError(f"Failed to read frame: {self._paths[i]}")
        return frame

    def read_next(self) -> Any:
        limit = self.length()
        if limit is not None and self._idx >= limit:
            return None
        frame = self.read(self._idx)
        self._idx += 1
        return frame

    def length(self) -> Optional[int]:
        base_len = len(self._paths)
        if self._length_limit is not None:
            return min(base_len, self._length_limit)
        return base_len

    def fps(self) -> Optional[float]:
        return self._fps

    def resolution(self) -> Optional[Tuple[int, int]]:
        return self._resolution

    def close(self) -> None:
        return None


class VideoSource(FrameSource):
    """Frame source backed by OpenCV VideoCapture.

    Note: CAP_PROP_POS_FRAMES seek 受 codec/GOP 影响，可能不精确。
    We still expose read(i) for convenience; treat it as best-effort.
    """

    def __init__(self, video_path: Path) -> None:
        self._require_cv2()
        import cv2  # type: ignore

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps is not None and self._fps <= 0:
            self._fps = None
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._frame_count <= 0:
            self._frame_count = None
        width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width and height:
            self._resolution = (int(width), int(height))
        else:
            self._resolution = None

        # INFO: 提示 seek 局限但不阻塞流程。
        logger.info(
            "VideoSource uses CAP_PROP_POS_FRAMES for seeking; random access may be imprecise."
        )

    @staticmethod
    def _require_cv2() -> None:
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenCV (cv2) is required to read video. Install it first."
            ) from exc

    def read(self, i: int) -> Any:
        import cv2  # type: ignore

        if i < 0:
            raise IndexError("Frame index must be non-negative.")
        # CAP_PROP_POS_FRAMES 可能落在最近可解码帧 (nearest decodable).
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise ValueError(f"Failed to read frame at index {i}.")
        return frame

    def read_next(self) -> Any:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def length(self) -> Optional[int]:
        return self._frame_count

    def fps(self) -> Optional[float]:
        return self._fps

    def resolution(self) -> Optional[Tuple[int, int]]:
        return self._resolution

    def close(self) -> None:
        # Release resources 以避免文件句柄泄漏。
        self._cap.release()


# --- Manifest I/O ---

def load_pairs(manifest_path: str | os.PathLike) -> List[PairConfig]:
    """Load pairs.yaml into a list of PairConfig.

    Args:
        manifest_path: Path to pairs.yaml (relative or absolute).

    Returns:
        List of PairConfig objects with repo-relative paths.

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        ValueError: If the manifest format is invalid.

    Example:
        pairs = load_pairs("data/manifests/pairs.yaml")
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = _load_yaml(manifest_path)
    if "pairs" not in data or not isinstance(data["pairs"], list):
        raise ValueError("Manifest format error: missing top-level 'pairs' list.")

    # pairs.yaml 位于 data/manifests/，parents[2] 即 repo root.
    root_dir = manifest_path.resolve().parents[2]
    pairs = []
    for entry in data["pairs"]:
        if not isinstance(entry, dict):
            raise ValueError("Manifest format error: each pair must be a dict.")
        pair = PairConfig(
            id=entry.get("id"),
            dataset=entry.get("dataset"),
            input_type=entry.get("input_type"),
            left=entry.get("left"),
            right=entry.get("right"),
            calib=entry.get("calib"),
            optional=bool(entry.get("optional", False)),
            meta=entry.get("meta") or {},
            root_dir=root_dir,
        )
        if not pair.id or not pair.left or not pair.right or not pair.input_type:
            raise ValueError(f"Incomplete pair entry: {entry}")
        pairs.append(pair)
    return pairs


def get_pair(pairs: Iterable[PairConfig] | Dict[str, PairConfig], pair_id: str) -> PairConfig:
    """Retrieve a PairConfig by id.

    Args:
        pairs: List of PairConfig or dict keyed by id.
        pair_id: Target pair id to fetch.

    Returns:
        PairConfig object.

    Raises:
        KeyError: If pair_id cannot be found.

    Example:
        pair = get_pair(pairs, "videos_campus4_c0_c1")
    """
    if isinstance(pairs, dict):
        if pair_id not in pairs:
            raise KeyError(f"Pair id not found: {pair_id}")
        return pairs[pair_id]
    for pair in pairs:
        if pair.id == pair_id:
            return pair
    raise KeyError(f"Pair id not found: {pair_id}")


def open_source(source_cfg: Dict[str, Any], root_dir: Optional[Path] = None) -> FrameSource:
    """Open a single source (video or frames) into a FrameSource.

    Args:
        source_cfg: Dict with keys: input_type, path, frame_pattern, index_csv, fps.
        root_dir: Repo root to resolve relative paths (auto-detected if None).

    Returns:
        FrameSource instance (VideoSource or FramesSource).

    Raises:
        FileNotFoundError: If the path is missing or frames are empty.
        ValueError: If input_type is unsupported or required fields missing.

    Example:
        src = open_source(
            {"input_type": "video", "path": "data/raw/Videos/Campus sequences/campus4-c0.avi"}
        )
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[2]

    input_type = source_cfg.get("input_type")
    path = source_cfg.get("path")
    if not input_type or not path:
        raise ValueError("source_cfg requires input_type and path.")

    source_path = (root_dir / path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {path}")

    if input_type == "video":
        return VideoSource(source_path)

    if input_type == "frames":
        frame_pattern = source_cfg.get("frame_pattern")
        index_csv = source_cfg.get("index_csv")
        if index_csv is None:
            default_index = root_dir / "data" / "manifests" / "epfl_frames_index.csv"
            if default_index.exists():
                index_csv = default_index.as_posix()

        # 优先 index CSV (stable ordering)，否则 fallback 到 glob pattern.
        frame_paths = _resolve_frame_paths(
            frame_dir=source_path,
            frame_pattern=frame_pattern,
            index_csv=index_csv,
            root_dir=root_dir,
        )
        return FramesSource(frame_paths, fps=source_cfg.get("fps"))

    raise ValueError(f"Unsupported input_type: {input_type}")


def open_pair(pair_cfg: PairConfig) -> Tuple[FrameSource, FrameSource]:
    """Open a pair into left/right FrameSource objects.

    Aligns frame counts for frames-based sources to avoid out-of-range access.

    Args:
        pair_cfg: PairConfig entry from manifest.

    Returns:
        (left_source, right_source)

    Raises:
        FileNotFoundError: If either source path is missing.
        ValueError: If input_type is unsupported or decode fails.

    Example:
        left, right = open_pair(pair)
    """
    left_pattern = pair_cfg.meta.get("frame_pattern")
    right_pattern = pair_cfg.meta.get("frame_pattern_right", left_pattern)
    left_cfg = {
        "input_type": pair_cfg.input_type,
        "path": pair_cfg.left,
        "frame_pattern": left_pattern,
        "index_csv": pair_cfg.meta.get("index_csv"),
        "fps": pair_cfg.meta.get("fps"),
    }
    right_cfg = {
        "input_type": pair_cfg.input_type,
        "path": pair_cfg.right,
        "frame_pattern": right_pattern,
        "index_csv": pair_cfg.meta.get("index_csv"),
        "fps": pair_cfg.meta.get("fps"),
    }

    left_source = open_source(left_cfg, root_dir=pair_cfg.root_dir)
    right_source = open_source(right_cfg, root_dir=pair_cfg.root_dir)

    left_len = left_source.length()
    right_len = right_source.length()

    if pair_cfg.input_type == "frames" and left_len is not None and right_len is not None:
        if left_len != right_len:
            min_len = min(left_len, right_len)
            # WARNING: 可通过长度对齐继续运行，避免 silent misalignment。
            # 若 pairs.yaml 里已标注问题，这里再次日志提醒。
            logger.warning(
                "Frame count mismatch for %s: left=%s right=%s, using min=%s",
                pair_cfg.id,
                left_len,
                right_len,
                min_len,
            )
            if isinstance(left_source, FramesSource) and isinstance(right_source, FramesSource):
                left_source.set_length_limit(min_len)
                right_source.set_length_limit(min_len)

    # INFO for traceability: 记录 pair 关键信息与粗粒度元数据。
    logger.info(
        "open_pair id=%s dataset=%s input_type=%s left=%s right=%s length=%s/%s res=%s/%s",
        pair_cfg.id,
        pair_cfg.dataset,
        pair_cfg.input_type,
        pair_cfg.left,
        pair_cfg.right,
        left_len,
        right_len,
        left_source.resolution(),
        right_source.resolution(),
    )
    return left_source, right_source


# --- Frame path resolution ---

def _resolve_frame_paths(
    frame_dir: Path,
    frame_pattern: Optional[str],
    index_csv: Optional[str],
    root_dir: Path,
) -> List[Path]:
    """Resolve frames in a directory using index CSV or glob pattern."""
    if index_csv:
        index_path = root_dir / index_csv if not Path(index_csv).is_absolute() else Path(index_csv)
        if index_path.exists():
            paths = _load_index_frames(index_path, frame_dir, root_dir)
            if paths:
                return paths

    if frame_pattern:
        matches = glob.glob(str(frame_dir / frame_pattern))
        frame_paths = [Path(p) for p in matches]
    else:
        frame_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            frame_paths.extend(Path(p) for p in glob.glob(str(frame_dir / ext)))

    if not frame_paths:
        raise FileNotFoundError(f"No frames found in directory: {frame_dir}")
    # 数字序号排序保证时序稳定；无法解析则字典序兜底。
    return sorted(frame_paths, key=_frame_sort_key)


def _load_index_frames(index_path: Path, frame_dir: Path, root_dir: Path) -> List[Path]:
    """Load frame paths from an index CSV, filtered to the target directory."""
    matched = []
    with index_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_path = row.get("frame_path")
            if not frame_path:
                continue
            row_path = Path(frame_path)
            if row_path.is_absolute():
                abs_path = row_path
            else:
                abs_path = (root_dir / row_path).resolve()
            if abs_path.parent == frame_dir.resolve():
                if not abs_path.exists():
                    # WARNING: index 缺失可跳过，但需提示。
                    logger.warning("Index frame missing on disk: %s", abs_path)
                    continue
                matched.append((row.get("frame_id"), abs_path))
    if not matched:
        return []
    return [
        p for _, p in sorted(matched, key=lambda item: _index_sort_key(item[0], item[1]))
    ]


def _index_sort_key(frame_id: Optional[str], path: Path):
    """Stable sorting for index CSV entries.

    优先 frame_id 数字排序；否则从文件名提取数字保证时序；
    无数字时回退字典序，确保稳定性。
    """
    if frame_id and frame_id.isdigit():
        return (0, int(frame_id))
    numbers = _extract_numbers(path.name)
    if numbers:
        return (0, numbers, path.name)
    return (1, path.name)


def _frame_sort_key(path: Path):
    """Sort key for globbed frames: numeric order if possible, else lexicographic."""
    numbers = _extract_numbers(path.name)
    if numbers:
        return (0, numbers, path.name)
    return (1, path.name)


def _extract_numbers(name: str) -> List[int]:
    return [int(n) for n in re.findall(r"\\d+", name)]


# --- YAML parsing (with fallback) ---

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML with PyYAML if available; otherwise use a minimal parser."""
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _load_simple_pairs_yaml(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping.")
    return data


def _load_simple_pairs_yaml(path: Path) -> Dict[str, Any]:
    """Minimal YAML reader for pairs.yaml when PyYAML is unavailable.

    Supports only the subset of YAML used by pairs.yaml (top-level list,
    nested meta, scalars, and simple lists).
    """
    pairs = []
    current = None
    current_meta = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "pairs:":
            continue
        if stripped.startswith("- "):
            if current:
                pairs.append(current)
            current = {}
            current_meta = None
            key, value = stripped[2:].split(":", 1)
            current[key.strip()] = _parse_scalar(value.strip())
            continue
        if stripped == "meta:":
            current_meta = {}
            if current is None:
                raise ValueError("meta section without a current pair entry.")
            current["meta"] = current_meta
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            target = current_meta if current_meta is not None and indent >= 6 else current
            if target is None:
                raise ValueError("Key-value pair without a current entry.")
            target[key.strip()] = _parse_scalar(value.strip())
    if current:
        pairs.append(current)
    return {"pairs": pairs}


def _parse_scalar(value: str):
    """Parse a simple YAML scalar without full YAML features."""
    if value == "" or value == "null":
        return None
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [_parse_scalar(p) for p in parts]
    if re.match(r"^-?\\d+$", value):
        return int(value)
    if re.match(r"^-?\\d+\\.\\d+$", value):
        return float(value)
    return value
