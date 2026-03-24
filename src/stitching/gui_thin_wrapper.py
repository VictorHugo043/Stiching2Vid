"""Desktop GUI thin wrapper for the existing stitching CLI.

The GUI intentionally stays thin:
- read existing pairs from pairs.yaml
- register new left/right video pairs into the repo + manifest
- launch scripts/run_baseline_video.py as a subprocess
- stream logs and summarize the resulting run bundle
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from stitching.io import PairConfig, load_pairs, open_pair
from stitching.method_b_presets import get_method_b_preset

from scripts.preprocess.split_sbs_stereo import (
    append_manifest_entries,
    path_relative_to_repo,
    probe_video,
)


@dataclass(frozen=True)
class MethodChoice:
    key: str
    label: str


METHOD_CHOICES: List[MethodChoice] = [
    MethodChoice("method_a_orb", "Method A / ORB"),
    MethodChoice("method_a_sift", "Method A / SIFT"),
    MethodChoice("method_b_accuracy_v1", "Method B / accuracy_v1"),
    MethodChoice("method_b_kp3072_v1", "Method B / kp3072_v1"),
]

PREVIEW_SIZE = (320, 180)


def sanitize_identifier(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", (value or "").strip().lower()).strip("_")
    return cleaned or fallback


def infer_default_fps(pair: PairConfig) -> float:
    meta_fps = pair.meta.get("fps")
    try:
        if meta_fps not in {None, ""}:
            fps_value = float(meta_fps)
            if fps_value > 0:
                return fps_value
    except (TypeError, ValueError):
        pass

    pair_id = pair.id.lower()
    dataset_name = pair.dataset.lower()
    if pair.input_type == "frames":
        if pair_id.startswith("kitti_raw") or "dynamicstereo" in pair_id:
            return 10.0
        return 30.0
    if "kitti" in dataset_name:
        return 10.0
    return 30.0


def build_registered_pair_entry(
    pair_id: str,
    dataset_name: str,
    left_rel: str,
    right_rel: str,
    left_info,
    right_info,
) -> Dict:
    same_resolution = (
        left_info.width == right_info.width
        and left_info.height == right_info.height
        and left_info.width is not None
        and left_info.height is not None
    )
    if same_resolution:
        resolution_meta = [int(left_info.width), int(left_info.height)]
    else:
        resolution_meta = {
            "left": [left_info.width, left_info.height],
            "right": [right_info.width, right_info.height],
        }

    fps_value = None
    if left_info.fps and left_info.fps > 0:
        fps_value = float(left_info.fps)
    elif right_info.fps and right_info.fps > 0:
        fps_value = float(right_info.fps)

    return {
        "id": pair_id,
        "dataset": dataset_name,
        "input_type": "video",
        "left": left_rel,
        "right": right_rel,
        "calib": None,
        "optional": False,
        "meta": {
            "scene": pair_id,
            "cameras": ["left", "right"],
            "length_left": left_info.frame_count,
            "length_right": right_info.frame_count,
            "fps": fps_value,
            "resolution": resolution_meta,
        },
    }


def open_path_in_file_manager(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    subprocess.Popen(["xdg-open", str(path)])


def _frame_to_png_base64(frame_bgr, size: tuple[int, int] = PREVIEW_SIZE) -> str:
    import cv2  # type: ignore
    import numpy as np

    target_w, target_h = size
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid frame shape for preview.")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    # cv2.imencode expects BGR ndarray input; converting to RGB here would
    # swap red/blue in the resulting PNG and cause a visible blue tint.
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode preview image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def load_pair_preview_png_data(pair: PairConfig) -> tuple[Optional[str], Optional[str], Optional[str]]:
    left_source = None
    right_source = None
    try:
        left_source, right_source = open_pair(pair)
        left_frame = left_source.read(0)
        right_frame = right_source.read(0)
        return (
            _frame_to_png_base64(left_frame),
            _frame_to_png_base64(right_frame),
            None,
        )
    except Exception as exc:
        return None, None, str(exc)
    finally:
        if left_source is not None:
            left_source.close()
        if right_source is not None:
            right_source.close()


class SubprocessRunner:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen[str]] = None
        self.thread: Optional[threading.Thread] = None
        self.lines: "queue.Queue[str]" = queue.Queue()

    def start(self, cmd: List[str], cwd: Path) -> None:
        if self.process is not None:
            raise RuntimeError("A run is already in progress.")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._pump_output, daemon=True)
        self.thread.start()

    def _pump_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)
        self.process.stdout.close()

    def read_available(self) -> List[str]:
        out: List[str] = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return out

    def poll(self) -> Optional[int]:
        if self.process is None:
            return None
        return self.process.poll()

    def terminate(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()

    def clear(self) -> None:
        self.process = None
        self.thread = None
        self.read_available()


class StitchingGuiApp:
    def __init__(self, root: tk.Tk, repo_root: Path, manifest_path: Path) -> None:
        self.root = root
        self.repo_root = repo_root
        self.manifest_path = manifest_path
        self.runner = SubprocessRunner()
        self.current_run_dir: Optional[Path] = None
        self.pairs: List[PairConfig] = []
        self.pair_map: Dict[str, PairConfig] = {}
        self.left_preview_photo: Optional[tk.PhotoImage] = None
        self.right_preview_photo: Optional[tk.PhotoImage] = None
        self.register_dialog: Optional[tk.Toplevel] = None

        self.pair_var = tk.StringVar()
        self.method_var = tk.StringVar(value=METHOD_CHOICES[0].key)
        self.geometry_mode_var = tk.StringVar(value="fixed_geometry")
        self.seam_policy_var = tk.StringVar(value="fixed")
        self.run_id_var = tk.StringVar()
        self.max_frames_var = tk.StringVar(value="6000")
        self.start_var = tk.StringVar(value="0")
        self.stride_var = tk.StringVar(value="1")
        self.fps_var = tk.StringVar(value="30")
        self.keyframe_every_var = tk.StringVar(value="5")
        self.seam_keyframe_every_var = tk.StringVar(value="10")
        self.seam_trigger_diff_var = tk.StringVar(value="18")
        self.seam_trigger_foreground_ratio_var = tk.StringVar(value="0.08")
        self.snapshot_every_var = tk.StringVar(value="1000")
        self.force_cpu_var = tk.BooleanVar(value=True)
        self.auto_open_run_dir_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="idle")

        self.log_text: tk.Text
        self.artefact_text: tk.Text
        self.pair_meta_label: ttk.Label
        self.left_preview_label: ttk.Label
        self.right_preview_label: ttk.Label
        self.open_run_button: ttk.Button
        self.geometry_keyframe_field: Optional[Dict[str, object]] = None
        self.seam_keyframe_field: Optional[Dict[str, object]] = None
        self._build_ui()
        self.refresh_pairs()
        self.geometry_mode_var.trace_add("write", self._on_mode_visibility_changed)
        self.seam_policy_var.trace_add("write", self._on_mode_visibility_changed)
        self._update_mode_visibility()
        self.root.after(200, self._poll_runner)

    def _build_ui(self) -> None:
        self.root.title("Two-View Video Stitching GUI")
        self.root.geometry("1200x860")

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        pair_frame = ttk.LabelFrame(outer, text="Existing Pair", padding=10)
        pair_frame.pack(fill=tk.X)
        pair_header = ttk.Frame(pair_frame)
        pair_header.pack(fill=tk.X)
        self.pair_combo = ttk.Combobox(
            pair_header,
            textvariable=self.pair_var,
            state="readonly",
            width=72,
        )
        self.pair_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.pair_combo.bind("<<ComboboxSelected>>", self._on_pair_changed)
        ttk.Button(pair_header, text="Refresh Pairs", command=self.refresh_pairs).pack(
            side=tk.LEFT,
            padx=(8, 0),
        )
        ttk.Button(pair_header, text="Register Pair... (Upload New Videos)", command=self._open_register_dialog).pack(
            side=tk.LEFT,
            padx=(8, 0),
        )
        self.pair_meta_label = ttk.Label(pair_frame, text="No pair selected.")
        self.pair_meta_label.pack(anchor=tk.W, pady=(8, 8))

        preview_row = ttk.Frame(pair_frame)
        preview_row.pack(fill=tk.X)
        left_frame = ttk.LabelFrame(preview_row, text="Left Frame 0", padding=6)
        right_frame = ttk.LabelFrame(preview_row, text="Right Frame 0", padding=6)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.left_preview_label = ttk.Label(left_frame, text="No preview", anchor=tk.CENTER)
        self.left_preview_label.pack(fill=tk.BOTH, expand=True)
        self.right_preview_label = ttk.Label(right_frame, text="No preview", anchor=tk.CENTER)
        self.right_preview_label.pack(fill=tk.BOTH, expand=True)

        run_frame = ttk.LabelFrame(outer, text="Run Config", padding=10)
        run_frame.pack(fill=tk.X, pady=(12, 0))

        grid = ttk.Frame(run_frame)
        grid.pack(fill=tk.X)
        self._grid_labeled_widget(grid, 0, 0, "Method", self._build_choice_combo(grid, self.method_var, [c.key for c in METHOD_CHOICES]))
        self._grid_labeled_widget(grid, 0, 1, "Geometry", self._build_choice_combo(grid, self.geometry_mode_var, ["fixed_geometry", "keyframe_update", "adaptive_update"]))
        self._grid_labeled_widget(grid, 0, 2, "Seam Policy", self._build_choice_combo(grid, self.seam_policy_var, ["fixed", "keyframe", "trigger"]))
        self._grid_labeled_widget(grid, 0, 3, "Run ID", ttk.Entry(grid, textvariable=self.run_id_var, width=28))

        self._grid_labeled_widget(grid, 1, 0, "Max Frames", ttk.Entry(grid, textvariable=self.max_frames_var, width=14))
        self._grid_labeled_widget(grid, 1, 1, "Start", ttk.Entry(grid, textvariable=self.start_var, width=14))
        self._grid_labeled_widget(grid, 1, 2, "Stride", ttk.Entry(grid, textvariable=self.stride_var, width=14))
        self._grid_labeled_widget(grid, 1, 3, "FPS", ttk.Entry(grid, textvariable=self.fps_var, width=14))

        self._grid_labeled_widget(grid, 2, 0, "Trigger Diff", ttk.Entry(grid, textvariable=self.seam_trigger_diff_var, width=14))
        self._grid_labeled_widget(grid, 2, 1, "FG Ratio", ttk.Entry(grid, textvariable=self.seam_trigger_foreground_ratio_var, width=14))
        self._grid_labeled_widget(grid, 2, 2, "Snapshot Every", ttk.Entry(grid, textvariable=self.snapshot_every_var, width=14))
        self._grid_labeled_widget(
            grid,
            2,
            3,
            "Force CPU",
            ttk.Checkbutton(grid, text="Enabled", variable=self.force_cpu_var),
        )

        self.geometry_keyframe_field = self._grid_labeled_widget(
            grid,
            3,
            0,
            "Keyframe Every",
            ttk.Entry(grid, textvariable=self.keyframe_every_var, width=14),
        )
        self.seam_keyframe_field = self._grid_labeled_widget(
            grid,
            3,
            1,
            "Seam Keyframe Every",
            ttk.Entry(grid, textvariable=self.seam_keyframe_every_var, width=14),
        )

        control_row = ttk.Frame(run_frame)
        control_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(control_row, text="Start Run", command=self._start_run).pack(side=tk.LEFT)
        ttk.Button(control_row, text="Stop Run", command=self._stop_run).pack(side=tk.LEFT, padx=(8, 0))
        self.open_run_button = ttk.Button(
            control_row,
            text="Open Run Folder",
            command=self._open_current_run_dir,
            state=tk.DISABLED,
        )
        self.open_run_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            control_row,
            text="Auto-open on finish",
            variable=self.auto_open_run_dir_var,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(control_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        bottom = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        log_frame = ttk.LabelFrame(bottom, text="Logs", padding=8)
        artefact_frame = ttk.LabelFrame(bottom, text="Artefacts", padding=8)
        bottom.add(log_frame, weight=3)
        bottom.add(artefact_frame, weight=2)

        self.log_text = tk.Text(log_frame, wrap="word", height=28)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.artefact_text = tk.Text(artefact_frame, wrap="word", height=28)
        self.artefact_text.pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _build_choice_combo(parent: ttk.Frame, variable: tk.StringVar, values: List[str]) -> ttk.Combobox:
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=22)
        combo.current(0)
        return combo

    @staticmethod
    def _grid_labeled_widget(parent: ttk.Frame, row: int, column: int, label: str, widget) -> Dict[str, object]:
        label_row = row * 2
        widget_row = label_row + 1
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(
            row=label_row,
            column=column,
            sticky="w",
            padx=6,
            pady=(4, 0),
        )
        widget.grid(
            row=widget_row,
            column=column,
            sticky="ew",
            padx=6,
            pady=(0, 6),
        )
        parent.grid_columnconfigure(column, weight=1)
        return {
            "label": label_widget,
            "widget": widget,
            "label_row": label_row,
            "widget_row": widget_row,
            "column": column,
        }

    @staticmethod
    def _set_field_visible(field: Optional[Dict[str, object]], visible: bool) -> None:
        if not field:
            return
        label_widget = field["label"]
        widget = field["widget"]
        if visible:
            label_widget.grid()
            widget.grid()
        else:
            label_widget.grid_remove()
            widget.grid_remove()

    @staticmethod
    def _add_labeled_entry(parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _add_browse_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            row,
            text="Browse",
            command=lambda: self._browse_video(variable),
        ).pack(side=tk.LEFT, padx=(6, 0))

    def _browse_video(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")],
        )
        if path:
            variable.set(path)

    def _on_mode_visibility_changed(self, *_args) -> None:
        self._update_mode_visibility()

    def _update_mode_visibility(self) -> None:
        self._set_field_visible(
            self.geometry_keyframe_field,
            self.geometry_mode_var.get().strip() == "keyframe_update",
        )
        self._set_field_visible(
            self.seam_keyframe_field,
            self.seam_policy_var.get().strip() == "keyframe",
        )

    def _set_preview_label(
        self,
        label: ttk.Label,
        image_data: Optional[str],
        fallback_text: str,
    ) -> Optional[tk.PhotoImage]:
        if image_data:
            photo = tk.PhotoImage(data=image_data)
            label.configure(image=photo, text="")
            return photo
        label.configure(image="", text=fallback_text)
        return None

    def _update_pair_preview(self, pair: PairConfig) -> None:
        left_data, right_data, error = load_pair_preview_png_data(pair)
        if error:
            self.left_preview_photo = None
            self.right_preview_photo = None
            self.left_preview_label.configure(image="", text=f"Preview unavailable\n{error}")
            self.right_preview_label.configure(image="", text="Preview unavailable")
            return
        self.left_preview_photo = self._set_preview_label(self.left_preview_label, left_data, "No preview")
        self.right_preview_photo = self._set_preview_label(self.right_preview_label, right_data, "No preview")

    def _open_register_dialog(self) -> None:
        if self.register_dialog is not None and self.register_dialog.winfo_exists():
            self.register_dialog.lift()
            self.register_dialog.focus_force()
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Pair")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        self.register_dialog = dialog

        pair_id_var = tk.StringVar()
        dataset_var = tk.StringVar(value="GUIUpload")
        left_path_var = tk.StringVar()
        right_path_var = tk.StringVar()

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        self._add_labeled_entry(body, "Pair ID", pair_id_var)
        self._add_labeled_entry(body, "Dataset", dataset_var)
        self._add_browse_row(body, "Left Video", left_path_var)
        self._add_browse_row(body, "Right Video", right_path_var)

        hint = ttk.Label(
            body,
            text="Videos will be copied into data/raw/Videos/gui_uploads/<pair_id>/ and then appended to pairs.yaml.",
            wraplength=440,
            justify=tk.LEFT,
        )
        hint.pack(anchor=tk.W, pady=(8, 0))

        button_row = ttk.Frame(body)
        button_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(
            button_row,
            text="Register",
            command=lambda: self._register_pair_from_dialog(
                dialog,
                pair_id_var,
                dataset_var,
                left_path_var,
                right_path_var,
            ),
        ).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=(8, 0))
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

    def _register_pair_from_dialog(
        self,
        dialog: tk.Toplevel,
        pair_id_var: tk.StringVar,
        dataset_var: tk.StringVar,
        left_path_var: tk.StringVar,
        right_path_var: tk.StringVar,
    ) -> None:
        pair_id = sanitize_identifier(pair_id_var.get(), "gui_upload_pair")
        dataset_name = dataset_var.get().strip() or "GUIUpload"
        left_src = Path(left_path_var.get().strip())
        right_src = Path(right_path_var.get().strip())

        if not left_src.exists() or not left_src.is_file():
            messagebox.showerror("Register Pair", "Left video path is invalid.", parent=dialog)
            return
        if not right_src.exists() or not right_src.is_file():
            messagebox.showerror("Register Pair", "Right video path is invalid.", parent=dialog)
            return
        if pair_id in self.pair_map:
            messagebox.showerror("Register Pair", f"Pair id already exists: {pair_id}", parent=dialog)
            return

        left_info = probe_video(left_src)
        right_info = probe_video(right_src)
        upload_dir = self.repo_root / "data" / "raw" / "Videos" / "gui_uploads" / pair_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        left_dst = upload_dir / f"left{left_src.suffix.lower()}"
        right_dst = upload_dir / f"right{right_src.suffix.lower()}"
        shutil.copy2(left_src, left_dst)
        shutil.copy2(right_src, right_dst)

        entry = build_registered_pair_entry(
            pair_id=pair_id,
            dataset_name=dataset_name,
            left_rel=path_relative_to_repo(left_dst, self.repo_root),
            right_rel=path_relative_to_repo(right_dst, self.repo_root),
            left_info=left_info,
            right_info=right_info,
        )
        backup_path = append_manifest_entries(self.manifest_path, [entry])
        self.refresh_pairs()
        self.pair_var.set(pair_id)
        self._on_pair_changed()
        self._append_log(
            f"[register] pair={pair_id} left={left_dst} right={right_dst} manifest_backup={backup_path}\n"
        )
        messagebox.showinfo("Register Pair", f"Registered pair: {pair_id}", parent=dialog)
        dialog.destroy()
        self.register_dialog = None

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _set_artefacts(self, text: str) -> None:
        self.artefact_text.delete("1.0", tk.END)
        self.artefact_text.insert("1.0", text)

    def refresh_pairs(self) -> None:
        self.pairs = sorted(load_pairs(self.manifest_path), key=lambda item: item.id)
        self.pair_map = {pair.id: pair for pair in self.pairs}
        pair_ids = [pair.id for pair in self.pairs]
        self.pair_combo["values"] = pair_ids
        if pair_ids and self.pair_var.get() not in self.pair_map:
            self.pair_var.set(pair_ids[0])
        self._on_pair_changed()

    def _on_pair_changed(self, *_args) -> None:
        pair = self.pair_map.get(self.pair_var.get())
        if pair is None:
            self.pair_meta_label.config(text="No pair selected.")
            self.left_preview_photo = None
            self.right_preview_photo = None
            self.left_preview_label.configure(image="", text="No preview")
            self.right_preview_label.configure(image="", text="No preview")
            return
        fps_value = infer_default_fps(pair)
        self.fps_var.set(str(int(fps_value)) if fps_value.is_integer() else str(fps_value))
        self.pair_meta_label.config(
            text=f"dataset={pair.dataset} | input_type={pair.input_type} | left={pair.left} | right={pair.right}"
        )
        self._update_pair_preview(pair)

    def _build_command(self, run_dir: Path, run_id: str) -> List[str]:
        pair_id = self.pair_var.get().strip()
        if not pair_id:
            raise ValueError("Please select a pair first.")

        cmd = [
            sys.executable,
            "-u",
            "scripts/run_baseline_video.py",
            "--pair",
            pair_id,
            "--manifest",
            str(self.manifest_path),
            "--out_dir",
            str(run_dir),
            "--run_id",
            run_id,
            "--geometry_mode",
            self.geometry_mode_var.get().strip(),
            "--seam_policy",
            self.seam_policy_var.get().strip(),
            "--start",
            self.start_var.get().strip() or "0",
            "--stride",
            self.stride_var.get().strip() or "1",
            "--snapshot_every",
            self.snapshot_every_var.get().strip() or "1000",
            "--blend",
            "feather",
        ]

        if self.geometry_mode_var.get().strip() == "keyframe_update":
            cmd.extend(["--keyframe_every", self.keyframe_every_var.get().strip() or "5"])

        max_frames = self.max_frames_var.get().strip()
        if max_frames:
            cmd.extend(["--max_frames", max_frames])

        seam_keyframe_every = self.seam_keyframe_every_var.get().strip()
        if self.seam_policy_var.get().strip() == "keyframe" and seam_keyframe_every:
            cmd.extend(["--seam_keyframe_every", seam_keyframe_every])

        fps_value = self.fps_var.get().strip()
        if fps_value:
            cmd.extend(["--fps", fps_value])

        if self.seam_policy_var.get() == "trigger":
            diff_threshold = self.seam_trigger_diff_var.get().strip() or "18"
            fg_ratio = self.seam_trigger_foreground_ratio_var.get().strip() or "0.08"
            cmd.extend(
                [
                    "--seam_trigger_diff_threshold",
                    diff_threshold,
                    "--foreground_mode",
                    "disagreement",
                    "--seam_trigger_foreground_ratio",
                    fg_ratio,
                ]
            )

        method_key = self.method_var.get().strip()
        if method_key == "method_a_orb":
            cmd.extend(
                [
                    "--feature",
                    "orb",
                    "--feature_backend",
                    "opencv_orb",
                    "--matcher_backend",
                    "opencv_bf_ratio",
                    "--geometry_backend",
                    "opencv_ransac",
                ]
            )
        elif method_key == "method_a_sift":
            cmd.extend(
                [
                    "--feature",
                    "sift",
                    "--feature_backend",
                    "opencv_sift",
                    "--matcher_backend",
                    "opencv_bf_ratio",
                    "--geometry_backend",
                    "opencv_ransac",
                ]
            )
        else:
            preset_name = "accuracy_v1" if method_key == "method_b_accuracy_v1" else "kp3072_v1"
            preset = get_method_b_preset(preset_name)
            cmd.extend(
                [
                    "--feature_backend",
                    "superpoint",
                    "--matcher_backend",
                    "lightglue",
                    "--geometry_backend",
                    "opencv_usac_magsac",
                    "--max_keypoints",
                    str(preset.max_keypoints),
                ]
            )
            if preset.resize_long_edge is not None:
                cmd.extend(["--resize_long_edge", str(preset.resize_long_edge)])
            if preset.depth_confidence is not None:
                cmd.extend(["--depth_confidence", str(preset.depth_confidence)])
            if preset.width_confidence is not None:
                cmd.extend(["--width_confidence", str(preset.width_confidence)])
            if preset.filter_threshold is not None:
                cmd.extend(["--filter_threshold", str(preset.filter_threshold)])
            if self.force_cpu_var.get():
                cmd.append("--force_cpu")

        return cmd

    def _build_request_payload(self, run_id: str, run_dir: Path, cmd: List[str]) -> Dict:
        pair = self.pair_map.get(self.pair_var.get())
        return {
            "run_id": run_id,
            "run_dir": str(run_dir.relative_to(self.repo_root)),
            "pair_id": self.pair_var.get(),
            "dataset": pair.dataset if pair else None,
            "input_type": pair.input_type if pair else None,
            "method": self.method_var.get(),
            "geometry_mode": self.geometry_mode_var.get(),
            "seam_policy": self.seam_policy_var.get(),
            "max_frames": self.max_frames_var.get().strip() or None,
            "start": self.start_var.get().strip() or "0",
            "stride": self.stride_var.get().strip() or "1",
            "fps": self.fps_var.get().strip() or None,
            "keyframe_every": self.keyframe_every_var.get().strip() or None,
            "seam_keyframe_every": self.seam_keyframe_every_var.get().strip() or None,
            "seam_trigger_diff_threshold": self.seam_trigger_diff_var.get().strip() or None,
            "seam_trigger_foreground_ratio": self.seam_trigger_foreground_ratio_var.get().strip() or None,
            "snapshot_every": self.snapshot_every_var.get().strip() or None,
            "force_cpu": bool(self.force_cpu_var.get()),
            "command": cmd,
        }

    def _start_run(self) -> None:
        if self.runner.process is not None:
            messagebox.showerror("Run", "A run is already in progress.")
            return

        pair_id = self.pair_var.get().strip()
        if not pair_id:
            messagebox.showerror("Run", "Please select a pair first.")
            return

        method_key = self.method_var.get().strip() or "method_a_orb"
        ts = time.strftime("%Y%m%d_%H%M%S")
        default_run_id = sanitize_identifier(f"{ts}_{pair_id}_{method_key}", "gui_run")
        run_id = sanitize_identifier(self.run_id_var.get(), default_run_id)
        self.run_id_var.set(run_id)

        run_dir = self.repo_root / "outputs" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_command(run_dir, run_id)
        request_payload = self._build_request_payload(run_id, run_dir, cmd)
        (run_dir / "gui_request.json").write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.current_run_dir = run_dir
        self.open_run_button.configure(state=tk.DISABLED)
        self.log_text.delete("1.0", tk.END)
        self._set_artefacts(f"run_dir={run_dir}\n")
        self._append_log(f"[gui] starting run_id={run_id}\n")
        self._append_log("[gui] command:\n" + " ".join(cmd) + "\n\n")
        self.status_var.set(f"running: {run_id}")

        try:
            self.runner.start(cmd, cwd=self.repo_root)
        except Exception as exc:
            messagebox.showerror("Run", str(exc))
            self.status_var.set("idle")

    def _stop_run(self) -> None:
        if self.runner.process is None:
            return
        self.runner.terminate()
        self._append_log("[gui] terminate requested\n")

    def _open_current_run_dir(self) -> None:
        if self.current_run_dir is None:
            messagebox.showerror("Open Run Folder", "No run directory is available yet.")
            return
        try:
            open_path_in_file_manager(self.current_run_dir)
        except Exception as exc:
            messagebox.showerror("Open Run Folder", str(exc))

    def _poll_runner(self) -> None:
        for line in self.runner.read_available():
            self._append_log(line)

        rc = self.runner.poll()
        if rc is not None and self.runner.process is not None:
            self.status_var.set(f"finished rc={rc}")
            self._append_log(f"\n[gui] process finished rc={rc}\n")
            self._set_artefacts(self._summarize_run_dir(self.current_run_dir))
            if self.current_run_dir is not None and self.current_run_dir.exists():
                self.open_run_button.configure(state=tk.NORMAL)
                if self.auto_open_run_dir_var.get():
                    try:
                        open_path_in_file_manager(self.current_run_dir)
                    except Exception as exc:
                        self._append_log(f"[gui] failed to open run dir: {exc}\n")
            self.runner.clear()

        self.root.after(200, self._poll_runner)

    def _summarize_run_dir(self, run_dir: Optional[Path]) -> str:
        if run_dir is None:
            return ""

        parts = [f"run_dir={run_dir}"]
        for name in [
            "gui_request.json",
            "logs.txt",
            "metrics_preview.json",
            "debug.json",
            "transforms.csv",
            "jitter_timeseries.csv",
            "stitched.mp4",
        ]:
            path = run_dir / name
            if path.exists():
                parts.append(str(path))

        snapshots_dir = run_dir / "snapshots"
        if snapshots_dir.exists():
            parts.append(str(snapshots_dir))

        metrics_path = run_dir / "metrics_preview.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                parts.append("")
                parts.append("metrics_preview:")
                for key in [
                    "pair_id",
                    "geometry_mode",
                    "feature_backend_effective",
                    "matcher_backend_effective",
                    "geometry_backend_effective",
                    "processed_frames",
                    "success_frames",
                    "mean_inliers",
                    "mean_inlier_ratio",
                    "approx_fps",
                ]:
                    if key in metrics:
                        parts.append(f"  {key}={metrics[key]}")
            except Exception:
                parts.append("metrics_preview: failed to parse")

        return "\n".join(parts) + "\n"


def launch_gui(repo_root: Path, manifest_path: Path) -> None:
    root = tk.Tk()
    app = StitchingGuiApp(root=root, repo_root=repo_root, manifest_path=manifest_path)
    root.mainloop()
