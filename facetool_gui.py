#!/usr/bin/env python3
"""facetool GUI - a desktop window for the facetool face detector.

Open a webcam, image or video, pick a detector, and watch faces get detected
live. Toggle blur and landmarks, record the result, save snapshots and crops.

Run:  python facetool_gui.py

Requires Tkinter (bundled with Python on Windows/macOS; on Debian/Ubuntu run
`sudo apt install python3-tk`) and Pillow (`pip install pillow`).
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import facetool
from facetool import Face

try:  # the GUI needs these; the Engine below does not
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    GUI_AVAILABLE = True
    GUI_IMPORT_ERROR = ""
except ImportError as _exc:  # pragma: no cover
    GUI_AVAILABLE = False
    GUI_IMPORT_ERROR = str(_exc)

APP_TITLE = "facetool"
OUTPUT_DIR = Path("facetool_output")


# --------------------------------------------------------------------------- #
# Engine: capture + detection on a background thread (no GUI code in here)
# --------------------------------------------------------------------------- #
class Settings:
    """Values the UI changes and the worker thread reads every frame."""

    def __init__(self) -> None:
        self.detector = "yunet"
        self.threshold = 0.7
        self.blur = False
        self.landmarks = False
        self.emotions = False
        self.version = 0  # bumped on every change so still images get re-processed

    def touch(self) -> None:
        self.version += 1


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.lock = threading.Lock()  # guards the published frame
        self._wlock = threading.Lock()  # guards the video writer
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.latest: Optional[np.ndarray] = None  # annotated frame (never mutated after publish)
        self.latest_raw: Optional[np.ndarray] = None
        self.latest_faces: List[Face] = []
        self.info = ""
        self.error: Optional[str] = None
        self.notice: Optional[str] = None  # non-fatal message for the UI (e.g. emotion model missing)
        self.finished = False

        self.recognizer = None
        self.detector = None
        self._det_key: Optional[str] = None
        self._det_thr: Optional[float] = None

        self._writer: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[str] = None
        self._fps_hint = 20.0

    # ---- control ----------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def recording(self) -> bool:
        return self._record_path is not None

    def start(self, source) -> None:
        """source: int (webcam index) or Path (image or video file)."""
        self.stop()
        with self.lock:
            self.latest = self.latest_raw = None
            self.latest_faces = []
            self.info = "Loading..."
        self.error, self.finished = None, False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3)
        self._thread = None
        self.stop_recording()

    def start_recording(self, path: str) -> None:
        with self._wlock:
            self._record_path, self._writer = str(path), None

    def stop_recording(self) -> None:
        with self._wlock:
            if self._writer is not None:
                self._writer.release()
            self._writer, self._record_path = None, None

    def state(self):
        with self.lock:
            return self.latest, self.latest_raw, list(self.latest_faces), self.info

    # ---- worker -----------------------------------------------------------
    def _ensure_detector(self) -> None:
        if self.detector is None or self._det_key != self.s.detector:
            self.detector = facetool.make_detector(self.s.detector, self.s.threshold)
            self._det_key, self._det_thr = self.s.detector, self.s.threshold
        elif self._det_thr != self.s.threshold:  # cheap live update for YuNet
            inner = getattr(self.detector, "det", None)
            if inner is not None and hasattr(inner, "setScoreThreshold"):
                inner.setScoreThreshold(float(self.s.threshold))
            self._det_thr = self.s.threshold

    def _ensure_recognizer(self) -> None:
        if self.recognizer is not None:
            return
        try:
            self.recognizer = facetool.EmotionRecognizer()
        except Exception as exc:  # keep detecting; switch the feature off, tell the user, allow a retry
            self.s.emotions = False
            self.notice = (
                f"Emotion model unavailable: {exc}\n\nTo use it offline, put the model file at:\n"
                f"{facetool.FER_PATH}"
            )

    def _process(self, raw: np.ndarray):
        self._ensure_detector()
        faces = self.detector.detect(raw)
        if self.s.emotions:
            self._ensure_recognizer()
            if self.s.emotions:
                facetool.label_emotions(raw, faces, self.recognizer)
        out = raw.copy()
        facetool.annotate(out, faces, self.s.blur, self.s.landmarks)
        return out, faces

    def _publish(self, out, raw, faces, info) -> None:
        with self.lock:
            self.latest, self.latest_raw, self.latest_faces, self.info = out, raw, faces, info

    def _run(self, source) -> None:
        try:
            if isinstance(source, Path) and source.suffix.lower() in facetool.IMAGE_EXTS:
                self._run_image(source)
            else:
                self._run_stream(source)
        except Exception as exc:  # surface any failure in the UI instead of dying silently
            self.error = str(exc)
        finally:
            self.stop_recording()
            self.finished = True

    def _run_image(self, path: Path) -> None:
        raw = cv2.imread(str(path))
        if raw is None:
            raise RuntimeError(f"Could not read image: {path}")
        while not self._stop.is_set():
            version = self.s.version
            out, faces = self._process(raw)
            self._publish(out, raw, faces, f"{len(faces)} face(s) | {self.detector.name}")
            self._write(out)
            while not self._stop.is_set() and version == self.s.version:
                time.sleep(0.05)  # idle until a setting changes

    def _run_stream(self, source) -> None:
        is_file = isinstance(source, Path)
        cap = cv2.VideoCapture(str(source) if is_file else int(source))
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open {'video ' + str(source) if is_file else 'webcam ' + str(source)}. "
                "Is the camera in use by another app?"
            )
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._fps_hint = src_fps if is_file and src_fps > 1 else 20.0
        period = 1.0 / src_fps if is_file and src_fps > 1 else 0.0  # play files at normal speed
        fps, prev, next_t = 0.0, time.time(), time.time()
        try:
            while not self._stop.is_set():
                ok, raw = cap.read()
                if not ok:
                    if not is_file:
                        raise RuntimeError("The camera stopped delivering frames.")
                    break  # end of video
                out, faces = self._process(raw)
                now = time.time()
                inst = 1.0 / max(now - prev, 1e-6)
                fps, prev = (0.9 * fps + 0.1 * inst) if fps else inst, now
                self._publish(out, raw, faces, f"{len(faces)} face(s) | {fps:4.1f} FPS | {self.detector.name}")
                self._write(out)
                if period:
                    next_t += period
                    delay = next_t - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.time()
        finally:
            cap.release()

    def _write(self, frame: np.ndarray) -> None:
        with self._wlock:
            if self._record_path is None:
                return
            if self._writer is None:
                h, w = frame.shape[:2]
                self._writer = cv2.VideoWriter(
                    self._record_path, cv2.VideoWriter_fourcc(*"mp4v"), self._fps_hint, (w, h)
                )
                if not self._writer.isOpened():
                    self._writer, self._record_path = None, None
                    self.error = "Could not start recording. Try a .mp4 or .avi filename."
                    return
            self._writer.write(frame)


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #
class App:
    POLL_MS = 30

    def __init__(self) -> None:
        self.settings = Settings()
        self.engine = Engine(self.settings)
        self._shown: Optional[np.ndarray] = None
        self._photo = None
        self.out_dir = OUTPUT_DIR

        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1100x680")
        self.root.minsize(820, 520)
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(self.POLL_MS, self._poll)

    # ---- layout -----------------------------------------------------------
    def _build(self) -> None:
        # create every variable first so no widget callback can run before they exist
        self.cam_index = tk.IntVar(value=0)
        self.det_var = tk.StringVar(value="yunet")
        self.thr_var = tk.DoubleVar(value=0.7)
        self.blur_var = tk.BooleanVar(value=False)
        self.lm_var = tk.BooleanVar(value=False)
        self.emo_var = tk.BooleanVar(value=False)

        side = ttk.Frame(self.root, padding=12)
        side.pack(side="left", fill="y")
        main = ttk.Frame(self.root)
        main.pack(side="right", fill="both", expand=True)

        # Source
        box = ttk.LabelFrame(side, text="Source", padding=8)
        box.pack(fill="x")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Button(row, text="Webcam", command=self._open_webcam).pack(side="left")
        ttk.Label(row, text=" camera #").pack(side="left")
        ttk.Spinbox(row, from_=0, to=9, width=3, textvariable=self.cam_index).pack(side="left")
        ttk.Button(box, text="Open image...", command=self._open_image).pack(fill="x", pady=(6, 0))
        ttk.Button(box, text="Open video...", command=self._open_video).pack(fill="x", pady=(4, 0))
        self.stop_btn = ttk.Button(box, text="Stop", command=self._stop_source, state="disabled")
        self.stop_btn.pack(fill="x", pady=(4, 0))

        # Detection
        det = ttk.LabelFrame(side, text="Detection", padding=8)
        det.pack(fill="x", pady=(10, 0))
        ttk.Label(det, text="Detector").pack(anchor="w")
        combo =ttk.Combobox(det, textvariable=self.det_var, values=["yunet", "haar"], state="readonly")
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._changed())
        self.thr_label = ttk.Label(det, text="Confidence threshold: 0.70")
        self.thr_label.pack(anchor="w", pady=(8, 0))
        ttk.Scale(det, from_=0.1, to=0.99, variable=self.thr_var, command=lambda _v: self._changed()).pack(fill="x")
        ttk.Label(det, text="(YuNet only. Higher = fewer false hits)", foreground="gray").pack(anchor="w")

        # Display
        dsp = ttk.LabelFrame(side, text="Display", padding=8)
        dsp.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(dsp, text="Blur faces", variable=self.blur_var, command=self._changed).pack(anchor="w")
        ttk.Checkbutton(dsp, text="Show landmarks", variable=self.lm_var, command=self._changed).pack(anchor="w")
        ttk.Checkbutton(dsp, text="Identify emotions", variable=self.emo_var, command=self._changed).pack(anchor="w")
        ttk.Label(dsp, text="Reads the visible expression, not\nfeelings. Works best with YuNet.",
                  foreground="gray").pack(anchor="w", padx=(22, 0))

        # Output
        out = ttk.LabelFrame(side, text="Save", padding=8)
        out.pack(fill="x", pady=(10, 0))
        self.snap_btn = ttk.Button(out, text="Save snapshot", command=self._snapshot, state="disabled")
        self.snap_btn.pack(fill="x")
        self.crop_btn = ttk.Button(out, text="Save face crops", command=self._crops, state="disabled")
        self.crop_btn.pack(fill="x", pady=(4, 0))
        self.rec_btn = ttk.Button(out, text="Start recording", command=self._toggle_record, state="disabled")
        self.rec_btn.pack(fill="x", pady=(4, 0))
        ttk.Button(out, text="Choose output folder...", command=self._choose_dir).pack(fill="x", pady=(8, 0))
        self.dir_label = ttk.Label(out, text=str(self.out_dir), foreground="gray", wraplength=190)
        self.dir_label.pack(anchor="w")

        # Preview + status
        self.canvas = tk.Canvas(main, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._render(self._shown))
        self.status = tk.StringVar(value="Open a webcam, image or video to start.")
        ttk.Label(main, textvariable=self.status, padding=(8, 4), anchor="w").pack(fill="x")
        self._render(None)

    # ---- settings ---------------------------------------------------------
    def _changed(self) -> None:
        s = self.settings
        s.detector = self.det_var.get()
        s.threshold = round(float(self.thr_var.get()), 2)
        s.blur, s.landmarks, s.emotions = self.blur_var.get(), self.lm_var.get(), self.emo_var.get()
        self.thr_label.config(text=f"Confidence threshold: {s.threshold:.2f}")
        s.touch()

    # ---- sources ----------------------------------------------------------
    def _begin(self, source, label: str, recordable: bool = True) -> None:
        self.engine.start(source)
        self.stop_btn.config(state="normal")
        self.snap_btn.config(state="normal")
        self.crop_btn.config(state="normal")
        # recording only makes sense for moving sources, not a single still image
        self.rec_btn.config(state="normal" if recordable else "disabled", text="Start recording")
        self.status.set(f"{label}: loading detector...")

    def _open_webcam(self) -> None:
        self._begin(int(self.cam_index.get()), f"Webcam {self.cam_index.get()}")

    def _open_image(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(facetool.IMAGE_EXTS))
        path = filedialog.askopenfilename(title="Open image", filetypes=[("Images", exts), ("All files", "*.*")])
        if path:
            self._begin(Path(path), Path(path).name, recordable=False)

    def _open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All files", "*.*")],
        )
        if path:
            self._begin(Path(path), Path(path).name)

    def _stop_source(self) -> None:
        self.engine.stop()
        self.stop_btn.config(state="disabled")
        self.rec_btn.config(text="Start recording", state="disabled")
        self.status.set("Stopped.")

    # ---- saving -----------------------------------------------------------
    def _choose_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.out_dir = Path(path)
            self.dir_label.config(text=str(self.out_dir))

    def _stamp(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def _snapshot(self) -> None:
        frame, _raw, _faces, _info = self.engine.state()
        if frame is None:
            self.status.set("Nothing to save yet.")
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"snapshot_{self._stamp()}.jpg"
        cv2.imwrite(str(path), frame)
        self.status.set(f"Saved {path}")

    def _crops(self) -> None:
        _frame, raw, faces, _info = self.engine.state()
        if raw is None or not faces:
            self.status.set("No faces to save in the current frame.")
            return
        n = facetool.save_crops(raw, faces, self.out_dir, f"crop_{self._stamp()}")
        self.status.set(f"Saved {n} face crop(s) to {self.out_dir}")

    def _toggle_record(self) -> None:
        if self.engine.recording:
            self.engine.stop_recording()
            self.rec_btn.config(text="Start recording")
            self.status.set("Recording saved.")
            return
        path = filedialog.asksaveasfilename(
            title="Record to...", defaultextension=".mp4", initialfile=f"recording_{self._stamp()}.mp4",
            initialdir=str(self.out_dir) if self.out_dir.exists() else None,
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")],
        )
        if path:
            self.engine.start_recording(path)
            self.rec_btn.config(text="Stop recording")

    # ---- preview ----------------------------------------------------------
    def _render(self, frame: Optional[np.ndarray]) -> None:
        c = self.canvas
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 10 or ch < 10:
            return
        c.delete("all")
        if frame is None:
            c.create_text(cw // 2, ch // 2, text="No source", fill="#888888", font=("TkDefaultFont", 16))
            return
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        small = cv2.resize(frame, (nw, nh), interpolation=interp)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        c.create_image(cw // 2, ch // 2, image=self._photo)

    def _poll(self) -> None:
        frame, _raw, _faces, info = self.engine.state()
        if frame is not None and frame is not self._shown:
            self._shown = frame
            self._render(frame)
        if self.engine.running and info:
            rec = "  |  REC" if self.engine.recording else ""
            self.status.set(info + rec)
        if self.engine.notice:
            msg, self.engine.notice = self.engine.notice, None
            self.emo_var.set(False)  # the engine already switched the feature off
            messagebox.showwarning(APP_TITLE, msg)
        if self.engine.error:
            msg, self.engine.error = self.engine.error, None
            self._stop_source()
            messagebox.showerror(APP_TITLE, msg)
        elif self.engine.finished and not self.engine.running and self.stop_btn.instate(["!disabled"]):
            self.stop_btn.config(state="disabled")
            self.rec_btn.config(text="Start recording", state="disabled")
            self.status.set("Finished. Open another source, or save a snapshot of the last frame.")
        self.root.after(self.POLL_MS, self._poll)

    def _close(self) -> None:
        self.engine.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    if not GUI_AVAILABLE:
        print(
            f"The GUI needs Tkinter and Pillow ({GUI_IMPORT_ERROR}).\n"
            "  Pillow:  pip install pillow\n"
            "  Tkinter: bundled on Windows/macOS; on Ubuntu/Debian run `sudo apt install python3-tk`.",
            file=sys.stderr,
        )
        return 1
    App().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
