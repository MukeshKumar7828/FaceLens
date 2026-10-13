#!/usr/bin/env python3
"""facetool - face detection for webcams, videos and images.

Detectors:
  yunet  (default) OpenCV's modern DNN detector. Handles angles and partial
         occlusion, returns confidence scores and 5 facial landmarks.
         The model (~230 KB) is downloaded automatically on first use.
  haar   Classic Haar cascade (fast, no download, less accurate).

Examples:
  python facetool.py                       # webcam
  python facetool.py photo.jpg             # image -> photo_out.jpg
  python facetool.py clip.mp4 -o out.mp4   # video -> annotated video
  python facetool.py --blur --landmarks    # privacy blur + landmarks, webcam

Live keys: q / Esc quit | s snapshot | b toggle blur | l toggle landmarks
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GREEN = (0, 255, 0)
RED = (0, 0, 255)


@dataclass
class Face:
    box: Tuple[int, int, int, int]  # x, y, w, h
    score: Optional[float] = None
    landmarks: Optional[np.ndarray] = None  # (N, 2) int points


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def ensure_yunet_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet model to {MODEL_PATH} ...", file=sys.stderr)
    urllib.request.urlretrieve(YUNET_URL, MODEL_PATH)
    return MODEL_PATH


class YuNetDetector:
    name = "yunet"

    def __init__(self, score_threshold: float = 0.7, nms_threshold: float = 0.3):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "This OpenCV build has no FaceDetectorYN. "
                "Upgrade with: pip install -U opencv-python (4.5.4+ required)."
            )
        self.det = cv2.FaceDetectorYN.create(
            str(ensure_yunet_model()), "", (320, 320), score_threshold, nms_threshold, 5000
        )

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, result = self.det.detect(frame)
        faces: List[Face] = []
        if result is None:
            return faces
        for r in result:
            x, y, bw, bh = (int(v) for v in r[:4])
            landmarks = r[4:14].reshape(5, 2).astype(int)  # eyes, nose, mouth corners
            faces.append(Face((x, y, bw, bh), float(r[14]), landmarks))
        return faces


class HaarDetector:
    name = "haar"

    def __init__(self, min_size: int = 60):
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} has no CascadeClassifier (removed in OpenCV 5). "
                "Install OpenCV 4.x: pip install 'opencv-python>=4.8,<5'"
            )
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.eye = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        if self.face.empty() or self.eye.empty():
            raise RuntimeError("Could not load Haar cascade files from OpenCV's data folder.")
        self.min_size = min_size

    def detect(self, frame: np.ndarray) -> List[Face]:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # even out lighting
        boxes = self.face.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_size, self.min_size)
        )
        faces: List[Face] = []
        for (x, y, w, h) in boxes:
            roi = gray[y : y + h // 2 + h // 8, x : x + w]  # eyes live in the upper half
            eyes = self.eye.detectMultiScale(roi, 1.1, 5, minSize=(w // 8, w // 8))
            pts = np.array([[x + ex + ew // 2, y + ey + eh // 2] for ex, ey, ew, eh in eyes[:2]])
            faces.append(Face((x, y, w, h), None, pts if len(pts) else None))
        return faces


def make_detector(kind: str, threshold: float):
    """Build the requested detector; YuNet falls back to Haar. Raises RuntimeError if none work."""
    if kind == "haar":
        return HaarDetector()
    try:
        return YuNetDetector(threshold)
    except Exception as exc:  # no network / old OpenCV -> fall back gracefully
        print(f"[warn] YuNet unavailable ({exc}); falling back to Haar.", file=sys.stderr)
        try:
            return HaarDetector()
        except Exception as exc2:
            raise RuntimeError(
                f"No detector could start. YuNet: {exc}. Haar: {exc2}. "
                f"To use YuNet offline, download the model manually to {MODEL_PATH}"
            ) from None


# --------------------------------------------------------------------------- #
# Drawing / effects
# --------------------------------------------------------------------------- #
def clamp_box(box, shape):
    x, y, w, h = box
    H, W = shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return x0, y0, x1, y1


def blur_face(img: np.ndarray, box) -> None:
    x0, y0, x1, y1 = clamp_box(box, img.shape)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    k = max(3, (max(x1 - x0, y1 - y0) // 3) | 1)  # odd kernel scaled to face size
    img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (k, k), 0)


def annotate(img: np.ndarray, faces: List[Face], blur: bool, landmarks: bool) -> None:
    for f in faces:  # blur first so boxes/labels stay crisp
        if blur:
            blur_face(img, f.box)
    for f in faces:
        x, y, w, h = f.box
        cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
        if f.score is not None:
            cv2.putText(img, f"{f.score:.2f}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
        if landmarks and f.landmarks is not None:
            for px, py in f.landmarks:
                cv2.circle(img, (int(px), int(py)), 3, RED, -1)


def hud(img: np.ndarray, text: str) -> None:
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.6
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(img, (0, 0), (tw + 20, th + base + 14), (0, 0, 0), -1)  # dark bar for contrast
    cv2.putText(img, text, (10, th + 7), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def save_crops(frame: np.ndarray, faces: List[Face], out_dir: Path, tag: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, f in enumerate(faces):
        x, y, w, h = f.box
        m = int(0.15 * max(w, h))  # a little margin around the face
        x0, y0, x1, y1 = clamp_box((x - m, y - m, w + 2 * m, h + 2 * m), frame.shape)
        if x1 > x0 and y1 > y0:
            cv2.imwrite(str(out_dir / f"{tag}_face{i}.jpg"), frame[y0:y1, x0:x1])
            saved += 1
    return saved


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_image(path: Path, detector, args) -> int:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return 1
    raw = frame.copy()
    faces = detector.detect(frame)
    annotate(frame, faces, args.blur, args.landmarks)
    hud(frame, f"{len(faces)} face(s) [{detector.name}]")
    out = Path(args.output) if args.output else path.with_name(f"{path.stem}_out{path.suffix}")
    cv2.imwrite(str(out), frame)
    print(f"{len(faces)} face(s) found -> {out}")
    if args.save_faces:
        n = save_crops(raw, faces, Path(args.save_faces), path.stem)
        print(f"Saved {n} face crop(s) to {args.save_faces}")
    if not args.no_display:
        cv2.imshow("facetool", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_stream(source, is_webcam: bool, detector, args) -> int:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}", file=sys.stderr)
        return 1

    writer = None
    blur, landmarks = args.blur, args.landmarks
    fps, prev, frames, snaps = 0.0, time.time(), 0, 0
    snap_dir = Path(args.save_faces or "snapshots")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:  # camera error or end of video
                break
            raw = frame.copy() if (args.save_faces or not args.no_display) else None
            faces = detector.detect(frame)
            annotate(frame, faces, blur, landmarks)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6)
            prev = now
            hud(frame, f"{len(faces)} face(s) | {fps:4.1f} FPS [{detector.name}]")

            if args.output:
                if writer is None:
                    h, w = frame.shape[:2]
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(
                        args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
                    )
                writer.write(frame)
            if args.save_faces and faces:  # continuous capture when a folder is given
                save_crops(raw, faces, snap_dir, f"f{frames:06d}")

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
            if args.no_display:
                continue

            cv2.imshow("facetool", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("b"):
                blur = not blur
            elif key == ord("l"):
                landmarks = not landmarks
            elif key == ord("s") and raw is not None:
                snap_dir.mkdir(parents=True, exist_ok=True)
                name = snap_dir / f"snapshot_{int(time.time())}_{snaps}.jpg"
                cv2.imwrite(str(name), frame)
                snaps += 1
                print(f"Saved {name}")
    except KeyboardInterrupt:
        pass
    finally:  # always release the camera / files
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Processed {frames} frame(s).")
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Face detection for webcam, video and images.")
    p.add_argument("source", nargs="?", default="0",
                   help="webcam index (default 0), video file, or image file")
    p.add_argument("-d", "--detector", choices=["yunet", "haar"], default="yunet")
    p.add_argument("-t", "--threshold", type=float, default=0.7,
                   help="YuNet confidence threshold (default 0.7)")
    p.add_argument("-o", "--output", help="write annotated image/video to this path")
    p.add_argument("--blur", action="store_true", help="blur detected faces")
    p.add_argument("--landmarks", action="store_true", help="draw eye/nose/mouth landmarks")
    p.add_argument("--save-faces", metavar="DIR", help="save cropped faces (and snapshots) here")
    p.add_argument("--no-display", action="store_true", help="headless: don't open a window")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        detector = make_detector(args.detector, args.threshold)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    src = args.source
    if src.isdigit():
        return run_stream(int(src), True, detector, args)
    path = Path(src)
    if not path.exists():
        print(f"File not found: {src}", file=sys.stderr)
        return 1
    if path.suffix.lower() in IMAGE_EXTS:
        return run_image(path, detector, args)
    return run_stream(str(path), False, detector, args)


if __name__ == "__main__":
    sys.exit(main())
