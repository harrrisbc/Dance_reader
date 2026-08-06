"""Background video decode + pose overlay for smooth desktop preview."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from app.pipeline.pose_track import COCO_EDGES

PREVIEW_MAX_W = 960
MAX_SEQUENTIAL_READS = 8
PLAYHEAD_HINT_HZ = 30  # worker emit cadence target while playing


@dataclass
class OverlayPose:
    time: float
    keypoints: list[list[float]]


class VideoDecodeWorker(QThread):
    """Owns VideoCapture; UI thread only paints and throttles chart playheads."""

    frame_ready = Signal(object, int, float, bool)  # QImage, idx, time_s, playing
    opened = Signal(float, int)  # fps, frame_count
    ended = Signal()
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cmds: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._alive = True
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._frame_count = 0
        self._pos = -1
        self._playing = False
        self._play_wall0 = 0.0
        self._play_frame0 = 0
        self._overlay: list[OverlayPose] = []
        self._draw_skeleton = True
        self._skel_style = "skeleton"  # off | joints | skeleton
        self._path: Path | None = None

    def request(self, cmd: str, payload: Any = None) -> None:
        self._cmds.put((cmd, payload))

    def shutdown(self) -> None:
        self.request("shutdown")
        self.wait(3000)

    def run(self) -> None:
        while self._alive:
            try:
                while True:
                    cmd, payload = self._cmds.get_nowait()
                    self._handle(cmd, payload)
            except queue.Empty:
                pass

            if not self._alive:
                break

            if self._playing and self._cap is not None:
                self._advance_clock()
            else:
                self.msleep(8)

        self._release()

    def _handle(self, cmd: str, payload: Any) -> None:
        if cmd == "shutdown":
            self._playing = False
            self._alive = False
            return
        if cmd == "open":
            self._open(Path(payload))
            return
        if cmd == "close":
            self._playing = False
            self._release()
            return
        if cmd == "play":
            if self._cap is None or self._frame_count <= 0:
                return
            start = int(payload) if payload is not None else max(0, self._pos)
            start = max(0, min(start, self._frame_count - 1))
            self._play_frame0 = start
            self._play_wall0 = time.perf_counter()
            self._playing = True
            # Emit current frame immediately
            self._emit_at(start, force_seek=True, playing=True)
            return
        if cmd == "pause":
            was = self._playing
            self._playing = False
            idx = int(payload) if payload is not None else max(0, self._pos)
            self._emit_at(idx, force_seek=True, playing=False)
            if was:
                pass
            return
        if cmd == "seek":
            self._playing = False
            idx = int(payload)
            self._emit_at(idx, force_seek=True, playing=False)
            return
        if cmd == "set_overlay":
            poses, style = payload
            self._overlay = poses or []
            if isinstance(style, bool):
                self._skel_style = "skeleton" if style else "off"
            else:
                self._skel_style = str(style or "skeleton")
            self._draw_skeleton = self._skel_style != "off"
            return

    def _open(self, path: Path) -> None:
        self._playing = False
        self._release()
        self._path = path
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.error.emit(f"Cannot open: {path}")
            return
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._pos = -1
        self.opened.emit(self._fps, self._frame_count)
        self._emit_at(0, force_seek=True, playing=False)

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._pos = -1

    def _advance_clock(self) -> None:
        assert self._cap is not None
        fps = max(self._fps, 1e-6)
        elapsed = time.perf_counter() - self._play_wall0
        target = self._play_frame0 + int(round(elapsed * fps))
        if target >= self._frame_count - 1:
            self._playing = False
            self._emit_at(self._frame_count - 1, force_seek=True, playing=False)
            self.ended.emit()
            return

        last = self._pos
        if target <= last:
            # Ahead of schedule — wait a bit
            self.msleep(4)
            return

        skip = target - last
        if skip > MAX_SEQUENTIAL_READS:
            self._emit_at(target, force_seek=True, playing=True)
        else:
            # Sequential reads to drop intermediate frames without random seek
            frame = None
            idx = last
            for _ in range(skip):
                ok, frame = self._cap.read()
                if not ok:
                    self._playing = False
                    self.ended.emit()
                    return
                idx += 1
            self._pos = idx
            if frame is not None:
                self._publish(frame, idx, playing=True)

        # Pace roughly to fps so we don't spin when decode is fast
        self.msleep(max(1, int(1000.0 / (fps * 1.5))))

    def _emit_at(self, frame_idx: int, *, force_seek: bool, playing: bool) -> None:
        if self._cap is None or self._frame_count <= 0:
            return
        frame_idx = max(0, min(frame_idx, self._frame_count - 1))
        need_seek = (
            force_seek
            or self._pos < 0
            or frame_idx < self._pos
            or frame_idx > self._pos + 1
        )
        if need_seek:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self._cap.read()
            if not ok:
                return
            self._pos = frame_idx
        elif frame_idx == self._pos + 1:
            ok, frame = self._cap.read()
            if not ok:
                return
            self._pos = frame_idx
        else:
            # Same frame — re-seek to refresh with current overlay settings
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self._cap.read()
            if not ok:
                return
            self._pos = frame_idx
        self._publish(frame, frame_idx, playing=playing)

    def _publish(self, frame: np.ndarray, frame_idx: int, *, playing: bool) -> None:
        t = frame_idx / max(self._fps, 1e-6)

        # Downscale first (cheaper overlay + paint)
        h0, w0 = frame.shape[:2]
        if w0 > PREVIEW_MAX_W:
            scale = PREVIEW_MAX_W / w0
            frame = cv2.resize(
                frame,
                (PREVIEW_MAX_W, max(1, int(h0 * scale))),
                interpolation=cv2.INTER_AREA,
            )

        draw = self._draw_skeleton and bool(self._overlay)

        if draw:
            frame = self._draw_poses(frame, t)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # Contiguous copy so QImage owns stable bytes across threads
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.frame_ready.emit(qimg, frame_idx, t, playing)

    def _draw_poses(self, frame: np.ndarray, t: float) -> np.ndarray:
        if not self._overlay or self._skel_style == "off":
            return frame
        out = frame
        h, w = out.shape[:2]
        pose = min(self._overlay, key=lambda p: abs(p.time - t))
        if abs(pose.time - t) > 0.35:
            return out
        color = (80, 180, 255)
        pts: list[tuple[int, int] | None] = []
        for kp in pose.keypoints:
            if len(kp) < 3 or kp[2] < 0.3:
                pts.append(None)
                continue
            x, y = int(kp[0] * w), int(kp[1] * h)
            pts.append((x, y))

        if self._skel_style == "skeleton":
            layer = out.copy()
            for a, b in COCO_EDGES:
                if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                    cv2.line(layer, pts[a], pts[b], color, 2, lineType=cv2.LINE_AA)
            cv2.addWeighted(layer, 0.65, out, 0.35, 0, out)

        for p in pts:
            if p:
                cv2.circle(out, p, 3, color, -1, lineType=cv2.LINE_AA)
        return out
