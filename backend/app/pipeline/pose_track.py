from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.pipeline.id_remapper import IdSlotRemapper

# COCO-17 skeleton edges for optional debug
COCO_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

# Defaults for analyze (accuracy over speed on GPU)
DEFAULT_MODEL = "yolo11s-pose.pt"
DEFAULT_STRIDE = 1
DEFAULT_MAX_PEOPLE = 2
KEYPOINT_EMA_ALPHA = 0.35


@dataclass
class TrackedPose:
    time: float
    track_id: int
    keypoints: np.ndarray  # (17, 3) x,y normalized, conf
    xyxy: np.ndarray  # pixel box


@dataclass
class TrackBuffer:
    track_id: int
    poses: list[TrackedPose] = field(default_factory=list)


def _ema_keypoints(
    prev: np.ndarray | None,
    cur: np.ndarray,
    *,
    alpha: float = KEYPOINT_EMA_ALPHA,
) -> np.ndarray:
    """Temporal smooth on XY; keep current confidence."""
    if prev is None or prev.shape != cur.shape:
        return cur.copy()
    out = cur.copy()
    for i in range(out.shape[0]):
        if cur[i, 2] < 0.2 or prev[i, 2] < 0.2:
            continue
        out[i, 0] = alpha * cur[i, 0] + (1.0 - alpha) * prev[i, 0]
        out[i, 1] = alpha * cur[i, 1] + (1.0 - alpha) * prev[i, 1]
    return out


def _bbox_area(xyxy: np.ndarray) -> float:
    return max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))


def _torso_com(norm_kpts: np.ndarray) -> tuple[float, float] | None:
    idxs = [5, 6, 11, 12]
    pts = []
    for i in idxs:
        if i < len(norm_kpts) and float(norm_kpts[i, 2]) >= 0.2:
            pts.append(norm_kpts[i, :2])
    if not pts:
        return None
    arr = np.mean(np.stack(pts, axis=0), axis=0)
    return float(arr[0]), float(arr[1])


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_keypoints(
    tracked_boxes: np.ndarray,
    raw_boxes: np.ndarray,
    raw_kpts: np.ndarray,
) -> list[np.ndarray | None]:
    """For each tracked box, attach best-IoU raw keypoints."""
    out: list[np.ndarray | None] = []
    used: set[int] = set()
    for tb in tracked_boxes:
        best_i = -1
        best_iou = 0.15
        for i, rb in enumerate(raw_boxes):
            if i in used:
                continue
            iou = _box_iou(tb, rb)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0:
            used.add(best_i)
            out.append(raw_kpts[best_i])
        else:
            out.append(None)
    return out


def resolve_device() -> str:
    """Prefer CUDA GPU when available (RTX / CUDA builds)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def load_model(model_name: str = DEFAULT_MODEL, device: str | None = None) -> YOLO:
    model = YOLO(model_name)
    dev = device or resolve_device()
    try:
        model.to(dev if dev != "0" else "cuda:0")
    except Exception:
        pass
    return model


def track_video(
    video_path: str,
    model: YOLO,
    *,
    stride: int = DEFAULT_STRIDE,
    max_people: int = DEFAULT_MAX_PEOPLE,
    progress_cb=None,
    device: str | None = None,
) -> tuple[dict[int, TrackBuffer], float, float, int, int]:
    """
    Run YOLO-Pose + Supervision ByteTrack, remapped to stable slots 0..max_people-1.
    Returns track buffers keyed by slot id, duration (s), fps, width, height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total / fps if fps > 0 and total > 0 else 0.0

    max_people = max(1, min(int(max_people), 4))
    device = device or resolve_device()
    tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps / max(stride, 1)))))
    remapper = IdSlotRemapper(max_slots=max_people, grace_sec=2.5)
    buffers: dict[int, TrackBuffer] = {}
    last_kpt: dict[int, np.ndarray] = {}
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        time_s = frame_idx / fps if fps > 0 else processed / 15.0
        h, w = frame.shape[:2]

        result = model(frame, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(result)

        raw_boxes = (
            detections.xyxy.copy()
            if detections.xyxy is not None and len(detections)
            else np.zeros((0, 4), dtype=np.float32)
        )

        raw_kpts = np.zeros((0, 17, 3), dtype=np.float32)
        if result.keypoints is not None and result.keypoints.data is not None:
            data = result.keypoints.data.cpu().numpy()
            if data.ndim == 3 and data.shape[0] > 0:
                raw_kpts = data.astype(np.float32)

        if len(detections):
            detections = tracker.update_with_detections(detections)
        else:
            detections = tracker.update_with_detections(sv.Detections.empty())

        frame_people: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        if len(detections) and detections.tracker_id is not None:
            matched = _match_keypoints(detections.xyxy, raw_boxes, raw_kpts)
            for i, tid in enumerate(detections.tracker_id):
                if tid is None:
                    continue
                kpt = matched[i] if i < len(matched) else None
                if kpt is None:
                    continue
                xyxy = detections.xyxy[i].copy()
                area = _bbox_area(xyxy)
                frame_people.append((int(tid), kpt, xyxy, area))

        frame_people.sort(key=lambda x: -x[3])
        frame_people = frame_people[:max_people]

        det_for_remap: list[tuple[int, tuple[float, float] | None]] = []
        prepared: list[tuple[int, np.ndarray, np.ndarray]] = []
        for tid_i, kpt, xyxy, _area in frame_people:
            norm = kpt.copy()
            norm[:, 0] = norm[:, 0] / max(w, 1)
            norm[:, 1] = norm[:, 1] / max(h, 1)
            com = _torso_com(norm)
            if com is None:
                cx = float((xyxy[0] + xyxy[2]) * 0.5 / max(w, 1))
                cy = float((xyxy[1] + xyxy[3]) * 0.5 / max(h, 1))
                com = (cx, cy)
            det_for_remap.append((tid_i, com))
            prepared.append((tid_i, norm, xyxy))

        raw_to_slot = remapper.update(det_for_remap, now=time_s)

        for tid_i, norm, xyxy in prepared:
            slot = raw_to_slot.get(tid_i)
            if slot is None:
                continue
            norm = _ema_keypoints(last_kpt.get(slot), norm)
            last_kpt[slot] = norm
            pose = TrackedPose(
                time=time_s,
                track_id=slot,
                keypoints=norm,
                xyxy=xyxy,
            )
            if slot not in buffers:
                buffers[slot] = TrackBuffer(track_id=slot)
            buffers[slot].poses.append(pose)

        processed += 1
        frame_idx += 1
        if progress_cb and total > 0:
            progress_cb(0.05 + 0.55 * (frame_idx / total), "姿態追蹤中")

    cap.release()

    if duration <= 0 and buffers:
        max_t = max(
            (p.time for buf in buffers.values() for p in buf.poses),
            default=0.0,
        )
        duration = max_t + (stride / fps if fps > 0 else 0.1)

    return buffers, duration, fps, width, height
