from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.pipeline.pose_track import TrackBuffer


def write_overlay(
    video_path: str,
    output_path: Path,
    buffers: dict[int, TrackBuffer],
    model: YOLO,
    *,
    stride: int = 2,
    progress_cb=None,
) -> Path:
    """Render annotated preview with Supervision (boxes + skeleton)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(fps / stride, 1.0), (width, height))

    tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps / max(stride, 1)))))
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    # VertexAnnotator / EdgeAnnotator for pose if available
    try:
        vertex_annotator = sv.VertexAnnotator(radius=3)
        edge_annotator = sv.EdgeAnnotator(thickness=2)
        has_pose_ann = True
    except AttributeError:
        vertex_annotator = None
        edge_annotator = None
        has_pose_ann = False

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        from app.pipeline.pose_track import resolve_device

        result = model(frame, verbose=False, device=resolve_device())[0]
        detections = sv.Detections.from_ultralytics(result)
        if len(detections):
            detections = tracker.update_with_detections(detections)
        else:
            detections = tracker.update_with_detections(sv.Detections.empty())

        labels: list[str] = []
        if len(detections) and detections.tracker_id is not None:
            labels = [f"舞者 {int(t)}" for t in detections.tracker_id]

        annotated = frame.copy()
        if len(detections):
            annotated = box_annotator.annotate(annotated, detections)
            if labels:
                annotated = label_annotator.annotate(annotated, detections, labels)

            if has_pose_ann and result.keypoints is not None:
                try:
                    kpts = sv.KeyPoints.from_ultralytics(result)
                    if edge_annotator is not None:
                        annotated = edge_annotator.annotate(annotated, kpts)
                    if vertex_annotator is not None:
                        annotated = vertex_annotator.annotate(annotated, kpts)
                except Exception:
                    pass

        writer.write(annotated)
        frame_idx += 1
        if progress_cb and total > 0:
            progress_cb(0.85 + 0.12 * (frame_idx / total), "產生預覽影片")

    cap.release()
    writer.release()
    return output_path
