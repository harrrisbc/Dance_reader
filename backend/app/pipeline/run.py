from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.jobs import JobStore
from app.pipeline.gvhmr_run import run_gvhmr_export
from app.pipeline.metrics import _actor_to_buffer, compute_metrics, enrich_metrics, frame_aspect
from app.pipeline.motifs import LAST_DETECT_STATS, detect_motif_types
from app.pipeline.pose_track import (
    DEFAULT_MAX_PEOPLE,
    DEFAULT_MODEL,
    DEFAULT_STRIDE,
    TrackBuffer,
    TrackedPose,
    load_model,  # kept so tests can assert the YOLO path is not called
    track_video,  # kept so tests can assert the YOLO path is not called
)
from app.schemas import Actor, AnalysisResult, PoseFrame


def _actor_label(index: int) -> str:
    return f"Dancer {index}"


def _gvhmr_tracks_to_actors(
    tracks: list[list[PoseFrame]],
    *,
    aspect: float,
) -> tuple[list[Actor], dict[str, TrackBuffer], dict[int, TrackBuffer]]:
    """Store SMPL-24 poses. Motif buffers are the COCO-17 bridge."""
    actors: list[Actor] = []
    keyed: dict[str, TrackBuffer] = {}
    buffers: dict[int, TrackBuffer] = {}
    idx = 1
    for tid, track_poses in enumerate(tracks):
        if not track_poses:
            continue
        raw = TrackBuffer(track_id=tid)
        for pf in track_poses:
            raw.poses.append(
                TrackedPose(
                    time=float(pf.time),
                    track_id=tid,
                    keypoints=np.asarray(pf.keypoints, dtype=np.float32),
                    xyxy=np.zeros(4, dtype=np.float32),
                )
            )
        actor_id = f"track_{tid}"
        bridged = [
            PoseFrame(
                time=p.time,
                keypoints=[[float(x), float(y), float(c)] for x, y, c in p.keypoints.tolist()],
            )
            for p in raw.poses
        ]
        coco = _actor_to_buffer(actor_id, bridged)
        actor = Actor(
            id=actor_id,
            label=_actor_label(idx),
            metrics=compute_metrics(coco, aspect=aspect),
            poses=bridged,
        )
        actor.poses = list(track_poses)
        actors.append(actor)
        keyed[actor_id] = coco
        buffers[tid] = coco
        idx += 1
    return actors, keyed, buffers


def analyze_video(
    video_path: Path,
    *,
    stride: int = DEFAULT_STRIDE,
    model_name: str = DEFAULT_MODEL,
    max_people: int = DEFAULT_MAX_PEOPLE,
    write_overlay_video: bool = False,
    artifacts_dir: Path | None = None,
    job_id: str | None = None,
    progress_cb=None,
) -> tuple[AnalysisResult, Path | None, dict[int, TrackBuffer]]:
    """In-process analysis for desktop UI (no JobStore required)."""

    def progress(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(p, msg)

    progress(0.1, "Running GVHMR…")
    tracks, ignored = run_gvhmr_export(video_path, max_people=min(max_people, 2))

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    duration = frame_count / fps if fps > 0 else 0.0

    progress(0.65, "Computing energy / speed / space metrics")
    aspect = frame_aspect(int(frame_w), int(frame_h))
    actors, keyed, buffers = _gvhmr_tracks_to_actors(tracks, aspect=aspect)

    progress(0.72, "Detecting motif types")
    motif_types = detect_motif_types(keyed, aspect=aspect)
    motif_diag = LAST_DETECT_STATS.detail()

    result = AnalysisResult(
        version=1,  # MetricPoint may gain fields; defaults keep old JSON loadable
        duration=float(duration),
        fps=float(fps),
        frame_width=int(frame_w),
        frame_height=int(frame_h),
        actors=actors,
        motifTypes=motif_types,
        manualMarks=[],
        motifDiag=motif_diag,
        poseBackend="gvhmr",
        coordFrame="gvhmr_world",
        staticCamera=True,
        ignoredTracks=int(ignored),
    )
    enrich_metrics(result)

    overlay_path = None
    if artifacts_dir is not None and job_id is not None:
        out_json = artifacts_dir / job_id / "analysis.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    progress(1.0, "Analysis complete")
    return result, overlay_path, buffers


def run_pipeline(
    job_id: str,
    video_path: Path,
    artifacts_dir: Path,
    store: JobStore,
    *,
    stride: int = DEFAULT_STRIDE,
    model_name: str = DEFAULT_MODEL,
    max_people: int = DEFAULT_MAX_PEOPLE,
    write_overlay_video: bool = False,
) -> None:
    def progress(p: float, msg: str) -> None:
        store.update(job_id, status="running", progress=p, message=msg)

    try:
        store.update(job_id, status="running", progress=0.02, message="Running GVHMR…")
        result, overlay_path, _buffers = analyze_video(
            video_path,
            stride=stride,
            model_name=model_name,
            max_people=max_people,
            write_overlay_video=write_overlay_video,
            artifacts_dir=artifacts_dir,
            job_id=job_id,
            progress_cb=progress,
        )
        done_msg = "Analysis complete"
        if write_overlay_video:
            done_msg = "Analysis complete. Overlay video is not produced in this version."
        store.update(
            job_id,
            status="done",
            progress=1.0,
            message=done_msg,
            result=result,
            overlay_path=overlay_path,
        )
    except Exception as exc:
        store.update(
            job_id,
            status="error",
            progress=1.0,
            message="Analysis failed",
            error=str(exc),
        )
