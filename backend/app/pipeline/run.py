from __future__ import annotations

import json
from pathlib import Path

from app.jobs import JobStore
from app.pipeline.annotate import write_overlay
from app.pipeline.metrics import compute_metrics, enrich_metrics, frame_aspect
from app.pipeline.motifs import LAST_DETECT_STATS, detect_motif_types
from app.pipeline.pose_track import (
    DEFAULT_MAX_PEOPLE,
    DEFAULT_MODEL,
    DEFAULT_STRIDE,
    TrackBuffer,
    load_model,
    track_video,
)
from app.schemas import Actor, AnalysisResult, PoseFrame


def _actor_label(index: int) -> str:
    return f"Dancer {index}"


def _buffers_to_actors(
    buffers: dict[int, TrackBuffer],
    *,
    aspect: float,
) -> tuple[list[Actor], dict[str, TrackBuffer]]:
    # Stable order by track id; drop very short tracks
    sorted_ids = sorted(buffers.keys())
    actors: list[Actor] = []
    keyed: dict[str, TrackBuffer] = {}
    idx = 1
    for tid in sorted_ids:
        buf = buffers[tid]
        if len(buf.poses) < 4:
            continue
        actor_id = f"track_{tid}"
        poses = [
            PoseFrame(
                time=p.time,
                keypoints=[[float(x), float(y), float(c)] for x, y, c in p.keypoints.tolist()],
            )
            for p in buf.poses
        ]
        metrics = compute_metrics(buf, aspect=aspect)
        actors.append(
            Actor(
                id=actor_id,
                label=_actor_label(idx),
                metrics=metrics,
                poses=poses,
            )
        )
        keyed[actor_id] = buf
        idx += 1
    return actors, keyed


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

    progress(0.02, "Loading model")
    model = load_model(model_name)

    progress(0.05, "Tracking poses")
    buffers, duration, fps, frame_w, frame_h = track_video(
        str(video_path),
        model,
        stride=stride,
        max_people=max_people,
        progress_cb=progress,
    )

    progress(0.65, "Computing energy / speed / space metrics")
    aspect = frame_aspect(int(frame_w), int(frame_h))
    actors, keyed = _buffers_to_actors(buffers, aspect=aspect)

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

    if write_overlay_video and actors and artifacts_dir is not None and job_id is not None:
        progress(0.85, "Writing preview video")
        overlay_path = artifacts_dir / job_id / "overlay.mp4"
        try:
            write_overlay(
                str(video_path),
                overlay_path,
                buffers,
                model,
                stride=max(stride, 1),
                progress_cb=progress,
            )
        except Exception:
            overlay_path = None

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
    write_overlay_video: bool = True,
) -> None:
    def progress(p: float, msg: str) -> None:
        store.update(job_id, status="running", progress=p, message=msg)

    try:
        store.update(job_id, status="running", progress=0.02, message="Loading model")
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
        store.update(
            job_id,
            status="done",
            progress=1.0,
            message="Analysis complete",
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
