from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from app.pipeline.mesh_cache import persist_sidecar_with_project, sha256_file
from app.pipeline.metrics import enrich_metrics
from app.pipeline.pose_track import TrackBuffer, TrackedPose
from app.schemas import AnalysisResult

APP_VERSION = "0.1.0"


class DanceProject(BaseModel):
    """User-facing project file (.dance.json)."""

    version: int = 1
    video_path: str
    analysis: AnalysisResult
    max_people: int | None = None
    pose_model: str | None = None  # e.g. yolo11s-pose.pt
    saved_at: str = ""
    app_version: str = APP_VERSION


def save_project(
    path: Path,
    project: DanceProject,
    *,
    smpl_sidecar_src: Path | None = None,
) -> None:
    if not project.saved_at:
        project.saved_at = datetime.now(timezone.utc).isoformat()
    path = Path(path)
    if path.suffix.lower() != ".json" and not str(path).endswith(".dance.json"):
        path = path.with_suffix(".dance.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if smpl_sidecar_src is not None and Path(smpl_sidecar_src).is_file():
        sha = project.analysis.smplSidecarSha256 or sha256_file(smpl_sidecar_src)
        rel = persist_sidecar_with_project(path, smpl_sidecar_src, sha)
        dest = path.parent / rel
        project.analysis.smplSidecar = rel
        project.analysis.smplSidecarSha256 = sha256_file(dest)
    path.write_text(
        json.dumps(project.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_project(path: Path) -> DanceProject:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    project = DanceProject.model_validate(data)
    # Old .dance.json may lack space metrics; rebuild all curves from poses.
    enrich_metrics(project.analysis)
    return project


def _parse_track_id(actor_id: str) -> int:
    # "track_0" → 0
    if actor_id.startswith("track_"):
        try:
            return int(actor_id.split("_", 1)[1])
        except ValueError:
            pass
    return abs(hash(actor_id)) % 10_000


def buffers_from_analysis(result: AnalysisResult) -> dict[str, TrackBuffer]:
    """Rebuild TrackBuffers from stored poses so 搵相似 works after Load."""
    out: dict[str, TrackBuffer] = {}
    for actor in result.actors:
        tid = _parse_track_id(actor.id)
        buf = TrackBuffer(track_id=tid)
        for pf in actor.poses:
            kpts = np.asarray(pf.keypoints, dtype=np.float32)
            if kpts.ndim != 2 or kpts.shape[1] < 3:
                continue
            if kpts.shape[1] > 3:
                kpts = kpts[:, :3]
            buf.poses.append(
                TrackedPose(
                    time=float(pf.time),
                    track_id=tid,
                    keypoints=kpts,
                    xyxy=np.zeros(4, dtype=np.float32),
                )
            )
        if buf.poses:
            out[actor.id] = buf
    return out
