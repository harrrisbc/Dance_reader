from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MetricPoint(BaseModel):
    time: float
    energy: float  # relative to this dancer's peak (0–1)
    energy_abs: float = 0.0  # smoothed Σ(v/S)² before peak-norm (torso²/s²)
    speed: float
    reach: float = 0.0
    level: float = 0.0
    travel: float = 0.0
    directness: float = 0.0  # 1 = straight path, 0 = winding; NaN when still
    # coco2d: screen 0–1 (COM x / foot y). smpl3d: pelvis world X / Z in metres.
    pos_x: float = 0.0
    pos_y: float = 0.0
    # Supporting-foot floor: screen 0–1 (coco2d) or world X/Z metres (smpl3d).
    pos_foot_x: float = 0.0
    pos_foot_y: float = 0.0


class PoseFrame(BaseModel):
    time: float
    keypoints: list[list[float]]  # [[x, y, conf], ...] normalized 0–1
    joints3d: list[list[float]] | None = None  # SMPL-24 meters, gvhmr_world; None on old files


class Actor(BaseModel):
    id: str
    label: str
    metrics: list[MetricPoint] = Field(default_factory=list)
    poses: list[PoseFrame] = Field(default_factory=list)


class MotifOccurrence(BaseModel):
    id: str
    actorId: str
    start: float
    end: float
    score: float | None = None  # display-only 0..1 match; optional for auto types
    tempo_ratio: float | None = None  # hit_dur/query_dur; >1 = slower; Find-similar only
    is_query: bool = False  # True for the Find-similar seed span


class MotifType(BaseModel):
    id: str
    name: str
    color: str
    occurrences: list[MotifOccurrence] = Field(default_factory=list)
    # Mean pairwise raw DTW within the type (auto-discovery); None for Find Similar
    meanIntraDist: float | None = None


class ManualMark(BaseModel):
    id: str
    start: float
    end: float
    label: str
    actorId: str | None = None
    motifTypeId: str | None = None


class AnalysisResult(BaseModel):
    version: Literal[1] = 1
    duration: float
    fps: float
    frame_width: int = 0
    frame_height: int = 0
    actors: list[Actor] = Field(default_factory=list)
    motifTypes: list[MotifType] = Field(default_factory=list)
    manualMarks: list[ManualMark] = Field(default_factory=list)
    # Last detect summary, e.g. "18 phrases · 4 clusters · 0 passed similarity 0.60"
    motifDiag: str | None = None
    poseBackend: str = "yolo"
    coordFrame: str = ""
    staticCamera: bool = False
    ignoredTracks: int = 0
    smplSidecar: str | None = None
    smplSidecarSha256: str | None = None
    meshCache: str | None = None
    meshCacheSha256: str | None = None
    # "smpl3d" when any actor used joints3d metrics; else "coco2d"
    metricsSpace: str = "coco2d"


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"]
    progress: float = 0.0
    message: str = ""
    result: AnalysisResult | None = None
    error: str | None = None
    overlayAvailable: bool = False


class RenameMotifBody(BaseModel):
    name: str


class AnalyzePathBody(BaseModel):
    path: str
    stride: int = 1
    model: str = "yolo11s-pose.pt"
    write_overlay: bool = False


def analysis_to_dict(result: AnalysisResult) -> dict[str, Any]:
    return result.model_dump()
