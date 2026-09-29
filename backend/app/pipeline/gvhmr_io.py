"""Parse the GVHMR adapter JSON. No torch import — safe in the app venv."""

from __future__ import annotations

from app.schemas import PoseFrame


def _as_pose(frame: dict) -> PoseFrame:
    joints = frame.get("joints3d") or []
    kpts = frame.get("keypoints") or []
    if len(joints) != 24 or any(len(j) != 3 for j in joints):
        raise ValueError("joints3d must be 24x3")
    if len(kpts) != 24 or any(len(k) != 3 for k in kpts):
        raise ValueError("keypoints must be 24x3")
    return PoseFrame(
        time=float(frame["time"]),
        joints3d=[[float(v) for v in j] for j in joints],
        keypoints=[[float(v) for v in k] for k in kpts],
    )


def parse_gvhmr_document(data: dict) -> tuple[list[list[PoseFrame]], int]:
    raw_tracks = data.get("tracks") or []
    parsed: list[list[PoseFrame]] = []
    for track in raw_tracks:
        frames = [_as_pose(f) for f in (track.get("frames") or [])]
        frames.sort(key=lambda p: p.time)
        if frames:
            parsed.append(frames)
    parsed.sort(key=lambda frames: (-len(frames), frames[0].time))
    ignored = int(data.get("ignoredTracks") or 0)
    return parsed, ignored
