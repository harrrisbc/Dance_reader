"""SMPL-24 kinematic tree for overlay. COCO-17 stays the old path."""

from __future__ import annotations

from app.pipeline.pose_track import COCO_EDGES

# Parent of each SMPL joint. -1 is the pelvis root.
SMPL_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)


def skeleton_edges(n_keypoints: int) -> list[tuple[int, int]]:
    if n_keypoints == 17:
        return list(COCO_EDGES)
    if n_keypoints == 24:
        return [(i, parent) for i, parent in enumerate(SMPL_PARENTS) if parent >= 0]
    return []


# COCO index → (SMPL index, confidence override or None to keep source conf)
_COCO_FROM_SMPL: tuple[tuple[int, float | None], ...] = (
    (15, None),  # nose ← head
    (15, 0.0),
    (15, 0.0),
    (15, 0.0),
    (15, 0.0),  # eyes, ears
    (16, None),
    (17, None),
    (18, None),
    (19, None),
    (20, None),
    (21, None),
    (1, None),
    (2, None),
    (4, None),
    (5, None),
    (7, None),
    (8, None),
)


def smpl_keypoints_to_coco17(keypoints: list[list[float]]) -> list[list[float]]:
    if len(keypoints) != 24:
        raise ValueError(f"expected 24 SMPL keypoints, got {len(keypoints)}")
    out: list[list[float]] = []
    for smpl_i, conf_override in _COCO_FROM_SMPL:
        x, y, conf = keypoints[smpl_i][:3]
        out.append([float(x), float(y), float(conf if conf_override is None else conf_override)])
    return out
