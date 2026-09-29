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
