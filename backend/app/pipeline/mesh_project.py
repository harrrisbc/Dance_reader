"""Project in-camera SMPL verts to pixel coords (numpy only, no Qt)."""

from __future__ import annotations

import numpy as np


def project_verts_to_pixels(
    verts_incam: np.ndarray,
    K: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Return (V, 2) float32 pixel u,v; NaN where z <= 0 or invalid."""
    del width, height  # callers pass source frame size for future clipping
    v = np.asarray(verts_incam, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("verts_incam must have shape (V, 3)")
    k = np.asarray(K, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError("K must have shape (3, 3)")

    uv = np.full((v.shape[0], 2), np.nan, dtype=np.float32)
    z = v[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return uv
    proj = (k @ v[valid].T).T
    uv[valid] = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
    return uv


def face_edge_pairs(faces: np.ndarray, *, stride: int = 8) -> np.ndarray:
    """Unique mesh edges from a strided subset of triangular faces -> (E, 2) int."""
    f = np.asarray(faces, dtype=np.int32)
    if f.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    step = max(1, int(stride))
    f = f[::step]
    edges = np.vstack(
        [
            f[:, [0, 1]],
            f[:, [1, 2]],
            f[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)
