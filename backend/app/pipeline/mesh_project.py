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


# Sum of BGR channels below this is treated as empty overlay (bake uses black bg).
MESH_OVERLAY_BLACK_THRESHOLD = 8


def blend_mesh_overlay(
    frame: np.ndarray,
    overlay: np.ndarray,
    *,
    alpha: float = 0.45,
    black_threshold: int = MESH_OVERLAY_BLACK_THRESHOLD,
) -> np.ndarray:
    """Alpha-composite a mesh-only overlay (black background) onto a video frame."""
    if frame.shape[:2] != overlay.shape[:2]:
        raise ValueError("frame and overlay must have the same height and width")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be BGR (H, W, 3)")
    if overlay.ndim != 3 or overlay.shape[2] != 3:
        raise ValueError("overlay must be BGR (H, W, 3)")

    strength = overlay.astype(np.int32).sum(axis=2)
    mask = strength > int(black_threshold)
    if not np.any(mask):
        return frame

    a = float(alpha)
    out = frame.copy()
    base = frame[mask].astype(np.float32)
    top = overlay[mask].astype(np.float32)
    out[mask] = (base * (1.0 - a) + top * a).astype(np.uint8)
    return out
