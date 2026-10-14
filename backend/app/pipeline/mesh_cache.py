from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SMPL_PARAM_KEYS = (
    "body_pose",
    "betas",
    "global_orient",
    "transl",
)


def _require_keys(keys: set[str], required: list[str], context: str) -> None:
    missing = [k for k in required if k not in keys]
    if missing:
        raise ValueError(f"{context}: missing required keys: {', '.join(missing)}")


def _archive_to_dict(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {k: np.asarray(archive[k]) for k in archive.files}


def load_smpl_sidecar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        _require_keys(keys, ["n_tracks"], "smpl sidecar")
        n_tracks = int(archive["n_tracks"])
        per_track: list[str] = []
        for i in range(n_tracks):
            per_track.extend(
                [
                    f"n_frames_{i}",
                    f"params_global_{i}",
                    f"params_incam_{i}",
                    f"K_{i}",
                ]
            )
        _require_keys(keys, per_track, "smpl sidecar")
        return _archive_to_dict(archive)


def load_mesh_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        _require_keys(keys, ["faces", "n_tracks"], "mesh cache")
        faces = np.asarray(archive["faces"])
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("mesh cache: faces must have shape (F, 3)")
        n_tracks = int(archive["n_tracks"])
        per_track: list[str] = []
        for i in range(n_tracks):
            per_track.extend(
                [
                    f"verts_world_{i}",
                    f"verts_incam_{i}",
                    f"times_{i}",
                    f"mask_{i}",
                ]
            )
        _require_keys(keys, per_track, "mesh cache")
        return _archive_to_dict(archive)
