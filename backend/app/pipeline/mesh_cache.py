from __future__ import annotations

import hashlib
import shutil
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


def mesh_cache_rel_name(dance_json_path: Path) -> str:
    return f"{Path(dance_json_path).stem}.dance.mesh.npz"


def persist_mesh_cache_with_project(
    dance_json_path: Path,
    src_mesh: Path,
    sha256_hex: str | None = None,
) -> str:
    """Copy baked mesh cache beside the project JSON; return relative filename."""
    dance_json_path = Path(dance_json_path)
    src_mesh = Path(src_mesh)
    rel_name = mesh_cache_rel_name(dance_json_path)
    dest = dance_json_path.parent / rel_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_mesh, dest)
    if sha256_hex is not None:
        if sha256_file(dest) != sha256_hex:
            raise ValueError("mesh cache copy sha256 mismatch")
    return rel_name


def persist_sidecar_with_project(
    dance_json_path: Path,
    src_sidecar: Path,
    sha256_hex: str | None = None,
) -> str:
    """Copy SMPL sidecar beside the project JSON; return relative filename."""
    dance_json_path = Path(dance_json_path)
    src_sidecar = Path(src_sidecar)
    rel_name = f"{dance_json_path.stem}.smpl.npz"
    dest = dance_json_path.parent / rel_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_sidecar, dest)
    if sha256_hex is not None:
        if sha256_file(dest) != sha256_hex:
            raise ValueError("sidecar copy sha256 mismatch")
    return rel_name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_smpl_sidecar(path: Path, tracks: list[dict[str, Any]]) -> None:
    """Write SMPL params for each track with g{i}_ / c{i}_ prefixed keys."""
    arrays: dict[str, np.ndarray] = {"n_tracks": np.int32(len(tracks))}
    for i, track in enumerate(tracks):
        global_params = track["global"]
        incam_params = track["incam"]
        k = np.asarray(track["K"], dtype=np.float32)
        n_frames: int | None = None
        for key in SMPL_PARAM_KEYS:
            g_arr = np.asarray(global_params[key], dtype=np.float32)
            c_arr = np.asarray(incam_params[key], dtype=np.float32)
            g_t = int(g_arr.shape[0])
            c_t = int(c_arr.shape[0])
            if n_frames is None:
                n_frames = g_t
            if g_t != n_frames:
                raise ValueError(
                    f"smpl sidecar track {i}: g{i}_{key} has T={g_t}, expected T={n_frames}"
                )
            if c_t != n_frames:
                raise ValueError(
                    f"smpl sidecar track {i}: c{i}_{key} has T={c_t}, expected T={n_frames}"
                )
            arrays[f"g{i}_{key}"] = g_arr
            arrays[f"c{i}_{key}"] = c_arr
        k_t = int(k.shape[0])
        if n_frames is None:
            n_frames = k_t
        elif k_t != n_frames:
            raise ValueError(
                f"smpl sidecar track {i}: K has T={k_t}, expected T={n_frames}"
            )
        arrays[f"K_{i}"] = k
        arrays[f"n_frames_{i}"] = np.int32(n_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_smpl_sidecar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        _require_keys(keys, ["n_tracks"], "smpl sidecar")
        n_tracks = int(archive["n_tracks"])
        per_track: list[str] = []
        for i in range(n_tracks):
            per_track.append(f"n_frames_{i}")
            per_track.append(f"K_{i}")
            for key in SMPL_PARAM_KEYS:
                per_track.append(f"g{i}_{key}")
                per_track.append(f"c{i}_{key}")
        _require_keys(keys, per_track, "smpl sidecar")
        return _archive_to_dict(archive)


def save_mesh_cache(
    path: Path,
    *,
    faces: np.ndarray,
    tracks: list[dict[str, Any]],
    overlay_relpath: str | None = None,
) -> None:
    """Write per-track mesh verts, times, and mask; shared faces."""
    faces_arr = np.asarray(faces, dtype=np.int32)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("mesh cache: faces must have shape (F, 3)")
    arrays: dict[str, np.ndarray] = {
        "faces": faces_arr,
        "n_tracks": np.int32(len(tracks)),
    }
    for i, track in enumerate(tracks):
        verts_world = np.asarray(track["verts_world"], dtype=np.float32)
        verts_incam = np.asarray(track["verts_incam"], dtype=np.float32)
        times = np.asarray(track["times"], dtype=np.float64)
        mask = np.asarray(track["mask"], dtype=bool)
        t = int(times.shape[0])
        if verts_world.shape[0] != t or verts_incam.shape[0] != t or mask.shape[0] != t:
            raise ValueError(f"mesh cache track {i}: verts/times/mask length mismatch")
        if verts_world.ndim != 3 or verts_incam.ndim != 3:
            raise ValueError(f"mesh cache track {i}: verts must be (T, V, 3)")
        arrays[f"verts_world_{i}"] = verts_world
        arrays[f"verts_incam_{i}"] = verts_incam
        arrays[f"times_{i}"] = times
        arrays[f"mask_{i}"] = mask
    if overlay_relpath is not None:
        arrays["overlay_relpath"] = np.array([overlay_relpath], dtype=np.str_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


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
