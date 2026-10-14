"""Call the GVHMR venv. This module must not import hmr4d."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.pipeline.gvhmr_io import parse_gvhmr_document
from app.schemas import PoseFrame

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_EXPORT_SCRIPT = _BACKEND_DIR / "tools" / "gvhmr_export.py"
_BAKE_SCRIPT = _BACKEND_DIR / "tools" / "gvhmr_mesh_bake.py"


class GvhmrNotConfigured(RuntimeError):
    pass


class MeshBakeError(RuntimeError):
    pass


def _require_paths() -> tuple[Path, Path]:
    py = os.environ.get("DANCE_GVHMR_PYTHON", "").strip()
    root = os.environ.get("DANCE_GVHMR_ROOT", "").strip()
    if not py or not Path(py).is_file():
        raise GvhmrNotConfigured("DANCE_GVHMR_PYTHON is missing or not a file")
    if not root or not Path(root).is_dir():
        raise GvhmrNotConfigured("DANCE_GVHMR_ROOT is missing or not a directory")
    weight = Path(root) / "inputs" / "checkpoints" / "yolo" / "yolov8x.pt"
    if not weight.is_file():
        raise GvhmrNotConfigured(f"GVHMR weight missing: {weight}")
    return Path(py), Path(root)


def _run_export_subprocess(
    python: Path,
    root: Path,
    video: Path,
    out: Path,
    max_people: int,
) -> dict:
    proc = subprocess.run(
        [
            str(python),
            str(_EXPORT_SCRIPT),
            "--video",
            str(video),
            "--out",
            str(out),
            "--max-people",
            str(max_people),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not out.is_file():
        tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise GvhmrNotConfigured(tail or f"GVHMR exited {proc.returncode}")
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise GvhmrNotConfigured(f"GVHMR JSON invalid: {exc}") from exc


def _parse_export_document(data: dict) -> tuple[list[list[PoseFrame]], int]:
    try:
        return parse_gvhmr_document(data)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise GvhmrNotConfigured(f"GVHMR JSON invalid: {exc}") from exc


def _sidecar_from_export(out: Path, data: dict) -> tuple[Path | None, str | None]:
    sidecar_name = data.get("smplSidecar")
    if not sidecar_name:
        return None, None
    candidate = out.parent / str(sidecar_name)
    if not candidate.is_file():
        return None, data.get("smplSidecarSha256")
    sha = data.get("smplSidecarSha256")
    return candidate, str(sha) if sha else None


def run_gvhmr_export(
    video: Path,
    *,
    max_people: int = 2,
    work_dir: Path | None = None,
) -> tuple[list[list[PoseFrame]], int, Path | None, str | None]:
    """Returns tracks, ignored, smpl_sidecar_path | None, sha256 | None."""
    python, root = _require_paths()

    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        out = work_dir / "gvhmr.json"
        data = _run_export_subprocess(python, root, video, out, max_people)
        tracks, ignored = _parse_export_document(data)
        sidecar_path, sidecar_sha = _sidecar_from_export(out, data)
        return tracks, ignored, sidecar_path, sidecar_sha

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gvhmr.json"
        data = _run_export_subprocess(python, root, video, out, max_people)
        tracks, ignored = _parse_export_document(data)
        sidecar_path, sidecar_sha = _sidecar_from_export(out, data)
        if sidecar_path is not None:
            fd, name = tempfile.mkstemp(suffix=".dance.smpl.npz")
            os.close(fd)
            persisted = Path(name)
            shutil.copy2(sidecar_path, persisted)
            sidecar_path = persisted
        return tracks, ignored, sidecar_path, sidecar_sha


def _mesh_bake_env() -> dict[str, str]:
    env = os.environ.copy()
    backend = str(_BACKEND_DIR)
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = backend if not existing else backend + os.pathsep + existing
    return env


def run_mesh_bake(*, smpl_sidecar: Path, mesh_out: Path, video: Path | None) -> Path:
    """Subprocess DANCE_GVHMR_PYTHON on gvhmr_mesh_bake.py; cwd=DANCE_GVHMR_ROOT."""
    python, root = _require_paths()
    cmd = [
        str(python),
        str(_BAKE_SCRIPT),
        "--smpl",
        str(smpl_sidecar),
        "--out",
        str(mesh_out),
    ]
    if video is not None:
        cmd.extend(["--video", str(video)])
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=_mesh_bake_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not mesh_out.is_file():
        tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise MeshBakeError(tail or f"mesh bake exited {proc.returncode}")
    return mesh_out
