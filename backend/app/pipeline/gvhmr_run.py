"""Call the GVHMR venv. This module must not import hmr4d."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from app.pipeline.gvhmr_io import parse_gvhmr_document
from app.schemas import PoseFrame

_EXPORT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "gvhmr_export.py"


class GvhmrNotConfigured(RuntimeError):
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


def run_gvhmr_export(video: Path, *, max_people: int = 2) -> tuple[list[list[PoseFrame]], int]:
    python, root = _require_paths()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gvhmr.json"
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
            data = json.loads(out.read_text(encoding="utf-8"))
            return parse_gvhmr_document(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise GvhmrNotConfigured(f"GVHMR JSON invalid: {exc}") from exc
