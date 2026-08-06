from __future__ import annotations

from pathlib import Path
from threading import Lock

from app.project_io import DanceProject, load_project
from app.schemas import AnalysisResult


class ReviewSession:
    """In-memory session for the local HTML review UI."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.project_path: Path | None = None
        self.video_path: Path | None = None
        self.analysis: AnalysisResult | None = None
        self.video_name: str = ""

    def clear(self) -> None:
        with self.lock:
            self.project_path = None
            self.video_path = None
            self.analysis = None
            self.video_name = ""

    def open_project_file(self, path: Path) -> DanceProject:
        project = load_project(path)
        video = Path(project.video_path)
        with self.lock:
            self.project_path = path
            self.analysis = project.analysis
            self.video_path = video if video.is_file() else None
            self.video_name = video.name if video.name else ""
        return project

    def set_analysis_json(self, analysis: AnalysisResult, video_path: str | None) -> None:
        with self.lock:
            self.analysis = analysis
            self.project_path = None
            if video_path:
                vp = Path(video_path)
                self.video_path = vp if vp.is_file() else None
                self.video_name = vp.name
            else:
                self.video_path = None
                self.video_name = ""

    def set_video_path(self, path: Path) -> None:
        with self.lock:
            self.video_path = path if path.is_file() else None
            self.video_name = path.name if path.is_file() else ""

    def get_video_path(self) -> Path | None:
        with self.lock:
            if self.video_path is not None and self.video_path.is_file():
                return self.video_path
            return None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "hasAnalysis": self.analysis is not None,
                "hasVideo": self.video_path is not None and self.video_path.is_file(),
                "projectPath": str(self.project_path) if self.project_path else None,
                "videoPath": str(self.video_path) if self.video_path else None,
                "videoName": self.video_name,
                "device": _device_info(),
                "analysis": self.analysis.model_dump() if self.analysis else None,
            }


_device_cache: dict | None = None


def _device_info() -> dict:
    global _device_cache
    if _device_cache is not None:
        return _device_cache
    device = "cpu"
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    _device_cache = {"device": device, "gpu": gpu}
    return _device_cache


review_session = ReviewSession()
