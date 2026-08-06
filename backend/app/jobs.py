from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.schemas import AnalysisResult, JobStatus


@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "排隊中"
    result: AnalysisResult | None = None
    error: str | None = None
    video_path: Path | None = None
    overlay_path: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def to_status(self) -> JobStatus:
        return JobStatus(
            id=self.id,
            status=self.status,  # type: ignore[arg-type]
            progress=self.progress,
            message=self.message,
            result=self.result,
            error=self.error,
            overlayAvailable=self.overlay_path is not None and self.overlay_path.exists(),
        )


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, video_path: Path) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, video_path=video_path)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        result: AnalysisResult | None = None,
        error: str | None = None,
        overlay_path: Path | None = None,
    ) -> None:
        job = self.get(job_id)
        if not job:
            return
        with job.lock:
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0.0, min(1.0, progress))
            if message is not None:
                job.message = message
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if overlay_path is not None:
                job.overlay_path = overlay_path


store = JobStore()


ProgressCb = Callable[[float, str], None]


def run_in_background(fn: Callable[[], None]) -> None:
    t = threading.Thread(target=fn, daemon=True)
    t.start()
