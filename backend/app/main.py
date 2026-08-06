from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.jobs import run_in_background, store
from app.pipeline.run import run_pipeline
from app.schemas import AnalyzePathBody, JobStatus, RenameMotifBody

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "uploads"
ARTIFACTS = ROOT / "artifacts"
UPLOADS.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Dance Analyser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    device = "cpu"
    gpu_name = None
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {"status": "ok", "device": device, "gpu": gpu_name}


@app.post("/api/analyze", response_model=JobStatus)
async def analyze(
    file: UploadFile | None = File(default=None),
    stride: int = Form(default=1),
    model: str = Form(default="yolo11s-pose.pt"),
    write_overlay: bool = Form(default=True),
) -> JobStatus:
    """Upload a video and start analysis job."""
    if file is None:
        raise HTTPException(status_code=400, detail="請上傳影片檔案")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    job = store.create(UPLOADS / "pending")
    dest = UPLOADS / f"{job.id}{suffix}"
    job.video_path = dest

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    s = max(1, min(int(stride), 5))
    run_in_background(
        lambda: run_pipeline(
            job.id,
            dest,
            ARTIFACTS,
            store,
            stride=s,
            model_name=model or "yolo11s-pose.pt",
            write_overlay_video=bool(write_overlay),
        )
    )
    return job.to_status()


@app.post("/api/analyze/path", response_model=JobStatus)
def analyze_path(body: AnalyzePathBody) -> JobStatus:
    """Analyze a video already on disk (dev / local path)."""
    path = Path(body.path)
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"找不到檔案：{body.path}")

    job = store.create(path)
    s = max(1, min(int(body.stride), 5))
    run_in_background(
        lambda: run_pipeline(
            job.id,
            path,
            ARTIFACTS,
            store,
            stride=s,
            model_name=body.model,
            write_overlay_video=body.write_overlay,
        )
    )
    return job.to_status()


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到工作")
    return job.to_status()


@app.get("/api/jobs/{job_id}/overlay")
def get_overlay(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if not job or not job.overlay_path or not job.overlay_path.exists():
        raise HTTPException(status_code=404, detail="尚無預覽影片")
    return FileResponse(
        job.overlay_path,
        media_type="video/mp4",
        filename=f"overlay_{job_id}.mp4",
    )


@app.get("/api/jobs/{job_id}/analysis")
def get_analysis(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到工作")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail="分析尚未完成")
    return job.result.model_dump()


@app.patch("/api/jobs/{job_id}/motif-types/{type_id}")
def rename_motif(job_id: str, type_id: str, body: RenameMotifBody) -> JobStatus:
    job = store.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="找不到分析結果")
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="名稱不可空白")
    found = False
    for mt in job.result.motifTypes:
        if mt.id == type_id:
            mt.name = name
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="找不到動作類型")
    # Persist
    out_json = ARTIFACTS / job_id / "analysis.json"
    if out_json.parent.exists():
        out_json.write_text(
            job.result.model_dump_json(indent=2),
            encoding="utf-8",
        )
    return job.to_status()
