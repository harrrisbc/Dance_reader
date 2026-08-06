from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.jobs import run_in_background, store
from app.pipeline.pose_track import resolve_pose_model
from app.pipeline.run import run_pipeline
from app.project_io import load_project
from app.review_session import review_session
from app.schemas import AnalyzePathBody, JobStatus, RenameMotifBody

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "uploads"
ARTIFACTS = ROOT / "artifacts"
STATIC = Path(__file__).resolve().parent / "static"
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

if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/review")


@app.get("/review")
def review_page() -> HTMLResponse:
    index = STATIC / "review" / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="找不到檢視頁面")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, object]:
    info = review_session.snapshot()["device"]
    return {"status": "ok", **info}


@app.get("/api/review/state")
def review_state() -> dict:
    return review_session.snapshot()


class OpenProjectBody(BaseModel):
    path: str


@app.post("/api/review/open")
def review_open(body: OpenProjectBody) -> dict:
    path = Path(body.path)
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"找不到專案：{body.path}")
    try:
        review_session.open_project_file(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"無法讀取專案：{exc}") from exc
    snap = review_session.snapshot()
    snap["videoPathHint"] = snap.get("videoPath")
    return snap


@app.post("/api/review/open-upload")
async def review_open_upload(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    dest = UPLOADS / f"review_project_{file.filename or 'project.dance.json'}"
    dest.write_bytes(raw)
    try:
        project = load_project(dest)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"無法讀取專案：{exc}") from exc

    video = Path(project.video_path)
    review_session.set_analysis_json(project.analysis, str(video) if video.is_file() else None)
    if video.is_file():
        review_session.set_video_path(video)
    snap = review_session.snapshot()
    snap["videoPathHint"] = project.video_path
    snap["hasVideo"] = bool(snap["hasVideo"])
    return snap


@app.post("/api/review/set-video")
async def review_set_video(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = UPLOADS / f"review_video{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    review_session.set_video_path(dest)
    return review_session.snapshot()


@app.get("/api/media")
def review_media(request: Request) -> Response:
    path = review_session.get_video_path()
    if path is None:
        raise HTTPException(status_code=404, detail="尚未指定影片")

    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(path, media_type=media_type, filename=path.name)

    # bytes=start-end
    try:
        units, _, rng = range_header.partition("=")
        if units != "bytes":
            raise ValueError("only bytes")
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except Exception as exc:
        raise HTTPException(status_code=416, detail="Invalid Range") from exc

    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        raise HTTPException(status_code=416, detail="Invalid Range")

    length = end - start + 1

    def iterfile():
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            chunk = 1024 * 256
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(iterfile(), status_code=206, media_type=media_type, headers=headers)


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
            model_name=resolve_pose_model(model),
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
            model_name=resolve_pose_model(body.model),
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
    # Also load into review session for fluent UI
    review_session.set_analysis_json(job.result, str(job.video_path) if job.video_path else None)
    if job.video_path and Path(job.video_path).is_file():
        review_session.set_video_path(Path(job.video_path))
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
    out_json = ARTIFACTS / job_id / "analysis.json"
    if out_json.parent.exists():
        out_json.write_text(
            job.result.model_dump_json(indent=2),
            encoding="utf-8",
        )
    return job.to_status()
