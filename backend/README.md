# Dance Analyser — Python analysis backend

YOLO-Pose + Supervision ByteTrack → energy/speed metrics → motif types.

## Setup

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Multipart video upload → job |
| POST | `/api/analyze/path` | Local path analyze (dev) |
| GET | `/api/jobs/{id}` | Job status + result |
| GET | `/api/jobs/{id}/overlay` | Annotated preview mp4 |
| GET | `/api/jobs/{id}/analysis` | `analysis.json` |
| PATCH | `/api/jobs/{id}/motif-types/{type_id}` | Rename motif type |

First run downloads `yolo11s-pose.pt` automatically.
