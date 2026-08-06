# Implementation notes

## Architecture pivot (2026-08-06)

Primary product is **Python desktop** (`python -m desktop`), not Vite/HTML.

- Pipeline: `backend/app/pipeline` (YOLO-Pose + Supervision)
- Desktop UI: `desktop/window.py` (PySide6)
- FastAPI + Vite: optional / legacy

## Run desktop

```powershell
cd <repo-root>
.\backend\.venv\Scripts\python.exe -m desktop
```

## analyze_video()

Defaults:

- Model: `yolo11s-pose.pt`
- Stride: `1`
- Max people: `2` (spinbox 1–4) via `IdSlotRemapper` — lost tracks reclaim the same 舞者 N within grace (~2.5s video time)
- Keypoint EMA in `pose_track.track_video`

## Metrics (`MetricPoint`)

Per actor, normalized 0–1 (except directness already ~0–1):

| Field | UI | Notes |
|-------|-----|--------|
| `energy` | 能量 | limb kinetic proxy |
| `speed` | 速度 | mean limb speed |
| `reach` | 伸展 | max head/wrist/ankle distance from torso COM ÷ shoulder width |
| `level` | 高低 | hip height in image Y (inverted + range-normalized) |
| `travel` | 走位 | rolling COM path length (~0.8s window) |
| `directness` | 直進 | net displacement ÷ path length in same window |

`enrich_metrics(result)` recomputes all curves from `actors[].poses` — called after analyze and on Load so old `.dance.json` without space fields still get space curves without re-running YOLO.

## Playback

Sequential `cap.read()` while playing (no seek-per-frame). Seek only on scrub/jump.
Preview capped ~960px; FastTransformation while playing.

## Timeline

Zoom: 放大 / 縮小 / 重設 + mouse wheel. `view_start` / `view_end` xlim.
Series toggles: 能量／速度／伸展／高低／走位／直進 (default on: 能量、速度、伸展).
Matplotlib CJK: Microsoft JhengHei / Noto Sans TC / Microsoft YaHei.

## Motif detection

Valley cuts + trajectory embeddings + 搵相似 query-by-example.

## Project save / load

- UI: **儲存專案** / **開啟專案** → `.dance.json`
- Schema: `app.project_io.DanceProject` (video_path + AnalysisResult + max_people)
- Load rebuilds `track_buffers` via `buffers_from_analysis` so 搵相似 works without re-running Supervision
- Load also runs `enrich_metrics` so space series appear for old projects
- If video path is missing, user is prompted to relocate the file
