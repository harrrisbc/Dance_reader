# Dance Analyser — Implementation Plan (Python-first)

**Decision (2026-08-06):** Local **single Python app**. HTML/Vite theatre UI is **legacy / optional**, not the primary product.

**Owner (design):** design chat  
**Owner (code):** programming agent + this handoff  
**UI language:** Traditional Chinese  
**Beauty / stage vibe:** not required

---

## Goal

One local process:

1. Open local video (+ YouTube via yt-dlp later)
2. YOLO-Pose + Supervision (multi-person)
3. Playback with skeleton overlay
4. Energy + speed charts; motif types (類型 A/B…) with rename
5. Manual marks

---

## Architecture (locked)

```
┌──────────────────────────────────────────┐
│  Desktop UI  (PySide6)                   │
│  open / play / analyze / charts / marks  │
└──────────────────┬───────────────────────┘
                   │ in-process call
┌──────────────────▼───────────────────────┐
│  backend/app/pipeline/*                  │
│  pose_track → metrics → motifs → overlay │
└──────────────────────────────────────────┘
```

- **No browser required** for daily use.
- FastAPI (`backend/app/main.py`) may remain for optional API / debugging; not required to launch the app.
- Existing Vite app under `src/` is frozen legacy until deleted later.

### Stack

| Piece | Choice |
|-------|--------|
| UI | PySide6 |
| Pose | ultralytics YOLO pose |
| Track / draw | supervision |
| Charts | matplotlib embedded in Qt |
| Clustering | sklearn (existing motifs.py) |

---

## Phases

### P0 — Desktop shell + existing pipeline

1. `desktop/main.py` entry: `python -m desktop`
2. Open file, scrub/play frames (OpenCV decode → QLabel)
3. Run analyze in background thread; progress label
4. Show energy/speed/space metrics for selected actor (series toggles)
5. List motif types; rename; seek to occurrence
6. Draw keypoints on current frame from analysis JSON

### P1 — Manual marks + multi-actor polish

### P2 — YouTube (yt-dlp); space metrics shipped (reach/level/travel/directness)

### P3 — Optionally remove Vite/FastAPI if unused

---

## Run

```powershell
cd backend
.\.venv\Scripts\activate
pip install -r requirements.txt
cd ..
python -m desktop
```

(Working directory / PYTHONPATH must include `backend` so `import app` works — entry script sets this.)

---

## Definition of done (this pivot)

- User runs one Python command → window opens  
- Analyze local mp4 with Supervision pipeline  
- See overlay or skeleton + charts + motif types in Traditional Chinese  
- No `npm run dev` required
