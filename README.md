# Dance Analyser

Local dance analysis: **YOLO-Pose (CUDA when available)** + a **PySide6 desktop** app for review.

UI language: **English**. Data stays on this machine; nothing is uploaded to the cloud.

## Start (recommended)

```powershell
cd C:\dev\Dance_reader\Dance_reader
.\backend\.venv\Scripts\python.exe -m desktop
```

Opens the Qt window (no browser).

### Typical workflow

1. **Open video** ??choose a local file (download YouTube clips first; no YouTube API yet).
2. Set **Max dancers** and **Model** (S fast default, or M for occlusion / side views), then **Analyze**.
3. After analysis: stacked metric charts, skeleton overlay, floor map, motif types.
4. Click a chart to seek; scroll to zoom the time window; **Find similar** for matches near the playhead.
5. **Save project** (`.dance.json`) / **Open project** to reload without re-analyzing.

Status bar shows motif diagnostics after analyze, e.g.  
`N phrases 繚 M clusters 繚 K passed similarity ?圳

### Overlay

**Skeleton** / **Joints only** / **Off**. Lines may cross when limbs overlap in the camera view.

## Chart units

| Curve | Unit |
|-------|------|
| Energy | Relative to own peak |
| Speed / Travel | Torso-heights / s |
| Reach | ? torso (guides at closed 1.2 / open 2.5) |
| Level | Stance ratio (low / mid / high bands) |
| Directness | Colour ribbon 0?? (blank when still) |
| L? / F? | Screen coords 0?? |

## Legacy web review

Opt-in only (kept for now, not the product UI):

```powershell
.\backend\.venv\Scripts\python.exe -m desktop --web
```

Serves `http://127.0.0.1:8000/review`. Prefer Qt for analysis and review.

## Analyze API (optional)

```powershell
cd backend
.\.venv\Scripts\activate
uvicorn app.main:app --port 8000
```

## Motif thresholds

Matching decides in **raw mask-aware DTW distance** (torso units, mean per visible dim).
Defaults: `CLUSTER_MAX_DIST=0.24`, `INTRA_MAX_DIST=0.26`, `CLUSTER_LINKAGE=complete`, `MOTIF_MIN_OCCURRENCES=3`.
Find Similar uses separate `SUBSEQ_*` gates. The UI shows a display-only 0-100% match score; that score is
**not** used as a gate. See `docs/IMPLEMENTATION_NOTES_MOTIF_OVERMERGE.md`.

```powershell
.\backend\.venv\Scripts\python.exe backend\scripts\calibrate_motifs.py path\to\file.dance.json
.\backend\.venv\Scripts\python.exe backend\scripts\calibrate_motifs.py path\to\hi.dance.json --negative
```
