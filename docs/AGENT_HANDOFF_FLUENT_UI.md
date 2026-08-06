# WORK ORDER — fluent review UX (GPU reality + charts + optional Web UI)

**To:** programming agent  
**From:** Design agent  
**User feedback:** playback still unusable; wants GPU if useful; needs fluent UX; graphs not ready (click / zoom each series); OK to move UI to HTML if Python is the limit; asks if baking graphs helps.

---

## 1. What uses CPU vs GPU today

| Work | Where | Device now |
|------|--------|------------|
| YOLO pose analyze | `pose_track.py` → Ultralytics | **GPU if CUDA** (`resolve_device()`). Machine has RTX 5070 / `torch.cuda.is_available() == True`. |
| Metrics / motifs | NumPy / sklearn | CPU (fine; cheap vs decode+charts) |
| Preview decode | `cv2.VideoCapture` | **CPU** soft decode |
| Pose overlay on preview | OpenCV draw | CPU |
| Timeline + floor map | matplotlib in Qt | **CPU**, redraw-heavy |
| Qt layout / pixmap | UI thread | CPU |

**Important:** Putting the preview on the GPU will **not** fix the main stutter. The expensive part while scrubbing/playing is (a) UI-thread work and (b) matplotlib redraws — not lack of CUDA on the video label. Analysis already benefits from the GPU; keep that path.

Do **not** spend a sprint wiring NVDEC/CUDA VideoCapture for preview until the chart/playhead path is fixed. Optional later only.

---

## 2. Baking graphs — yes, for playback; no, not for interaction

**Baking = pre-render the full metric curves once** (or when analysis/zoom window changes), then during play only move a playhead.

| Goal | Baking helps? |
|------|----------------|
| Smooth play while charts visible | **Yes — highest ROI on current desktop** |
| Scrub without full redraw | Yes (blit line over baked image / static artists) |
| Per-panel zoom, pan, brush, hover | **No** — need a real interactive chart engine |
| “Click this graph and zoom easily” | Need re-render or Web charts (Plotly / uPlot / ECharts) |

So: bake (or freeze axes + blit) for **play mode**; use an interactive chart library for **explore mode**. Do not expect baking alone to make matplotlib feel like a DAW timeline.

---

## 3. Product decision (design)

User priority is **fluent review** (video + graphs), not “stay on PySide at all costs”.

**Chosen direction: Hybrid local app**

1. **Keep Python** for analyze (YOLO on GPU), `.dance.json`, motifs, enrich_metrics.  
2. **Move review UI to local HTML** served by FastAPI (or Qt WebEngine loading that page) — video element + interactive charts.  
3. Desktop window can remain a thin shell later; first ship: `python -m desktop` may open browser to `http://127.0.0.1:…` **or** embed WebEngine. Prefer one entry point documented in README.

**Do not** port YOLO/analysis to JavaScript.  
**Do not** rebuild Vite as a remote SaaS — stay localhost, same files on disk.

If hybrid is too large for one pass, interim desktop stopgap (same handoff file, Phase A only): pyqtgraph or plotly-qt + baked playhead + prior `AGENT_HANDOFF_PLAYBACK.md` P0/P1. Phase B still required for the graph UX the user described.

---

## 4. Graph UX requirements (must-have)

Replace “one matplotlib figure, shared scroll zoom only” with:

- **One panel per enabled metric** (already the data model).
- **Each panel:** wheel zoom X (time), optional Y lock (fixed unit ranges from space-metrics handoff), drag-pan on that panel’s time window (linked X across panels by default).
- **Click** anywhere on any panel → seek video to that time.
- **Brush / drag-select** on a panel → zoom time window to selection (Reset restores full duration).
- Playhead line synced to video; while playing, charts update playhead only (baked/static series).
- Floor map stays separate; playhead dot throttled; not redrawn every video frame.

Suggested web stack (local): **uPlot** or **ECharts** (fast time series) or **Plotly** (faster to ship, heavier). Prefer one library; Traditional Chinese labels.

Video: HTML `<video src="file://…">` or better **blob / range-served by FastAPI** (`/media?path=`) so browser security is happy. Sync charts via `timeupdate` + seek API.

---

## 5. Implementation phases

### Phase A — make current desktop tolerable (short)

Still do `docs/AGENT_HANDOFF_PLAYBACK.md` P0/P1 if not done:

- Clock sync + frame skip  
- Decode off UI thread  
- **Bake / freeze chart series while playing**; playhead ≤10 Hz; no `refresh_chart` on play ticks  

This unblocks daily use while Phase B lands.

### Phase B — local Web review UI (main)

- FastAPI: open project / analyze status / serve `AnalysisResult` JSON + video bytes  
- HTML review page: video + stacked interactive charts + motif list + 搵相似 hooks as needed  
- Reuse existing pipeline; no re-analyze for chart interaction  
- Document: analyze still GPU; UI is browser; data stays local  

### Phase C — polish (later)

- Optional GPU decode for preview  
- Homography floor map  
- Embed Web UI inside Qt WebEngine for single-window feel  

---

## 6. Acceptance

- User can play a typical rehearsal MP4 with analysis open without the whole UI freezing.  
- Each metric panel supports click-to-seek and wheel/brush zoom without feeling laggy.  
- Playing does not replot all series every frame (baked or equivalent).  
- Analyze still uses CUDA when available (status text may show `cuda` / device).  
- No emotion labels; space metric units from prior handoffs still apply in the new charts.

## Out of scope this order

- Rewriting analysis in the browser  
- Cloud upload  
- Switching away from YOLO for “GPU usage” theatre — analyze already uses GPU
