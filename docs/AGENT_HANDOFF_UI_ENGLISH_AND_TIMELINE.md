# WORK ORDER — skeleton clarity, English UI, timeline click alignment

**To:** programming agent  
**From:** Design agent  
**User feedback (3 items):**

1. “Will there be no bone overlapping on the video now?” (after motif-accuracy handoff)  
2. Switch app UI to **English** (easier to understand).  
3. **Timeline click area does not match** what is shown — seek goes to wrong time.

---

## 1. Bone overlap on video — clarify + optional polish

### What the motif handoff does **not** do

`AGENT_HANDOFF_MOTIF_ACCURACY.md` adds **bone vectors inside trajectory embeddings** for matching gestures. That is **math only**. It does **not** change how skeletons are drawn on the preview video.

### Current preview behaviour

| Surface | Skeleton |
|---------|----------|
| Qt desktop (`desktop/playback.py`) | **Selected dancer only**; COCO-17 lines + joints; every other frame skipped while playing |
| Web review (`backend/app/static/review/`) | **No skeleton overlay on video yet** |

So:

- Two dancers’ skeletons should **not** stack on desktop unless we later draw all actors.  
- **Self-overlap** (arm crosses torso → lines cross on screen) is **expected** for 2D pose; not a bug from motif work.  
- Motif accuracy work will **not** remove that unless we add separate overlay rules.

### If user wants cleaner video overlay (optional, same PR or follow-up)

- Toggle: **Joints only** / **Skeleton** / **Off**  
- Lower line alpha (e.g. 0.6) so crossed limbs are less visually muddy  
- Distinct color per dancer when “show all dancers” is added  
- Do **not** promise true occlusion (3D depth) without a depth model

Document in UI tooltip: “Lines may cross when limbs overlap in the camera view.”

---

## 2. English UI (product change)

**Decision:** Default UI language is **English**. Traditional Chinese strings can be removed or moved to a future i18n file; no bilingual requirement for this pass.

### Files to update (user-visible strings)

| Area | Files |
|------|--------|
| Web review | `backend/app/static/review/index.html`, `review.js`, `review.css` (hints only) |
| Qt desktop | `desktop/window.py`, `desktop/__main__.py`, `desktop/playback.py` (error strings) |
| Actor default labels | `backend/app/pipeline/run.py` — `_actor_label` → `"Dancer {n}"` |
| Series labels | `CHART_SERIES` in `window.py`; `SERIES` in `review.js` |

### Suggested English copy (charts)

| Was | EN |
|-----|-----|
| 能量 | Energy |
| 速度 | Speed |
| 伸展 | Reach |
| 高低 | Level |
| 走位 | Travel |
| 直進 | Directness |
| 左右 / 前後 | L–R / F–B |
| 播放 / 暫停 | Play / Pause |
| 分析 | Analyze |
| Find similar | Find similar |
| 走位圖（畫面座標） | Floor map (screen coords) |

Keep `lang="en"` on HTML. README can note “UI in English”.

**Out of scope:** full gettext/i18n; only replace hard-coded strings.

---

## 3. Timeline click misalignment (bug)

User reports: clicking the chart does not seek to the time under the cursor / visible playhead.

### Web review (`review.js`) — likely causes

1. **Fixed uPlot width at create time** (`opts.width = host.clientWidth`) but CSS sets `.uplot { width: 100% }` → chart scales visually; click math uses stale pixel width → **horizontal offset**.  
2. Click uses `plot.over.getBoundingClientRect()` + `clientX` but should use coordinates relative to uPlot **plot area** (`bbox`), not the full root/over if padding differs.  
3. No **ResizeObserver** → after window resize, plots stay wrong size until rebuild.

### Web fix (required)

```javascript
// On create + on resize:
function plotWidth(host) { return host.clientWidth || 480; }
// After layout:
plot.setSize({ width: plotWidth(host), height: 90 });

// Click — prefer uPlot’s own mapping:
host.addEventListener("click", (ev) => {
  const rect = plot.root.getBoundingClientRect();
  const x = ev.clientX - rect.left - plot.bbox.left;
  const t = plot.posToVal(x, "x");
  ...
});

// Or attach to plot.over and use ev.offsetX minus 0 if over === plot area

new ResizeObserver(() => {
  for each plot: setSize({ width: plotWidth(host), height: 90 });
}).observe(chartStack);
```

Also align **CSS playhead** (`.ph-line`) with the same `valToPos` after resize.

Add **click on full panel** (not only `.u-over`) mapped through the same function so the clickable region matches the visible chart box.

### Qt desktop (`window.py` + matplotlib) — likely causes

1. `fig.set_size_inches(...)` vs `FigureCanvas` widget size / **device pixel ratio** mismatch on HiDPI → mouse x ≠ data x.  
2. `tight_layout` + left `set_title(..., loc="left")` — clicks outside axes are ignored; user may click the “empty” margin thinking it is the timeline.  
3. After **blit** bake, visual playhead may not match axes if background was copied from wrong bbox.

### Qt fix (required)

- On canvas `resizeEvent`: set figure size to `width/dpi`, `height/dpi` to match widget pixels.  
- Set explicit `subplots_adjust(left=..., right=..., top=..., bottom=...)` instead of relying on `tight_layout` for hit-testing consistency.  
- Expand hit target: treat click anywhere in canvas Y-band of a row as that row’s axes (map display coords → data coords with `ax.transData.inverted().transform`).  
- Verify `on_chart_click`: `event.xdata` matches slider after click at 0%, 50%, 100% of visible xlim.

### Acceptance (timeline)

- Click at visual centre of a chart row → video seeks to that time (± one frame).  
- After resizing the window, click still accurate (web + Qt).  
- Playhead line and click target use the same x scale.

---

## Priority

1. **P0** — Timeline click fix (web + Qt)  
2. **P0** — English UI  
3. **P1** — Skeleton overlay tooltip + optional joints-only toggle (clarify overlap is not fixed by motif work)

## Out of scope

- Motif mirror/DTW (separate handoff)  
- 3D occlusion / depth-aware skeleton
