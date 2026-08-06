# WORK ORDER — Web review: skeleton lag, timeline alignment, chart kinds

**To:** programming agent  
**From:** Design agent  
**Date:** 2026-08-06  
**User reports (still broken after prior “done” marks):**

1. Bones on web UI still **lagging**  
2. Timeline UI **not lined up** (click / playhead / drawn curves disagree)  
3. Timeline charts must use the **agreed visual kinds** (not all plain lines)

Primary files: `backend/app/static/review/review.js`, `review.css`, `index.html`  
Reference (already correct on Qt): `desktop/window.py` → `CHART_SERIES` + `_plot_series_row`

Bump `?v=` on the script tag in `index.html` after shipping so browsers don’t keep a stale cache.

---

## Why

Web review still draws every series as `kind: "line"`. Qt already implements area / reach refs / level bands / ribbon. Skeleton runs off `timeupdate` (~4–12 Hz) and the overlay canvas is stretched to the **wrap box**, while `<video>` letterboxes — so bones feel late *and* mis-registered. Click seek attaches to the **panel wrap** but maps X via `plot.over`; y-axis gutter + panel title break alignment.

---

## 1. Skeleton lag + registration (P0)

### Root causes in current `review.js`

| Issue | Detail |
|-------|--------|
| Update cadence | `drawSkeletonOverlay()` only on `timeupdate` (throttled 80 ms) → looks like lag vs video |
| Letterbox mismatch | `#skeletonOverlay { inset:0 }` covers `.video-wrap`; video keeps aspect → keypoints map to full wrap, not the painted video rectangle |
| Cost | Glow pass + stroke pass every tick; “All dancers” multiplies work |
| Pose lookup | OK (binary search), not the main lag |

### Required changes

1. **Align canvas to the actual video content box** (object-fit / letterbox math):

```text
videoW, videoH = video.videoWidth, video.videoHeight
dispW, dispH = video.getBoundingClientRect()
scale = min(dispW/videoW, dispH/videoH)
drawW = videoW * scale, drawH = videoH * scale
ox = (dispW - drawW) / 2, oy = (dispH - drawH) / 2
```

Size/position the canvas (or draw with offset) so `(kp[0]*drawW + ox, kp[1]*drawH + oy)` lands on the person.

2. **Drive overlay with `requestAnimationFrame` while playing** (and on `seeked` / pause once). Do **not** rely on `timeupdate` for skeleton. Cap at display refresh; skip a frame if previous draw unfinished.

3. **Cheap draw while playing:**
   - Default selected dancer only (All dancers remains optional).
   - One stroke pass (drop glow) while `!video.paused`; full style on pause/seek.
   - Joints-only mode already exists — keep it.

4. Optional: if still heavy, downsample pose rate display to 30 Hz but **sample `video.currentTime` every rAF** so registration stays tight (late bone = wrong time, not just low FPS).

### Acceptance (skeleton)

- Playing 1080p with analysis: skeleton tracks limbs within ~1 frame visually.  
- No systematic offset (bones not floating beside the body after letterbox).  
- Pause / seek: overlay snaps immediately.  
- “Off” still hides overlay.

---

## 2. Timeline not lined up (P0)

### Root causes

1. Click on **`.panel` wrap** → `seekFromPointer` uses `plot.over.getBoundingClientRect()`; clicks on **title / Y-axis / padding** still fire with wrong `left`.  
2. uPlot Y axis `size: 42` — plot area ≠ panel left edge; playhead CSS uses `valToPos` inside `.over` (OK) but wrap-level click does not.  
3. Possible residual: create plot when `host.clientWidth === 0`, then CSS stretch; `resizePlots` helps but click may still use stale scale until RO fires.  
4. Wheel zoom uses same `over` math — same class of bug if pointer is outside `.over`.

### Required changes

1. Attach seek / wheel **only to `plot.over`** (or map with `plot.bbox`):

```javascript
// Prefer:
const left = clientX - plot.over.getBoundingClientRect().left;
// Clamp to [0, plot.bbox.width]; ignore if outside over
const t = plot.posToVal(left, "x");
```

If supporting click on full panel height for UX, convert using:

```javascript
const x = clientX - plot.root.getBoundingClientRect().left - plot.bbox.left;
```

and reject when `x` outside `[0, plot.bbox.width]`.

2. **Identical X for playhead and click:** both must use the same `posToVal` / `valToPos` on the same plot after `setSize`. After every `setSize` / `setScale`, redraw playheads.

3. **Lock fixed Y ranges** from the design table (do not expand ymin/ymax from data except when values exceed the lock — same as Qt). Today web expands freely → curves look “shifted” vs expected bands.

4. Shared X: all panels already share `state.view`; after brush-zoom call `drawPlayheads`.

5. Visual check fixture: mark times 0%, 50%, 100% of visible window — click those pixels → seek within ±1 frame; playhead must sit under the cursor.

### Acceptance (timeline)

- Click centre of a curve row → seek matches that time.  
- Playhead sits on the same X as the click.  
- After window resize, still accurate.  
- Y-axis gutter clicks do not jump to wrong times (ignore or map correctly).

---

## 3. Chart kinds (P0) — match agreed design

Web `SERIES` currently forces `kind: "line"`. Port Qt behaviour.

| Series | kind | Chart treatment | Y label / range |
|--------|------|-----------------|-----------------|
| Energy | `area` | Filled area + thin stroke | rel. peak · **0–1 locked** |
| Speed | `line` | Line | shoulders/s · **0–4** |
| Reach | `reach` | Line + ref lines at **1.5** (“closed”) and **3.0** (“open”) | × shoulder · **1–3.5** |
| Level | `level` | Line over bands 低 0–0.45 / 中 0.45–0.8 / 高 0.8–1.05 | stance · **0–1.05** |
| Travel | `area` | Filled area + thin stroke | shoulders/s · **0–2** |
| Directness | `ribbon` | **No Y axis / no line** — vertical colour ribbon; gaps where NaN (standing still) | — |
| L–R / F–B | `pos` | Line | screen · **0–1** |

### uPlot implementation notes

- **area:** `fill` under series (uPlot `fill` + `paths` or `bands`) + stroke.  
- **reach:** extra series or `hooks.draw` for hlines + text.  
- **level:** `hooks.drawClear` / background rects for three bands, then line.  
- **ribbon:** don’t plot Y series; in `hooks.draw`, for each segment with finite values, fill a mid-band rect coloured by value (1 cool → 0 warm), skip NaN. Hide Y axis (`axes[1].show = false`).  
- Defaults on: Energy, Speed, Reach (unchanged).

Mirror the same kinds if Qt drifts; **web is primary review UI**.

### Acceptance (kinds)

- Energy/Travel read as filled effort, not thin lines.  
- Reach shows closed/open guides.  
- Level shows three background bands.  
- Directness is a ribbon with gaps when still — not a misleading flat “1.0” line.  
- Locked Y ranges don’t jump when zooming time.

---

## Priority order

1. Timeline click + playhead alignment + locked Y  
2. Skeleton letterbox + rAF  
3. Chart kinds (area / reach / level / ribbon)

## Out of scope

- Motif accuracy / mirror DTW  
- Homography true stage floor  
- Rewriting analysis pipeline  
- English copy (already done)

## Done when

Update `docs/IMPLEMENTATION_NOTES_WEB_TIMELINE_POLISH.md` + mark this row **done** in `AGENT_HANDOFF_INDEX.md`.
