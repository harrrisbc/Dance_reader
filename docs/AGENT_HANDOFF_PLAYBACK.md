# WORK ORDER — smooth video preview (keep PySide6)

**To:** programming agent  
**From:** Design agent  
**Do not** switch UI framework. The lag is architecture, not platform.

## Diagnosis (current `desktop/window.py`)

Every `QTimer` tick (~30–60 Hz) runs **on the UI thread** and does, in order:

1. `cv2.VideoCapture.read()` (software decode)
2. `draw_poses` (copy frame + nearest-pose search + draw)
3. resize → BGR→RGB → `QImage.copy()` → `QPixmap.scaled` → `setPixmap`
4. `update_playhead` → `matplotlib` `canvas.draw_idle()` on the **stacked timeline** (many axes)
5. `floor_canvas.draw_idle()` when 走位圖 is open

If that chain takes longer than the timer interval, Qt queues ticks, the UI freezes, and the video looks like dropped frames. After space-metrics UI (small multiples + floor map), step 4–5 got much heavier.

Also: playback advances **one frame per timer tick**, not by wall clock. When the machine falls behind, time stretches instead of skipping — feels stuttery even when decode is fine.

## Verdict

**Stay on PySide6 + OpenCV for analysis.** Do not migrate to Electron / Flutter / web player for this. Fix the playback path first; only revisit if a measured profile still fails after the changes below.

## Fix plan (ordered by impact)

### P0 — Stop redrawing charts every video frame

While `playing`:

- Update the video pixmap every tick (or every decoded frame).
- Update timeline playhead lines + floor dot at most **~10 Hz** (e.g. every 100 ms), or only when `floor_toggle` is on / charts visible.
- Prefer blitting / `set_xdata` + lightweight draw if already available; never call a full `refresh_chart` during play.
- On pause / seek / scrub: full chart update is fine.

### P0 — Clock-based playback (skip frames)

Replace “timer → next index” with wall-clock sync:

```
elapsed = now - play_start_wall
target_frame = play_start_frame + round(elapsed * fps)
if target_frame > last_shown + 1: skip ahead (sequential reads or one seek)
```

Goal: stay in sync with real time; drop frames rather than stretch time. Cap work per tick (e.g. max N sequential reads, then seek).

### P1 — Decode off the UI thread

- Background `QThread` (or similar) owns `VideoCapture`, decodes, applies pose overlay, produces a ready `QImage` / RGB buffer.
- UI thread only: `setPixmap` + throttled playhead.
- Keep seek / pause via a small command queue so the worker never blocks the GUI.

### P1 — Cheap preview path while playing

Already partially done (`PREVIEW_MAX_W`, `FastTransformation`). Tighten:

- While playing: skip pose overlay every other frame, or draw skeleton only for selected actor.
- Avoid `frame.copy()` when possible; draw on a downscaled buffer.
- Prefer `QImage` without unnecessary `.copy()` if lifetime is safe for one paint.

### P2 — Optional later (only if still needed)

- `QMediaPlayer` / Qt Multimedia for hardware decode of the *preview* stream, with a separate OpenCV path for analysis / pose overlay. More complex sync; do not start here.
- Do **not** rewrite the app in another toolkit for smoothness alone.

## Acceptance

- Play 1080p (or typical rehearsal MP4) with analysis loaded: UI stays responsive (lists, buttons, scrub).
- Video tracks wall clock within ~100 ms; under load it skips frames instead of slowing down.
- Timeline / floor map may update more slowly while playing; catch up fully on pause.
- No framework change; Traditional Chinese UI strings unchanged.

## Out of scope

- Changing analysis pipeline (YOLO) for preview smoothness.
- Web/Electron player.
- Emotion labels / space-metric formula work (separate handoffs).
