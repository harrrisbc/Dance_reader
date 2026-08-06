# WORK ORDER — Always-visible current time readout

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Priority:** P1 — UX; pairs well with floor-live fix

## Why

User wants to always know **what time the video is at**. Today:

- Under the video there is only a **frame slider** — no seconds label.
- Chart playheads show time only inside the zoomed window; when zoomed in, the marker can leave the visible chart, and there is still no numeric clock.

A permanent time box solves both “where am I?” and “what second is this occurrence?” without fighting zoom.

## Product decision

Ship a **live time readout** next to the transport (slider / Play), always on screen when a video or project is open. Do **not** rely on the chart playhead alone.

Format (English UI):

```text
0:08.42 / 0:16.20
```

- Left = current time, right = duration.
- Minutes:seconds with **two decimal places** on the seconds (choreography needs sub-second).
- If duration &lt; 1 minute, still fine as `0:08.42 / 0:16.20` (leading `0:` is OK) — keep one format always.
- Monospace or fixed-width-ish font so the digits do not jitter layout while playing.

Optional second line is **not** needed. Frame number is not required in the label (slider already is frame-based).

## Scope

- `desktop/window.py` — add `QLabel` (or similar), wire updates from the same path that advances playhead / frames
- Tiny README note under typical workflow if you touch docs

## Changes

1. Add a **time label** beside the video slider row (preferred: under the video, same row as Play is already in the toolbar — put current/duration **immediately left or right of the slider** so eyes stay on transport).
2. Update it whenever display time changes: play frames, pause, seek, slider move, chart click, occurrence click, project load. Reuse `update_playhead` or the frame-paint callback — do not invent a second clock.
3. While playing, update at least as often as the chart throttle (~10 Hz) is fine; every painted frame is also fine if the label `setText` is cheap (it is).
4. When no video is loaded: show `--:--.-- / --:--.--` or hide the label.
5. **Do not** auto-pan the chart in this handoff. If the playhead leaves the zoomed window, the time box still tells the truth; auto-follow zoom can be a later option.

## Acceptance

- [ ] With a video open, the time box is always visible and shows current / duration.
- [ ] During Play the left number advances smoothly enough to read (~10 Hz+).
- [ ] Seek via slider, chart click, or occurrence list updates the label immediately.
- [ ] After Analyze / Open project, duration matches `analysis.duration` (or video duration if that is what the slider uses — be consistent with `current_time()` / `frame_count/fps`).
- [ ] Playback smoothness unchanged (label updates must not force chart/floor full redraws).

## Out of scope

- Overlaying timecode on the video pixels
- SMPTE / frame-number primary display
- Auto-scrolling the chart to keep the playhead in view (follow-up if user asks)
- Floor map work (separate handoff `AGENT_HANDOFF_FLOOR_LIVE.md`)
