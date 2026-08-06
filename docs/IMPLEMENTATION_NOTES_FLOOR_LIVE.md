# Implementation notes — Floor map live during play

**Handoff:** [AGENT_HANDOFF_FLOOR_LIVE.md](./AGENT_HANDOFF_FLOOR_LIVE.md)  
**Date:** 2026-08-06

## What shipped

Removed the early `return` in `desktop/window.py` `update_playhead` that skipped the floor
scatter while `self.playing`. Floor now updates on the same `PLAYHEAD_MIN_INTERVAL_S`
(~10 Hz) throttle as the chart: `set_offsets` + `draw_idle()` only — still no
`refresh_floor_map()` per tick.

## How to test

1. Open a project with metrics / floor map on.
2. Press Play — red floor playhead should track the dancer (≈10 Hz is fine).
3. Pause — dot matches current time; seek / chart click updates immediately.
4. Uncheck Floor map — no floor work while playing.
