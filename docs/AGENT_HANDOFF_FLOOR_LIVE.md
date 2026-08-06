# WORK ORDER — Floor map must track playhead while playing

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Priority:** P0 — user-visible playback bug

## Why

User: floor map only moves when paused; during Play it stays frozen.

This is not a mysterious race — `desktop/window.py` `update_playhead` **explicitly returns before updating the floor dot while `self.playing`**:

```python
# Floor map: skip while playing (catch up on pause)
if self.playing and not force:
    return
```

That came from the playback smoothness handoff (avoid `floor_canvas.draw_idle()` every video frame). Charts were later fixed with ~10 Hz throttle + blit; the floor path was left on the “skip entirely” shortcut. The constant comment still says `PLAYHEAD_MIN_INTERVAL_S` is for “chart/floor”, but floor never uses it.

**Product decision:** floor map is a live orientation aid during playback, same as the chart playhead. Throttle is fine; freezing until pause is not.

## Scope

- `desktop/window.py` — `update_playhead` (and only what is needed for a cheap floor blit if you add one)
- Optional: one line in `docs/IMPLEMENTATION_NOTES_PLAYBACK.md` if you touch docs

## Changes

1. **Remove the early return** that skips floor updates while playing.
2. While playing, update the floor playhead on the **same throttle** as the chart (`PLAYHEAD_MIN_INTERVAL_S`, already applied at the top of `update_playhead`).
3. Keep the cheap path: `set_offsets` on the existing scatter + `draw_idle()` (or blit over a baked floor background if draw_idle is still heavy). Do **not** call `refresh_floor_map()` every tick — that clears and rebuilds the whole figure.
4. On pause / seek (`force=True`), behaviour stays as today (full catch-up).
5. If floor toggle is off, still skip (existing guard).

## Acceptance

- [ ] Press Play: floor red/playhead dot moves continuously with the dancer (≈10 Hz is OK; must not stay stuck on the pause position).
- [ ] Pause: dot lands on the correct current position.
- [ ] Seek / chart click: floor updates immediately.
- [ ] Video playback stays smooth — no return to “UI freezes while floor redraws every frame”. If smoothness regresses, switch floor to blit-over-baked-background rather than skipping updates.
- [ ] Floor toggle off: no floor work while playing.

## Out of scope

- Floor homography / metres
- Changing chart blit behaviour
- Motif / metrics math
