# Implementation notes — Always-visible time readout

**Handoff:** [AGENT_HANDOFF_TIME_READOUT.md](./AGENT_HANDOFF_TIME_READOUT.md)  
**Date:** 2026-08-06

## What shipped

- `desktop/window.py`: monospace `time_label` immediately left of the frame slider.
- Format: `0:08.42 / 0:16.20` via `format_timecode` / `format_time_readout`.
- Empty / no video: `--:--.-- / --:--.--`.
- Updated from `update_playhead` **before** the chart/floor throttle so the clock tracks every painted frame without forcing redraws.
- Duration uses `_duration_s` (video `frame_count/fps`, may extend slightly after analyze — same as timeline).

## How to test

1. Open a video — label shows `0:00.00 / …`.
2. Play — left side advances; chart/floor still ~10 Hz.
3. Seek via slider, chart click, or occurrence — label jumps immediately.
4. Close / failed open — placeholder returns.
