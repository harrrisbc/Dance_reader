# Fluent review UI notes

See `docs/AGENT_HANDOFF_FLUENT_UI.md`.

## Shipped

### Phase A (Qt)
- Playback worker (prior handoff)
- Chart series baked; playhead blit ≤10 Hz while playing; floor map skipped during play

### Phase B (default entry)
- `python -m desktop` → FastAPI + browser `/review`
- uPlot stacked panels: click seek, wheel zoom, brush zoom, linked X, fixed Y units
- HTML `<video>` + `/api/media` Range streaming
- Floor map canvas; Traditional Chinese UI
- Analyze remains Python/CUDA (`/api/health` / badge)

### Qt fallback
`python -m desktop --qt`
