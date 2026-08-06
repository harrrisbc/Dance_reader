# Playback smoothness (AGENT_HANDOFF_PLAYBACK)

## Shipped

- `desktop/playback.py` — `VideoDecodeWorker` owns OpenCV decode off the UI thread
- Wall-clock playback with frame skip (max 8 sequential reads, then seek)
- Chart / floor playhead updates throttled to ~10 Hz while playing; full catch-up on pause/seek
- While playing: downscale before overlay; skeleton every other frame; selected actor only
- Stay on PySide6 — no framework change

## Run

```powershell
.\backend\.venv\Scripts\python.exe -m desktop
```
