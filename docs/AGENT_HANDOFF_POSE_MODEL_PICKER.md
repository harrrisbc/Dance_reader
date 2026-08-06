# WORK ORDER — Analyze model picker (YOLO S / M)

**To:** programming agent  
**From:** Design agent  
**Date:** 2026-08-06

## Why

Default is `yolo11s-pose.pt` — fine for speed. For clips with fans, occlusion, or strong
profile turns, `yolo11m-pose.pt` can hold wrists/elbows more stably. The user wants a
visible choice at analyze time; do not force M for everyone (slower, larger download).

This is **not** a fix for the Reach explosion (already fixed via torso normalizer). It is
optional quality for harder footage.

## Scope

| File | Change |
|------|--------|
| `backend/app/pipeline/pose_track.py` | Keep `DEFAULT_MODEL = "yolo11s-pose.pt"`; add `POSE_MODELS` map |
| `desktop/window.py` | Combo next to Max dancers; pass into `AnalyzeWorker` / `analyze_video` |
| `backend/app/project_io.py` | Optional: store `pose_model` on `DanceProject` so reopen remembers choice |
| `README.md` / short note | One line: S default, M for hard clips |

API (`AnalyzePathBody.model`) already accepts a string — wire desktop to the same values.

## Changes

### 1. Allowed models (whitelist)

```python
POSE_MODELS = {
    "S": "yolo11s-pose.pt",  # default — balanced
    "M": "yolo11m-pose.pt",  # slower, better under occlusion / side view
}
```

Do **not** expose N / L / X in the UI this pass (keeps the control simple). Reject unknown
strings; fall back to S.

### 2. Qt UI

- Label: **Model**
- Control: `QComboBox` with items:
  - `S — fast (default)` → `yolo11s-pose.pt`
  - `M — more accurate` → `yolo11m-pose.pt`
- Tooltip:  
  `M is slower but often better when fans or side views hide joints. Does not replace re-opening projects after the torso-scale fix.`
- Place near **Max dancers** on the top bar.
- Default selection: **S**.

### 3. Analyze path

`AnalyzeWorker(..., model_name=...)` → `analyze_video(..., model_name=...)`.  
First run of M will download weights via Ultralytics if missing — status text may show
`Loading model` longer; that is expected.

### 4. Persist (nice-to-have, same PR if cheap)

`DanceProject.pose_model: str | None = None` — save the weight filename used. On load, set the
combo from it. Old projects without the field → S.

### 5. Status after analyze

Include model in the done line, e.g.  
`Done: 1 dancer, 4 motif types (yolo11m-pose) — unsaved`

## Acceptance

- Default analyze still uses `yolo11s-pose.pt` with no user action.
- Selecting M and Analyze runs with `yolo11m-pose.pt` (check status / logs).
- Switching back to S works.
- Invalid / empty model cannot crash analyze (falls back to S).
- English UI strings only.

## Out of scope

- Auto-picking M from GPU VRAM
- Batch A/B compare UI
- Changing motif thresholds when model changes
- Web review UI
