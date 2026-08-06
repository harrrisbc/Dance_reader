# Implementation notes — viewpoint-robust pose normalizer

## Fixture

`backend/uploads/review_project_07 走路+武拜廟.dance.json`  
(16.2 s, 1920×1080, 1 dancer; repeated move near 8 s / 10.3 s / 12 s)

## Root cause (fixed)

1. **Shoulder-width divisor** collapsed on profile turns (min ≈ 0.002) → Reach spikes to 168.
2. **Aspect distortion**: `x/W`, `y/H` made horizontal distances ~1.78× too small on 16:9.
3. Chart Y locks (Reach 1–3.5) clipped **100%** of the broken Reach curve.

## Fix

- Aspect-correct into square-pixel space: `x' = x * (W/H)` (`metrics.to_square_pixels`).
- Scale by **torso height** `|shoulder_mid − hip_mid|`, floored by `0.5 × median torso` over the track.
- Degenerate frames: carry forward last valid metrics; motif phrases with &gt;35% missing features are dropped.
- Units: Speed / Travel = **torso/s**; Reach = **× torso**. Locked ranges: Reach 0.5–4, Speed 0–7, Travel 0–2.5. Reach guides at 1.2 / 2.5.

## Calibration after fix (same fixture)

| Metric | Value |
|--------|-------|
| Reach min / median / max | 1.63 / 2.01 / **2.21** (0% outside 0.5–4) |
| Pairwise sim median / p95 / max | 0.396 / 0.655 / 0.774 |
| Probe 8 s ↔ 10.3 s | **0.646** |
| Probe 8 s ↔ 12 s | 0.419 |
| `detect_motif_types` | **4 types**; Type C includes 8.0–8.5 and 10.0–10.6 |

## Retuned thresholds (DTW scale, sane features)

| Constant | Value |
|----------|-------|
| `SIMILARITY_MIN` | **0.40** |
| `CLUSTER_DISTANCE` | **0.50** (sim ≥ ~0.50) |
| `INTRA_SIM_MIN` | **0.45** |
| `COSINE_PREFILTER` | 0.55 (reject → 0.0) |

Re-open / re-analyze projects so `enrich_metrics` recomputes curves. Old saved Reach spikes are overwritten on load.

```powershell
.\backend\.venv\Scripts\python.exe backend\scripts\calibrate_motifs.py path\to\file.dance.json
```
