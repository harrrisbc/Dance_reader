# Implementation notes — W2 feature correctness

**Handoff:** [AGENT_HANDOFF_MATCH_W2_FEATURES.md](./AGENT_HANDOFF_MATCH_W2_FEATURES.md)  
**Date:** 2026-08-06  
**Depends on:** W1 tooling (fixtures + fast DTW + distance reporting)

## Expected direction (stated before measuring)

Per standing lesson 1:

| Change | Expected effect |
|--------|-----------------|
| A1 mask-aware DTW (RMS / visible dims) | Raw distances shrink ~√D vs full L2; occluded frames stop looking like “arms at torso”; padded frames stop giving free alignment |
| A1 low-vis imputation ≥ median | Warping paths avoid low-visibility cells when a well-observed route exists |
| A2 scale-normalized energy + KEY_INDICES | Walking-toward-camera no longer inflates energy; valleys may move; fewer/different phrases |
| A3 Savitzky–Golay (~83 ms) | Speed max drops toward p95; accents preserved better than a boxcar |
| A4 drop reach/level penalty | Distances change slightly; if separation collapses *only* from this, reintroduce with derived weights — otherwise leave out |
| Motion gate on torso/s speed | Gate may drop more idle phrases once motion is truly in torso/s |

**Thresholds not tuned in this wave.** Over-merge under current `1/(1+d)` gates is expected if distances compress; Wave 3 owns the cuts.

## What shipped

### A1 — occlusion masks
- `pose_feature` / `pose_feature_rich` return `(feat, mask)`; occluded joints/bones are `mask=False` (coords left 0, never counted as evidence).
- `Segment` carries `seq_mask` / `seq_mirror_mask`.
- `_sample_sequence` pads with **zero-visibility** frames (no last-frame repeat as evidence).
- Mask-aware frame cost = RMS over jointly-visible dims; if shared KEY joints &lt; `MIN_SHARED_VISIBLE_JOINTS` (4), cost = `max(partial, median_good)`.
- `segment_path_visibility()` for probe diagnostics.

### A2 — energy
- `limb_energy_and_speed()`: KEY_INDICES only; `(v/scale)²` sum. Charts still peak-normalize per actor; comment notes absolute torso²/s² is available for future cross-clip use.

### A3 — smoothing
- `smooth_series()` Savitzky–Golay, window ≈ 83 ms scaled by fps (odd, ≥5).
- Applied identically in `compute_metrics` (display) and `_energy_series` (segmenter).

### A4 — penalty removed
- `segment_similarity` is cosine gate + raw mask-aware DTW only. Reach/level means kept on `Segment` for diagnostics but unused in matching.

### Tests
- `backend/tests/test_features_w2.py` — mask visibility, low-vis ≥ median, KEY_INDICES energy, pad mask.
- W1 unmasked DTW identity tests still pass.

## Calibration results (thresholds untouched)

### Before (W1 baseline, positive fixture)

| | |
|--|--|
| Phrases | raw=15 after_motion=15 (drop=0) |
| Pairwise sim | min 0.324 · median 0.396 · max 0.774 |
| Pairwise raw DTW | min 0.284 · median 1.488 · max 2.069 |
| detect_motif_types | **4** types |
| Probe 8/10/12 | sim 0.412 / 0.419 / 0.338 · dtw 1.40 / 1.37 / 1.95 |

### After (W2, positive)

| | |
|--|--|
| Phrases | raw=**8** after_motion=**8** (drop=0) — valleys moved; gate unchanged |
| Speed | median 1.58 · p95 3.60 · max **6.23** (was max 6.75) |
| Pairwise sim | min 0.760 · median 0.802 · max 0.846 |
| Pairwise raw DTW | min 0.182 · median 0.248 · max 0.317 |
| detect_motif_types | **1** type (8 occ) — current gates over-merge |
| Probe near 8/10/12 | sim 0.788 / 0.785 / 0.823 · dtw 0.269 / 0.275 / 0.215 |
| Path low-vis | **0%** on all three probe pairs (len=14) |

### Negative `hi.dance.json`

| | Before W2 | After W2 |
|--|-----------|----------|
| Phrases | 4→3 after motion | **2** after motion (drop=0) |
| Types @ current gates | 1 | 1 |
| Pairwise | 3 pairs, sim median ~0.64 | 1 pair, sim **0.919**, dtw **0.088** |

Motion gate: **no change in drop count** on either fixture (0). Phrase *counts* fell because energy valleys changed (A2/A3), not because the gate tightened.

### Occlusion path check
Probe warping paths show `path_low_vis=0%`. Low-visibility imputation is not attracting the path on this footage.

### A4 verdict
Did **not** reintroduce reach/level weights. The dominant distribution shift is the RMS-per-dim distance scale (≈1/√D vs old L2), not the missing ~0.08/0.15 penalties. Separation for “same phrase vs unrelated” must be rebuilt in Wave 3 distance units against both fixtures — absolute thresholds can now be meaningful because the distance is dimensionless mean-per-dim deviation.

## Wave 3 handoff note
Current `SIMILARITY_MIN=0.40` / `CLUSTER_DISTANCE=0.50` / `INTRA_SIM_MIN=0.45` are meaningless on the new scale (everything clears them). Sweep in **raw DTW distance**; require the negative fixture to be able to return **zero** types.

## Out of scope (as ordered)
W3 thresholds, W4 subsequence, `level` redefinition, `SEQ_FRAMES` raise — not started.
