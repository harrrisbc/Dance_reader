# WORK ORDER — pose normalizer is numerically broken (measured on real project)

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06
**Severity:** P0 — this invalidates motif matching, 搵相似, and the Reach / Speed / Travel charts.
**Related:** [AGENT_HANDOFF_MOTIF_THRESHOLDS.md](./AGENT_HANDOFF_MOTIF_THRESHOLDS.md) — done, but the thresholds cannot work until this is fixed.

## Evidence

Measured on a real user project: `07 走路+武拜廟.dance.json`
(16.2 s, 60 fps, 1 dancer, 969 poses, 1920×1080).

The user reported three frames of the same move (~8 s, ~10 s, ~12 s) that are not grouped.
`detect_motif_types()` returns **0 types**, and every one of the 78 segment pairs fails:

| Pairwise similarity over all 13 segments | |
|---|---|
| median | 0.100 |
| p95 | 0.324 |
| **max** | **0.527** |
| `SIMILARITY_MIN` | 0.55 |

Max similarity in the entire clip is below the threshold, so zero is the only possible outcome.
The three user-flagged segments:

| Pair | sim | cos | DTW dist |
|------|-----|-----|----------|
| 8 s vs 10 s | 0.000 | 0.396 (prefilter rejected) | **12.06** |
| 8 s vs 12 s | 0.245 | 0.833 | 2.92 |
| 10 s vs 12 s | 0.000 | 0.547 (prefilter rejected) | **11.99** |

A DTW frame distance of 12 on torso-normalized features that should sit around 0.3 is not a
tuning problem. The features are numerically blown up.

## Root cause 1 — the divisor collapses and is never clamped

`pose_feature` / `_torso_frame` / `_reach_raw` all divide by **shoulder width in the image**.
When the dancer turns toward profile, the two shoulders project onto nearly the same point.

Measured shoulder width across the clip (normalized image units):

| | value |
|---|---|
| min | **0.00196** |
| p5 | 0.01292 |
| median | 0.04505 |
| max | 0.07860 |
| **max / min** | **40×** |

102 of 969 frames fall below 0.02; 31 fall below 0.01.

The guard is `scale = float(np.hypot(...)) or 1e-3`, which only triggers on an exact `0.0`.
A value of `0.00196` is used as-is, inflating every feature in that frame by roughly **85×**.

The stored Reach curve shows the damage directly:

| Reach (should be ~1–3 × body scale) | |
|---|---|
| median | 6.17 |
| p95 | 22.87 |
| **max** | **168.89** |

Worst spikes: 168.9 at t=4.70 s, 144.0 at t=9.57 s, 97.0 at t=7.15 s — all frames where shoulder
width nearly vanished. One exploded frame inside a phrase is enough to wreck its DTW distance
against every other phrase, which is exactly what the 12.06 above is.

**Torso height (shoulder-mid → hip-mid) over the same clip varies only 2.1× (0.085–0.179).**
It is the stable choice, confirmed on real data rather than assumed.

## Root cause 2 — X and Y are in different units (aspect-ratio distortion)

Keypoints are normalized as `x / frame_width`, `y / frame_height`. On 1920×1080 that compresses
every horizontal distance by **1.778×** relative to vertical ones, so the body shape is squashed
in feature space before anything else happens.

Measured torso-height / shoulder-width ratio on this clip:

| | ratio |
|---|---|
| as stored | **3.06** |
| after multiplying x by `frame_width / frame_height` | **1.76** |
| anatomically expected | ~1.2–1.6 |

The corrected number lands in the right neighbourhood; the raw one does not. This also means
horizontal motion is systematically under-weighted in matching relative to vertical motion.

## Root cause 3 — the Reach chart is fully clipped

Locked Y range is 1.0–3.5, but **100% of this clip's Reach values exceed 3.5** (median 6.17).
The Reach panel the user has been looking at is pinned to the top of its axis for the entire
video. Same class of problem for Speed and Travel, which are labelled "shoulders/s" and inherit
the same unstable divisor.

## Changes

### 1. Aspect-correct before anything else (P0)

Convert keypoints to a **square-pixel space** at the point of use: multiply the normalized `x` by
`frame_width / frame_height` (or normalize both axes by height). Do this once, centrally, so
`pose_feature`, `pose_feature_rich`, metrics, and the floor map all agree.

`AnalysisResult.frame_width` / `frame_height` already exist. For older projects missing them,
fall back to 16:9 and note it.

### 2. Normalize by torso height, with a hard floor (P0)

```
torso = |shoulder_mid - hip_mid|          # in square-pixel space
scale = max(torso, floor)                 # floor from a per-track percentile, not a constant
```

Prefer a per-track reference such as the **median torso height over the track**, so a few bad
frames cannot move the scale. Fall back to shoulder width only when hips are invisible, and clamp
that too.

Never divide by a raw per-frame measurement without a floor — that is the bug.

### 3. Mark degenerate frames instead of emitting garbage (P0)

When the torso is not measurable (low confidence, or scale below the floor), do not emit an
inflated value. Carry forward the previous valid value and flag the frame as low confidence, the
same treatment already applied to dropped detections for `level`.

Consider excluding phrases containing many degenerate frames from clustering entirely.

### 4. Re-derive units and chart ranges (P0)

After the fix, Reach is in **torso heights**, not shoulder widths. Recompute sensible locked
ranges from real data — do not keep 1.0–3.5 without checking. Relabel the axes: Speed and Travel
become `torso/s`.

Add a sanity assertion in the calibration script: if more than a few percent of frames fall
outside the locked chart range, fail loudly rather than silently clipping.

### 5. Re-run calibration and re-tune thresholds (P0)

The similarity distribution will change completely once features are sane. The current
0.55 / 0.45 / 0.50 values were tuned against broken features and are meaningless now.

Use the calibration script from the thresholds handoff on this exact project file. Target: the
~8 s, ~10 s and ~12 s phrases group together, while unrelated phrases do not.

`07 走路+武拜廟.dance.json` is a good regression fixture — it contains a repeated move, real
turning, and degenerate-shoulder frames.

### 6. Rotation tolerance (P1, only after the above)

Once the normalizer is stable, re-measure how much of the remaining gap is genuine out-of-plane
rotation. Synthetic testing suggested a ~30° turn costs roughly 0.13 of similarity even with a
sane scale. If real data confirms it, reduce the weight of the X component in the DTW frame
distance (start near 0.5 relative to Y). Do not zero X out — that destroys left/right
discrimination.

## Acceptance

- No Reach value above ~4 torso heights on this clip; no spikes above 20.
- Shoulder-width-driven explosions gone: no frame's features inflated by a near-zero divisor.
- `detect_motif_types()` returns at least one type on `07 走路+武拜廟.dance.json`.
- The ~8 s / ~10 s / ~12 s phrases land in the same type, or 搵相似 from one finds the others.
- Locked chart ranges contain the data; Reach panel is readable, not pinned to the top.
- Calibration output recorded in `docs/IMPLEMENTATION_NOTES_VIEWPOINT.md`, including the new
  thresholds and the measured similarity distribution.

## Out of scope

- 3D pose lifting / multi-view
- Changing the similarity metric shape
- Web review UI (retired)
