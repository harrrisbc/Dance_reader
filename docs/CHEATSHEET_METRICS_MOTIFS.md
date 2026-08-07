# Cheat sheet — how metrics & motifs are calculated

Quick reference for the **current** pipeline (`metrics.py` + `motifs.py`).  
粵語版：[CHEATSHEET_METRICS_MOTIFS_YUE.md](./CHEATSHEET_METRICS_MOTIFS_YUE.md)  
Units assume keypoints are first corrected to **square-pixel** space (`x' = x · W/H`).

---

## 0. Shared geometry (everything sits on this)

| Symbol | Meaning |
|--------|---------|
| Torso centre `(cx, cy)` | Midpoint of shoulders + hips (COCO 5,6,11,12) |
| Scale `S` | Torso height (shoulder mid → hip mid), with a **floor** from the track median so near-zero never blows features |
| Visible | Keypoint confidence ≥ `TORSO_CONF` (0.2); occluded joints are **masked out**, not written as (0,0) |

Pose feature for matching: joint XY relative to centre, ÷ `S`, plus unit bone vectors (rich feature). Mirror = left↔right swap of the same feature.

---

## 1. Per-frame curves (charts)

Computed in `compute_metrics` → `MetricPoint`. Energy & speed are **Savitzky–Golay** smoothed (~83 ms) before charts **and** before phrase cutting (same curve).

| Field | Chart label | Formula (plain language) | Unit on chart |
|-------|-------------|--------------------------|---------------|
| `energy` | Energy | Sum of `(joint velocity / S)²` over limb joints (`KEY_INDICES`), then ÷ that dancer’s **own peak** | 0–1 relative |
| `speed` | Speed | Mean `\|v\| / S` over the same joints | torso heights / s |
| `reach` | Reach | Max distance of head / wrists / ankles from torso centre, ÷ `S` | × torso (guides ~1.2 closed / ~2.5 open) |
| `level` | Level | Mean `\|ankle_y − hip_y\|` ÷ this track’s **90th-percentile** leg length, clipped to 0–1 | stance ratio (*see note*) |
| `travel` | Travel | Foot-path length over ~**0.8 s** window, ÷ mean `S`, ÷ window seconds | torso heights / s |
| `directness` | Directness | `net displacement / path length` in that window; **blank (NaN)** if almost still | 0 = winding, 1 = straight |
| `pos_x` | L–R | Torso COM x in the frame | 0 = left, 1 = right |
| `pos_y` | F–B | Supporting (lower) ankle y in the frame | 0 = top/far, 1 = bottom/near |

**Travel / directness path** (foot track, aspect-corrected):

```text
path = Σ hypot(Δx·aspect, Δy)     over the rolling window
net  = hypot( end−start )
travel     = (path / S_window) / Δt
directness = net / path            if path ≥ min threshold, else blank
```

**Notes**

- Degenerate / missing poses **carry forward** last good reach / level / position (avoids spikes).
- `enrich_metrics` rebuilds all curves from stored poses when you open an old `.dance.json`.
- **Level is still** leg-length / p90. A redesign to `(supporting_ankle_y − hip_y) / torso_height` is pending (`AGENT_HANDOFF_LEVEL_DEFINITION.md`).
- Floor map uses `pos_x` / `pos_y` — **screen coords**, not stage metres.

---

## 2. Motifs — automatic types (“Repeated motifs”)

Pipeline: **phrase → feature → distance → cluster → keep**.

```text
poses
  → cut phrases at Energy valleys (true rests)
  → drop idle phrases (motion gate)
  → resample each phrase to 14 frames (+ visibility mask)
  → rich pose feature (+ mirrored twin)
  → pairwise distance (cosine prefilter, then mask-aware DTW)
  → complete-linkage clustering
  → keep types with enough occurrences and tight enough spread
```

### 2.1 Cut phrases

1. Build the same smoothed energy as the chart (before peak-normalize).
2. Find local minima that sit below nearby peak × `VALLEY_FRAC` (0.55) — real rests, not tiny wiggles.
3. Phrase length clamped roughly **0.65–2.8 s**; if valleys fail, fall back to sliding ~1.2 s windows.
4. **Motion gate:** keep only phrases with enough speed (`MOTION_SPEED_MIN` / relative to peak).

### 2.2 Represent a phrase

- Resample to `SEQ_FRAMES = 14` pose features (tempo is **discarded** at this step).
- Also store the **mirror** sequence (left/right same gesture).
- Padding frames are **zero-visibility** (not fake poses).

### 2.3 Distance between two phrases

```text
1. Cosine on bag embeddings; if both same & mirror < 0.55 → reject (huge distance)
2. Else DTW on frame features, taking the better of (A vs B) and (A vs mirror B)
3. Frame cost = RMS over jointly visible dims only
   - too few shared joints → cost ≥ median (occlusion must not be “cheap”)
4. Decision uses raw distance d  (lower = more similar)
```

UI **% match** is display-only: `score = clip(1 − d/0.50, 0, 1)` — **never** used as a gate.

### 2.4 Cluster → motif type

| Gate | Default | Role |
|------|---------|------|
| `CLUSTER_LINKAGE` | `complete` | Max pairwise link must clear the cut (stops A~B~C chains) |
| `CLUSTER_MAX_DIST` | **0.24** | Agglomerative cut |
| `INTRA_MAX_DIST` | **0.26** | Mean pairwise inside a kept type |
| `MOTIF_MIN_OCCURRENCES` | **3** | After NMS; short no-repeat clips can return **zero** types |

**Precision > recall** for auto types: better to miss a borderline repeat than merge “looks like” with “same move”. Split true repeats can still be found with Find Similar.

---

## 3. Find Similar (user-directed)

Different path from auto types on purpose.

1. Query = selected occurrence span, or a window centred on the playhead (**not** forced to a valley cut).
2. **Subsequence DTW** slides the query along the whole track (open begin/end).
3. Keep local minima with NMS; show tempo ratio `hit_dur / query_dur`.
4. Reject if tempo ratio outside **[~0.5, 2.0]** (same shape, different speed is OK within 2×).
5. Strong hits: `d ≤ 0.16`; softer extras up to `d ≤ 0.30`.

So: auto discovery is conservative; Find Similar is where recall lives.

---

## 4. One-page mental model

| You want… | Look at… | Rough meaning |
|-----------|----------|---------------|
| How hard the body is working | Energy | Peak-relative kinetic proxy |
| How fast limbs move | Speed | torso/s |
| How open the body is | Reach | × torso from centre |
| How high/low the stance is | Level | Leg extension vs this clip’s tall stance (*until redesign*) |
| How much they travel on the floor | Travel | Foot path / torso / s |
| Straight vs circling path | Directness | net / path |
| Where on screen | L–R / F–B + floor map | Image coords |
| “This phrase appears again” (catalogue) | Motif types | Clustered valley phrases |
| “Find this bit anywhere” | Find Similar | Subsequence DTW + tempo label |

---

## 5. Code map

| Step | File |
|------|------|
| Scale, features, curves | `backend/app/pipeline/metrics.py` |
| Phrases, DTW, cluster, Find Similar | `backend/app/pipeline/motifs.py` |
| Analyze wiring | `backend/app/pipeline/run.py` |
| Rebuild metrics on load | `backend/app/project_io.py` → `enrich_metrics` |
| Threshold calibration | `backend/scripts/calibrate_motifs.py` |

Deeper changelogs: `docs/IMPLEMENTATION_NOTES_MATCH_W*.md`, `docs/IMPLEMENTATION_NOTES_MOTIF_OVERMERGE.md`.
